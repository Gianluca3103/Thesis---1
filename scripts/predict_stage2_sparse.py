"""Reconstruct Stage-I-allowed LiDAR sites with radar/faulty-conditioned Stage II."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import load_faulty, synthetic_fault
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import (
    load_stage1, load_stage2, sha256, stage1_frame,
)


def _write_ply(path: Path, xyz: np.ndarray, probabilities: np.ndarray) -> None:
    colors = np.clip(probabilities * 255, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        for name in ("x", "y", "z"):
            stream.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            stream.write(f"property uchar {name}\n")
        stream.write("end_header\n")
        for point, color in zip(xyz, colors):
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                         f"{int(color)} {int(color)} {255-int(color)}\n")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path,
                        help="Pre-aligned N×4 float32 .bin files; omit for synthetic faults")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.limit < 1 or args.sampling_steps < 1 or
            not 0 < args.occupancy_threshold < 1 or
            args.split == "test" and args.faulty_root is None):
        parser.error("Inference requires CUDA and real faulty LiDAR for the test split")
    stage1, saved1 = load_stage1(args.stage1_checkpoint, args.device,
                                  args.radar_variant)
    model, saved2 = load_stage2(args.stage2_checkpoint, args.device)
    if (saved2["stage1_sha256"] != sha256(args.stage1_checkpoint)
            or saved2["radar_variant"] != args.radar_variant):
        raise ValueError("Stage-II checkpoint was trained with a different Stage I")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=args.faulty_root is None,
                             data_cache_root=(args.data_cache_root
                                              if args.faulty_root is None else None))
    indices = random.Random(args.random_seed).sample(
        range(len(dataset)), min(args.limit, len(dataset)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    alpha_bars = cosine_alpha_bars(device=args.device)
    summary = []
    for index in indices:
        row = dataset[index]
        radar = row["radar"].to(args.device)
        output = stage1_frame(stage1, radar, float(saved1.get("radius_m", 3.2)))
        conditions = model.conditions(output, float(saved2["gate_threshold"]))
        faulty = (load_faulty(args.faulty_root, args.split, row["frame_id"])
                  if args.faulty_root else synthetic_fault(
                      row["clean_lidar"], row["frame_id"], 0,
                      seed=int(saved2["fault_seed"]),
                      dropout=float(saved2["fault_dropout"]),
                      sector_deg=float(saved2["fault_sector_deg"])))
        faulty = faulty[:, :3].to(args.device)
        predicted = model.sample(faulty, conditions, alpha_bars,
                                 steps=args.sampling_steps,
                                 occupancy_threshold=args.occupancy_threshold)
        xyz = predicted["points_xyz"].cpu().numpy()
        scores = predicted["probabilities"].cpu().numpy()
        stem = args.output_root / f"{row['frame_id']}_stage2"
        _write_ply(stem.with_suffix(".ply"), xyz, scores)
        np.savez_compressed(stem.with_suffix(".npz"), xyz=xyz,
                            probability=scores,
                            coordinates=predicted["coordinates"].cpu().numpy())
        stats = {"frame_id": row["frame_id"],
                 "allowed_cells": predicted["allowed_cells"],
                 "reconstructed_points": predicted["predicted_cells"],
                 "faulty_input_points": len(faulty)}
        summary.append(stats)
        print(json.dumps(stats), flush=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
