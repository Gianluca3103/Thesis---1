"""Evaluate a full-region LiDAR LPCR checkpoint with R2LDM geometric metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.lpcr_eval import (
    metric_voxel_centroids, point_cloud_metrics, reconstruct_full_region,
)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.stage2_io import load_stage2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--halo", type=int, default=3)
    parser.add_argument("--export-pairs", action="store_true",
                        help="Save gen/ and gt/ float32 XYZ .bin pairs for the official R2LDM evaluator")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1 or not 0 < args.occupancy_threshold < 1:
        parser.error("Use a positive limit and occupancy threshold in (0,1)")
    model, saved = load_stage2(args.lpcr_checkpoint, args.device, phase="lpcr")
    if not saved.get("lpcr_full_region"):
        raise ValueError("Checkpoint is not a full-region LPCR run")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    indices = random.Random(args.seed).sample(
        range(len(dataset)), min(args.limit, len(dataset)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.export_pairs:
        (args.output_root / "gen").mkdir(exist_ok=True)
        (args.output_root / "gt").mkdir(exist_ok=True)
    records = []
    lower = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    upper = np.asarray(GRID.maximum_xyz, dtype=np.float32)
    for index in tqdm(indices, desc="LPCR full-region evaluation", unit="frame"):
        row = dataset[index]
        clean = row["clean_lidar"][:, :3].to(args.device)
        predicted, visited = reconstruct_full_region(
            model, clean, threshold=args.occupancy_threshold,
            tile_size=args.tile_size, halo=args.halo)
        pred_xyz = predicted.numpy().astype(np.float32)
        clean_xyz = clean.cpu().numpy().astype(np.float32)
        clean_xyz = clean_xyz[((clean_xyz >= lower) & (clean_xyz < upper)).all(1)]
        metrics = point_cloud_metrics(metric_voxel_centroids(pred_xyz),
                                      metric_voxel_centroids(clean_xyz))
        records.append({"frame_id": row["frame_id"],
                        "candidate_cells": visited, **metrics})
        if args.export_pairs:
            pred_xyz.tofile(args.output_root / "gen" / f"{row['frame_id']}.bin")
            clean_xyz.tofile(args.output_root / "gt" / f"{row['frame_id']}.bin")
    fields = ("chamfer_squared_l2", "hausdorff_squared_l2", "precision",
              "recall", "fscore_5cm", "predicted_points", "target_points")
    mean = {key: (float(np.mean(values)) if values else None)
            for key in fields
            for values in ([row[key] for row in records if row[key] is not None],)}
    result = {"checkpoint": str(args.lpcr_checkpoint), "epoch": saved["epoch"],
              "split": args.split, "frames": len(records),
              "roi_xyz_m": [*GRID.minimum_xyz, *GRID.maximum_xyz],
              "metric_voxel_m": 0.125, "fscore_radius_m": 0.05,
              "occupancy_threshold": args.occupancy_threshold,
              "empty_predictions": sum(row["predicted_points"] == 0 for row in records),
              "mean": mean, "per_frame": records}
    (args.output_root / "metrics.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_frame"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
