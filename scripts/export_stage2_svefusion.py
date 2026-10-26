"""Export matched VoD validation LiDAR for a frozen SVEFusion detector.

Radar, calibration, labels and detector preprocessing remain in the detector's
existing VoD data root. Only LiDAR point files vary by condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar, load_vod_radar_to_lidar,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.detector_export import (
    detector_clouds, radar_sector_indices,
)
from models.radar_lidar_stage2.fault_data import (
    FAULT_NAMES, FAULT_PLAN, FAULT_POLICY_VERSION, SevereFogCache, severe_fault,
    severe_fov_filter,
)
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import (
    load_stage1, load_stage2, sha256, stage1_frame,
)


def _write_cloud(path: Path, points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f"Expected finite XYZI cloud: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        np.ascontiguousarray(points, dtype="<f4").tofile(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--fault-cache-root", type=Path, required=True)
    parser.add_argument("--stage2-cache-root", type=Path)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--fault-type", choices=("auto", *FAULT_NAMES), default="auto")
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--limit", type=int, help="Export first N frames for a pipeline check")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available()
            or not 0 < args.occupancy_threshold < 1
            or not 1 <= args.sampling_steps <= 1000
            or (args.limit is not None and args.limit < 1)):
        parser.error("CUDA, a valid occupancy threshold, and 1..1000 sampling steps are required")

    if args.stage2_cache_root:
        saved1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        stage1 = None
    else:
        stage1, saved1 = load_stage1(args.stage1_checkpoint, args.device,
                                      args.radar_variant)
    model, saved2 = load_stage2(args.stage2_checkpoint, args.device)
    if (saved1.get("radar_variant") != args.radar_variant
            or saved2.get("radar_variant") != args.radar_variant
            or saved2.get("stage1_sha256") != sha256(args.stage1_checkpoint)
            or saved2.get("fault_policy_version") != FAULT_POLICY_VERSION
            or tuple(tuple(item) for item in saved2.get("fault_plan", ())) != FAULT_PLAN):
        raise ValueError("Stage I, Stage II, radar variant or severe fault policy differ")

    dataset = paired_dataset(args.vod_root, "val", args.radar_variant,
                             data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    fog = SevereFogCache(args.fault_cache_root, "val", ids,
                         seed=int(saved2["fault_seed"]))
    cache = None
    if args.stage2_cache_root:
        expected = cache_identity(args.stage1_checkpoint, args.vod_root,
                                  args.radar_variant, float(saved1.get("radius_m", 3.2)),
                                  float(saved2["gate_threshold"]), "val", ids)
        cache = Stage2FixedCache(args.stage2_cache_root / "val", expected)
        cache.check_complete()

    identity = {
        "version": 1, "split": "val", "frame_ids": ids,
        "exporter_sha256": sha256(Path(__file__)),
        "cloud_helper_sha256": sha256(
            Path(__file__).resolve().parents[1]
            / "models/radar_lidar_stage2/detector_export.py"),
        "vod_root": str(args.vod_root.resolve()),
        "stage1_sha256": sha256(args.stage1_checkpoint),
        "stage2_sha256": sha256(args.stage2_checkpoint),
        "fog_manifest_sha256": sha256(args.fault_cache_root / "val" / "manifest.json"),
        "stage2_cache_manifest_sha256": (
            sha256(args.stage2_cache_root / "val" / "manifest.json")
            if args.stage2_cache_root else None),
        "radar_variant": args.radar_variant,
        "fault_type": args.fault_type,
        "fault_seed": int(saved2["fault_seed"]),
        "sampling_steps": args.sampling_steps,
        "occupancy_threshold": args.occupancy_threshold,
        "random_seed": args.random_seed,
        "gate_threshold": float(saved2["gate_threshold"]),
        "clean_lidar": "original full XYZI VoD scan",
        "faulty_lidar": "full-scan total/FoV faults; fog dropout in the cached radar sector",
        "completed_lidar": "faulty plus Stage-II missing-voxel points; generated intensity zero",
        "radar": "unmodified seven-channel five-frame VoD in SVEFusion",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_config = args.output_root / "run_config.json"
    if run_config.exists():
        if json.loads(run_config.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Existing export has different settings: {run_config}")
    else:
        _write_json(run_config, identity)

    alpha_bars = cosine_alpha_bars(device=args.device)
    count = min(len(dataset), args.limit or len(dataset))
    for index in range(count):
        frame = dataset.frames[index]
        frame_id = frame.frame_id
        faulty_path = args.output_root / "faulty" / f"{frame_id}.bin"
        completed_path = args.output_root / "reconstructed" / f"{frame_id}.bin"
        stat_path = args.output_root / "frames" / f"{frame_id}.json"
        if stat_path.is_file() and faulty_path.is_file() and completed_path.is_file():
            prior = json.loads(stat_path.read_text(encoding="utf-8"))
            if (faulty_path.stat().st_size == 16 * prior["faulty_full_points"]
                    and completed_path.stat().st_size == 16 * prior["completed_full_points"]):
                continue

        row = dataset[index]
        clean_full = load_vod_lidar(frame.lidar_path)
        lidar_from_radar = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path)
        sector = radar_sector_indices(clean_full, lidar_from_radar)
        clean_sector = row["clean_lidar"].numpy()
        if not np.array_equal(clean_full[sector], clean_sector):
            raise ValueError(f"Stage-II clean sector does not match raw scan: {frame_id}")
        faulty_sector, fault_name = severe_fault(
            row["clean_lidar"], frame_id, 0, fog,
            seed=int(saved2["fault_seed"]),
            fault_type=None if args.fault_type == "auto" else args.fault_type)

        if cache is None:
            output = stage1_frame(stage1, row["radar"].to(args.device),
                                  float(saved1.get("radius_m", 3.2)))
        else:
            output, _ = cache.load(frame_id, args.device)
        conditions = model.conditions(output, float(saved2["gate_threshold"]))
        if len(conditions.hierarchy[0]):
            seed = int.from_bytes(hashlib.sha256(
                f"{args.random_seed}:{frame_id}".encode()).digest()[:8], "little")
            torch.manual_seed(seed)
            prediction = model.sample(
                faulty_sector.to(args.device), conditions, alpha_bars,
                steps=args.sampling_steps,
                occupancy_threshold=args.occupancy_threshold,
                missing_only=True)
            additions = prediction["points_xyz"].cpu().numpy()
            allowed = prediction["allowed_cells"]
        else:
            additions = np.empty((0, 3), dtype=np.float32)
            allowed = 0
        full_retained = None
        if fault_name == "total_loss":
            full_retained = np.empty((0, 3), dtype=np.float32)
        elif fault_name == "fov_filter":
            full_retained = severe_fov_filter(clean_full)[:, :3]
        faulty, completed = detector_clouds(
            clean_full, sector, faulty_sector.numpy(), additions,
            retained_full_xyz=full_retained)
        _write_cloud(faulty_path, faulty)
        _write_cloud(completed_path, completed)
        _write_json(stat_path, {
            "frame_id": frame_id, "fault": fault_name,
            "raw_clean_points": len(clean_full), "sector_points": len(sector),
            "retained_sector_points": len(faulty_sector),
            "faulty_full_points": len(faulty),
            "allowed_cells": allowed, "reconstructed_points": len(additions),
            "completed_full_points": len(completed),
        })
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"Exported {index + 1}/{count} validation frames", flush=True)

    complete = all(
        (args.output_root / "frames" / f"{frame_id}.json").is_file()
        and (args.output_root / "faulty" / f"{frame_id}.bin").is_file()
        and (args.output_root / "reconstructed" / f"{frame_id}.bin").is_file()
        for frame_id in ids)
    _write_json(args.output_root / "export_manifest.json", {
        **identity, "exported_frames": count, "complete": complete,
        "clean_points_root": str(dataset.frames[0].lidar_path.parent.resolve()),
        "radar_points_root": str(dataset.frames[0].radar_path.parent.resolve()),
        "lidar_calibration_root": str(
            dataset.frames[0].lidar_calibration_path.parent.resolve()),
        "radar_calibration_root": str(
            dataset.frames[0].radar_calibration_path.parent.resolve()),
        "faulty_points_root": str((args.output_root / "faulty").resolve()),
        "reconstructed_points_root": str((args.output_root / "reconstructed").resolve()),
    })
    print(f"Export {'complete' if complete else 'partial'}: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
