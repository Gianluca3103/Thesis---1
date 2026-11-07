"""Score sampled occupied additions on all eligible mild-fault validation cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.eligible_validation import evaluate_all_eligible
from models.radar_lidar_stage2.fault_region import (
    FaultRegionCache, cache_identity as region_identity)
from models.radar_lidar_stage2.geometry_gate import RADIUS_M, VERSION
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import load_stage2, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available()
            or not 1 <= args.sampling_steps <= 1000
            or not 0 < args.occupancy_threshold < 1):
        parser.error("Evaluation requires CUDA and valid diffusion settings")

    model, saved = load_stage2(args.checkpoint, args.device)
    if (saved.get("geometry_gate_version") != VERSION
            or not saved.get("ray_offset_baseline")
            or saved.get("radius_m") != RADIUS_M):
        raise ValueError("Expected the 1.6 m geometry-only Stage II checkpoint")
    dataset = paired_dataset(
        args.vod_root, "val", saved["radar_variant"], include_clean=True,
        data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    manifest = json.loads((args.faulty_root / "val" / "manifest.json")
                          .read_text(encoding="utf-8"))
    if manifest.get("frame_ids") != frame_ids:
        raise ValueError("Mild-fault manifest frame order differs from validation")
    for key, path in (
            ("faulty_val_manifest_sha256", args.faulty_root / "val" / "manifest.json"),
            ("support_val_manifest_sha256", args.support_cache_root / "val" / "manifest.json"),
            ("region_val_manifest_sha256", args.fault_region_root / "val" / "manifest.json"),
            ("region_val_data_sha256", args.fault_region_root / "val" / "regions.json")):
        if saved.get(key) != sha256(path):
            raise ValueError(f"Checkpoint and validation cache differ: {path}")
    support = RadarSupportCache(
        args.support_cache_root / "val",
        support_identity("val", frame_ids, saved["radar_variant"],
                         data_cache_root=args.data_cache_root,
                         radius_m=RADIUS_M))
    support.check_complete()
    regions = FaultRegionCache(
        args.fault_region_root / "val",
        region_identity("val", frame_ids,
                        args.faulty_root / "val" / "manifest.json",
                        args.data_cache_root / "val" / "manifest.json"))
    if regions.regions is None:
        raise FileNotFoundError("Injector footprint cache is incomplete")
    report = evaluate_all_eligible(
        model, dataset, faulty_root=args.faulty_root, support_cache=support,
        region_cache=regions, fault_names=manifest["frame_faults"],
        alpha_bars=cosine_alpha_bars(device=args.device),
        gate_threshold=float(saved["gate_threshold"]),
        occupancy_threshold=args.occupancy_threshold,
        sampling_steps=args.sampling_steps, device=args.device)
    report.update({"checkpoint": str(args.checkpoint),
                   "checkpoint_epoch": int(saved["epoch"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Epoch {saved['epoch']}, {report['frames']} validation frames: "
          f"eligible occupancy IoU={report['eligible_iou']:.2%}, "
          f"IoU@0.2m={report['eligible_iou_0p2m']:.2%}; "
          f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
