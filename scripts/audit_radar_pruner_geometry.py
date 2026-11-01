"""Measure the exact radar rule's recall ceiling before pruning or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    missing_target, quantize, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import supported_coordinates
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FOG2_TYPES, MILD_FOG2_VERSION)


def _one(rows: list[dict]) -> dict:
    return rows[0]


def _rates(count: dict) -> dict:
    return {**count,
            "geometry_full_recall_ceiling": (count["eligible_missing"] /
                                             count["all_missing"]
                                             if count["all_missing"] else 1.0),
            "keep_all_addition_precision": (count["eligible_missing"] /
                                            count["addition_candidates"]
                                            if count["addition_candidates"] else 0.0),
            "mean_addition_candidates_per_frame": (
                count["addition_candidates"] / max(count["frames"], 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    cache = RadarSupportCache(args.support_cache_root / args.split,
                              identity(args.split, ids, args.radar_variant,
                                       data_cache_root=args.data_cache_root))
    cache.check_complete()
    manifest = json.loads((args.faulty_root / args.split / "manifest.json").read_text(
        encoding="utf-8"))
    if (manifest.get("version") != MILD_FOG2_VERSION or
            manifest.get("radar_variant") != args.radar_variant or
            manifest.get("frame_ids") != ids):
        parser.error("Mild-fog2 cache differs from the radar support dataset")
    groups = {name: {"frames": 0, "supported_cells": 0,
                     "addition_candidates": 0, "all_missing": 0,
                     "eligible_missing": 0}
              for name in ("all", *MILD_FOG2_TYPES)}
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=_one, num_workers=0)
    for index, row in enumerate(tqdm(loader, desc="Radar geometry audit", unit="frame")):
        if args.limit is not None and index >= args.limit:
            break
        frame_id = row["frame_id"]
        faulty = load_faulty(args.faulty_root, args.split, frame_id)
        missing = missing_target(row["clean_lidar"][:, :3], faulty)
        coords = supported_coordinates(cache.load(frame_id))
        observed = target_mask(coords, quantize(faulty, 1), 1)
        eligible = int(target_mask(coords, missing, 1).sum())
        for name in ("all", manifest["frame_faults"][frame_id]):
            group = groups[name]
            group["frames"] += 1
            group["supported_cells"] += len(coords)
            group["addition_candidates"] += len(coords) - int(observed.sum())
            group["all_missing"] += len(missing)
            group["eligible_missing"] += eligible
    report = {"geometry_rule": "3 distinct radar voxel centres within 3.0 m",
              "split": args.split,
              "fault_types": {name: _rates(group) for name, group in groups.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, group in report["fault_types"].items():
        print(f"{name}: {group['frames']} frames, "
              f"ceiling={group['geometry_full_recall_ceiling']:.2%}, "
              f"keep-all P={group['keep_all_addition_precision']:.2%}, "
              f"mean candidates/frame={group['mean_addition_candidates_per_frame']:,.0f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
