"""Report how many gate-eligible clean cells a radar decoder could generate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.radar_lidar_crossmodal_v1.meta_cache import MetaOccupancyCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.meta_cache_root / args.split
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("support_policy") != "radar12vox_1p6m_stage2_roi_v1":
        raise ValueError("Expected geometry for the >=12 radar voxels / 1.6 m gate")
    cache = MetaOccupancyCache(root, manifest)
    selected = range(min(args.limit, len(cache))) if args.limit else range(len(cache))
    totals = {"eligible_clean_voxels": 0, "decoder_representable_clean_voxels": 0,
              "supported_decoder_cells": 0, "eligible_clean_points": 0,
              "decoder_representable_clean_points": 0}
    for index in selected:
        row = cache[index]
        totals["eligible_clean_voxels"] += len(row["observed_coords"])
        totals["decoder_representable_clean_voxels"] += int(
            (row["support"] & row["occupied"]).sum())
        totals["supported_decoder_cells"] += int(row["support"].sum())
        totals["eligible_clean_points"] += row["clean_supported_points"]
        totals["decoder_representable_clean_points"] += row[
            "represented_clean_points_supported"]
    count = len(selected)
    result = {"split": args.split, "frames": count, **totals,
              "voxel_coverage_ceiling": (
                  totals["decoder_representable_clean_voxels"] /
                  totals["eligible_clean_voxels"] if totals["eligible_clean_voxels"] else None),
              "all_supported_decoder_cells_exact_iou": (
                  totals["decoder_representable_clean_voxels"] /
                  (totals["supported_decoder_cells"] +
                   totals["eligible_clean_voxels"] -
                   totals["decoder_representable_clean_voxels"]))
              if (totals["supported_decoder_cells"] +
                  totals["eligible_clean_voxels"] -
                  totals["decoder_representable_clean_voxels"]) else None,
              "mean_supported_decoder_cells": totals["supported_decoder_cells"] / count
              if count else None}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
