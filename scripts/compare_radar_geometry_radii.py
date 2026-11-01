"""Geometry-only Stage-I baselines for three radar voxels within 1.6 or 3.2 m.

Every supported Stage-II cell is kept. Clean occupancy and missing-cell
addition metrics are reported separately. Conditional eligible recall is 100%
by construction; full-target recall exposes the actual geometric ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    missing_target, quantize)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import (
    unique_radar_voxel_centers)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FOG2_TYPES, MILD_FOG2_VERSION)
from models.radar_lidar_stage2.sparse_geometry import GRID


RADII_M = (1.6, 3.2)
REQUIRED_RADAR_VOXELS = 3
SHAPE = GRID.shape_zyx


def _all_centers() -> np.ndarray:
    flat = np.arange(int(np.prod(SHAPE)), dtype=np.int64)
    z, rem = divmod(flat, SHAPE[1] * SHAPE[2])
    y, x = divmod(rem, SHAPE[2])
    xyz = np.stack((x, y, z), axis=1)
    return (np.asarray(GRID.minimum_xyz, dtype=np.float64)
            + (xyz + 0.5) * np.asarray(GRID.voxel_xyz, dtype=np.float64))


CENTERS = _all_centers()


def third_nearest_distance(radar) -> np.ndarray:
    """Metres to the third distinct radar voxel centre for every ROI cell."""
    radar_centers = unique_radar_voxel_centers(radar)
    if len(radar_centers) < REQUIRED_RADAR_VOXELS:
        return np.full(SHAPE, np.inf, dtype=np.float64)
    tree = cKDTree(radar_centers)
    distances = np.empty(len(CENTERS), dtype=np.float64)
    for start in range(0, len(CENTERS), 65536):
        stop = min(start + 65536, len(CENTERS))
        nearest, _ = tree.query(CENTERS[start:stop],
                                k=REQUIRED_RADAR_VOXELS,
                                distance_upper_bound=max(RADII_M) + 1e-9,
                                workers=1)
        distances[start:stop] = nearest[:, -1]
    return distances.reshape(SHAPE)


def _inside(mask: np.ndarray, coords) -> int:
    if not len(coords):
        return 0
    rows = coords.detach().cpu().numpy()
    return int(mask[rows[:, 0], rows[:, 1], rows[:, 2]].sum())


def _empty_counts() -> dict[str, int]:
    return {"frames": 0, "radar_supported_cells": 0,
            "clean_lidar_voxels": 0, "supported_clean_lidar_voxels": 0,
            "missing_lidar_voxels": 0, "supported_missing_lidar_voxels": 0,
            "addition_candidates": 0}


def _summarize(counts: dict[str, int], *, has_faulty: bool) -> dict:
    clean = counts["clean_lidar_voxels"]
    supported_clean = counts["supported_clean_lidar_voxels"]
    total = counts["radar_supported_cells"]
    result = {**counts,
              "clean_occupancy": {
                  "tp": supported_clean, "selected": total, "target": clean,
                  "precision": supported_clean / total if total else 0.0,
                  "full_recall": supported_clean / clean if clean else 1.0,
                  "eligible_recall": 1.0,
              },
              "mean_supported_cells_per_frame": total / max(counts["frames"], 1)}
    if has_faulty:
        missing = counts["missing_lidar_voxels"]
        eligible = counts["supported_missing_lidar_voxels"]
        candidates = counts["addition_candidates"]
        result["missing_additions"] = {
            "tp": eligible, "selected": candidates, "target": missing,
            "precision": eligible / candidates if candidates else 0.0,
            "full_recall": eligible / missing if missing else 1.0,
            "eligible_recall": 1.0,
            "eligible_target": eligible,
            "mean_addition_candidates_per_frame": (
                candidates / max(counts["frames"], 1)),
        }
    return result


def _one(rows: list[dict]) -> dict:
    return rows[0]


def _selected_indices(dataset, match_report: Path | None,
                      limit: int | None) -> list[int]:
    if match_report is None:
        return list(range(min(len(dataset), limit or len(dataset))))
    report = json.loads(match_report.read_text(encoding="utf-8"))
    sampled = report["sampled_frame_ids"]
    heldout = sampled[int(report["calibration_frames"]):]
    ids = {frame.frame_id: i for i, frame in enumerate(dataset.frames)}
    if not heldout or len(set(heldout)) != len(heldout) or set(heldout) - set(ids):
        raise ValueError("Matched report has missing or duplicate held-out frame IDs")
    return [ids[frame_id] for frame_id in heldout[:limit]]


def evaluate(dataset, indices: list[int], faulty_root: Path | None,
             frame_faults: dict[str, str] | None) -> dict:
    groups = {name: {radius: _empty_counts() for radius in RADII_M}
              for name in (("all", *MILD_FOG2_TYPES) if faulty_root else ("all",))}
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=_one, num_workers=0)
    for row in tqdm(loader, desc="Geometry-only radar comparison", unit="frame"):
        frame_id = row["frame_id"]
        faulty = (load_faulty(faulty_root, "val", frame_id)
                  if faulty_root else None)
        clean = quantize(row["clean_lidar"][:, :3], 1)
        missing = (missing_target(row["clean_lidar"][:, :3], faulty)
                   if faulty is not None else None)
        observed = quantize(faulty, 1) if faulty is not None else None
        distance = third_nearest_distance(row["radar"])
        for radius in RADII_M:
            mask = distance <= radius + 1e-9
            supported = int(mask.sum())
            supported_clean = _inside(mask, clean)
            supported_missing = _inside(mask, missing) if missing is not None else 0
            observed_inside = _inside(mask, observed) if observed is not None else 0
            names = ("all", frame_faults[frame_id]) if frame_faults else ("all",)
            for name in names:
                count = groups[name][radius]
                count["frames"] += 1
                count["radar_supported_cells"] += supported
                count["clean_lidar_voxels"] += len(clean)
                count["supported_clean_lidar_voxels"] += supported_clean
                if faulty is not None:
                    count["missing_lidar_voxels"] += len(missing)
                    count["supported_missing_lidar_voxels"] += supported_missing
                    count["addition_candidates"] += supported - observed_inside
    return {name: {str(radius): _summarize(count, has_faulty=faulty_root is not None)
                   for radius, count in by_radius.items()}
            for name, by_radius in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--faulty-root", type=Path,
                        help="Mild-fog2 cache; enables missing-addition metrics")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--match-report", type=Path,
                        help="Evaluate the held-out frame IDs from a Stage-I sweep JSON")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    dataset = paired_dataset(args.vod_root, "val", args.radar_variant,
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_faults = None
    if args.faulty_root:
        manifest = json.loads((args.faulty_root / "val" / "manifest.json").read_text(
            encoding="utf-8"))
        if (manifest.get("version") != MILD_FOG2_VERSION or
                manifest.get("radar_variant") != args.radar_variant or
                manifest.get("frame_ids") != [f.frame_id for f in dataset.frames]):
            parser.error("Mild-fog2 cache differs from the validation dataset")
        frame_faults = manifest["frame_faults"]
    indices = _selected_indices(dataset, args.match_report, args.limit)
    counts = evaluate(dataset, indices, args.faulty_root, frame_faults)
    report = {"rule": "3 distinct radar voxel centres within radius",
              "radii_m": RADII_M,
              "voxel_size_xyz_m": GRID.voxel_xyz,
              "metric_scope": "Stage-II ROI; exact voxel matches",
              "frame_count": len(indices),
              "frame_ids": [dataset.frames[i].frame_id for i in indices],
              "matched_report": str(args.match_report) if args.match_report else None,
              "faulty_root": str(args.faulty_root) if args.faulty_root else None,
              "results": counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, radii in counts.items():
        for radius, row in radii.items():
            clean = row["clean_occupancy"]
            line = (f"{name} r={radius}m: clean P={clean['precision']:.2%}, "
                    f"full R={clean['full_recall']:.2%}, "
                    f"eligible R={clean['eligible_recall']:.2%}")
            if "missing_additions" in row:
                added = row["missing_additions"]
                line += (f"; additions P={added['precision']:.2%}, "
                         f"full R={added['full_recall']:.2%}, "
                         f"eligible R={added['eligible_recall']:.2%}, "
                         f"mean cells/frame={added['mean_addition_candidates_per_frame']:,.0f}")
            print(line, flush=True)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
