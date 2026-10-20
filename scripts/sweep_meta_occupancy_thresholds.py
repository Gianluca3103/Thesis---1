"""Evaluate radar-only occupancy IoU across thresholds on one VoD split.

Each frame is decoded once. The existing version-3 geometry cache supplies
occupancy, known-free, radar support, and clean-scan voxel coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.meta_cache import (
    VERSION as CACHE_VERSION, MetaOccupancyCache, MetaOccupancyDataset,
    _source_hash, collate_meta_cache,
)
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    scan_overlap_counts_observed, scan_overlap_rates,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.train_meta_occupancy import load_meta_student


DEFAULT_THRESHOLDS = tuple(round(index / 10, 2) for index in range(1, 10))
CSV_COLUMNS = (
    "threshold", "iou_known_supported", "iou_full_support", "iou_exact",
    "iou_at_0_2m", "f1_exact", "f1_at_0_2m", "predicted", "observed",
    "tp_exact", "tp_at_0_2m", "known_intersection", "known_union",
    "missed_occupied_voxels",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _check_manifest(path: Path, saved: dict, dataset, split: str,
                    vod_root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    radius_m = saved.get("radius_m", saved.get("meta_config", {}).get("radius_m"))
    expected = {
        "version": CACHE_VERSION,
        "split": split,
        "frame_ids": [frame.frame_id for frame in dataset.frames],
        "vod_root": str(vod_root.resolve()),
        "radar_variant": saved["radar_variant"],
        "radar_initial_sha256": saved["radar_sha256"],
        "radius_m": radius_m,
        "source_hash": _source_hash(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Geometry cache {key} differs from checkpoint or split")
    if "lidar_sha256" in saved and manifest.get("lidar_initial_sha256") != saved["lidar_sha256"]:
        raise ValueError("Geometry cache LiDAR encoder differs from checkpoint")
    if "meta_config" in saved and manifest.get("max_candidates") != saved["meta_config"]["max_candidates"]:
        raise ValueError("Geometry cache neighborhood differs from checkpoint")
    return manifest


@torch.no_grad()
def _evaluate(model, loader, thresholds: tuple[float, ...], device: str) -> list[dict]:
    totals = [{
        "threshold": threshold, "predicted": 0, "observed": 0,
        "tp_exact": 0, "tp_at_0_2m": 0, "known_intersection": 0,
        "known_union": 0, "missed_occupied_voxels": 0,
    } for threshold in thresholds]
    for raw in tqdm(loader, desc="Threshold sweep", unit="batch"):
        radar = raw["radar"].to(device, non_blocking=True)
        radar_valid = raw["radar_valid"].to(device, non_blocking=True)
        output = model(radar, radar_valid)
        coordinates = output.confidence_logits.coords.cpu()
        if not torch.equal(coordinates, raw["candidate_coords"].long()):
            raise ValueError("Cached candidates differ from the live decoder")
        score = output.confidence_logits.features[:, 0].sigmoid().cpu().numpy()
        support = raw["support"].numpy()
        occupied = raw["occupied"].numpy()
        known_free = raw["known_free"].numpy()
        known = support & (occupied | known_free)
        observed = raw["observed_coords"]
        observed_count = len(observed)
        missed = max(observed_count - int(np.count_nonzero(support & occupied)), 0)
        for row in totals:
            selected = support & (score >= row["threshold"])
            intersection = int(np.count_nonzero(selected & occupied))
            known_union = int(np.count_nonzero((selected | occupied) & known))
            scan = scan_overlap_counts_observed(
                coordinates[torch.from_numpy(selected)], observed)
            if scan["tp_exact"] != intersection:
                raise ValueError("Cached occupancy and exact scan overlap disagree")
            row["predicted"] += scan["predicted"]
            row["observed"] += scan["observed"]
            row["tp_exact"] += scan["tp_exact"]
            row["tp_at_0_2m"] += scan["tp_at_0_2m"]
            row["known_intersection"] += intersection
            row["known_union"] += known_union
            row["missed_occupied_voxels"] += missed
    for row in totals:
        row["iou_known_supported"] = _ratio(
            row["known_intersection"], row["known_union"])
        row["iou_full_support"] = _ratio(
            row["known_intersection"],
            row["known_union"] + row["missed_occupied_voxels"])
        row.update(scan_overlap_rates(row))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--meta-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1 or args.batch_size < 1
            or args.num_workers < 0 or not args.thresholds
            or any(not 0 < value < 1 for value in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds)):
        parser.error("Use positive limits/batch size, nonnegative workers, and unique thresholds in (0,1)")
    thresholds = tuple(sorted(args.thresholds))
    model, saved = load_meta_student(args.checkpoint, args.device)
    points = paired_dataset(args.vod_root, args.split, saved["radar_variant"],
                            data_cache_root=args.data_cache_root)
    cache_root = args.meta_cache_root / args.split
    manifest = _check_manifest(cache_root / "manifest.json", saved, points,
                               args.split, args.vod_root)
    cache = MetaOccupancyCache(cache_root, manifest)
    cache.check_complete()
    joined = MetaOccupancyDataset(points, cache)
    selected = Subset(joined, range(min(args.limit, len(joined)))) if args.limit else joined
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_meta_cache,
                        pin_memory=args.device.startswith("cuda"))
    rows = _evaluate(model, loader, thresholds, args.device)
    best = {name: max(rows, key=lambda row: (row[name] if row[name] is not None else -1,
                                            row["threshold"]))["threshold"]
            for name in ("iou_known_supported", "iou_full_support", "iou_exact",
                         "iou_at_0_2m")}
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": saved["epoch"],
        "split": args.split, "frames": len(selected),
        "radar_variant": saved["radar_variant"],
        "radius_m": manifest["radius_m"], "best_thresholds": best, "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({name: row[name] for name in CSV_COLUMNS} for row in rows)
    json_path = args.output_root / "threshold_sweep.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("threshold  known IoU  full IoU  exact IoU  IoU@0.2m", flush=True)
    for row in rows:
        print(f"{row['threshold']:>9.2f}  {row['iou_known_supported'] or 0:>9.4f}  "
              f"{row['iou_full_support'] or 0:>8.4f}  {row['iou_exact'] or 0:>9.4f}  "
              f"{row['iou_at_0_2m'] or 0:>9.4f}", flush=True)
    print(f"Best thresholds: {best}", flush=True)
    print(f"Saved {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
