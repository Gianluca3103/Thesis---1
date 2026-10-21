"""Compare five- and 20-scan meta students on identical voxel evaluation sites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import decode_keys, encode_keys
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.meta_cache import (
    MetaOccupancyCache, MetaOccupancyDataset, collate_meta_cache,
)
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    candidate_support, scan_overlap_counts_observed, scan_overlap_rates,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.train_meta_occupancy import load_meta_student
from scripts.sweep_meta_occupancy_thresholds import (
    CSV_COLUMNS, DEFAULT_THRESHOLDS, _check_manifest, _ratio,
)


class PairedFrames:
    def __init__(self, five: MetaOccupancyDataset, twenty: MetaOccupancyDataset):
        if [frame.frame_id for frame in five.frames] != [frame.frame_id for frame in twenty.frames]:
            raise ValueError("Five- and 20-scan validation frames differ")
        self.five, self.twenty = five, twenty

    def __len__(self):
        return len(self.five)

    def __getitem__(self, index):
        return self.five[index], self.twenty[index]


def _collate(rows):
    five, twenty = zip(*rows)
    return collate_meta_cache(list(five)), collate_meta_cache(list(twenty))


def _union_labels(five_coords, twenty_coords, five, twenty):
    keys5 = encode_keys(five_coords, GRID.shape_zyx).numpy()
    keys20 = encode_keys(twenty_coords, GRID.shape_zyx).numpy()
    if len(np.unique(keys5)) != len(keys5) or len(np.unique(keys20)) != len(keys20):
        raise ValueError("Decoder emitted duplicate voxel coordinates")
    common, i5, i20 = np.intersect1d(keys5, keys20, return_indices=True,
                                     assume_unique=True)
    for name in ("occupied", "known_free"):
        if len(common) and not np.array_equal(five[name].numpy()[i5],
                                              twenty[name].numpy()[i20]):
            raise ValueError(f"Five- and 20-scan caches disagree on {name} labels")
    keys = np.union1d(keys5, keys20)
    sites = decode_keys(torch.from_numpy(keys).long(), GRID.shape_zyx)
    index5, index20 = np.searchsorted(keys, keys5), np.searchsorted(keys, keys20)
    occupied = np.zeros(len(keys), dtype=np.bool_)
    known_free = np.zeros(len(keys), dtype=np.bool_)
    occupied[index5], occupied[index20] = five["occupied"].numpy(), twenty["occupied"].numpy()
    known_free[index5], known_free[index20] = (five["known_free"].numpy(),
                                               twenty["known_free"].numpy())
    if np.any(occupied & known_free):
        raise ValueError("An evaluation voxel is both occupied and known free")
    return sites, index5, index20, occupied, known_free


def _totals(thresholds):
    return [{"threshold": threshold, "predicted": 0, "observed": 0,
             "tp_exact": 0, "tp_at_0_2m": 0, "known_intersection": 0,
             "known_union": 0, "missed_occupied_voxels": 0}
            for threshold in thresholds]


@torch.no_grad()
def _evaluate(five_model, twenty_model, loader, thresholds, radius_m, device):
    totals = {"official5": _totals(thresholds), "processed20": _totals(thresholds)}
    for five, twenty in tqdm(loader, desc="Common-support sweep", unit="frame"):
        if (five["frame_id"] != twenty["frame_id"] or
                not torch.equal(five["clean_lidar"], twenty["clean_lidar"])):
            raise ValueError("Paired clean LiDAR or frame IDs differ")
        outputs = {}
        for name, model, batch in (("official5", five_model, five),
                                   ("processed20", twenty_model, twenty)):
            output = model(batch["radar"].to(device, non_blocking=True),
                           batch["radar_valid"].to(device, non_blocking=True))
            coords = output.confidence_logits.coords.cpu()
            if not torch.equal(coords, batch["candidate_coords"].long()):
                raise ValueError(f"Cached candidates differ from {name} decoder")
            scores = output.confidence_logits.features[:, 0].sigmoid().cpu().numpy()
            outputs[name] = coords, scores
            del output
        sites, index5, index20, occupied, known_free = _union_labels(
            outputs["official5"][0], outputs["processed20"][0], five, twenty)
        support = (candidate_support(sites, five["radar"], five["radar_valid"],
                                     radius_m).numpy()
                   & candidate_support(sites, twenty["radar"], twenty["radar_valid"],
                                       radius_m).numpy())
        observed = five["observed_coords"]
        observed = observed[candidate_support(observed, twenty["radar"],
                                              twenty["radar_valid"], radius_m)]
        known = support & (occupied | known_free)
        missed = max(len(observed) - int(np.count_nonzero(support & occupied)), 0)
        for name, indices in (("official5", index5), ("processed20", index20)):
            scores = np.full(len(sites), -np.inf, dtype=np.float32)
            scores[indices] = outputs[name][1]
            for row in totals[name]:
                selected = support & (scores >= row["threshold"])
                intersection = int(np.count_nonzero(selected & occupied))
                union = int(np.count_nonzero((selected | occupied) & known))
                scan = scan_overlap_counts_observed(
                    sites[torch.from_numpy(selected)], observed)
                if scan["tp_exact"] != intersection:
                    raise ValueError("Common-domain labels disagree with exact scan matching")
                row["predicted"] += scan["predicted"]
                row["observed"] += scan["observed"]
                row["tp_exact"] += scan["tp_exact"]
                row["tp_at_0_2m"] += scan["tp_at_0_2m"]
                row["known_intersection"] += intersection
                row["known_union"] += union
                row["missed_occupied_voxels"] += missed
    for rows in totals.values():
        for row in rows:
            row["iou_known_supported"] = _ratio(row["known_intersection"],
                                                row["known_union"])
            row["iou_full_support"] = _ratio(
                row["known_intersection"],
                row["known_union"] + row["missed_occupied_voxels"])
            row.update(scan_overlap_rates(row))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--five-checkpoint", type=Path, required=True)
    parser.add_argument("--twenty-checkpoint", type=Path, required=True)
    parser.add_argument("--five-data-cache-root", type=Path, required=True)
    parser.add_argument("--twenty-data-cache-root", type=Path, required=True)
    parser.add_argument("--five-meta-cache-root", type=Path, required=True)
    parser.add_argument("--twenty-meta-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.num_workers < 0 or args.limit is not None and args.limit < 1
            or not args.thresholds or any(not 0 < value < 1 for value in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds)):
        parser.error("Invalid workers, limit, or thresholds")
    models = {}
    joined = {}
    radius = None
    for name, checkpoint, data_root, cache_root in (
            ("official5", args.five_checkpoint, args.five_data_cache_root,
             args.five_meta_cache_root),
            ("processed20", args.twenty_checkpoint, args.twenty_data_cache_root,
             args.twenty_meta_cache_root)):
        model, saved = load_meta_student(checkpoint, args.device)
        expected_variant = "radar_5frames" if name == "official5" else "radar_20frames_verified_doppler_radial"
        if saved["radar_variant"] != expected_variant:
            raise ValueError(f"{name} checkpoint uses {saved['radar_variant']}, expected {expected_variant}")
        points = paired_dataset(args.vod_root, args.split, saved["radar_variant"],
                                data_cache_root=data_root)
        manifest_path = cache_root / args.split / "manifest.json"
        manifest = _check_manifest(manifest_path, saved, points, args.split,
                                   args.vod_root, "meta_occupancy")
        if radius is not None and manifest["radius_m"] != radius:
            raise ValueError("Five- and 20-scan support radii differ")
        radius = manifest["radius_m"]
        cache = MetaOccupancyCache(cache_root / args.split, manifest)
        cache.check_complete()
        models[name] = model
        joined[name] = MetaOccupancyDataset(points, cache)
    paired = PairedFrames(joined["official5"], joined["processed20"])
    selected = Subset(paired, range(min(args.limit, len(paired)))) if args.limit else paired
    loader = DataLoader(selected, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate,
                        pin_memory=args.device.startswith("cuda"))
    thresholds = tuple(sorted(args.thresholds))
    totals = _evaluate(models["official5"], models["processed20"],
                       loader, thresholds, radius, args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {"split": args.split, "frames": len(selected), "radius_m": radius,
              "evaluation_region": "intersection of 3.2m radar support; union of decoder candidate voxels",
              "five_checkpoint": str(args.five_checkpoint.resolve()),
              "twenty_checkpoint": str(args.twenty_checkpoint.resolve()),
              "rows": totals}
    json_path = args.output_root / "common_support_sweep.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = args.output_root / "common_support_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", *CSV_COLUMNS))
        writer.writeheader()
        for name, rows in totals.items():
            writer.writerows({"model": name, **{key: row[key] for key in CSV_COLUMNS}}
                             for row in rows)
    for metric in ("iou_known_supported", "iou_full_support", "iou_exact", "iou_at_0_2m"):
        best = {name: max(rows, key=lambda row: row[metric]
                          if row[metric] is not None else -1)
                for name, rows in totals.items()}
        left, right = best["official5"], best["processed20"]
        if left[metric] is None or right[metric] is None:
            print(f"{metric}: undefined for at least one model", flush=True)
        else:
            print(f"{metric}: official5={left[metric]:.6f} @ {left['threshold']:.2f}; "
                  f"processed20={right[metric]:.6f} @ {right['threshold']:.2f}; "
                  f"delta(5-20)={left[metric] - right[metric]:+.6f}", flush=True)
        if 0.5 in thresholds:
            fixed5 = next(row[metric] for row in totals["official5"]
                          if row["threshold"] == 0.5)
            fixed20 = next(row[metric] for row in totals["processed20"]
                           if row["threshold"] == 0.5)
            if fixed5 is not None and fixed20 is not None:
                print(f"  at 0.50: official5={fixed5:.6f}; "
                      f"processed20={fixed20:.6f}; delta={fixed5 - fixed20:+.6f}",
                      flush=True)
    print(f"Saved {json_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
