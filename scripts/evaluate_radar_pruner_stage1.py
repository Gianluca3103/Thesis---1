"""Calibrate a three-radar-voxel pruner on disjoint held-out frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    missing_target, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import load_radar_pruner
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FOG2_TYPES, MILD_FOG2_VERSION)


STEPS = 10000


def _one(rows: list[dict]) -> dict:
    return rows[0]


def _collect(model, dataset, indices, support_cache, faulty_root,
             device, frame_faults) -> dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=_one, num_workers=0)
    groups = {name: {"positive": np.zeros(STEPS + 1, dtype=np.int64),
                     "negative": np.zeros(STEPS + 1, dtype=np.int64),
                     "all_missing": 0, "eligible_missing": 0,
                     "candidates": 0, "frames": 0}
              for name in ("all", *MILD_FOG2_TYPES)}
    with torch.no_grad():
        for row in tqdm(loader, desc="Radar pruner held-out evaluation", unit="frame"):
            frame_id = row["frame_id"]
            faulty = load_faulty(faulty_root, "val", frame_id).to(device)
            clean = row["clean_lidar"][:, :3].to(device)
            radar = row["radar"].to(device)
            target = missing_target(clean, faulty)
            output = model(radar, faulty, support_cache.load(frame_id))
            positive = target_mask(output["coords"], target, 1)
            bin_index = torch.clamp(
                (output["logits"].float().sigmoid() * STEPS).long(), 0, STEPS)
            pos_bins = torch.bincount(bin_index[positive],
                                       minlength=STEPS + 1).cpu().numpy()
            neg_bins = torch.bincount(bin_index[~positive],
                                       minlength=STEPS + 1).cpu().numpy()
            for name in ("all", frame_faults[frame_id]):
                group = groups[name]
                group["positive"] += pos_bins
                group["negative"] += neg_bins
                group["all_missing"] += len(target)
                group["eligible_missing"] += int(positive.sum())
                group["candidates"] += len(output["coords"])
                group["frames"] += 1
    return groups


def _curve(group: dict) -> dict:
    tp = group["positive"][::-1].cumsum()[::-1]
    fp = group["negative"][::-1].cumsum()[::-1]
    selected = tp + fp
    precision = np.divide(tp, selected, out=np.zeros(len(tp), dtype=float),
                          where=selected > 0)
    eligible = max(group["eligible_missing"], 1)
    full = max(group["all_missing"], 1)
    return {"tp": tp, "selected": selected, "precision": precision,
            "eligible_recall": tp / eligible, "full_recall": tp / full}


def _point(curve: dict, index: int | None) -> dict | None:
    if index is None:
        return None
    return {"threshold": index / STEPS, "tp": int(curve["tp"][index]),
            "selected": int(curve["selected"][index]),
            "precision": float(curve["precision"][index]),
            "eligible_recall": float(curve["eligible_recall"][index]),
            "full_recall": float(curve["full_recall"][index])}


def _choose(curve: dict, objective: str, constraint: str,
            minimum: float) -> int | None:
    feasible = np.flatnonzero((curve[constraint] >= minimum) &
                              (curve["selected"] > 0))
    if not len(feasible):
        return None
    return int(max(feasible, key=lambda i: (curve[objective][i],
                                            curve["precision"][i], i)))


def _describe(group: dict, thresholds: dict[str, int | None] | None = None) -> dict:
    curve = _curve(group)
    result = {"frames": group["frames"], "all_missing": group["all_missing"],
              "eligible_missing": group["eligible_missing"],
              "geometry_full_recall_ceiling": (group["eligible_missing"] /
                                                group["all_missing"]
                                                if group["all_missing"] else 1.0),
              "all_candidate_cells": group["candidates"],
              "mean_candidate_cells": group["candidates"] / max(group["frames"], 1),
              "keep_all": _point(curve, 0)}
    if thresholds is None:
        thresholds = {
            "max_recall_at_20pct_precision": _choose(
                curve, "full_recall", "precision", 0.20),
            "max_precision_at_90pct_eligible_recall": _choose(
                curve, "precision", "eligible_recall", 0.90),
            "max_precision_at_95pct_eligible_recall": _choose(
                curve, "precision", "eligible_recall", 0.95),
            "max_precision_at_90pct_full_recall": _choose(
                curve, "precision", "full_recall", 0.90),
            "max_precision_at_95pct_full_recall": _choose(
                curve, "precision", "full_recall", 0.95),
        }
        result["calibration_operating_points"] = {
            name: _point(curve, index) for name, index in thresholds.items()}
    else:
        result["calibrated_operating_points"] = {
            name: _point(curve, index) for name, index in thresholds.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 2:
        parser.error("At least two validation frames are needed")
    model, saved = load_radar_pruner(args.checkpoint, args.device)
    radius_m = float(saved["support_radius_m"])
    target_recall = float(saved.get("target_eligible_recall", 0.95))
    dataset = paired_dataset(args.vod_root, "val", saved["radar_variant"],
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    cache = RadarSupportCache(args.support_cache_root / "val",
                              identity("val", ids, saved["radar_variant"],
                                       data_cache_root=args.data_cache_root,
                                       radius_m=radius_m))
    cache.check_complete()
    manifest = json.loads((args.faulty_root / "val" / "manifest.json").read_text(
        encoding="utf-8"))
    if (manifest.get("version") != MILD_FOG2_VERSION or
            manifest.get("radar_variant") != saved["radar_variant"] or
            manifest.get("frame_ids") != ids):
        parser.error("Mild-fog2 validation cache differs from the checkpoint dataset")
    excluded = set(saved.get("validation_frame_ids", ()))
    available = [i for i, frame_id in enumerate(ids) if frame_id not in excluded]
    selected = random.Random(args.seed).sample(available,
                                               min(args.limit, len(available)))
    midpoint = len(selected) // 2
    splits = {"calibration": selected[:midpoint], "heldout": selected[midpoint:]}
    groups = {name: _collect(model, dataset, indices, cache, args.faulty_root,
                             args.device, manifest["frame_faults"])
              for name, indices in splits.items()}
    report = {"checkpoint": str(args.checkpoint),
              "geometry_rule": ("3 distinct radar voxel centres within "
                                f"{radius_m:.1f} m"),
              "target_eligible_recall": target_recall,
              "sampled_frames": len(selected),
              "calibration_frames": midpoint,
              "heldout_frames": len(selected) - midpoint,
              "excluded_model_selection_frames": len(excluded),
              "sampled_frame_ids": [ids[i] for i in selected],
              "fault_types": {}}
    for fault in ("all", *MILD_FOG2_TYPES):
        cal = groups["calibration"][fault]
        held = groups["heldout"][fault]
        calibration = _describe(cal)
        indices = {name: (None if point is None else round(
            point["threshold"] * STEPS))
            for name, point in calibration["calibration_operating_points"].items()}
        report["fault_types"][fault] = {
            "calibration": calibration,
            "heldout": _describe(held, indices)}
        key = f"max_precision_at_{int(round(target_recall * 100))}pct_eligible_recall"
        report["fault_types"][fault]["primary_heldout"] = (
            report["fault_types"][fault]["heldout"]
            ["calibrated_operating_points"][key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output}; {len(selected)} frames = {midpoint} calibration "
          f"+ {len(selected) - midpoint} held-out", flush=True)
    for fault, results in report["fault_types"].items():
        held = results["heldout"]
        print(f"{fault}: geometry ceiling={held['geometry_full_recall_ceiling']:.2%}, "
              f"keep-all precision={held['keep_all']['precision']:.2%}, "
              f"mean candidates/frame={held['mean_candidate_cells']:,.0f}")
        row = results["primary_heldout"]
        if row is not None:
            print(f"  calibration-selected at {target_recall:.0%} eligible recall: "
                  f"P={row['precision']:.2%}, "
                  f"R_eligible={row['eligible_recall']:.2%}, "
                  f"R_full={row['full_recall']:.2%}, "
                  f"cells/frame={row['selected'] / max(held['frames'], 1):,.0f}")


if __name__ == "__main__":
    main()
