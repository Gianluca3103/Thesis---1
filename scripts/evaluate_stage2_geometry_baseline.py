"""Calibrate and evaluate fixed-gate Stage II on disjoint mild-fault frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.geometry_gate import (
    RADIUS_M, VERSION, geometry_gate_output, offset_targets)
from models.radar_lidar_stage2.sparse_geometry import GRID, voxelize_means
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import load_stage2, sha256


def _counts(hist_positive: np.ndarray, hist_negative: np.ndarray,
            target: int, full_target: int, threshold_bin: int) -> dict:
    tp = int(hist_positive[threshold_bin:].sum())
    selected = tp + int(hist_negative[threshold_bin:].sum())
    precision = tp / selected if selected else 0.0
    recall = tp / target if target else 0.0
    return {"tp": tp, "selected": selected, "eligible_target": target,
            "full_missing_target": full_target, "precision": precision,
            "eligible_recall": recall,
            "full_missing_recall": tp / full_target if full_target else 0.0,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0),
            "threshold": threshold_bin / len(hist_positive)}


def _select(hist_positive: np.ndarray, hist_negative: np.ndarray,
            target: int) -> dict:
    tp = np.cumsum(hist_positive[::-1], dtype=np.int64)[::-1]
    selected = tp + np.cumsum(hist_negative[::-1], dtype=np.int64)[::-1]
    precision = np.divide(tp, selected, out=np.zeros_like(tp, dtype=float),
                          where=selected > 0)
    recall = tp / max(target, 1)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    valid90 = np.flatnonzero(recall >= 0.90)
    valid20 = np.flatnonzero(precision >= 0.20)
    return {"best_f1": int(np.argmax(f1)),
            "best_precision_at_90_eligible_recall": (
                int(valid90[np.argmax(precision[valid90])]) if len(valid90) else None),
            "max_recall_at_20_precision": (
                int(valid20[np.argmax(recall[valid20])]) if len(valid20) else None)}


def _empty_hist(bins: int) -> dict:
    return {"positive": np.zeros(bins, dtype=np.int64),
            "negative": np.zeros(bins, dtype=np.int64),
            "eligible_target": 0, "full_missing_target": 0,
            "frames": 0, "candidate_cells": 0, "frame_ids": []}


def _summary(values: dict, choices: dict) -> dict:
    target = values["eligible_target"]
    full = values["full_missing_target"]
    return {"frames": values["frames"], "frame_ids": values["frame_ids"],
            "eligible_target": target, "full_missing_target": full,
            "geometry_full_missing_recall_ceiling": target / full if full else 0.0,
            "mean_candidate_cells": values["candidate_cells"] / max(values["frames"], 1),
            "keep_all": _counts(values["positive"], values["negative"],
                                target, full, 0),
            "calibration_selected": {name: (
                _counts(values["positive"], values["negative"], target,
                        full, bin_index) if bin_index is not None else None)
                for name, bin_index in choices.items()}}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512,
                        help="Frames after the 64 model-selection frames; 512 by default")
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--bins", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available()
            or args.limit < 2 or args.sampling_steps < 1 or args.bins < 16):
        parser.error("Evaluation requires CUDA, at least two frames and valid bins/steps")
    model, saved = load_stage2(args.checkpoint, args.device)
    if (saved.get("geometry_gate_version") != VERSION
            or not saved.get("ray_offset_baseline")
            or saved.get("radius_m") != RADIUS_M):
        raise ValueError("Expected the 1.6 m geometry epsilon/ray/offset checkpoint")
    variant = saved["radar_variant"]
    dataset = paired_dataset(args.vod_root, "val", variant, include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    if saved.get("faulty_val_manifest_sha256") != sha256(
            args.faulty_root / "val" / "manifest.json"):
        raise ValueError("Fault cache differs from checkpoint")
    fault_manifest = json.loads((args.faulty_root / "val" / "manifest.json")
                                .read_text(encoding="utf-8"))
    frame_faults = fault_manifest["frame_faults"]
    if saved.get("support_val_manifest_sha256") != sha256(
            args.support_cache_root / "val" / "manifest.json"):
        raise ValueError("Support cache differs from checkpoint")
    expected = support_identity("val", frame_ids, variant,
                                data_cache_root=args.data_cache_root,
                                radius_m=RADIUS_M)
    support = RadarSupportCache(args.support_cache_root / "val", expected)
    support.check_complete()
    available = len(dataset) - 64
    if available < 2:
        raise ValueError("No validation frames remain after model selection")
    count = min(args.limit, available)
    selected_indices = np.linspace(64, len(dataset) - 1, count).round().astype(int)
    if len(np.unique(selected_indices)) != count:
        raise RuntimeError("Evaluation frame selection repeated a frame")
    hist = {split: _empty_hist(args.bins)
            for split in ("calibration", "heldout")}
    by_fault = {split: {fault: _empty_hist(args.bins)
                        for fault in fault_manifest["fault_types"]}
                for split in hist}
    alpha_bars = cosine_alpha_bars(device=args.device)
    for ordinal, index in enumerate(tqdm(selected_indices,
                                         desc="Geometry Stage-II validation",
                                         unit="frame")):
        split = "calibration" if ordinal % 2 == 0 else "heldout"
        row = dataset[int(index)]
        frame_id = row["frame_id"]
        radar = row["radar"].to(args.device)
        faulty = load_faulty(args.faulty_root, "val", frame_id).to(args.device)
        clean = row["clean_lidar"][:, :3].to(args.device)
        mask = support.load(frame_id)
        output = geometry_gate_output(radar, faulty, mask)
        conditions = model.conditions(output, float(saved["gate_threshold"]))
        fine = conditions.hierarchy[0]
        clean_coords, _ = voxelize_means(clean, grid=GRID)
        faulty_coords, _ = voxelize_means(faulty, grid=GRID)
        clean_keys = encode_keys(clean_coords, GRID.shape_zyx)
        faulty_keys = encode_keys(faulty_coords, GRID.shape_zyx)
        full_missing = int((~torch.isin(clean_keys, faulty_keys)).sum())
        if len(fine):
            occupied, _, _ = offset_targets(clean, fine)
            seed = int.from_bytes(hashlib.sha256(
                f"3103:{frame_id}".encode()).digest()[:8], "little") % (2**63)
            torch.manual_seed(seed)
            prediction = model.sample(
                faulty, conditions, alpha_bars, steps=args.sampling_steps,
                occupancy_threshold=0.5, missing_only=True)
            scores = prediction["candidate_probabilities"].cpu().numpy()
            if len(scores) != len(fine):
                raise RuntimeError("Diffusion did not score every allowed voxel")
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise RuntimeError("Diffusion produced invalid candidate probabilities")
            positive = occupied.cpu().numpy()
            index_bins = np.minimum((scores * args.bins).astype(np.int64),
                                    args.bins - 1)
            for row_hist in (hist[split], by_fault[split][frame_faults[frame_id]]):
                row_hist["positive"] += np.bincount(index_bins[positive],
                                                     minlength=args.bins)
                row_hist["negative"] += np.bincount(index_bins[~positive],
                                                     minlength=args.bins)
                row_hist["eligible_target"] += int(positive.sum())
        for row_hist in (hist[split], by_fault[split][frame_faults[frame_id]]):
            row_hist["full_missing_target"] += full_missing
            row_hist["candidate_cells"] += len(fine)
            row_hist["frames"] += 1
            row_hist["frame_ids"].append(frame_id)
    calibration = hist["calibration"]
    choices = _select(calibration["positive"], calibration["negative"],
                      calibration["eligible_target"])
    report = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": saved["epoch"],
              "objective": saved["diffusion_objective"], "radar_variant": variant,
              "sampling_steps": args.sampling_steps, "histogram_bins": args.bins,
              "model_selection_frames_excluded": 64,
              "splits": {}, "by_fault": {}, "calibration_choices": {}}
    for split, values in hist.items():
        report["splits"][split] = _summary(values, choices)
        report["by_fault"][split] = {
            fault: _summary(row, choices)
            for fault, row in by_fault[split].items()}
    report["calibration_choices"] = {key: (value / args.bins
                                              if value is not None else None)
                                      for key, value in choices.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": report["splits"]["calibration"],
                      "heldout": report["splits"]["heldout"],
                      "report": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
