"""Measure frozen LPCR's clean-latent ceiling on the fault-gated Stage-II ROI.

This is an oracle diagnostic: clean LiDAR is encoded before decoding. It is
never a valid reconstruction input at deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.clean_selected_metrics import full_frame_voxels
from models.radar_lidar_stage2.clean_thresholds import (
    THRESHOLD_STEPS, score_histograms)
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.evaluate_stage2_conditional_reconstruction import (
    _frame_candidates, counts, curve)
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, _CleanRows, _batch, _rows)
from scripts.train_stage2_conditional_from_scratch import fresh_model


def _store() -> dict:
    return {"positive": torch.zeros(THRESHOLD_STEPS + 1, dtype=torch.long),
            "negative": torch.zeros(THRESHOLD_STEPS + 1, dtype=torch.long),
            "baseline": {"tp": 0, "predicted": 0, "target": 0},
            "frames": 0}


class IndexedRows:
    """Keep split indices stable when DataLoader uses worker processes."""

    def __init__(self, rows: _CleanRows):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {**self.rows[index], "validation_index": index}


def _selected(row: dict, eligible: int) -> dict:
    added = row["added"]
    true = row["true_additions"]
    return {**row, "addition_precision": true / added if added else 0.0,
            "addition_eligible_recall": true / eligible if eligible else 0.0}


def _ceiling(baseline: dict, eligible: int) -> dict:
    tp = baseline["tp"] + eligible
    predicted = baseline["predicted"] + eligible
    target = baseline["target"]
    union = predicted + target - tp
    return {"true_additions": eligible, "tp": tp, "predicted": predicted,
            "target": target, "iou": tp / union if union else 0.0}


def summarize(stores: dict[str, dict]) -> dict:
    rows = {name: curve(value["positive"], value["negative"],
                        value["baseline"])
            for name, value in stores.items()}
    best_index = max(range(len(rows["calibration"])),
                     key=lambda index: (rows["calibration"][index]["iou"],
                                        rows["calibration"][index]["threshold"]))
    result = {"calibration_selected_threshold":
              rows["calibration"][best_index]["threshold"],
              "selection_scope": "calibration; reported on heldout and all_validation",
              "groups": {}}
    for name, value in stores.items():
        eligible = int(value["positive"].sum())
        candidates = eligible + int(value["negative"].sum())
        result["groups"][name] = {
            "frames": value["frames"], "candidate_cells": candidates,
            "eligible_missing_clean_cells": eligible,
            "faulty_baseline": rows[name][-1],
            "gate_perfect_classifier_ceiling": _ceiling(value["baseline"], eligible),
            "lpcr_clean_latent_at_0p5": _selected(
                rows[name][THRESHOLD_STEPS // 2], eligible),
            "lpcr_clean_latent_calibrated": _selected(
                rows[name][best_index], eligible)}
    return result


@torch.no_grad()
def evaluate(model, loader, *, device: str) -> dict:
    stores = {name: _store() for name in
              ("calibration", "heldout", "all_validation")}
    for rows in tqdm(loader, desc="Fault-gated frozen LPCR oracle", unit="batch"):
        hierarchy, clean_points, point_ids = _batch(rows, device)
        fine = hierarchy[0]
        if len(hierarchy[-1]):
            clean_latent, _ = model.lidar_encoder(
                clean_points, hierarchy[-1], point_ids)
            logits, _ = model.lpcr(
                clean_latent, hierarchy, clean_latent.new_ones((len(fine), 1)))
        else:
            logits = torch.empty((0,), device=device)
        for frame, row in enumerate(rows):
            candidates, scores = _frame_candidates(
                fine, logits, frame, row["faulty_xyz"], device)
            empty = torch.empty((0, 4), dtype=torch.long)
            clean, faulty, _ = full_frame_voxels(
                row["full_clean_xyz"], row["faulty_xyz"], empty)
            target = torch.isin(
                encode_keys(candidates, GRID.shape_zyx),
                encode_keys(clean.to(device), GRID.shape_zyx))
            positive, negative = score_histograms(scores.sigmoid(), target)
            baseline = counts(faulty, clean)
            role = ("calibration" if
                    ((row["validation_index"] // 20) % 5 == 0) else "heldout")
            for name in (role, "all_validation"):
                store = stores[name]
                store["frames"] += 1
                store["positive"] += positive
                store["negative"] += negative
                for key in baseline:
                    store["baseline"][key] += baseline[key]
    return summarize(stores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.num_workers < 0 or
            (args.limit is not None and args.limit < 1) or
            args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("Invalid batch size, workers, limit, or CUDA unavailable")
    dataset = paired_dataset(
        args.vod_root, "val", "radar_20frames_verified_doppler_radial",
        include_clean=True, data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    support = RadarSupportCache(
        args.support_cache_root / "val",
        support_identity("val", ids, "radar_20frames_verified_doppler_radial",
                         data_cache_root=args.data_cache_root,
                         radius_m=RADIUS_M,
                         min_radar_voxels=MIN_RADAR_VOXELS))
    support.check_complete()
    regions = FaultRegionCache(
        args.fault_region_root / "val",
        cache_identity("val", ids,
                       args.faulty_root / "val" / "manifest.json",
                       args.data_cache_root / "val" / "manifest.json"))
    if regions.regions is None:
        raise FileNotFoundError("Validation injector footprints are not built")
    source_rows = _CleanRows(
        dataset, support, args.limit, args.faulty_root, "val",
        include_full_clean=True, fault_regions=regions)
    # The validation split uses its fixed frame ordering for the same
    # calibration/heldout roles as the deployment evaluator.
    loader = DataLoader(IndexedRows(source_rows), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=_rows, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    model = fresh_model(args.lpcr_checkpoint, args.device).eval()
    report = evaluate(model, loader, device=args.device)
    report.update({"kind": "oracle_clean_latent_frozen_LPCR",
                   "not_deployment_reconstruction": True,
                   "validation_frames": len(source_rows),
                   "lpcr_checkpoint_sha256": sha256(args.lpcr_checkpoint),
                   "radar_rule": ">=12 distinct radar voxels within 1.6 m",
                   "gate": "radar support intersect synthetic injector fault region"})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n",
                                encoding="utf-8")
    print(f"Clean-latent LPCR oracle; {len(source_rows)} validation frames; "
          f"threshold={report['calibration_selected_threshold']:.3f}")
    for name, value in report["groups"].items():
        print(f"{name}: faulty IoU={value['faulty_baseline']['iou']:.2%}, "
              f"gate ceiling={value['gate_perfect_classifier_ceiling']['iou']:.2%}, "
              f"LPCR@0.5={value['lpcr_clean_latent_at_0p5']['iou']:.2%}, "
              f"LPCR calibrated={value['lpcr_clean_latent_calibrated']['iou']:.2%}, "
              f"eligible missing={value['eligible_missing_clean_cells']:,}")
    print(f"Report: {args.output_json}")


if __name__ == "__main__":
    main()
