"""Evaluate Stage II with only faulty LiDAR and radar in the prediction path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.addition import addition_context
from models.radar_lidar_stage2.decoder_occupancy_loss import error_sums
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, full_frame_voxels)
from models.radar_lidar_stage2.clean_thresholds import (
    THRESHOLD_STEPS, score_histograms, sweep_rows)
from models.radar_lidar_stage2.eligible_occupancy_metrics import (
    occupancy_counts as tolerant_occupancy_counts)
from models.radar_lidar_stage2.eligible_occupancy_metrics import summarize_counts
from models.radar_lidar_stage2.fault_only_gate import POLICY
from models.radar_lidar_stage2.sparse_geometry import GRID, sorted_unique
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.conditional_stage2 import (
    deployment_loader, load_model, predict_batch, reference_clean)


def counts(predicted: torch.Tensor, clean: torch.Tensor) -> dict[str, int]:
    source = encode_keys(predicted.long(), GRID.shape_zyx)
    target = encode_keys(clean.long(), GRID.shape_zyx)
    return {"tp": int(torch.isin(source, target).sum()),
            "predicted": len(predicted), "target": len(clean)}


def curve(positive: torch.Tensor, negative: torch.Tensor,
          baseline: dict[str, int]) -> list[dict]:
    result = []
    for row in sweep_rows(positive, negative):
        tp = baseline["tp"] + row["tp"]
        predicted = baseline["predicted"] + row["predicted"]
        target = baseline["target"]
        union = predicted + target - tp
        result.append({"threshold": row["threshold"],
                       "added": row["predicted"], "true_additions": row["tp"],
                       "tp": tp, "predicted": predicted, "target": target,
                       "iou": tp / union if union else 0.0,
                       "f1": 2 * tp / (predicted + target)
                       if predicted + target else 0.0})
    tp, predicted, target = (baseline[key]
                             for key in ("tp", "predicted", "target"))
    union = predicted + target - tp
    result.append({"threshold": 1 + 1 / THRESHOLD_STEPS,
                   "added": 0, "true_additions": 0, "tp": tp,
                   "predicted": predicted, "target": target,
                   "iou": tp / union if union else 0.0,
                   "f1": 2 * tp / (predicted + target)
                   if predicted + target else 0.0})
    return result


def _frame_candidates(fine: torch.Tensor, logits: torch.Tensor, frame: int,
                      faulty_xyz: torch.Tensor, device: str):
    selected = fine[:, 0] == frame
    cells = fine[selected].clone()
    cells[:, 0] = 0
    observed = addition_context(faulty_xyz.to(device), cells).observed
    return cells[~observed], logits[selected][~observed]


def _within_gate(voxels: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Keep occupied cells inside one frame's complete active coordinate gate."""
    gate_keys = encode_keys(gate, GRID.shape_zyx)
    voxel_keys = encode_keys(voxels, GRID.shape_zyx)
    return voxels[torch.isin(voxel_keys, gate_keys)]


def _iou_from_matches(matches: dict, *, tolerant: bool = False) -> float:
    tp = matches["tolerant_tp_0p2m" if tolerant else "exact_tp"]
    union = matches["predicted"] + matches["eligible_target"] - tp
    return tp / union if union else 0.0


def _empty_active_counts() -> dict[str, int]:
    return {"frames": 0, "predicted": 0, "eligible_target": 0,
            "exact_tp": 0, "tolerant_tp_0p2m": 0,
            "radar_supported_cells": 0, "active_cells": 0}


def _active_summary(values: dict[str, int]) -> dict:
    result = summarize_counts(values)
    predicted, target = values["predicted"], values["eligible_target"]
    exact, tolerant = values["exact_tp"], values["tolerant_tp_0p2m"]
    return {"frames": values["frames"], **result,
            "mean_radar_supported_cells": (values["radar_supported_cells"] /
                                           values["frames"] if values["frames"] else 0.0),
            "mean_active_cells": (values["active_cells"] / values["frames"]
                                  if values["frames"] else 0.0),
            "precision": exact / predicted if predicted else 0.0,
            "recall": exact / target if target else 0.0,
            "f1": 2 * exact / (predicted + target) if predicted + target else 0.0,
            "f1_0p2m": 2 * tolerant / (predicted + target)
            if predicted + target else 0.0}


@torch.no_grad()
def sweep(model, loader, ids: list[str], raw_root: Path, alpha_bars,
          *, device: str, seed: int, steps: int,
          sampler: str = "gaussian_ddim", timestep: int = 499,
          fixed_half_tolerance: bool = False,
          decoder_loss_fp_weight: float | None = None):
    role = {frame_id: ("calibration" if (index // 20) % 5 == 0 else "heldout")
            for index, frame_id in enumerate(ids)}
    stores = {name: {"positive": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                     "negative": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                     "baseline": {"tp": 0, "predicted": 0, "target": 0},
                     "fixed_half_tolerant": {
                         "predicted": 0, "eligible_target": 0,
                         "exact_tp": 0, "tolerant_tp_0p2m": 0},
                     "decoder_errors": {
                         "positive_error_sum": 0.0,
                         "negative_error_sum": 0.0,
                         "positive_count": 0,
                         "negative_count": 0},
                     "frames": 0}
              for name in ("calibration", "heldout")}
    for rows in tqdm(loader, desc="Conditional Stage II validation sweep",
                     unit="batch"):
        output = predict_batch(model, rows, alpha_bars, device=device,
                               seed=seed, steps=steps, sampler=sampler,
                               timestep=timestep)
        for frame, row in enumerate(rows):
            store = stores[role[row["frame_id"]]]
            store["frames"] += 1
            clean_xyz = reference_clean(raw_root, "val", row["frame_id"])
            empty = output["fine"].new_empty((0, 4)).cpu()
            clean, faulty, _ = full_frame_voxels(
                clean_xyz, row["faulty_xyz"], empty)
            current = counts(faulty, clean)
            for key in store["baseline"]:
                store["baseline"][key] += current[key]
            candidates, logits = _frame_candidates(
                output["fine"], output["logits"], frame,
                row["faulty_xyz"], device)
            target = torch.isin(
                encode_keys(candidates, GRID.shape_zyx),
                encode_keys(clean.to(device), GRID.shape_zyx))
            positive, negative = score_histograms(logits.sigmoid(), target)
            store["positive"] += positive
            store["negative"] += negative
            if decoder_loss_fp_weight is not None:
                parts = error_sums(logits, target)
                for key in store["decoder_errors"]:
                    value = parts[key]
                    store["decoder_errors"][key] += (
                        float(value) if isinstance(value, torch.Tensor)
                        else value)
            if fixed_half_tolerance:
                # The 0.5 here is the occupancy decision threshold. The
                # distance tolerance for matching voxel centers is 0.2 m.
                bucket = (torch.floor(logits.sigmoid() * THRESHOLD_STEPS)
                          .long().clamp(0, THRESHOLD_STEPS))
                additions = candidates[bucket >= THRESHOLD_STEPS // 2].cpu()
                completed = sorted_unique(
                    torch.cat((faulty, additions), 0), GRID.shape_zyx)
                matched = tolerant_occupancy_counts(
                    completed, clean, tolerance_m=0.2)
                for key, value in matched.items():
                    store["fixed_half_tolerant"][key] += value
    curves = {name: curve(store["positive"], store["negative"],
                          store["baseline"])
              for name, store in stores.items()}
    index = max(range(THRESHOLD_STEPS + 2),
                key=lambda i: (curves["calibration"][i]["iou"],
                               curves["calibration"][i]["threshold"]))
    result = {"threshold": curves["calibration"][index]["threshold"],
            "selected_index": index,
            "frames": {name: store["frames"] for name, store in stores.items()},
            "calibration": curves["calibration"][index],
            "heldout": curves["heldout"][index],
            "curves": curves}
    if fixed_half_tolerance:
        result["fixed_half_tolerant"] = {
            name: store["fixed_half_tolerant"] for name, store in stores.items()}
    if decoder_loss_fp_weight is not None:
        result["generated_decoder_loss"] = {}
        for name, store in stores.items():
            parts = store["decoder_errors"]
            positive = (parts["positive_error_sum"] /
                        max(parts["positive_count"], 1))
            negative = (parts["negative_error_sum"] /
                        max(parts["negative_count"], 1))
            result["generated_decoder_loss"][name] = {
                "loss": positive + decoder_loss_fp_weight * negative,
                "positive_mean_error": positive,
                "negative_mean_error": negative,
                "positive_count": parts["positive_count"],
                "negative_count": parts["negative_count"]}
    return result, role


@torch.no_grad()
def fixed_threshold_validation(model, loader, raw_root: Path, alpha_bars,
                               *, device: str, seed: int, steps: int,
                               decoder_loss_fp_weight: float | None = None) -> dict:
    """Validate generated occupancy at 0.5 without a threshold sweep."""
    totals = {"frames": 0, "tp": 0, "predicted": 0, "target": 0,
              "tolerant_tp_0p2m": 0, "faulty_tp": 0,
              "faulty_predicted": 0, "faulty_tolerant_tp_0p2m": 0}
    fault_area = {
        phase: {key: 0 for key in ("predicted", "eligible_target",
                                    "exact_tp", "tolerant_tp_0p2m")}
        for phase in ("faulty", "completed")}
    errors = {"positive_error_sum": 0.0, "negative_error_sum": 0.0,
              "positive_count": 0, "negative_count": 0}
    for rows in tqdm(loader, desc="Conditional fixed-0.5 validation",
                     unit="batch"):
        output = predict_batch(model, rows, alpha_bars, device=device,
                               seed=seed, steps=steps)
        for frame, row in enumerate(rows):
            candidates, logits = _frame_candidates(
                output["fine"], output["logits"], frame,
                row["faulty_xyz"], device)
            additions = candidates[logits >= 0].cpu()
            clean_xyz = reference_clean(raw_root, "val", row["frame_id"])
            clean, faulty, completed = full_frame_voxels(
                clean_xyz, row["faulty_xyz"], additions)
            current = counts(completed, clean)
            baseline = counts(faulty, clean)
            tolerant = tolerant_occupancy_counts(
                completed, clean, tolerance_m=0.2)
            baseline_tolerant = tolerant_occupancy_counts(
                faulty, clean, tolerance_m=0.2)
            if (tolerant["predicted"] != current["predicted"] or
                    tolerant["eligible_target"] != current["target"] or
                    tolerant["exact_tp"] != current["tp"] or
                    baseline_tolerant["predicted"] != baseline["predicted"] or
                    baseline_tolerant["eligible_target"] != baseline["target"] or
                    baseline_tolerant["exact_tp"] != baseline["tp"]):
                raise RuntimeError("Exact and 0.2 m validation counts differ")
            gate = output["fine"][output["fine"][:, 0] == frame].cpu().clone()
            gate[:, 0] = 0
            gated_clean = _within_gate(clean, gate)
            gated_faulty = _within_gate(faulty, gate)
            gated_completed = _within_gate(completed, gate)
            gated_metrics = {
                "faulty": tolerant_occupancy_counts(
                    gated_faulty, gated_clean, tolerance_m=0.2),
                "completed": tolerant_occupancy_counts(
                    gated_completed, gated_clean, tolerance_m=0.2)}
            for phase, matches in gated_metrics.items():
                for key, value in matches.items():
                    fault_area[phase][key] += value
            totals["frames"] += 1
            for key in ("tp", "predicted", "target"):
                totals[key] += current[key]
            totals["tolerant_tp_0p2m"] += tolerant["tolerant_tp_0p2m"]
            totals["faulty_tp"] += baseline["tp"]
            totals["faulty_predicted"] += baseline["predicted"]
            totals["faulty_tolerant_tp_0p2m"] += (
                baseline_tolerant["tolerant_tp_0p2m"])
            if decoder_loss_fp_weight is not None:
                target = torch.isin(
                    encode_keys(candidates, GRID.shape_zyx),
                    encode_keys(clean.to(device), GRID.shape_zyx))
                parts = error_sums(logits, target)
                for key in errors:
                    value = parts[key]
                    errors[key] += (float(value) if isinstance(value, torch.Tensor)
                                    else value)
    union = totals["predicted"] + totals["target"] - totals["tp"]
    tolerant_union = (totals["predicted"] + totals["target"] -
                      totals["tolerant_tp_0p2m"])
    baseline_union = (totals["faulty_predicted"] + totals["target"] -
                      totals["faulty_tp"])
    baseline_tolerant_union = (
        totals["faulty_predicted"] + totals["target"] -
        totals["faulty_tolerant_tp_0p2m"])
    gate_summary = {
        phase: {**matches,
                "iou": _iou_from_matches(matches),
                "iou_0p2m": _iou_from_matches(matches, tolerant=True)}
        for phase, matches in fault_area.items()}
    result = {"threshold": 0.5, **totals,
              "iou": totals["tp"] / union if union else 0.0,
              "iou_0p2m": (totals["tolerant_tp_0p2m"] / tolerant_union
                            if tolerant_union else 0.0),
              "faulty_baseline_iou": (totals["faulty_tp"] / baseline_union
                                      if baseline_union else 0.0),
              "faulty_baseline_iou_0p2m": (
                  totals["faulty_tolerant_tp_0p2m"] /
                  baseline_tolerant_union if baseline_tolerant_union else 0.0),
              "fault_area": gate_summary}
    result["full_iou_improvement_pp"] = (
        100 * (result["iou"] - result["faulty_baseline_iou"]))
    result["full_iou_0p2m_improvement_pp"] = (
        100 * (result["iou_0p2m"] - result["faulty_baseline_iou_0p2m"]))
    if decoder_loss_fp_weight is not None:
        positive = errors["positive_error_sum"] / max(errors["positive_count"], 1)
        negative = errors["negative_error_sum"] / max(errors["negative_count"], 1)
        result["generated_decoder_loss"] = {
            "loss": positive + decoder_loss_fp_weight * negative,
            "positive_mean_error": positive,
            "negative_mean_error": negative,
            "positive_count": errors["positive_count"],
            "negative_count": errors["negative_count"]}
    return result


@torch.no_grad()
def selected_metrics(model, loader, raw_root: Path, split: str, alpha_bars,
                     *, device: str, seed: int, steps: int,
                     threshold_index: int, role: dict | None,
                     fault_names: dict[str, str],
                     sampler: str = "gaussian_ddim",
                     timestep: int = 499) -> dict:
    groups = ("calibration", "heldout") if role is not None else ("test",)
    overall = {name: {"faulty": empty_accumulator(),
                      "completed": empty_accumulator(),
                      "active": _empty_active_counts()}
               for name in groups}
    by_fault = {name: {"faulty": empty_accumulator(),
                       "completed": empty_accumulator(),
                       "active": _empty_active_counts()}
                for name in sorted(set(fault_names.values()))}
    for rows in tqdm(loader, desc=f"Conditional Stage II {split} metrics",
                     unit="batch"):
        output = predict_batch(model, rows, alpha_bars, device=device,
                               seed=seed, steps=steps, sampler=sampler,
                               timestep=timestep)
        for frame, row in enumerate(rows):
            candidates, logits = _frame_candidates(
                output["fine"], output["logits"], frame,
                row["faulty_xyz"], device)
            bucket = (torch.floor(logits.sigmoid() * THRESHOLD_STEPS)
                      .long().clamp(0, THRESHOLD_STEPS))
            additions = candidates[bucket >= threshold_index].cpu()
            clean_xyz = reference_clean(raw_root, split, row["frame_id"])
            clean, faulty, completed = full_frame_voxels(
                clean_xyz, row["faulty_xyz"], additions)
            # Within the filtered active region, score additions against clean
            # cells that were not already observed in the faulty LiDAR.
            candidate_cpu = candidates.cpu()
            eligible = candidate_cpu[torch.isin(
                encode_keys(candidate_cpu, GRID.shape_zyx),
                encode_keys(clean, GRID.shape_zyx))]
            active_counts = tolerant_occupancy_counts(
                additions, eligible, tolerance_m=0.2)
            group = role[row["frame_id"]] if role is not None else "test"
            for store in (overall[group], by_fault[fault_names[row["frame_id"]]]):
                add_frame(store["faulty"], faulty, clean)
                add_frame(store["completed"], completed, clean)
                store["active"]["frames"] += 1
                store["active"]["radar_supported_cells"] += row["radar_supported_cells"]
                store["active"]["active_cells"] += row["active_cells"]
                for key, value in active_counts.items():
                    store["active"][key] += value
    if role is not None:
        overall["all_validation"] = {
            phase: {key: (overall["calibration"][phase][key] +
                          overall["heldout"][phase][key])
                    for key in overall["calibration"][phase]}
            for phase in ("faulty", "completed", "active")}
    return {"overall": {name: completion_summary(
                value["faulty"], value["completed"])
                | {"active_additions": _active_summary(value["active"])}
                for name, value in overall.items()},
            "by_fault": {name: completion_summary(
                value["faulty"], value["completed"])
                | {"active_additions": _active_summary(value["active"])}
                for name, value in by_fault.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.num_workers < 0 or not 1 <= args.steps <= 1000 or
            any(value is not None and value < 1 for value in
                (args.val_limit, args.test_limit))):
        parser.error("CUDA, nonnegative workers, and 1–1000 steps required")
    model, identity, saved = load_model(
        args.lpcr_checkpoint, args.checkpoint, args.device)
    sampler = identity.get("sampler", "gaussian_ddim")
    timestep = identity.get("eval_timestep", 499)
    if ((sampler == "faulty_latent_one_step" and args.steps != 1) or
            (sampler == "gaussian_ddim" and args.steps < 2) or
            sampler not in ("faulty_latent_one_step", "gaussian_ddim")):
        parser.error("Sampling steps differ from checkpoint sampler")
    alpha_bars = cosine_alpha_bars(device=args.device)
    val_loader, val_ids, val_manifest = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, "val", identity, args.num_workers, args.val_limit,
        fault_region_root=args.fault_region_root)
    sweep_report, role = sweep(
        model, val_loader, val_ids, args.data_cache_root, alpha_bars,
        device=args.device, seed=args.seed, steps=args.steps,
        sampler=sampler, timestep=timestep,
        decoder_loss_fp_weight=identity.get("decoder_false_positive_weight"))
    validation = selected_metrics(
        model, val_loader, args.data_cache_root, "val", alpha_bars,
        device=args.device, seed=args.seed, steps=args.steps,
        threshold_index=sweep_report["selected_index"], role=role,
        fault_names=val_manifest["frame_faults"], sampler=sampler,
        timestep=timestep)
    for group in ("calibration", "heldout"):
        actual = validation["overall"][group]["completed"]
        expected = sweep_report[group]
        if (actual["exact_tp"] != expected["tp"] or
                actual["predicted"] != expected["predicted"] or
                actual["eligible_target"] != expected["target"]):
            raise RuntimeError(f"{group} sweep and selected metrics differ")
    test_loader, test_ids, test_manifest = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, "test", identity, args.num_workers, args.test_limit,
        fault_region_root=args.fault_region_root)
    if set(val_ids) & set(test_ids):
        raise ValueError("Official test and validation frames overlap")
    test = selected_metrics(
        model, test_loader, args.data_cache_root, "test", alpha_bars,
        device=args.device, seed=args.seed, steps=args.steps,
        threshold_index=sweep_report["selected_index"], role=None,
        fault_names=test_manifest["frame_faults"], sampler=sampler,
        timestep=timestep)
    report = {"model": ("conditional_faulty_latent_to_clean_one_step"
                        if sampler == "faulty_latent_one_step" else
                        "conditional_epsilon_denoiser_full_noise_DDIM"),
              "oracle_clean_latent": False,
              "addition_head": "disabled_pending_training_on_generated_latents",
              "prediction_inputs": ["faulty_LiDAR", "measured_radar",
                                    "radar_geometry_gate", "Gaussian_noise"],
              "clean_LiDAR_role": "metric_reference_only",
              "oracle_injector_fault_gate": args.fault_region_root is not None,
              "fault_gate_policy": POLICY if
                  args.fault_region_root is not None else None,
              "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_epoch": int(saved["epoch"]),
              "sampling_steps": args.steps,
              "sampler": sampler, "eval_timestep": timestep,
              "training_target": identity.get("training_target", "pure_epsilon"),
              "threshold_selection": "full_scene_exact_IoU_on_validation_calibration_frames",
              "val_limit": args.val_limit, "test_limit": args.test_limit,
              "validation_note": (
                  "Checkpoint selected using full validation at fixed 0.5 "
                  "every few epochs; threshold swept once after training "
                  "using validation calibration frames. Threshold-heldout "
                  "frames also participated in checkpoint selection. "
                  "Official test was untouched."
                  if identity.get("checkpoint_selection") ==
                  "fixed_0p5_full_validation_exact_iou" else
                  "Threshold uses calibration frames; heldout frames selected "
                  "the checkpoint across epochs. Official test was untouched."
                  if identity.get("checkpoint_selection") else
                  "All validation frames may have selected the denoiser "
                  "checkpoint; heldout only means held out from threshold selection."),
              "sweep": sweep_report, "validation": validation, "test": test}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.output_json)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Checkpoint epoch {saved['epoch']}; {sampler}; "
          f"threshold={sweep_report['threshold']:.3f}",
          flush=True)
    if "generated_decoder_loss" in sweep_report:
        print("Generated decoder occupancy loss on validation: "
              f"calibration="
              f"{sweep_report['generated_decoder_loss']['calibration']['loss']:.4f}, "
              f"heldout="
              f"{sweep_report['generated_decoder_loss']['heldout']['loss']:.4f}",
              flush=True)
    for name, value in (("validation heldout", validation["overall"]["heldout"]),
                        ("all validation", validation["overall"]["all_validation"]),
                        ("official test", test["overall"]["test"])):
        before, after, delta = (value[key] for key in
                                ("faulty", "completed", "improvement"))
        print(f"{name} ({before['frames']} frames, "
              f"{value['added_cells']:,} additions): "
              f"exact IoU {before['iou']:.2%} -> {after['iou']:.2%} "
              f"({delta['iou'] * 100:+.2f} pp); "
              f"exact F1 {before['f1']:.2%} -> {after['f1']:.2%} "
              f"({delta['f1'] * 100:+.2f} pp); "
              f"IoU@0.2m {before['iou_0p2m']:.2%} -> "
              f"{after['iou_0p2m']:.2%} "
              f"({delta['iou_0p2m'] * 100:+.2f} pp); "
              f"F1@0.2m {before['f1_0p2m']:.2%} -> "
              f"{after['f1_0p2m']:.2%} "
              f"({delta['f1_0p2m'] * 100:+.2f} pp)", flush=True)
        active = value["active_additions"]
        print(f"  Active-gate additions: "
              f"P={active['precision']:.2%}, R={active['recall']:.2%}, "
              f"IoU={active['eligible_iou']:.2%}, "
              f"IoU@0.2m={active['eligible_iou_0p2m']:.2%}, "
              f"mean active/radar cells={active['mean_active_cells']:,.0f}/"
              f"{active['mean_radar_supported_cells']:,.0f}, "
              f"predicted={active['predicted']:,}, "
              f"eligible targets={active['eligible_target']:,}", flush=True)
    for fault, value in test["by_fault"].items():
        active = value["active_additions"]
        print(f"test {fault}: exact IoU "
              f"{value['faulty']['iou']:.2%} -> "
              f"{value['completed']['iou']:.2%}; "
              f"active cells/frame {active['mean_active_cells']:,.0f}/"
              f"{active['mean_radar_supported_cells']:,.0f} radar-supported; "
              f"addition P={active['precision']:.2%}, "
              f"R={active['recall']:.2%}", flush=True)
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
