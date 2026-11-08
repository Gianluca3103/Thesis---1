"""Evaluate Stage II with only faulty LiDAR and radar in the prediction path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.addition import addition_context
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, full_frame_voxels)
from models.radar_lidar_stage2.clean_thresholds import (
    THRESHOLD_STEPS, score_histograms, sweep_rows)
from models.radar_lidar_stage2.sparse_geometry import GRID
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


@torch.no_grad()
def sweep(model, loader, ids: list[str], raw_root: Path, alpha_bars,
          *, device: str, seed: int, steps: int,
          sampler: str = "gaussian_ddim", timestep: int = 499):
    role = {frame_id: ("calibration" if (index // 20) % 5 == 0 else "heldout")
            for index, frame_id in enumerate(ids)}
    stores = {name: {"positive": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                     "negative": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                     "baseline": {"tp": 0, "predicted": 0, "target": 0},
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
    curves = {name: curve(store["positive"], store["negative"],
                          store["baseline"])
              for name, store in stores.items()}
    index = max(range(THRESHOLD_STEPS + 2),
                key=lambda i: (curves["calibration"][i]["iou"],
                               curves["calibration"][i]["threshold"]))
    return {"threshold": curves["calibration"][index]["threshold"],
            "selected_index": index,
            "frames": {name: store["frames"] for name, store in stores.items()},
            "calibration": curves["calibration"][index],
            "heldout": curves["heldout"][index],
            "curves": curves}, role


@torch.no_grad()
def selected_metrics(model, loader, raw_root: Path, split: str, alpha_bars,
                     *, device: str, seed: int, steps: int,
                     threshold_index: int, role: dict | None,
                     fault_names: dict[str, str],
                     sampler: str = "gaussian_ddim",
                     timestep: int = 499) -> dict:
    groups = ("calibration", "heldout") if role is not None else ("test",)
    overall = {name: {"faulty": empty_accumulator(),
                      "completed": empty_accumulator()}
               for name in groups}
    by_fault = {name: {"faulty": empty_accumulator(),
                       "completed": empty_accumulator()}
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
            group = role[row["frame_id"]] if role is not None else "test"
            for store in (overall[group], by_fault[fault_names[row["frame_id"]]]):
                add_frame(store["faulty"], faulty, clean)
                add_frame(store["completed"], completed, clean)
    if role is not None:
        overall["all_validation"] = {
            phase: {key: (overall["calibration"][phase][key] +
                          overall["heldout"][phase][key])
                    for key in overall["calibration"][phase]}
            for phase in ("faulty", "completed")}
    return {"overall": {name: completion_summary(
                value["faulty"], value["completed"])
                for name, value in overall.items()},
            "by_fault": {name: completion_summary(
                value["faulty"], value["completed"])
                for name, value in by_fault.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
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
        args.faulty_root, "val", identity, args.num_workers, args.val_limit)
    sweep_report, role = sweep(
        model, val_loader, val_ids, args.data_cache_root, alpha_bars,
        device=args.device, seed=args.seed, steps=args.steps,
        sampler=sampler, timestep=timestep)
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
        args.faulty_root, "test", identity, args.num_workers, args.test_limit)
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
              "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256(args.checkpoint),
              "checkpoint_epoch": int(saved["epoch"]),
              "sampling_steps": args.steps,
              "sampler": sampler, "eval_timestep": timestep,
              "training_target": identity.get("training_target", "pure_epsilon"),
              "threshold_selection": "full_scene_exact_IoU_on_validation_calibration_frames",
              "val_limit": args.val_limit, "test_limit": args.test_limit,
              "validation_note": "All validation frames may have selected the denoiser checkpoint; heldout only means held out from threshold selection.",
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
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
