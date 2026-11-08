"""Select addition threshold by full-ROI IoU, then score validation and test."""

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
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, full_frame_voxels)
from models.radar_lidar_stage2.clean_thresholds import (
    THRESHOLD_STEPS, score_histograms, sweep_rows)
from models.radar_lidar_stage2.reliable_addition import decoded_addition_batch
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, _CleanRows, _rows)
from scripts.train_stage2_reliable_addition import (
    REVISION, combine_counts, exact_counts, load_base_model)


def full_roi_curve(positive: torch.Tensor, negative: torch.Tensor,
                   baseline: dict[str, int]) -> list[dict]:
    """Add thresholded candidate cells to the observed faulty baseline."""
    rows = sweep_rows(positive, negative)
    result = []
    for row in rows:
        tp = baseline["tp"] + row["tp"]
        predicted = baseline["predicted"] + row["predicted"]
        target = baseline["target"]
        union = predicted + target - tp
        result.append({"threshold": row["threshold"], "added": row["predicted"],
                       "true_additions": row["tp"], "tp": tp,
                       "predicted": predicted, "target": target,
                       "precision": tp / predicted if predicted else 0.0,
                       "recall": tp / target if target else 0.0,
                       "f1": 2 * tp / (predicted + target)
                       if predicted + target else 0.0,
                       "iou": tp / union if union else 0.0})
    # A threshold above one explicitly means no additions. This guarantees
    # that the selected full-ROI operating point can retain the faulty scan.
    tp = baseline["tp"]
    predicted = baseline["predicted"]
    target = baseline["target"]
    union = predicted + target - tp
    result.append({"threshold": 1 + 1 / (len(rows) - 1),
                   "added": 0, "true_additions": 0, "tp": tp,
                   "predicted": predicted, "target": target,
                   "precision": tp / predicted if predicted else 0.0,
                   "recall": tp / target if target else 0.0,
                   "f1": 2 * tp / (predicted + target)
                   if predicted + target else 0.0,
                   "iou": tp / union if union else 0.0})
    return result


def _loader(args, split: str, variant: str, batch_size: int):
    dataset = paired_dataset(args.vod_root, split, variant,
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    expected = support_identity(
        split, frame_ids, variant, data_cache_root=args.data_cache_root,
        radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS)
    cache = RadarSupportCache(args.support_cache_root / split, expected)
    cache.check_complete()
    manifest = json.loads((args.faulty_root / split / "manifest.json")
                          .read_text(encoding="utf-8"))
    if manifest.get("split") != split or manifest.get("frame_ids") != frame_ids:
        raise ValueError(f"{split} cached faults differ from official frames")
    if split == "test":
        training = json.loads((args.faulty_root / "train" / "manifest.json")
                              .read_text(encoding="utf-8"))
        for key in ("version", "seed", "radar_variant", "fault_types",
                    "assignment_policy", "partial_loss_policy"):
            if manifest.get(key) != training.get(key):
                raise ValueError(f"Test mild-fault policy differs: {key}")
    loader = DataLoader(
        _CleanRows(dataset, cache, None, args.faulty_root, split,
                   include_full_clean=True),
        batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_rows,
        pin_memory=True, persistent_workers=args.num_workers > 0)
    return loader, frame_ids, manifest


@torch.no_grad()
def _sweep(model, loader, frame_ids: list[str], alpha_bars,
           *, device: str, seed: int, eval_timestep: int) -> tuple[dict, dict]:
    role = {frame_id: ("calibration" if (index // 20) % 5 == 0 else "heldout")
            for index, frame_id in enumerate(frame_ids)}
    parts = {name: {"positive": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                    "negative": torch.zeros(THRESHOLD_STEPS + 1,
                                            dtype=torch.long),
                    "baseline": {"predicted": 0, "target": 0, "tp": 0},
                    "frames": 0}
             for name in ("calibration", "heldout")}
    generator = torch.Generator(device=device).manual_seed(seed)
    for rows in tqdm(loader, desc="Reliable addition validation sweep",
                     unit="batch"):
        batch = decoded_addition_batch(
            model, rows, alpha_bars, generator, device,
            eval_timestep=eval_timestep, train_head=False, labels=False)
        fine, logits, masks = (batch[key] for key in ("fine", "logits", "masks"))
        for frame, row in enumerate(rows):
            part = parts[role[row["frame_id"]]]
            part["frames"] += 1
            selected = fine[:, 0] == frame
            empty = fine.new_empty((0, 4))
            clean, faulty, _ = full_frame_voxels(
                row["full_clean_xyz"], row["faulty_xyz"], empty)
            combine_counts(part["baseline"], exact_counts(faulty, clean))
            if masks is None:
                continue
            candidates = selected & masks["candidates"]
            positive, negative = score_histograms(
                logits[candidates].sigmoid(), masks["occupied"][candidates])
            part["positive"] += positive
            part["negative"] += negative
    all_parts = {"positive": parts["calibration"]["positive"] +
                parts["heldout"]["positive"],
                "negative": parts["calibration"]["negative"] +
                parts["heldout"]["negative"],
                "baseline": {key: parts["calibration"]["baseline"][key] +
                             parts["heldout"]["baseline"][key]
                             for key in ("predicted", "target", "tp")},
                "frames": len(frame_ids)}
    parts["all_validation"] = all_parts
    curves = {name: full_roi_curve(part["positive"], part["negative"],
                                   part["baseline"])
              for name, part in parts.items()}
    index = max(range(THRESHOLD_STEPS + 2),
                key=lambda i: (curves["calibration"][i]["iou"],
                               curves["calibration"][i]["precision"],
                               curves["calibration"][i]["threshold"]))
    return {"index": index, "threshold": index / THRESHOLD_STEPS,
            "selected": {name: curve[index] for name, curve in curves.items()},
            "curves": curves,
            "frames": {name: part["frames"] for name, part in parts.items()}}, role


@torch.no_grad()
def _selected(model, loader, alpha_bars, *, device: str, seed: int,
              eval_timestep: int, threshold: float, role: dict | None,
              fault_names: dict[str, str]) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    accumulators = {name: {"faulty": empty_accumulator(),
                           "completed": empty_accumulator()}
                    for name in (("calibration", "heldout") if role is not None
                                 else ("test",))}
    by_fault = {name: {"faulty": empty_accumulator(),
                       "completed": empty_accumulator()}
                for name in sorted(set(fault_names.values()))}
    index = round(threshold * THRESHOLD_STEPS)
    for rows in tqdm(loader, desc="Reliable addition selected metrics",
                     unit="batch"):
        batch = decoded_addition_batch(
            model, rows, alpha_bars, generator, device,
            eval_timestep=eval_timestep, train_head=False, labels=False)
        fine, logits, masks = (batch[key] for key in ("fine", "logits", "masks"))
        chosen = (masks["candidates"] &
                  (torch.floor(logits.sigmoid() * THRESHOLD_STEPS).long()
                   .clamp(0, THRESHOLD_STEPS) >= index)
                  if masks is not None else
                  torch.zeros(len(fine), device=device, dtype=torch.bool))
        for frame, row in enumerate(rows):
            selected = fine[(fine[:, 0] == frame) & chosen].cpu().clone()
            selected[:, 0] = 0
            clean, faulty, completed = full_frame_voxels(
                row["full_clean_xyz"], row["faulty_xyz"], selected)
            name = role[row["frame_id"]] if role is not None else "test"
            for store in (accumulators[name],
                          by_fault[fault_names[row["frame_id"]]]):
                add_frame(store["faulty"], faulty, clean)
                add_frame(store["completed"], completed, clean)
    if role is not None:
        accumulators["all_validation"] = {
            stage: {key: accumulators["calibration"][stage][key] +
                    accumulators["heldout"][stage][key]
                    for key in accumulators["calibration"][stage]}
            for stage in ("faulty", "completed")}
    return {"overall": {name: completion_summary(
                values["faulty"], values["completed"])
                for name, values in accumulators.items()},
            "by_fault": {name: completion_summary(
                values["faulty"], values["completed"])
                for name, values in by_fault.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--addition-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required")
    saved = torch.load(args.addition_checkpoint, map_location="cpu",
                       weights_only=False)
    head_identity = saved.get("identity")
    model, base_identity = load_base_model(
        args.lpcr_checkpoint, args.base_checkpoint, args.device)
    if (not isinstance(head_identity, dict) or
            head_identity.get("revision") != REVISION or
            head_identity.get("base_checkpoint_sha256") != sha256(args.base_checkpoint)
            or head_identity.get("lpcr_checkpoint_sha256") !=
            sha256(args.lpcr_checkpoint) or
            head_identity.get("val_limit") is not None):
        raise ValueError("Expected full-validation addition checkpoint and base")
    for split in ("train", "val"):
        for root, key in ((args.data_cache_root, "data_manifest_sha256"),
                          (args.support_cache_root, "support_manifest_sha256"),
                          (args.faulty_root, "faulty_manifest_sha256")):
            if base_identity[key][split] != sha256(root / split / "manifest.json"):
                raise ValueError(f"{split} cache differs from the base model")
    model.addition_head.load_state_dict(saved["addition_head"])
    model.eval().requires_grad_(False)
    alpha_bars = cosine_alpha_bars(device=args.device)
    val_loader, val_ids, val_manifest = _loader(
        args, "val", base_identity["radar_variant"], base_identity["batch_size"])
    sweep, role = _sweep(
        model, val_loader, val_ids, alpha_bars, device=args.device,
        seed=int(base_identity["seed"]),
        eval_timestep=int(base_identity["eval_timestep"]))
    val = _selected(
        model, val_loader, alpha_bars, device=args.device,
        seed=int(base_identity["seed"]),
        eval_timestep=int(base_identity["eval_timestep"]),
        threshold=sweep["threshold"], role=role,
        fault_names=val_manifest["frame_faults"])
    test_loader, test_ids, test_manifest = _loader(
        args, "test", base_identity["radar_variant"], base_identity["batch_size"])
    if set(test_ids) & set(val_ids):
        raise ValueError("Test frames overlap validation")
    test = _selected(
        model, test_loader, alpha_bars, device=args.device,
        seed=int(base_identity["seed"]),
        eval_timestep=int(base_identity["eval_timestep"]),
        threshold=sweep["threshold"], role=None,
        fault_names=test_manifest["frame_faults"])
    report = {"model": "frozen_clean_latent_denoiser_plus_reliable_addition_head",
              "oracle_clean_latent": True,
              "threshold_selected_by": "calibration_full_ROI_exact_IoU",
              "threshold": sweep["threshold"],
              "base_checkpoint": str(args.base_checkpoint),
              "base_checkpoint_sha256": sha256(args.base_checkpoint),
              "addition_checkpoint": str(args.addition_checkpoint),
              "addition_checkpoint_sha256": sha256(args.addition_checkpoint),
              "addition_epoch": int(saved["epoch"]),
              "full_ROI": "Stage-II 0.2m x 0.2m x 0.25m voxel grid",
              "sweep": sweep, "validation": val, "test": test}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.output_json)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Addition epoch {saved['epoch']}; full-ROI threshold "
          f"{sweep['threshold']:.3f} selected on "
          f"{sweep['frames']['calibration']} validation frames", flush=True)
    for name, values in (("validation heldout", val["overall"]["heldout"]),
                         ("all validation", val["overall"]["all_validation"]),
                         ("official test", test["overall"]["test"])):
        before, after, delta = (values[key] for key in
                                ("faulty", "completed", "improvement"))
        print(f"{name} ({before['frames']} frames, "
              f"{values['added_cells']:,} additions): "
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
    for fault, values in test["by_fault"].items():
        print(f"test {fault}: full-ROI IoU "
              f"{values['faulty']['iou']:.2%} -> "
              f"{values['completed']['iou']:.2%} "
              f"({values['improvement']['iou'] * 100:+.2f} pp)", flush=True)
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
