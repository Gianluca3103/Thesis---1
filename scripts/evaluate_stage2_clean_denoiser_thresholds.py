"""Sweep eligible occupancy thresholds for the clean-LiDAR denoiser.

The checkpoint was selected on validation noise MSE. Calibration and held-out
below apply only to occupancy-threshold selection; they are not an independent
test set for model selection.
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
from models.radar_lidar_stage2.clean_thresholds import (
    THRESHOLD_STEPS, score_histograms, selected_indices, sweep_rows)
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, REVISION, _batch, _CleanRows, _rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available() or args.num_workers < 0:
        parser.error("CUDA is required and num-workers must be nonnegative")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    if not isinstance(identity, dict) or identity.get("revision") != REVISION:
        raise ValueError("Expected a clean 12-voxel/1.6 m denoiser checkpoint")
    if identity.get("lpcr_sha256") != sha256(args.lpcr_checkpoint):
        raise ValueError("Pretrained LPCR checkpoint differs from training")
    if identity.get("radius_m") != RADIUS_M or identity.get(
            "min_distinct_radar_voxels") != MIN_RADAR_VOXELS:
        raise ValueError("Checkpoint uses a different radar support rule")
    for split in ("train", "val"):
        if identity["data_manifest_sha256"][split] != sha256(
                args.data_cache_root / split / "manifest.json"):
            raise ValueError(f"Fixed {split} frames differ from training")
        if identity["support_manifest_sha256"][split] != sha256(
                args.support_cache_root / split / "manifest.json"):
            raise ValueError(f"Radar support {split} differs from training")
    if identity.get("val_limit") is not None:
        raise ValueError("Full threshold sweep requires a full-validation checkpoint")

    saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu",
                            weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected the frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(args.device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from the architecture")
    model.diffusion.load_state_dict(saved["diffusion"])
    model.eval().requires_grad_(False)

    dataset = paired_dataset(args.vod_root, "val", identity["radar_variant"],
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    expected = support_identity(
        "val", frame_ids, identity["radar_variant"],
        data_cache_root=args.data_cache_root,
        radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS)
    cache = RadarSupportCache(args.support_cache_root / "val", expected)
    cache.check_complete()
    loader = DataLoader(
        _CleanRows(dataset, cache, None), batch_size=identity["batch_size"],
        shuffle=False, num_workers=args.num_workers, collate_fn=_rows,
        pin_memory=True, persistent_workers=args.num_workers > 0)
    # Adjacent frames can be correlated. Hold out four of every five 20-frame
    # blocks when selecting occupancy thresholds.
    role = {frame_id: ("calibration" if (index // 20) % 5 == 0 else "heldout")
            for index, frame_id in enumerate(frame_ids)}
    histograms = {name: {"positive": torch.zeros(THRESHOLD_STEPS + 1,
                                                dtype=torch.long),
                         "negative": torch.zeros(THRESHOLD_STEPS + 1,
                                                dtype=torch.long),
                         "frames": 0, "candidate_cells": 0,
                         "latent_sites": 0}
                  for name in ("calibration", "heldout")}
    alpha_bars = cosine_alpha_bars(device=args.device)
    timestep = int(identity["eval_timestep"])
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(identity["seed"]))
    with torch.no_grad():
        for rows in tqdm(loader, desc="Clean denoiser threshold sweep",
                         unit="batch"):
            hierarchy, clean, point_ids = _batch(rows, args.device)
            fine, coarse = hierarchy[0], hierarchy[-1]
            calibration_frames = torch.tensor(
                [role[row["frame_id"]] == "calibration" for row in rows],
                dtype=torch.bool, device=args.device)
            fine_calibration = calibration_frames[fine[:, 0].long()]
            coarse_calibration = calibration_frames[coarse[:, 0].long()]
            for name, frame_mask, fine_mask, coarse_mask in (
                    ("calibration", calibration_frames, fine_calibration,
                     coarse_calibration),
                    ("heldout", ~calibration_frames, ~fine_calibration,
                     ~coarse_calibration)):
                store = histograms[name]
                store["frames"] += int(frame_mask.sum())
                store["candidate_cells"] += int(fine_mask.sum())
                store["latent_sites"] += int(coarse_mask.sum())
            if not len(coarse):
                continue
            x0, _ = model.lidar_encoder(clean, coarse, point_ids)
            alpha = alpha_bars[timestep]
            epsilon = torch.randn(x0.shape, dtype=x0.dtype,
                                  device=args.device, generator=generator)
            xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
            prediction = model.diffusion(
                xt, torch.zeros_like(x0), torch.zeros_like(x0),
                x0.new_ones((len(x0), 1)),
                torch.zeros(len(x0), dtype=torch.bool, device=args.device),
                torch.full((len(rows),), timestep, device=args.device,
                           dtype=torch.long), coarse)
            recovered = (xt - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
            logits, _ = model.lpcr(recovered.clamp(-1, 1), hierarchy,
                                   x0.new_ones((len(fine), 1)))
            target, _, _ = offset_targets(clean, fine, point_ids)
            probabilities = logits.sigmoid()
            for name, frame_mask in (("calibration", fine_calibration),
                                     ("heldout", ~fine_calibration)):
                positive, negative = score_histograms(
                    probabilities[frame_mask], target[frame_mask])
                store = histograms[name]
                store["positive"] += positive
                store["negative"] += negative

    calibration = histograms["calibration"]
    heldout = histograms["heldout"]
    curves = {name: sweep_rows(store["positive"], store["negative"])
              for name, store in histograms.items()}
    curves["all_validation"] = sweep_rows(
        calibration["positive"] + heldout["positive"],
        calibration["negative"] + heldout["negative"])
    indices = selected_indices(curves["calibration"])
    selected = {label: (None if index is None else {
        "threshold": curves["calibration"][index]["threshold"],
        "calibration": curves["calibration"][index],
        "heldout": curves["heldout"][index],
        "all_validation": curves["all_validation"][index]})
        for label, index in indices.items()}
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": int(saved["epoch"]),
        "model_revision": REVISION, "sampling": "one_step_clean_x0_at_fixed_t",
        "eval_timestep": timestep, "alpha_bar": float(alpha),
        "threshold_steps": THRESHOLD_STEPS,
        "target_definition": "clean_LiDAR_occupied_voxels_within_12_radar_voxels_1p6m_gate",
        "prediction_definition": "frozen_LPCR_occupancy_from_one_step_denoised_clean_latent",
        "validation_note": ("All frames were used to select the checkpoint by noise MSE. "
                            "Held-out here means held out from threshold selection only, "
                            "not an independent model test set."),
        "split_rule": "one calibration block and four held-out blocks per five consecutive 20-frame blocks",
        "frames": {name: store["frames"] for name, store in histograms.items()},
        "mean_candidate_cells": {
            name: store["candidate_cells"] / max(store["frames"], 1)
            for name, store in histograms.items()},
        "mean_latent_sites": {
            name: store["latent_sites"] / max(store["frames"], 1)
            for name, store in histograms.items()},
        "training_validation_at_checkpoint": saved.get("record", {}).get("val"),
        "selected": selected, "curves": curves,
        "calibration_frame_ids": [item for item in frame_ids
                                  if role[item] == "calibration"],
        "heldout_frame_ids": [item for item in frame_ids
                              if role[item] == "heldout"]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.output_json)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Checkpoint epoch {saved['epoch']}: {calibration['frames']} calibration, "
          f"{heldout['frames']} threshold-held-out frames", flush=True)
    for label, value in selected.items():
        if value is None:
            print(f"{label}: no calibration threshold satisfies constraint",
                  flush=True)
            continue
        row = value["heldout"]
        print(f"{label}: threshold={value['threshold']:.3f}, "
              f"heldout P={row['precision']:.2%}, R={row['recall']:.2%}, "
              f"F1={row['f1']:.2%}, IoU={row['iou']:.2%}, "
              f"predicted={row['predicted']:,}, target={row['target']:,}",
              flush=True)
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
