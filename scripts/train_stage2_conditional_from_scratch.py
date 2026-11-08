"""Train fresh conditional epsilon denoiser; validate from Gaussian noise."""

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
from models.radar_lidar_stage2.clean_thresholds import THRESHOLD_STEPS
from models.radar_lidar_stage2.fault_only_gate import POLICY
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.conditional_stage2 import (
    CONDITIONED_FRESH_REVISION, CONDITIONED_FRESH_FAULT_ONLY_REVISION,
    deployment_loader)
from scripts.evaluate_stage2_conditional_reconstruction import sweep
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, _CleanRows, _atomic_save,
    _rows, _run_epoch)


def full_iou_at_half(sweep_report: dict) -> dict:
    index = THRESHOLD_STEPS // 2
    rows = [sweep_report["curves"][name][index]
            for name in ("calibration", "heldout")]
    tp = sum(row["tp"] for row in rows)
    predicted = sum(row["predicted"] for row in rows)
    target = sum(row["target"] for row in rows)
    union = predicted + target - tp
    baseline = [sweep_report["curves"][name][-1]
                for name in ("calibration", "heldout")]
    base_tp = sum(row["tp"] for row in baseline)
    base_predicted = sum(row["predicted"] for row in baseline)
    base_target = sum(row["target"] for row in baseline)
    base_union = base_predicted + base_target - base_tp
    result = {"threshold": 0.5, "tp": tp, "predicted": predicted,
              "target": target, "iou": tp / union if union else 0.0,
              "faulty_baseline_iou": (base_tp / base_union
                                      if base_union else 0.0)}
    if "fixed_half_tolerant" in sweep_report:
        tolerant = {key: sum(sweep_report["fixed_half_tolerant"][role][key]
                             for role in ("calibration", "heldout"))
                    for key in ("predicted", "eligible_target", "exact_tp",
                                "tolerant_tp_0p2m")}
        if (tolerant["predicted"] != predicted or
                tolerant["eligible_target"] != target or
                tolerant["exact_tp"] != tp):
            raise RuntimeError("Exact and 0.2 m validation counts differ")
        tolerant_union = (predicted + target -
                          tolerant["tolerant_tp_0p2m"])
        result["iou_0p2m"] = (tolerant["tolerant_tp_0p2m"] /
                              tolerant_union if tolerant_union else 0.0)
    return result


def fresh_model(lpcr_path: Path, device: str) -> SparseResidualStage2:
    pretrained = torch.load(lpcr_path, map_location="cpu", weights_only=False)
    if (pretrained.get("version") != VERSION or
            pretrained.get("model_revision") != LPCR_REVISION or
            pretrained.get("phase") != "lpcr" or
            not pretrained.get("lpcr_full_region")):
        raise ValueError("Expected the frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(device)
    fresh_diffusion = {key: value.clone() for key, value in
                       model.diffusion.state_dict().items()}
    fresh_radar = {key: value.clone() for key, value in
                   model.radar_adapter.state_dict().items()}
    missing, unexpected = model.load_state_dict(pretrained["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from Stage II architecture")
    model.diffusion.load_state_dict(fresh_diffusion)
    model.radar_adapter.load_state_dict(fresh_radar)
    model.requires_grad_(False)
    model.diffusion.requires_grad_(True)
    model.radar_adapter.requires_grad_(True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-steps", type=int, default=64)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.lr <= 0 or not 2 <= args.validation_steps <= 1000 or
            any(value is not None and value < 1 for value in
                (args.train_limit, args.val_limit))):
        parser.error("Invalid settings or CUDA unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = fresh_model(args.lpcr_checkpoint, args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    alpha_bars = cosine_alpha_bars(device=args.device)

    train_dataset = paired_dataset(
        args.vod_root, "train", "radar_20frames_verified_doppler_radial",
        include_clean=True, data_cache_root=args.data_cache_root)
    train_ids = [frame.frame_id for frame in train_dataset.frames]
    expected = support_identity(
        "train", train_ids, "radar_20frames_verified_doppler_radial",
        data_cache_root=args.data_cache_root, radius_m=RADIUS_M,
        min_radar_voxels=MIN_RADAR_VOXELS)
    support = RadarSupportCache(args.support_cache_root / "train", expected)
    support.check_complete()
    train_regions = None
    if args.fault_region_root is not None:
        train_regions = FaultRegionCache(
            args.fault_region_root / "train",
            cache_identity("train", train_ids,
                           args.faulty_root / "train" / "manifest.json",
                           args.data_cache_root / "train" / "manifest.json"))
        if train_regions.regions is None:
            raise FileNotFoundError("Train injector footprints are not built")
    train_loader = DataLoader(
        _CleanRows(train_dataset, support, args.train_limit,
                   args.faulty_root, "train", fault_regions=train_regions),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_rows, pin_memory=True,
        persistent_workers=args.num_workers > 0)
    identity = {
        "revision": (CONDITIONED_FRESH_FAULT_ONLY_REVISION
                     if train_regions is not None else CONDITIONED_FRESH_REVISION),
        "lpcr_sha256": sha256(args.lpcr_checkpoint),
        "radar_variant": "radar_20frames_verified_doppler_radial",
        "radius_m": RADIUS_M,
        "min_distinct_radar_voxels": MIN_RADAR_VOXELS,
        "data_manifest_sha256": {
            split: sha256(args.data_cache_root / split / "manifest.json")
            for split in ("train", "val")},
        "support_manifest_sha256": {
            split: sha256(args.support_cache_root / split / "manifest.json")
            for split in ("train", "val")},
        "faulty_manifest_sha256": {
            split: sha256(args.faulty_root / split / "manifest.json")
            for split in ("train", "val")},
        "train_limit": args.train_limit, "val_limit": args.val_limit,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "validation_steps": args.validation_steps,
        "validation_model_inputs": "faulty_LiDAR_radar_Gaussian_noise",
        "fresh_diffusion_and_radar_adapter": True}
    if train_regions is not None:
        val_dataset_for_regions = paired_dataset(
            args.vod_root, "val", identity["radar_variant"],
            include_clean=False, data_cache_root=args.data_cache_root)
        val_region_ids = [frame.frame_id for frame in val_dataset_for_regions.frames]
        val_regions = FaultRegionCache(
            args.fault_region_root / "val",
            cache_identity("val", val_region_ids,
                           args.faulty_root / "val" / "manifest.json",
                           args.data_cache_root / "val" / "manifest.json"))
        if val_regions.regions is None:
            raise FileNotFoundError("Validation injector footprints are not built")
        identity.update({"fault_region_policy": POLICY,
                         "fault_region_sha256": {
                             "train": sha256(train_regions.regions_path),
                             "val": sha256(val_regions.regions_path)}})
    val_loader, val_ids, _ = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, "val", identity, args.num_workers, args.val_limit,
        fault_region_root=args.fault_region_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    start, best_iou = 1, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Resume checkpoint belongs to another scratch run")
        model.diffusion.load_state_dict(saved["diffusion"])
        model.radar_adapter.load_state_dict(saved["radar_adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["epoch"]) + 1
        best_iou = float(saved["best_val_full_iou_at_0p5"])
    if start > args.epochs:
        print(f"Already complete: epoch {start - 1}", flush=True)
        return

    for epoch in range(start, args.epochs + 1):
        train = _run_epoch(
            model, tqdm(train_loader, desc=f"Conditional train {epoch:03d}",
                        unit="batch"), optimizer, alpha_bars,
            device=args.device, epoch=epoch, eval_timestep=499,
            seed=args.seed, condition_faulty_radar=True,
            train_radar_adapter=True)
        model.eval()
        validation, _ = sweep(
            model, val_loader, val_ids, args.data_cache_root, alpha_bars,
            device=args.device, seed=args.seed, steps=args.validation_steps,
            fixed_half_tolerance=True)
        full = full_iou_at_half(validation)
        improved = full["iou"] > best_iou
        best_iou = max(best_iou, full["iou"])
        record = {"epoch": epoch, "train_noise_mse": train["noise_mse"],
                  "val_full_roi_at_0p5": full,
                  "val_frames": len(val_ids),
                  "val_calibration_selected": validation["calibration"],
                  "val_threshold_heldout": validation["heldout"],
                  "best_val_full_iou_at_0p5": best_iou}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"identity": identity, "epoch": epoch,
                      "diffusion": model.diffusion.state_dict(),
                      "radar_adapter": model.radar_adapter.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "best_val_full_iou_at_0p5": best_iou,
                      "record": record}
        _atomic_save(args.output_root / "last.ckpt", checkpoint)
        if improved:
            _atomic_save(args.output_root / "best_iou.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: train_noise_MSE={train['noise_mse']:.4f} "
              f"val_full_exact_IoU={full['iou']:.2%} "
              f"val_full_IoU@0.2m={full['iou_0p2m']:.2%} "
              f"faulty_full_exact_IoU={full['faulty_baseline_iou']:.2%} "
              f"heldout_calibrated_exact_IoU={validation['heldout']['iou']:.2%} "
              f"calibrated_probability_threshold={validation['threshold']:.3f} "
              f"occupancy_probability_threshold=0.5 "
              f"best_fixed_threshold_exact_IoU={best_iou:.2%}", flush=True)
    print(f"Best checkpoint: {args.output_root / 'best_iou.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
