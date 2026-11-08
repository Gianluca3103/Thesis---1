"""Train faulty-LiDAR latent to clean-LiDAR latent restoration on a fixed gate."""

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
from models.radar_lidar_stage2.clean_denoising import frame_mean
from models.radar_lidar_stage2.fault_only_gate import POLICY
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.conditional_sampling import (
    faulty_to_clean_noise_target)
from models.radar_lidar_stage2.sparse_residual import (
    SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.conditional_stage2 import (
    CONDITIONED_FAULTY_START_REVISION, CONDITIONED_FAULT_ONLY_REVISION,
    deployment_loader)
from scripts.evaluate_stage2_conditional_reconstruction import sweep
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, _CleanRows, _atomic_save,
    _batch, _conditioned_inputs, _rows)
from scripts.train_stage2_conditional_from_scratch import (
    fresh_model, full_iou_at_half)


def loss_epoch(model: SparseResidualStage2, loader: DataLoader,
               optimizer: torch.optim.Optimizer | None,
               alpha_bars: torch.Tensor, *, device: str, epoch: int,
               seed: int, timestep: int) -> dict:
    """One objective: noise-equivalent MSE to the clean latent."""
    training = optimizer is not None
    model.eval()
    model.diffusion.train(training)
    model.radar_adapter.train(training)
    # Fixed validation noise makes successive epoch losses comparable.
    generator = torch.Generator(device=device).manual_seed(
        seed + epoch if training else seed)
    weighted_loss = 0.0
    frames = 0
    for rows in loader:
        hierarchy, clean, clean_ids = _batch(rows, device)
        coarse = hierarchy[-1]
        frames += len(rows)
        if not len(coarse):
            continue
        with torch.no_grad():
            clean_latent, _ = model.lidar_encoder(clean, coarse, clean_ids)
        faulty_latent, faulty_present, radar_latent = _conditioned_inputs(
            model, rows, hierarchy, device,
            train_radar_adapter=training)
        epsilon = torch.randn(faulty_latent.shape, device=device,
                              dtype=faulty_latent.dtype, generator=generator)
        alpha = alpha_bars[timestep]
        xt, target = faulty_to_clean_noise_target(
            faulty_latent, clean_latent, epsilon, alpha)
        with torch.set_grad_enabled(training):
            predicted = model.diffusion(
                xt, faulty_latent, radar_latent,
                faulty_latent.new_ones((len(coarse), 1)), faulty_present,
                torch.full((len(rows),), timestep, device=device,
                           dtype=torch.long), coarse)
            loss = frame_mean((predicted - target).square().mean(1),
                              coarse[:, 0].long(), len(rows))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters()
                     if parameter.requires_grad), 1.0)
                optimizer.step()
        weighted_loss += float(loss.detach()) * len(rows)
    return {"noise_equivalent_mse": weighted_loss / frames, "frames": frames}


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
    parser.add_argument("--eval-timestep", type=int, default=499)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.lr <= 0 or not 1 <= args.eval_timestep < 999 or
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
        "revision": (CONDITIONED_FAULT_ONLY_REVISION if train_regions is not None
                     else CONDITIONED_FAULTY_START_REVISION),
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
        "sampler": "faulty_latent_one_step",
        "eval_timestep": args.eval_timestep,
        "training_target": "noise_equivalent_faulty_to_clean_latent",
        "validation_model_inputs": "faulty_LiDAR_radar_noisy_faulty_latent",
        "fresh_diffusion_and_radar_adapter": True,
        "comparison_peer": ("conditional_faulty_start_12vox_1p6m"
                            if train_regions is not None else
                            "stage2_conditional_scratch_12vox_1p6m_20261108")}
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
        identity.update({
            "fault_region_policy": POLICY,
            "fault_region_sha256": {
                "train": sha256(train_regions.regions_path),
                "val": sha256(val_regions.regions_path)}})
    else:
        val_regions = None
    val_loader, val_ids, _ = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, "val", identity, args.num_workers, args.val_limit,
        fault_region_root=args.fault_region_root)
    val_dataset = paired_dataset(
        args.vod_root, "val", identity["radar_variant"], include_clean=True,
        data_cache_root=args.data_cache_root)
    if [frame.frame_id for frame in val_dataset.frames][:len(val_ids)] != val_ids:
        raise ValueError("Validation loss and deployment frame IDs differ")
    val_support = RadarSupportCache(
        args.support_cache_root / "val",
        support_identity(
            "val", [frame.frame_id for frame in val_dataset.frames],
            identity["radar_variant"], data_cache_root=args.data_cache_root,
            radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS))
    val_support.check_complete()
    val_loss_loader = DataLoader(
        _CleanRows(val_dataset, val_support, args.val_limit,
                   args.faulty_root, "val", fault_regions=val_regions),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_rows, pin_memory=True,
        persistent_workers=args.num_workers > 0)
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
        train = loss_epoch(
            model, tqdm(train_loader, desc=f"Conditional train {epoch:03d}",
                        unit="batch"), optimizer, alpha_bars, device=args.device,
            epoch=epoch, seed=args.seed, timestep=args.eval_timestep)
        val_loss = loss_epoch(
            model, tqdm(val_loss_loader, desc=f"Conditional val loss {epoch:03d}",
                        unit="batch"), None, alpha_bars, device=args.device,
            epoch=epoch, seed=args.seed, timestep=args.eval_timestep)
        validation, _ = sweep(
            model, val_loader, val_ids, args.data_cache_root, alpha_bars,
            device=args.device, seed=args.seed, steps=1,
            sampler="faulty_latent_one_step", timestep=args.eval_timestep,
            fixed_half_tolerance=True)
        full = full_iou_at_half(validation)
        tolerant_counts = {
            key: sum(validation["fixed_half_tolerant"][role][key]
                     for role in ("calibration", "heldout"))
            for key in ("predicted", "eligible_target", "exact_tp",
                        "tolerant_tp_0p2m")}
        if (tolerant_counts["predicted"] != full["predicted"] or
                tolerant_counts["eligible_target"] != full["target"] or
                tolerant_counts["exact_tp"] != full["tp"]):
            raise RuntimeError("Exact and 0.2 m validation counts differ")
        tolerant_union = (tolerant_counts["predicted"] +
                          tolerant_counts["eligible_target"] -
                          tolerant_counts["tolerant_tp_0p2m"])
        full["iou_0p2m"] = (tolerant_counts["tolerant_tp_0p2m"] /
                            tolerant_union if tolerant_union else 0.0)
        improved = full["iou"] > best_iou
        best_iou = max(best_iou, full["iou"])
        record = {"epoch": epoch,
                  "train_noise_equivalent_mse": train["noise_equivalent_mse"],
                  "val_noise_equivalent_mse": val_loss["noise_equivalent_mse"],
                  "loss_terms": ["noise_equivalent_mse"],
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
        print(f"Epoch {epoch:03d}: train_loss(noise_equivalent_MSE)="
              f"{train['noise_equivalent_mse']:.4f} "
              f"val_loss(noise_equivalent_MSE)="
              f"{val_loss['noise_equivalent_mse']:.4f} "
              f"val_exact_IoU(occupancy_threshold=0.5)={full['iou']:.2%} "
              f"val_IoU@0.2m(occupancy_threshold=0.5)="
              f"{full['iou_0p2m']:.2%} "
              f"faulty_baseline={full['faulty_baseline_iou']:.2%} "
              f"best={best_iou:.2%}", flush=True)
    print(f"Best checkpoint: {args.output_root / 'best_iou.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
