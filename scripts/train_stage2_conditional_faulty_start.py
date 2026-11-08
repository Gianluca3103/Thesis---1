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
from models.radar_lidar_stage2.conditional_sampling import (
    faulty_to_clean_noise_target)
from models.radar_lidar_stage2.sparse_residual import (
    SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.conditional_stage2 import (
    CONDITIONED_FAULTY_START_REVISION, deployment_loader)
from scripts.evaluate_stage2_conditional_reconstruction import sweep
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, _CleanRows, _atomic_save,
    _batch, _conditioned_inputs, _rows)
from scripts.train_stage2_conditional_from_scratch import (
    fresh_model, full_iou_at_half)


def train_epoch(model: SparseResidualStage2, loader: DataLoader,
                optimizer: torch.optim.Optimizer, alpha_bars: torch.Tensor,
                *, device: str, epoch: int, seed: int,
                timestep: int) -> dict:
    """Learn a clean latent from noisy faulty latent, using clean only as label."""
    model.eval()
    model.diffusion.train()
    model.radar_adapter.train()
    generator = torch.Generator(device=device).manual_seed(seed + epoch)
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
            model, rows, hierarchy, device, train_radar_adapter=True)
        epsilon = torch.randn(faulty_latent.shape, device=device,
                              dtype=faulty_latent.dtype, generator=generator)
        alpha = alpha_bars[timestep]
        xt, target = faulty_to_clean_noise_target(
            faulty_latent, clean_latent, epsilon, alpha)
        predicted = model.diffusion(
            xt, faulty_latent, radar_latent,
            faulty_latent.new_ones((len(coarse), 1)), faulty_present,
            torch.full((len(rows),), timestep, device=device, dtype=torch.long),
            coarse)
        loss = frame_mean((predicted - target).square().mean(1),
                          coarse[:, 0].long(), len(rows))
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
    train_loader = DataLoader(
        _CleanRows(train_dataset, support, args.train_limit,
                   args.faulty_root, "train"),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_rows, pin_memory=True,
        persistent_workers=args.num_workers > 0)
    identity = {
        "revision": CONDITIONED_FAULTY_START_REVISION,
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
        "comparison_peer": "stage2_conditional_scratch_12vox_1p6m_20261108"}
    val_loader, val_ids, _ = deployment_loader(
        args.vod_root, args.data_cache_root, args.support_cache_root,
        args.faulty_root, "val", identity, args.num_workers, args.val_limit)
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
        train = train_epoch(
            model, tqdm(train_loader, desc=f"Conditional train {epoch:03d}",
                        unit="batch"), optimizer, alpha_bars, device=args.device,
            epoch=epoch, seed=args.seed, timestep=args.eval_timestep)
        model.eval()
        validation, _ = sweep(
            model, val_loader, val_ids, args.data_cache_root, alpha_bars,
            device=args.device, seed=args.seed, steps=1,
            sampler="faulty_latent_one_step", timestep=args.eval_timestep)
        full = full_iou_at_half(validation)
        improved = full["iou"] > best_iou
        best_iou = max(best_iou, full["iou"])
        record = {"epoch": epoch,
                  "train_noise_equivalent_mse": train["noise_equivalent_mse"],
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
        print(f"Epoch {epoch:03d}: train_noise_equivalent_MSE="
              f"{train['noise_equivalent_mse']:.4f} "
              f"val_full_IoU@0.5={full['iou']:.2%} "
              f"faulty_baseline={full['faulty_baseline_iou']:.2%} "
              f"best={best_iou:.2%}", flush=True)
    print(f"Best checkpoint: {args.output_root / 'best_iou.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
