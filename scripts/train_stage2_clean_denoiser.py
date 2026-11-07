"""Train epsilon-only clean LiDAR denoising on a fixed radar-supported grid.

This is a clean-input diagnostic. The frozen LPCR encoder supplies x0 from
clean LiDAR inside the gate; validation decodes the denoised x0 for occupancy
metrics. No faulty LiDAR, fault mask, ray loss, offset loss, or occupancy loss
enters training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import supported_coordinates
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.clean_denoising import (
    batched_hierarchy, clean_points_in_support, frame_mean,
    occupancy_counts, occupancy_metrics)
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256


REVISION = "stage2_clean_12vox_1p6m_epsilon_denoiser_v1"
REVERSE_REVISION = "stage2_clean_12vox_1p6m_fullnoise_reverse_5train_3val_v1"
RADIUS_M = 1.6
MIN_RADAR_VOXELS = 12
RADAR_VARIANT = "radar_20frames_verified_doppler_radial"


class _CleanRows:
    def __init__(self, dataset, cache: RadarSupportCache, limit: int | None):
        self.dataset = dataset
        self.cache = cache
        self.limit = len(dataset) if limit is None else min(len(dataset), limit)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> dict:
        row = self.dataset[index]
        mask = self.cache.load(row["frame_id"])
        return {"frame_id": row["frame_id"],
                "fine_zyx": supported_coordinates(mask),
                "clean_xyz": clean_points_in_support(row["clean_lidar"][:, :3], mask)}


def _rows(rows: list[dict]) -> list[dict]:
    return rows


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch(rows: list[dict], device: str):
    hierarchy = batched_hierarchy([row["fine_zyx"] for row in rows], device)
    clouds = [row["clean_xyz"] for row in rows]
    clean = torch.cat(clouds).to(device, non_blocking=True)
    point_ids = torch.cat([torch.full((len(cloud),), frame, dtype=torch.long)
                           for frame, cloud in enumerate(clouds)]).to(device)
    return hierarchy, clean, point_ids


def reverse_schedule(count: int, *, total: int = 1000,
                     final_timestep: int = 199) -> torch.Tensor:
    """Nested five/three-step DDIM schedules with a shared terminal noise level."""
    if count < 2 or total <= final_timestep or final_timestep < 1:
        raise ValueError("Invalid reverse-step schedule")
    schedule = torch.linspace(total - 1, final_timestep, count).round().long()
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("Reverse timesteps must be distinct")
    return schedule


def reverse_step(xt: torch.Tensor, predicted_epsilon: torch.Tensor,
                 alpha: torch.Tensor, next_alpha: torch.Tensor | None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Deterministic DDIM update; bound only the estimated clean latent."""
    x0_hat = ((xt - (1 - alpha).sqrt() * predicted_epsilon)
              / alpha.sqrt()).clamp(-1, 1)
    if next_alpha is None:
        return x0_hat, None
    following = (next_alpha.sqrt() * x0_hat
                 + (1 - next_alpha).sqrt() * predicted_epsilon)
    return x0_hat, following


def _run_epoch(model: SparseResidualStage2, loader: DataLoader,
               optimizer: torch.optim.Optimizer | None,
               alpha_bars: torch.Tensor, *, device: str, epoch: int,
               eval_timestep: int, seed: int,
               reverse_train_steps: int = 0,
               reverse_val_steps: int = 0) -> dict:
    training = optimizer is not None
    model.eval()
    model.diffusion.train(training)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + epoch if training else seed)
    loss_sum = latent_mse_sum = 0.0
    frames = fine_sites = latent_sites = present_latents = 0
    counts = {"tp": 0, "predicted": 0, "target": 0}
    oracle = dict(counts)
    description = f"Clean denoiser {'train' if training else 'val'} {epoch:03d}"
    reverse_count = reverse_train_steps if training else reverse_val_steps
    schedule = (reverse_schedule(reverse_count, total=len(alpha_bars)).to(device)
                if reverse_count else None)
    for rows in tqdm(loader, desc=description, unit="batch"):
        hierarchy, clean, point_ids = _batch(rows, device)
        frames += len(rows)
        fine_sites += len(hierarchy[0])
        coarse = hierarchy[-1]
        latent_sites += len(coarse)
        if not len(coarse):
            continue
        with torch.no_grad():
            x0, present = model.lidar_encoder(clean, coarse, point_ids)
        present_latents += int(present.sum())
        latent_ids = coarse[:, 0].long()
        zero_condition = torch.zeros_like(x0)
        site_probability = x0.new_ones((len(x0), 1))
        absent = torch.zeros(len(x0), dtype=torch.bool, device=device)
        if reverse_count:
            # Start from Gaussian noise. Each epsilon target is computed for the
            # *actual rolled-out state*, not for a different forward-noised x0.
            # Detaching the DDIM transition keeps the five sparse U-Net passes
            # from retaining a five-step GPU autograd graph.
            xt = torch.randn(x0.shape, dtype=x0.dtype,
                             device=device, generator=generator)
            losses = []
            if training:
                optimizer.zero_grad(set_to_none=True)
            for index, timestep in enumerate(schedule):
                alpha = alpha_bars[timestep]
                epsilon_target = (xt - alpha.sqrt() * x0) / (1 - alpha).sqrt()
                steps = timestep.expand(len(rows))
                with torch.set_grad_enabled(training):
                    predicted_noise = model.diffusion(
                        xt, zero_condition, zero_condition,
                        site_probability, absent, steps, coarse)
                    step_loss = frame_mean(
                        (predicted_noise - epsilon_target).square().mean(1),
                        latent_ids, len(rows))
                    if training:
                        (step_loss / reverse_count).backward()
                losses.append(step_loss.detach())
                next_alpha = (alpha_bars[schedule[index + 1]]
                              if index + 1 < reverse_count else None)
                with torch.no_grad():
                    recovered, following = reverse_step(
                        xt, predicted_noise.detach(), alpha, next_alpha)
                    if following is not None:
                        xt = following
            loss = torch.stack(losses).mean()
            if training:
                torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
                optimizer.step()
        else:
            if training:
                steps = torch.randint(len(alpha_bars), (len(rows),),
                                      device=device, generator=generator)
            else:
                steps = torch.full((len(rows),), eval_timestep,
                                   device=device, dtype=torch.long)
            alpha = alpha_bars[steps[latent_ids], None]
            epsilon = torch.randn(x0.shape, dtype=x0.dtype,
                                  device=device, generator=generator)
            xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
            with torch.set_grad_enabled(training):
                predicted_noise = model.diffusion(
                    xt, zero_condition, zero_condition,
                    site_probability, absent, steps, coarse)
                loss = frame_mean((predicted_noise - epsilon).square().mean(1),
                                  latent_ids, len(rows))
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
                    optimizer.step()
        loss_sum += float(loss.detach()) * len(rows)
        if not training:
            with torch.no_grad():
                if not reverse_count:
                    recovered = ((xt - (1 - alpha).sqrt() * predicted_noise)
                                 / alpha.sqrt())
                latent_mse = frame_mean((recovered - x0).square().mean(1),
                                        latent_ids, len(rows))
                latent_mse_sum += float(latent_mse) * len(rows)
                # Both decoder passes and labels use exactly the gated sites.
                site_features = x0.new_ones((len(hierarchy[0]), 1))
                denoised_logits, _ = model.lpcr(recovered.clamp(-1, 1),
                                                hierarchy, site_features)
                oracle_logits, _ = model.lpcr(x0, hierarchy, site_features)
                target, _, _ = offset_targets(clean, hierarchy[0], point_ids)
                for store, logits in ((counts, denoised_logits),
                                      (oracle, oracle_logits)):
                    current = occupancy_counts(logits, target)
                    for key in store:
                        store[key] += current[key]
    output = {"frames": frames,
              "noise_mse": loss_sum / max(frames, 1),
              "mean_candidate_cells": fine_sites / max(frames, 1),
              "mean_latent_sites": latent_sites / max(frames, 1),
              "present_latent_fraction": present_latents / max(latent_sites, 1)}
    if not training:
        output.update({"recovered_clean_latent_mse": latent_mse_sum / max(frames, 1),
                       "eligible_occupancy": occupancy_metrics(counts),
                       "frozen_lpcr_oracle": occupancy_metrics(oracle),
                       "occupancy_threshold": 0.5,
                       "eval_timestep": eval_timestep if not reverse_count else None})
    if reverse_count:
        output.update({"sampling": "sequential_DDIM_from_Gaussian_noise",
                       "reverse_steps": reverse_count,
                       "reverse_schedule": schedule.tolist(),
                       "reverse_terminal_timestep": int(schedule[-1]),
                       "conditioning": "gate_coordinates_only"})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default=RADAR_VARIANT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--eval-timestep", type=int, default=499,
                        help="Fixed zero-based diffusion timestep for validation")
    parser.add_argument("--reverse-train-steps", type=int, default=0,
                        help="Sequential DDIM steps from Gaussian noise during training")
    parser.add_argument("--reverse-val-steps", type=int, default=0,
                        help="Sequential DDIM steps from Gaussian noise during validation")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.lr <= 0 or not 0 <= args.eval_timestep < 1000 or
            ((args.reverse_train_steps, args.reverse_val_steps) not in
             ((0, 0), (5, 3))) or
            any(value is not None and value < 1 for value in
                (args.train_limit, args.val_limit))):
        parser.error("Invalid training, validation, or diffusion settings")
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("This sparse U-Net run requires CUDA and MinkowskiEngine")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu",
                            weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected the pretrained full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(args.device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from this Stage II architecture")
    model.requires_grad_(False)
    model.diffusion.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.diffusion.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    alpha_bars = cosine_alpha_bars(device=args.device)

    loaders = {}
    support_hashes = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        dataset = paired_dataset(args.vod_root, split, args.radar_variant,
                                 include_clean=True,
                                 data_cache_root=args.data_cache_root)
        expected = support_identity(
            split, [frame.frame_id for frame in dataset.frames],
            args.radar_variant, data_cache_root=args.data_cache_root,
            radius_m=RADIUS_M, min_radar_voxels=MIN_RADAR_VOXELS)
        cache = RadarSupportCache(args.support_cache_root / split, expected)
        cache.check_complete()
        support_hashes[split] = sha256(args.support_cache_root / split /
                                       "manifest.json")
        rows = _CleanRows(dataset, cache, limit)
        loaders[split] = DataLoader(
            rows, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.num_workers, collate_fn=_rows,
            pin_memory=True, persistent_workers=args.num_workers > 0)

    identity = {"revision": REVERSE_REVISION if args.reverse_train_steps else REVISION,
                "lpcr_sha256": sha256(args.lpcr_checkpoint),
                "support_manifest_sha256": support_hashes,
                "data_manifest_sha256": {
                    split: sha256(args.data_cache_root / split / "manifest.json")
                    for split in ("train", "val")},
                "radar_variant": args.radar_variant,
                "radius_m": RADIUS_M,
                "min_distinct_radar_voxels": MIN_RADAR_VOXELS,
                "train_limit": args.train_limit, "val_limit": args.val_limit,
                "batch_size": args.batch_size, "lr": args.lr,
                "eval_timestep": args.eval_timestep, "seed": args.seed}
    if args.reverse_train_steps:
        identity.update({"reverse_train_steps": args.reverse_train_steps,
                         "reverse_val_steps": args.reverse_val_steps,
                         "reverse_train_schedule": reverse_schedule(5).tolist(),
                         "reverse_val_schedule": reverse_schedule(3).tolist(),
                         "sampling": "sequential_DDIM_from_Gaussian_noise",
                         "conditioning": "gate_coordinates_only"})
    start, best = 1, float("inf")
    best_iou = -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Resume checkpoint belongs to a different denoising run")
        model.diffusion.load_state_dict(saved["diffusion"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_val_noise_mse"])
        if args.reverse_train_steps:
            best_iou = float(saved["best_val_iou_at_0p5"])
    history = args.output_root / "history.jsonl"
    if start > args.epochs:
        print(f"Already complete: epoch {start - 1}; {args.output_root}", flush=True)
        return
    print(f"Clean-only epsilon denoising on >=12 distinct radar voxels within 1.6 m; "
          f"{len(loaders['train'].dataset)} train and {len(loaders['val'].dataset)} "
          f"validation frames. Reverse train/val steps="
          f"{args.reverse_train_steps}/{args.reverse_val_steps}. "
          "No fault input or task losses.", flush=True)
    for epoch in range(start, args.epochs + 1):
        train = _run_epoch(model, loaders["train"], optimizer, alpha_bars,
                           device=args.device, epoch=epoch,
                           eval_timestep=args.eval_timestep, seed=args.seed,
                           reverse_train_steps=args.reverse_train_steps,
                           reverse_val_steps=args.reverse_val_steps)
        val = _run_epoch(model, loaders["val"], None, alpha_bars,
                         device=args.device, epoch=epoch,
                         eval_timestep=args.eval_timestep, seed=args.seed,
                         reverse_train_steps=args.reverse_train_steps,
                         reverse_val_steps=args.reverse_val_steps)
        improved = val["noise_mse"] < best
        if improved:
            best = val["noise_mse"]
        improved_iou = (args.reverse_train_steps > 0 and
                        val["eligible_occupancy"]["iou"] > best_iou)
        if improved_iou:
            best_iou = val["eligible_occupancy"]["iou"]
        record = {"epoch": epoch, "train": train, "val": val,
                  "best_val_noise_mse": best,
                  "validation_scope": f"{len(loaders['val'].dataset)} clean frames"}
        if args.reverse_train_steps:
            record["best_val_iou_at_0p5"] = best_iou
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"identity": identity, "epoch": epoch,
                      "best_val_noise_mse": best,
                      "diffusion": model.diffusion.state_dict(),
                      "optimizer": optimizer.state_dict(), "record": record}
        if args.reverse_train_steps:
            checkpoint["best_val_iou_at_0p5"] = best_iou
        _atomic_save(args.output_root / "last.ckpt", checkpoint)
        if improved:
            _atomic_save(args.output_root / "best.ckpt", checkpoint)
        if improved_iou:
            _atomic_save(args.output_root / "best_iou.ckpt", checkpoint)
        print(json.dumps(record), flush=True)
    print(f"Best checkpoint: {args.output_root / 'best.ckpt'}", flush=True)
    if args.reverse_train_steps:
        print(f"Best occupancy IoU checkpoint: "
              f"{args.output_root / 'best_iou.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
