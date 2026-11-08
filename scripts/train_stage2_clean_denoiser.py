"""Train epsilon-only clean LiDAR latent recovery on a fixed radar gate.

The original diagnostic uses one forward-noised clean latent per update. The
full-noise trial rolls out five/three DDIM reverse steps. Its optional
faulty/radar conditioning uses cached mild faults and measured radar features;
clean LiDAR remains the target only. The fresh conditional fault-only trial
can additionally train with ordinary clean-occupancy BCE through frozen LPCR.
No ray or offset loss enters training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import supported_coordinates
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.clean_denoising import (
    batched_hierarchy, clean_points_in_support, frame_mean,
    occupancy_counts, occupancy_metrics)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.fault_only_gate import fault_only_support
from models.radar_lidar_stage2.fault_region import FaultRegionCache
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.geometry_gate import geometry_gate_output
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256


REVISION = "stage2_clean_12vox_1p6m_epsilon_denoiser_v1"
REVERSE_REVISION = "stage2_clean_12vox_1p6m_fullnoise_reverse_5train_3val_v1"
CONDITIONED_REVERSE_REVISION = (
    "stage2_clean_12vox_1p6m_faulty_radar_reverse_5train_3val_v1")
CONDITIONED_ONE_STEP_REVISION = (
    "stage2_clean_12vox_1p6m_faulty_radar_one_step_epsilon_v1")
RADIUS_M = 1.6
MIN_RADAR_VOXELS = 12
RADAR_VARIANT = "radar_20frames_verified_doppler_radial"


class _CleanRows:
    def __init__(self, dataset, cache: RadarSupportCache, limit: int | None,
                 faulty_root: Path | None = None, split: str | None = None,
                 *, include_full_clean: bool = False,
                 fault_regions: FaultRegionCache | None = None):
        self.dataset = dataset
        self.cache = cache
        self.limit = len(dataset) if limit is None else min(len(dataset), limit)
        self.faulty_root = faulty_root
        self.split = split
        self.include_full_clean = include_full_clean
        self.fault_regions = fault_regions

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> dict:
        row = self.dataset[index]
        mask = self.cache.load(row["frame_id"])
        if self.fault_regions is not None:
            mask = fault_only_support(mask, self.fault_regions.get(row["frame_id"]))
        result = {"frame_id": row["frame_id"],
                  "fine_zyx": supported_coordinates(mask),
                  "clean_xyz": clean_points_in_support(
                      row["clean_lidar"][:, :3], mask)}
        if self.include_full_clean:
            result["full_clean_xyz"] = row["clean_lidar"][:, :3]
        if self.faulty_root is not None:
            result.update({"radar": row["radar"],
                           "faulty_xyz": load_faulty(
                               self.faulty_root, self.split, row["frame_id"]),
                           "support_mask": mask})
        return result


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


def _conditioned_inputs(model: SparseResidualStage2, rows: list[dict],
                        hierarchy: tuple[torch.Tensor, ...],
                        device: str, *, train_radar_adapter: bool = False
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode faulty LiDAR and measured radar without exposing clean targets."""
    faults = [row["faulty_xyz"] for row in rows]
    faulty = torch.cat(faults).to(device, non_blocking=True)
    faulty_ids = torch.cat([
        torch.full((len(cloud),), frame, dtype=torch.long)
        for frame, cloud in enumerate(faults)]).to(device)
    coarse = hierarchy[-1]
    radar_parts = []
    with torch.no_grad():
        faulty_latent, faulty_present = model.lidar_encoder(
            faulty, coarse, faulty_ids)
    for frame, row in enumerate(rows):
        radar = row["radar"].to(device, non_blocking=True)
        # This full-scan, empty-fault call supplies radar statistics only.
        # Its allowed cells are exactly the precomputed >=12/1.6 m mask;
        # observed faulty cells are deliberately not removed here.
        with torch.no_grad():
            output = geometry_gate_output(
                radar, faulty[:0], {"kind": "full_scan"},
                row["support_mask"], MIN_RADAR_VOXELS)
        with torch.set_grad_enabled(train_radar_adapter):
            condition = model.conditions(output, 0.5)
        frame_coarse = coarse[coarse[:, 0] == frame][:, 1:]
        if not torch.equal(condition.hierarchy[-1][:, 1:], frame_coarse):
            raise ValueError("Radar features do not align with gated latent sites")
        radar_parts.append(condition.radar_features)
    radar_latent = torch.cat(radar_parts)
    if radar_latent.shape != faulty_latent.shape:
        raise ValueError("Faulty LiDAR and radar latent features differ")
    return faulty_latent, faulty_present, radar_latent


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


def resolve_sampling_strategy(revision: str, requested: str) -> str:
    """Pin a checkpoint's default sampler while allowing a crossed comparison."""
    if revision not in (REVISION, REVERSE_REVISION,
                        CONDITIONED_ONE_STEP_REVISION):
        raise ValueError("Expected a supported clean denoiser checkpoint")
    if requested not in ("checkpoint", "one_step", "reverse3"):
        raise ValueError("Unknown clean denoiser sampling strategy")
    if requested == "checkpoint":
        return ("reverse3" if revision == REVERSE_REVISION else "one_step")
    return requested


@torch.no_grad()
def sample_clean_latent(model: SparseResidualStage2, x0: torch.Tensor,
                        coarse: torch.Tensor, alpha_bars: torch.Tensor,
                        generator: torch.Generator, *, batch_size: int,
                        strategy: str, eval_timestep: int,
                        conditions: tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor] | None = None
                        ) -> torch.Tensor:
    """Use the same fixed gate and decoder with either clean-noise or full-noise input."""
    if conditions is None:
        faulty = radar = torch.zeros_like(x0)
        present = torch.zeros(len(x0), dtype=torch.bool, device=x0.device)
    else:
        faulty, present, radar = conditions
        if (faulty.shape != x0.shape or radar.shape != x0.shape or
                present.shape != (len(x0),)):
            raise ValueError("Conditioning differs from clean latent sites")
    probability = x0.new_ones((len(x0), 1))
    if strategy == "one_step":
        alpha = alpha_bars[eval_timestep]
        epsilon = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                              generator=generator)
        xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
        prediction = model.diffusion(
            xt, faulty, radar, probability, present,
            torch.full((batch_size,), eval_timestep, device=x0.device,
                       dtype=torch.long), coarse)
        return (xt - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    if strategy != "reverse3":
        raise ValueError("Unknown clean denoiser sampling strategy")
    xt = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                     generator=generator)
    schedule = reverse_schedule(3, total=len(alpha_bars)).to(x0.device)
    for index, timestep in enumerate(schedule):
        prediction = model.diffusion(
            xt, faulty, radar, probability, present,
            timestep.expand(batch_size), coarse)
        following_alpha = (alpha_bars[schedule[index + 1]]
                           if index + 1 < len(schedule) else None)
        recovered, following = reverse_step(
            xt, prediction, alpha_bars[timestep], following_alpha)
        if following is not None:
            xt = following
    return recovered


def _run_epoch(model: SparseResidualStage2, loader: DataLoader,
               optimizer: torch.optim.Optimizer | None,
               alpha_bars: torch.Tensor, *, device: str, epoch: int,
               eval_timestep: int, seed: int,
               reverse_train_steps: int = 0,
               reverse_val_steps: int = 0,
               condition_faulty_radar: bool = False,
               train_radar_adapter: bool = False,
               occupancy_bce: bool = False,
               eval_random_timesteps: bool = False,
               eval_loss_only: bool = False) -> dict:
    training = optimizer is not None
    if occupancy_bce and (not training or reverse_train_steps):
        raise ValueError("Occupancy BCE requires single-pass training")
    if eval_loss_only and (training or not eval_random_timesteps or
                           reverse_val_steps):
        raise ValueError("Loss-only validation requires random single-pass timesteps")
    model.eval()
    model.diffusion.train(training)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + epoch if training else seed)
    loss_sum = latent_mse_sum = bce_sum = total_sum = 0.0
    frames = fine_sites = latent_sites = present_latents = 0
    counts = {"tp": 0, "predicted": 0, "target": 0}
    oracle = dict(counts)
    reverse_count = reverse_train_steps if training else reverse_val_steps
    schedule = (reverse_schedule(reverse_count, total=len(alpha_bars)).to(device)
                if reverse_count else None)
    for rows in loader:
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
        if condition_faulty_radar:
            faulty_condition, faulty_present, radar_condition = (
                _conditioned_inputs(
                    model, rows, hierarchy, device,
                    train_radar_adapter=training and train_radar_adapter))
        else:
            faulty_condition = radar_condition = torch.zeros_like(x0)
            faulty_present = torch.zeros(
                len(x0), dtype=torch.bool, device=device)
        site_probability = x0.new_ones((len(x0), 1))
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
                        xt, faulty_condition, radar_condition,
                        site_probability, faulty_present, steps, coarse)
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
            if training or eval_random_timesteps:
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
                    xt, faulty_condition, radar_condition,
                    site_probability, faulty_present, steps, coarse)
                loss = frame_mean((predicted_noise - epsilon).square().mean(1),
                                  latent_ids, len(rows))
                if training:
                    total_loss = loss
                    if occupancy_bce:
                        # A second, moderate noise level gives the ordinary
                        # occupancy BCE a stable x0 estimate on every batch.
                        # Noise MSE still trains over the full timestep range.
                        bce_steps = torch.full(
                            (len(rows),), eval_timestep, device=device,
                            dtype=torch.long)
                        bce_alpha = alpha_bars[eval_timestep]
                        bce_noise = torch.randn(
                            x0.shape, dtype=x0.dtype, device=device,
                            generator=generator)
                        bce_xt = (bce_alpha.sqrt() * x0 +
                                  (1 - bce_alpha).sqrt() * bce_noise)
                        bce_prediction = model.diffusion(
                            bce_xt, faulty_condition, radar_condition,
                            site_probability, faulty_present, bce_steps, coarse)
                        bce_x0 = ((bce_xt - (1 - bce_alpha).sqrt() *
                                   bce_prediction) / bce_alpha.sqrt())
                        site_features = x0.new_ones((len(hierarchy[0]), 1))
                        logits, _ = model.lpcr(
                            bce_x0.clamp(-1, 1), hierarchy, site_features)
                        target, _, _ = offset_targets(
                            clean, hierarchy[0], point_ids)
                        bce_loss = F.binary_cross_entropy_with_logits(
                            logits, target.to(logits.dtype))
                        total_loss = loss + bce_loss
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters()
                         if parameter.requires_grad), 1.0)
                    optimizer.step()
                    if occupancy_bce:
                        bce_sum += float(bce_loss.detach()) * len(rows)
                        total_sum += float(total_loss.detach()) * len(rows)
        loss_sum += float(loss.detach()) * len(rows)
        if not training and not eval_loss_only:
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
    if occupancy_bce:
        output.update({"occupancy_bce": bce_sum / max(frames, 1),
                       "total_loss": total_sum / max(frames, 1),
                       "bce_timestep": eval_timestep})
    if not training and not eval_loss_only:
        output.update({"recovered_clean_latent_mse": latent_mse_sum / max(frames, 1),
                       "eligible_occupancy": occupancy_metrics(counts),
                       "frozen_lpcr_oracle": occupancy_metrics(oracle),
                       "occupancy_threshold": 0.5,
                       "eval_timestep": eval_timestep if not reverse_count else None})
    if eval_loss_only:
        output.update({"validation_timestep_sampling": "uniform_over_training_schedule",
                       "validation_seed": seed})
    if reverse_count:
        output.update({"sampling": "sequential_DDIM_from_Gaussian_noise",
                       "reverse_steps": reverse_count,
                       "reverse_schedule": schedule.tolist(),
                       "reverse_terminal_timestep": int(schedule[-1]),
                       "conditioning": ("faulty_LiDAR_and_measured_radar"
                                        if condition_faulty_radar else
                                        "gate_coordinates_only")})
    elif condition_faulty_radar:
        output["conditioning"] = "faulty_LiDAR_and_measured_radar"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path)
    parser.add_argument("--condition-faulty-radar", action="store_true")
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
    if args.condition_faulty_radar and args.faulty_root is None:
        parser.error("Faulty/radar conditioning requires --faulty-root")
    if args.faulty_root is not None and not args.condition_faulty_radar:
        parser.error("--faulty-root requires --condition-faulty-radar")
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
    train_radar_adapter = bool(args.condition_faulty_radar and
                               args.reverse_train_steps == 0)
    if train_radar_adapter:
        model.radar_adapter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=args.lr,
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
        rows = _CleanRows(dataset, cache, limit,
                          args.faulty_root, split)
        loaders[split] = DataLoader(
            rows, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.num_workers, collate_fn=_rows,
            pin_memory=True, persistent_workers=args.num_workers > 0)

    revision = (CONDITIONED_REVERSE_REVISION
                if args.condition_faulty_radar and args.reverse_train_steps
                else CONDITIONED_ONE_STEP_REVISION if args.condition_faulty_radar
                else REVERSE_REVISION if args.reverse_train_steps else REVISION)
    identity = {"revision": revision,
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
                         "conditioning": ("faulty_LiDAR_and_measured_radar"
                                          if args.condition_faulty_radar else
                                          "gate_coordinates_only")})
    if args.condition_faulty_radar:
        identity["faulty_manifest_sha256"] = {
            split: sha256(args.faulty_root / split / "manifest.json")
            for split in ("train", "val")}
        if train_radar_adapter:
            identity["radar_adapter_trainable"] = True
            identity["conditioning"] = "faulty_LiDAR_and_measured_radar"
    start, best = 1, float("inf")
    best_iou = -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Resume checkpoint belongs to a different denoising run")
        model.diffusion.load_state_dict(saved["diffusion"])
        if train_radar_adapter:
            model.radar_adapter.load_state_dict(saved["radar_adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_val_noise_mse"])
        if args.reverse_train_steps or args.condition_faulty_radar:
            best_iou = float(saved["best_val_iou_at_0p5"])
    history = args.output_root / "history.jsonl"
    if start > args.epochs:
        print(f"Already complete: epoch {start - 1}; {args.output_root}", flush=True)
        return
    for epoch in range(start, args.epochs + 1):
        train = _run_epoch(model, loaders["train"], optimizer, alpha_bars,
                           device=args.device, epoch=epoch,
                           eval_timestep=args.eval_timestep, seed=args.seed,
                           reverse_train_steps=args.reverse_train_steps,
                           reverse_val_steps=args.reverse_val_steps,
                           condition_faulty_radar=args.condition_faulty_radar,
                           train_radar_adapter=train_radar_adapter)
        val = _run_epoch(model, loaders["val"], None, alpha_bars,
                         device=args.device, epoch=epoch,
                         eval_timestep=args.eval_timestep, seed=args.seed,
                         reverse_train_steps=args.reverse_train_steps,
                         reverse_val_steps=args.reverse_val_steps,
                         condition_faulty_radar=args.condition_faulty_radar,
                         train_radar_adapter=train_radar_adapter)
        improved = val["noise_mse"] < best
        if improved:
            best = val["noise_mse"]
        improved_iou = ((args.reverse_train_steps > 0 or
                         args.condition_faulty_radar) and
                        val["eligible_occupancy"]["iou"] > best_iou)
        if improved_iou:
            best_iou = val["eligible_occupancy"]["iou"]
        record = {"epoch": epoch, "train": train, "val": val,
                  "best_val_noise_mse": best,
                  "validation_scope": f"{len(loaders['val'].dataset)} clean frames"}
        if args.reverse_train_steps or args.condition_faulty_radar:
            record["best_val_iou_at_0p5"] = best_iou
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"identity": identity, "epoch": epoch,
                      "best_val_noise_mse": best,
                      "diffusion": model.diffusion.state_dict(),
                      "optimizer": optimizer.state_dict(), "record": record}
        if train_radar_adapter:
            checkpoint["radar_adapter"] = model.radar_adapter.state_dict()
        if args.reverse_train_steps or args.condition_faulty_radar:
            checkpoint["best_val_iou_at_0p5"] = best_iou
        _atomic_save(args.output_root / "last.ckpt", checkpoint)
        if improved:
            _atomic_save(args.output_root / "best.ckpt", checkpoint)
        if improved_iou:
            _atomic_save(args.output_root / "best_iou.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: "
              f"train_loss={train['noise_mse']:.4f} "
              f"val_loss={val['noise_mse']:.4f} "
              f"exact_IoU={val['eligible_occupancy']['iou']:.2%}",
              flush=True)


if __name__ == "__main__":
    main()
