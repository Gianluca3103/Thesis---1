"""Train radar-seeded latent diffusion with a frozen Minkowski LiDAR decoder.

The synthetic injector footprint defines the reconstruction area in this
diagnostic. It is not an inferred fault mask and is unavailable on real faults.
Clean LiDAR is used only for supervision, never to choose decoder coordinates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lidar_minkowski_ae import SparseVoxelAutoencoder, _keys
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import RadarSupportCache, identity as support_identity
from models.radar_lidar_stage2.clean_selected_metrics import (
    completion_summary, empty_accumulator, full_frame_voxels, summarize)
from models.radar_lidar_stage2.eligible_occupancy_metrics import occupancy_counts
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.minkowski_completion import (
    SparseLatentDenoiser, decode_codes, gather_codes, gather_logits,
    radar_anchor_features, radar_parent_sites, voxelize_bxyz)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.cache_pointnetpp_stage2_features import RadarFeatureCache
from scripts.train_stage2_clean_denoiser import _CleanRows, _rows


REVISION = "stage2_minkowski_decoder_pointnetpp768_partial_only_v1"
RADAR_VARIANT = "radar_20frames_verified_doppler_radial"


def _atomic_save(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _dataset(args, split: str, *, limit: int | None):
    base = paired_dataset(args.vod_root, split, RADAR_VARIANT,
                          include_clean=True, data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in base.frames]
    support = RadarSupportCache(
        args.support_cache_root / split,
        support_identity(split, ids, RADAR_VARIANT,
                         data_cache_root=args.data_cache_root,
                         radius_m=1.6, min_radar_voxels=12))
    support.check_complete()
    regions = FaultRegionCache(
        args.fault_region_root / split,
        cache_identity(split, ids,
                       args.faulty_root / split / "manifest.json",
                       args.data_cache_root / split / "manifest.json"))
    if regions.regions is None:
        raise FileNotFoundError(f"Missing {split} injector footprints")
    features = RadarFeatureCache(args.radar_feature_root, split,
                                 sha256(args.radar_checkpoint))
    if features.feature_channels != 768 or features.manifest["frame_ids"] != ids:
        raise ValueError(f"{split} radar features are not width-768 aligned frames")
    return _CleanRows(base, support, limit, args.faulty_root, split,
                      include_full_clean=True, fault_regions=regions,
                      radar_features=features)


def _clouds(rows: list[dict], name: str, device: torch.device):
    points = [torch.as_tensor(row[name], dtype=torch.float32, device=device)
              for row in rows]
    xyz = torch.cat(points)
    batch = torch.cat([torch.full((len(point),), i, dtype=torch.long,
                                  device=device) for i, point in enumerate(points)])
    return voxelize_bxyz(xyz, batch)


def _prepare(rows: list[dict], autoencoder, device: torch.device,
             *, training_target: bool):
    fine, parents = radar_parent_sites(
        [torch.as_tensor(row["fine_zyx"]) for row in rows], device)
    if not len(fine):
        return None
    faulty = _clouds(rows, "faulty_xyz", device)
    if not len(faulty):
        raise RuntimeError("This trial requires nonempty faulty LiDAR")
    with torch.no_grad():
        faulty_latent = autoencoder.encode(faulty)
        condition, faulty_present = gather_codes(
            faulty_latent.C, faulty_latent.F, parents)
        if training_target:
            clean = _clouds(rows, "full_clean_xyz", device)
            if not len(clean):
                raise RuntimeError("Training requires nonempty clean LiDAR")
            clean_latent = autoencoder.encode(clean)
            target, clean_present = gather_codes(
                clean_latent.C, clean_latent.F, parents)
            clean_at_gate = torch.isin(_keys(fine), _keys(clean))
        else:
            target = clean_present = clean_at_gate = None
    anchors = [(row["learned_radar_xyz"].to(device, non_blocking=True),
                row["learned_radar_features"].to(device, non_blocking=True))
               for row in rows]
    radar = radar_anchor_features(parents, anchors)
    observed_at_gate = torch.isin(_keys(fine), _keys(faulty))
    return (fine, parents, target, clean_present, condition, faulty_present,
            radar, clean_at_gate, observed_at_gate)


def _sample(model, parents, faulty, present, radar, alpha_bars,
            frame_ids, *, steps: int, seed: int, gradient_steps: int):
    """Full-noise DDIM; only final gradient_steps retain an autograd graph."""
    schedule = torch.linspace(999, 0, steps, device=parents.device).round().long()
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("Sampling steps must be distinct")
    noise = torch.empty_like(faulty)
    for batch, frame_id in enumerate(frame_ids):
        digest = hashlib.blake2b(f"{seed}:{frame_id}".encode(),
                                 digest_size=8).digest()
        generator = torch.Generator(device=faulty.device).manual_seed(
            int.from_bytes(digest, "little") % 2**63)
        selected = parents[:, 0] == batch
        noise[selected] = torch.randn((int(selected.sum()), faulty.shape[1]),
                                      generator=generator, device=faulty.device)
    xt = noise
    for index, step in enumerate(schedule):
        with torch.set_grad_enabled(index >= steps - gradient_steps):
            epsilon = model(parents, xt, faulty, radar, present,
                            step.expand(len(frame_ids)))
            alpha = alpha_bars[step]
            x0 = ((xt - (1 - alpha).sqrt() * epsilon) /
                  alpha.sqrt()).clamp(-2, 2)
            if index + 1 == steps:
                return x0
            next_alpha = alpha_bars[schedule[index + 1]]
            xt = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
    raise AssertionError("Empty sampling schedule")


def _losses(rows, autoencoder, model, alpha_bars, device, *, latent_scale,
            rollout_steps, gradient_steps, seed, generated_bce,
            deterministic_noise: bool):
    prepared = _prepare(rows, autoencoder, device, training_target=True)
    if prepared is None:
        return None
    (fine, parents, target, _, faulty, faulty_present, radar,
     clean_at_gate, observed_at_gate) = prepared
    generator = None
    if deterministic_noise:
        digest = hashlib.blake2b(
            f"{seed}:".encode() +
            ",".join(row["frame_id"] for row in rows).encode(),
            digest_size=8).digest()
        generator = torch.Generator(device=device).manual_seed(
            int.from_bytes(digest, "little") % 2**63)
    batch_time = torch.randint(0, 1000, (len(rows),), device=device,
                               generator=generator)
    alpha = alpha_bars[batch_time[parents[:, 0].long()], None]
    epsilon = torch.randn(target.shape, device=device, dtype=target.dtype,
                          generator=generator)
    xt = alpha.sqrt() * (target / latent_scale) + (1 - alpha).sqrt() * epsilon
    predicted = model(parents, xt, faulty / latent_scale, radar,
                      faulty_present, batch_time)
    mse = F.mse_loss(predicted, epsilon)
    bce = mse.new_zeros(())
    if generated_bce:
        x0 = _sample(model, parents, faulty / latent_scale, faulty_present,
                     radar, alpha_bars, [row["frame_id"] for row in rows],
                     steps=rollout_steps, seed=seed,
                     gradient_steps=gradient_steps)
        decoded = decode_codes(autoencoder, parents, x0 * latent_scale)
        gate_logits = gather_logits(decoded.coordinates, decoded.logits, fine)
        allowed = ~observed_at_gate
        if bool(allowed.any()):
            bce = F.binary_cross_entropy_with_logits(
                gate_logits[allowed], clean_at_gate[allowed].float())
    return mse, bce


def _epoch(loader, autoencoder, model, optimizer, alpha_bars, device, args,
           epoch, *, label):
    model.train(optimizer is not None)
    total = mse_total = bce_total = frames = 0
    for rows in tqdm(loader, desc=f"Minkowski Stage II {label} {epoch:03d}",
                     unit="batch", dynamic_ncols=True):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            result = _losses(
                rows, autoencoder, model, alpha_bars, device,
                latent_scale=args.latent_scale,
                rollout_steps=args.rollout_steps,
                gradient_steps=args.gradient_steps if optimizer is not None else 0,
                seed=args.seed + epoch if optimizer is not None else args.seed,
                generated_bce=optimizer is not None,
                deterministic_noise=optimizer is None)
            if result is None:
                continue
            mse, bce = result
            loss = mse + args.bce_weight * bce
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite Stage II loss on {rows[0]['frame_id']}")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        frames += len(rows)
        mse_total += float(mse.detach()) * len(rows)
        bce_total += float(bce.detach()) * len(rows)
        total += float(loss.detach()) * len(rows)
    if frames == 0:
        raise RuntimeError(f"No radar-gated partial-loss frames in {label}")
    return {"frames": frames, "noise_mse": mse_total / max(frames, 1),
            "generated_bce": bce_total / max(frames, 1),
            "total_loss": total / max(frames, 1)}


def _to_bzyx(coords: torch.Tensor) -> torch.Tensor:
    return coords[:, [0, 3, 2, 1]].long().cpu()


def _add_occupancy_counts(accumulator: dict, predicted: torch.Tensor,
                          target: torch.Tensor) -> None:
    """Exact/0.2 m IoU without the unrelated, expensive Chamfer pass."""
    counts = occupancy_counts(predicted, target, tolerance_m=0.2)
    accumulator["frames"] += 1
    for key, value in counts.items():
        accumulator[key] += value
    accumulator["empty_prediction_frames"] += int(len(predicted) == 0)
    accumulator["empty_target_frames"] += int(len(target) == 0)


def _evaluate(loader, autoencoder, model, alpha_bars, device, args,
              *, threshold: float | None, split: str) -> dict:
    """Score all frames, or collect exact full-scene threshold counts."""
    model.eval()
    if threshold is not None:
        missing = empty_accumulator()
        faulty_total = empty_accumulator()
        completed_total = empty_accumulator()
        bce_sum = bce_sites = 0
    else:
        bins = {name: {"all": torch.zeros(1001, dtype=torch.long),
                       "true": torch.zeros(1001, dtype=torch.long),
                       "baseline_tp": 0, "baseline_predicted": 0,
                       "target": 0, "frames": 0}
                for name in ("calibration", "heldout")}
    with torch.no_grad():
        for rows in tqdm(loader, desc=f"Minkowski Stage II {split}"
                         + (" sweep" if threshold is None else " metrics"),
                         unit="batch", dynamic_ncols=True):
            prepared = _prepare(rows, autoencoder, device,
                                training_target=False)
            if prepared is None:
                fine = torch.empty((0, 4), dtype=torch.int32, device=device)
                probability = torch.empty(0, dtype=torch.float32, device=device)
                clean_at_gate = observed_at_gate = torch.empty(
                    0, dtype=torch.bool, device=device)
            else:
                (fine, parents, _, _, faulty, present, radar,
                 _, observed_at_gate) = prepared
                x0 = _sample(model, parents, faulty / args.latent_scale, present,
                             radar, alpha_bars, [row["frame_id"] for row in rows],
                             steps=args.validation_steps, seed=args.seed,
                             gradient_steps=0)
                decoded = decode_codes(autoencoder, parents, x0 * args.latent_scale)
                gate_logits = gather_logits(decoded.coordinates, decoded.logits,
                                            fine)
                if not bool(torch.isfinite(gate_logits).all()):
                    raise RuntimeError(f"Nonfinite generated logits on {rows[0]['frame_id']}")
                probability = gate_logits.sigmoid()
                # Ground truth enters only after all model scores are fixed.
                clean = _clouds(rows, "full_clean_xyz", device)
                clean_at_gate = torch.isin(_keys(fine), _keys(clean))
                if threshold is not None:
                    allowed = ~observed_at_gate
                    if bool(allowed.any()):
                        bce_sum += float(F.binary_cross_entropy_with_logits(
                            gate_logits[allowed],
                            clean_at_gate[allowed].float(),
                            reduction="sum"))
                        bce_sites += int(allowed.sum())
            for batch, row in enumerate(rows):
                selected = (fine[:, 0] == batch) & ~observed_at_gate
                local = fine[selected].clone()
                local[:, 0] = 0
                local_prob = probability[selected]
                local_truth = clean_at_gate[selected]
                if threshold is None:
                    group = ("calibration" if bins["calibration"]["frames"] <
                             args.calibration_frames else "heldout")
                    stat = bins[group]
                    index = (local_prob * 1000).long().clamp(0, 1000).cpu()
                    stat["all"] += torch.bincount(index, minlength=1001)
                    stat["true"] += torch.bincount(index[local_truth.cpu()],
                                                   minlength=1001)
                    clean, faulty_scene, _ = full_frame_voxels(
                        row["full_clean_xyz"], row["faulty_xyz"],
                        _to_bzyx(local[:0]))
                    stat["baseline_tp"] += int(torch.isin(
                        _keys(faulty_scene[:, [0, 3, 2, 1]]),
                        _keys(clean[:, [0, 3, 2, 1]])).sum())
                    stat["baseline_predicted"] += len(faulty_scene)
                    stat["target"] += len(clean)
                    stat["frames"] += 1
                    continue
                predicted = _to_bzyx(local[local_prob >= threshold])
                target = _to_bzyx(local[local_truth])
                _add_occupancy_counts(missing, predicted, target)
                clean, faulty_scene, completed = full_frame_voxels(
                    row["full_clean_xyz"], row["faulty_xyz"], predicted)
                _add_occupancy_counts(faulty_total, faulty_scene, clean)
                _add_occupancy_counts(completed_total, completed, clean)
    if threshold is not None:
        return {"split": split, "threshold": threshold,
                "validation_generated_bce": bce_sum / max(bce_sites, 1),
                "missing_cell_additions": summarize(missing),
                "full_scene": completion_summary(faulty_total, completed_total)}
    return _select_threshold(bins)


def _select_threshold(bins: dict) -> dict:
    """Select full-scene exact IoU on the first 20% of validation frames."""
    def curve(stat):
        predicted = torch.flip(torch.cumsum(torch.flip(stat["all"], (0,)), 0), (0,))
        tp = torch.flip(torch.cumsum(torch.flip(stat["true"], (0,)), 0), (0,))
        truth = stat["target"]
        numerator = stat["baseline_tp"] + tp
        denominator = stat["baseline_predicted"] + predicted + truth - numerator
        iou = numerator.double() / denominator.clamp_min(1).double()
        return iou, predicted, tp

    calibration, _, _ = curve(bins["calibration"])
    selected = int(torch.argmax(calibration))
    # A no-addition policy is a valid outcome if this frozen decoder cannot help.
    if bins["calibration"]["baseline_tp"] / max(
            bins["calibration"]["baseline_predicted"] +
            bins["calibration"]["target"] -
            bins["calibration"]["baseline_tp"], 1) >= float(calibration[selected]):
        selected = 1001
    result = {"threshold": selected / 1000 if selected <= 1000 else 1.001,
              "selection_metric": "calibration_full_scene_exact_iou",
              "groups": {}}
    for name, stat in bins.items():
        iou, predicted, tp = curve(stat)
        if selected == 1001:
            added, added_tp = 0, 0
            chosen_iou = stat["baseline_tp"] / max(
                stat["baseline_predicted"] + stat["target"] -
                stat["baseline_tp"], 1)
        else:
            added, added_tp = int(predicted[selected]), int(tp[selected])
            chosen_iou = float(iou[selected])
        result["groups"][name] = {
            "frames": stat["frames"], "full_exact_iou": chosen_iou,
            "baseline_exact_iou": stat["baseline_tp"] / max(
                stat["baseline_predicted"] + stat["target"] -
                stat["baseline_tp"], 1),
            "added": added, "added_tp": added_tp,
            "calibration_best_iou": float(iou.max()) if name == "calibration" else None}
    return result


def _build_loader(args, split, *, limit, batch_size):
    dataset = _dataset(args, split, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=args.workers, collate_fn=_rows,
                      pin_memory=True, persistent_workers=args.workers > 0), len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--radar-feature-root", type=Path, required=True)
    parser.add_argument("--radar-checkpoint", type=Path, required=True)
    parser.add_argument("--lidar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--skip-final-evaluation", action="store_true",
                        help="Only for the isolated CUDA smoke test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--latent-scale", type=float, default=4.0)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--validation-steps", type=int, default=64)
    parser.add_argument("--reconstruction-val-every", type=int, default=5)
    parser.add_argument("--calibration-frames", type=int, default=260)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--seed", type=int, default=3103)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.epochs < 1 or
            args.lr <= 0 or args.bce_weight < 0 or args.latent_scale <= 0 or
            not 2 <= args.rollout_steps <= 1000 or
            not 1 <= args.gradient_steps <= args.rollout_steps or
            not 2 <= args.validation_steps <= 1000 or
            args.reconstruction_val_every < 1 or args.calibration_frames < 1):
        parser.error("Invalid training, rollout, or validation settings")
    if not torch.cuda.is_available():
        parser.error("CUDA and MinkowskiEngine are required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    saved_ae = torch.load(args.lidar_checkpoint, map_location="cpu",
                          weights_only=False)
    if (saved_ae.get("grid") != asdict(GRID) or
            saved_ae.get("model") != {"width": 32, "levels": 1}):
        raise ValueError("Expected the audited stride-2 width-32 Minkowski AE")
    autoencoder = SparseVoxelAutoencoder(width=32, levels=1).to(device)
    autoencoder.load_state_dict(saved_ae["model_state"])
    autoencoder.eval().requires_grad_(False)
    model = SparseLatentDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    alpha_bars = cosine_alpha_bars(device=device)
    identity = {"revision": REVISION, "grid": asdict(GRID),
                "lidar_checkpoint_sha256": sha256(args.lidar_checkpoint),
                "radar_checkpoint_sha256": sha256(args.radar_checkpoint),
                "radar_width": 768, "gate": "12 distinct radar voxels within 1.6 m, partial-loss injector interior",
                "latent_sites": "radar-gate stride-2 parents, independent of clean/faulty LiDAR",
                "training": {key: getattr(args, key) for key in (
                    "batch_size", "lr", "bce_weight", "latent_scale",
                    "rollout_steps", "gradient_steps", "validation_steps", "seed")},
                "manifests": {split: {name: sha256(root / split / "manifest.json")
                                      for name, root in (
                                          ("fixed", args.data_cache_root),
                                          ("support", args.support_cache_root),
                                          ("faulty", args.faulty_root),
                                          ("regions", args.fault_region_root),
                                          ("radar_features", args.radar_feature_root))}
                              for split in ("train", "val", "test")}}
    args.output_root.mkdir(parents=True, exist_ok=True)
    first, best = 1, -1.0
    checkpoint = args.checkpoint if args.phase == "evaluate" else args.resume
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Stage II checkpoint belongs to another experiment")
        model.load_state_dict(saved["model"])
        best = float(saved["best_val_iou"])
        if args.phase == "train":
            optimizer.load_state_dict(saved["optimizer"])
            first = int(saved["epoch"]) + 1
    elif args.phase == "evaluate":
        parser.error("Evaluation requires --checkpoint")
    elif (args.output_root / "last.ckpt").exists():
        parser.error("Existing checkpoint needs --resume or a new output directory")
    val_loader, n_val = _build_loader(args, "val", limit=args.val_limit,
                                      batch_size=args.batch_size)
    if args.phase == "train":
        train_loader, n_train = _build_loader(args, "train", limit=args.train_limit,
                                              batch_size=args.batch_size)
        print(f"Minkowski Stage II: {sum(p.numel() for p in model.parameters()):,} "
              f"trainable parameters; {n_train} train / {n_val} val frames; "
              "frozen audited LiDAR decoder; radar width 768", flush=True)
        for epoch in range(first, args.epochs + 1):
            train = _epoch(train_loader, autoencoder, model, optimizer,
                           alpha_bars, device, args, epoch, label="train")
            val = _epoch(val_loader, autoencoder, model, None,
                         alpha_bars, device, args, epoch, label="val")
            full = None
            if epoch == 1 or epoch % args.reconstruction_val_every == 0 or epoch == args.epochs:
                full = _evaluate(val_loader, autoencoder, model, alpha_bars,
                                 device, args, threshold=0.5, split="val")
                current = full["full_scene"]["completed"]["iou"]
                improved = current > best
                best = max(best, current)
            else:
                improved = False
            record = {"epoch": epoch, "train": train, "val": val,
                      "reconstruction": full, "best_full_exact_iou": best}
            with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            payload = {"identity": identity, "epoch": epoch,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "best_val_iou": best}
            _atomic_save(args.output_root / "last.ckpt", payload)
            if improved:
                _atomic_save(args.output_root / "best_iou.ckpt", payload)
            text = (f"Epoch {epoch:03d}: train_MSE={train['noise_mse']:.4f} "
                    f"train_BCE={train['generated_bce']:.4f} "
                    f"val_MSE={val['noise_mse']:.4f}")
            if full is not None:
                scene = full["full_scene"]
                missing = full["missing_cell_additions"]
                text += (f" missing_exact_IoU={missing['iou']:.2%} "
                         f"missing_IoU@0.2m={missing['iou_0p2m']:.2%} "
                         f"val_BCE={full['validation_generated_bce']:.4f} "
                         f"val_total={val['noise_mse'] + args.bce_weight * full['validation_generated_bce']:.4f} "
                         f"full_exact_IoU={scene['faulty']['iou']:.2%}->"
                         f"{scene['completed']['iou']:.2%} "
                         f"full_IoU@0.2m={scene['faulty']['iou_0p2m']:.2%}->"
                         f"{scene['completed']['iou_0p2m']:.2%}")
            print(text, flush=True)
        checkpoint = args.output_root / "best_iou.ckpt"
        if not checkpoint.is_file():
            raise RuntimeError("Training did not produce a best IoU checkpoint")
        print(f"Best checkpoint: {checkpoint}", flush=True)
        selected_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(selected_model["model"])
        if args.skip_final_evaluation:
            print("Smoke training and fixed validation completed.", flush=True)
            return
    print("Selecting threshold once on validation calibration frames.", flush=True)
    sweep = _evaluate(val_loader, autoencoder, model, alpha_bars, device,
                      args, threshold=None, split="val")
    (args.output_root / "threshold_sweep.json").write_text(
        json.dumps(sweep, indent=2) + "\n", encoding="utf-8")
    selected = sweep["threshold"]
    reports = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
               "threshold_sweep": sweep}
    for split in ("val", "test"):
        loader = val_loader if split == "val" else _build_loader(
            args, "test", limit=None, batch_size=args.batch_size)[0]
        reports[split] = _evaluate(loader, autoencoder, model, alpha_bars,
                                   device, args, threshold=selected, split=split)
        scene = reports[split]["full_scene"]
        print(f"{split}: full exact IoU {scene['faulty']['iou']:.2%} -> "
              f"{scene['completed']['iou']:.2%}; missing exact IoU "
              f"{reports[split]['missing_cell_additions']['iou']:.2%}", flush=True)
    report_path = args.output_root / "selected_val_test.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
