"""Fit the old Stage-II addition head to reject reliable-region additions.

The conditioned one-step denoiser and frozen LPCR are unchanged. Clean LiDAR
provides occupancy labels only. This remains
an oracle-assisted clean-latent diagnostic, not deployable reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.clean_selected_metrics import full_frame_voxels
from models.radar_lidar_stage2.reliable_addition import (
    addition_losses, decoded_addition_batch)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    CONDITIONED_ONE_STEP_REVISION, MIN_RADAR_VOXELS, RADIUS_M,
    _CleanRows, _rows)


REVISION = "stage2_conditioned_12vox_healthy_addition_head_v2"


def exact_counts(predicted: torch.Tensor, clean: torch.Tensor) -> dict[str, int]:
    pred_keys = encode_keys(predicted.long(), GRID.shape_zyx)
    clean_keys = encode_keys(clean.long(), GRID.shape_zyx)
    return {"predicted": len(predicted), "target": len(clean),
            "tp": int(torch.isin(pred_keys, clean_keys).sum())}


def combine_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key in ("predicted", "target", "tp"):
        total[key] += current[key]


def iou(counts: dict[str, int]) -> float:
    union = counts["predicted"] + counts["target"] - counts["tp"]
    return counts["tp"] / union if union else 0.0


def run_epoch(model: SparseResidualStage2, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None,
              alpha_bars: torch.Tensor, *, device: str, epoch: int,
              seed: int, eval_timestep: int,
              weights: tuple[float, float, float]) -> dict:
    training = optimizer is not None
    model.eval()
    model.addition_head.train(training)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + epoch if training else seed)
    loss_sum = 0.0
    frames = 0
    sites = {name: 0 for name in ("positive", "negative", "healthy")}
    baseline = {"predicted": 0, "target": 0, "tp": 0}
    completed = dict(baseline)
    for rows in loader:
        result = decoded_addition_batch(
            model, rows, alpha_bars, generator, device,
            eval_timestep=eval_timestep, train_head=training)
        fine, logits, masks = (result[key] for key in ("fine", "logits", "masks"))
        frames += len(rows)
        if masks is None:
            if training:
                continue
            chosen = torch.zeros(len(fine), dtype=torch.bool, device=device)
        else:
            losses = addition_losses(
                logits, fine, masks, len(rows),
                positive_weight=weights[0], negative_weight=weights[1],
                healthy_weight=weights[2])
            loss_sum += float(losses["total"].detach()) * len(rows)
            for key in sites:
                sites[key] += losses[f"{key}_sites"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.addition_head.parameters(), 1.0)
                optimizer.step()
            else:
                chosen = masks["candidates"] & (logits.sigmoid() >= 0.5)
        if training:
            continue
        for index, row in enumerate(rows):
            frame = fine[:, 0] == index
            predicted = fine[frame & chosen].detach().cpu().clone()
            predicted[:, 0] = 0
            target, faulty, final = full_frame_voxels(
                row["full_clean_xyz"], row["faulty_xyz"], predicted)
            combine_counts(baseline, exact_counts(faulty, target))
            combine_counts(completed, exact_counts(final, target))
    output = {"frames": frames, "loss": loss_sum / max(frames, 1),
              "mean_sites": {key: value / max(frames, 1)
                             for key, value in sites.items()}}
    if not training:
        output.update({"baseline": {**baseline, "iou": iou(baseline)},
                       "completed_at_0p5": {**completed,
                                             "iou": iou(completed)}})
    return output


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_base_model(lpcr_path: Path, base_path: Path,
                    device: str) -> tuple[SparseResidualStage2, dict]:
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    identity = base.get("identity")
    if (not isinstance(identity, dict) or
            identity.get("revision") != CONDITIONED_ONE_STEP_REVISION or
            identity.get("lpcr_sha256") != sha256(lpcr_path)):
        raise ValueError("Expected matching conditioned one-step base checkpoint")
    saved_lpcr = torch.load(lpcr_path, map_location="cpu", weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from architecture")
    model.diffusion.load_state_dict(base["diffusion"])
    model.radar_adapter.load_state_dict(base["radar_adapter"])
    model.requires_grad_(False)
    model.addition_head.requires_grad_(True)
    return model, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--healthy-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    weights = (args.positive_weight, args.negative_weight,
               args.healthy_weight)
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or
            args.lr <= 0 or any(weight < 0 for weight in weights) or
            any(limit is not None and limit < 1 for limit in
                (args.train_limit, args.val_limit))):
        parser.error("Invalid training settings or CUDA unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, base_identity = load_base_model(
        args.lpcr_checkpoint, args.base_checkpoint, args.device)
    if (args.batch_size != base_identity["batch_size"] or
            args.seed != base_identity["seed"]):
        raise ValueError("Batch size and seed must match the frozen denoiser "
                         "to preserve its validation noise sequence")
    for split in ("train", "val"):
        for label, root, key in (
                ("fixed", args.data_cache_root, "data_manifest_sha256"),
                ("radar support", args.support_cache_root,
                 "support_manifest_sha256"),
                ("mild faults", args.faulty_root, "faulty_manifest_sha256")):
            if base_identity[key][split] != sha256(root / split / "manifest.json"):
                raise ValueError(f"{label} {split} cache differs from base model")
    loaders = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        dataset = paired_dataset(
            args.vod_root, split, base_identity["radar_variant"],
            include_clean=True, data_cache_root=args.data_cache_root)
        frame_ids = [frame.frame_id for frame in dataset.frames]
        expected = support_identity(
            split, frame_ids, base_identity["radar_variant"],
            data_cache_root=args.data_cache_root, radius_m=RADIUS_M,
            min_radar_voxels=MIN_RADAR_VOXELS)
        cache = RadarSupportCache(args.support_cache_root / split, expected)
        cache.check_complete()
        rows = _CleanRows(
            dataset, cache, limit, args.faulty_root, split,
            include_full_clean=(split == "val"))
        loaders[split] = DataLoader(
            rows, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.num_workers, collate_fn=_rows,
            pin_memory=True, persistent_workers=args.num_workers > 0)
    identity = {"revision": REVISION,
                "base_checkpoint_sha256": sha256(args.base_checkpoint),
                "lpcr_checkpoint_sha256": sha256(args.lpcr_checkpoint),
                "base_epoch": int(torch.load(
                    args.base_checkpoint, map_location="cpu",
                    weights_only=False)["epoch"]),
                "batch_size": args.batch_size, "lr": args.lr,
                "weights": weights, "seed": args.seed,
                "train_limit": args.train_limit, "val_limit": args.val_limit,
                "eval_timestep": int(base_identity["eval_timestep"]),
                "radar_variant": base_identity["radar_variant"]}
    optimizer = torch.optim.AdamW(model.addition_head.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    args.output_root.mkdir(parents=True, exist_ok=True)
    start, best_loss, best_iou = 1, float("inf"), -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Resume checkpoint belongs to another addition run")
        model.addition_head.load_state_dict(saved["addition_head"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["epoch"]) + 1
        best_loss = float(saved["best_val_loss"])
        best_iou = float(saved["best_val_full_iou_at_0p5"])
    alpha_bars = cosine_alpha_bars(device=args.device)
    if start > args.epochs:
        print(f"Already complete: epoch {start - 1}", flush=True)
        return
    for epoch in range(start, args.epochs + 1):
        train = run_epoch(model, loaders["train"], optimizer, alpha_bars,
                          device=args.device, epoch=epoch, seed=args.seed,
                          eval_timestep=identity["eval_timestep"],
                          weights=weights)
        val = run_epoch(model, loaders["val"], None, alpha_bars,
                        device=args.device, epoch=epoch, seed=args.seed,
                        eval_timestep=identity["eval_timestep"],
                        weights=weights)
        loss_improved = val["loss"] < best_loss
        iou_improved = val["completed_at_0p5"]["iou"] > best_iou
        best_loss = min(best_loss, val["loss"])
        best_iou = max(best_iou, val["completed_at_0p5"]["iou"])
        record = {"epoch": epoch, "train": train, "val": val,
                  "best_val_loss": best_loss,
                  "best_val_full_iou_at_0p5": best_iou}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"identity": identity, "epoch": epoch,
                      "addition_head": model.addition_head.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "best_val_loss": best_loss,
                      "best_val_full_iou_at_0p5": best_iou,
                      "record": record}
        _save(args.output_root / "last.ckpt", checkpoint)
        if loss_improved:
            _save(args.output_root / "best.ckpt", checkpoint)
        if iou_improved:
            _save(args.output_root / "best_iou.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: train_loss={train['loss']:.4f} "
              f"val_loss={val['loss']:.4f} "
              f"full_ROI_IoU@0.5={val['completed_at_0p5']['iou']:.2%} "
              f"faulty_baseline={val['baseline']['iou']:.2%}", flush=True)
    print(f"Best full-ROI IoU checkpoint: {args.output_root / 'best_iou.ckpt'}",
          flush=True)


if __name__ == "__main__":
    main()
