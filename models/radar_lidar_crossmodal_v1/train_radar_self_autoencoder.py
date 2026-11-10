"""Train a true filtered-radar -> filtered-radar point-cloud autoencoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .filtered_radar_cache import FilteredRadarCache
from .radar_self_autoencoder import (
    SLOTS, RadarSelfAutoencoder, collate_filtered_radar, decoded_points,
    reconstruction_counts, reconstruction_loss, target_slots)


VERSION = "radar_self_point_autoencoder_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _epoch(model, loader, optimizer, device, *, threshold: float, name: str):
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in ("total_loss", "bce", "offset", "attributes")}
    counts = {name: 0 for name in (
        "frames", "observed_points", "represented_points", "observed_voxels",
        "represented_voxels", "predicted_points", "target_points",
        "predicted_voxels", "target_voxels", "voxel_tp", "point_tp_0p2m",
        "chamfer_frames", "attribute_matches", "rcs_abs_sum",
        "radial_abs_sum", "compensated_radial_abs_sum", "scan_age_abs_sum")}
    chamfer_sum = 0.0
    for batch in tqdm(loader, desc=name, unit="batch", dynamic_ncols=True):
        radar = batch["radar"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            prediction = model(radar, valid)
            targets = target_slots(prediction, radar, valid, model.slots)
            loss, terms = reconstruction_loss(prediction, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
        frames = len(batch["frame_id"])
        counts["frames"] += frames
        counts["observed_points"] += targets["observed_points"]
        counts["represented_points"] += targets["represented_points"]
        counts["observed_voxels"] += len(targets["observed_coords"])
        counts["represented_voxels"] += targets["represented_voxels"]
        sums["total_loss"] += float(loss.detach()) * frames
        for key, value in terms.items():
            sums[key] += float(value) * frames
        if not training:
            points, occupied = decoded_points(prediction, threshold=threshold,
                                              slots=model.slots)
            measured = reconstruction_counts(points, occupied, radar, valid)
            chamfer_sum += measured.pop("chamfer_squared_sum")
            for key, value in measured.items():
                counts[key] += value
    n = counts["frames"]
    report = {key: value / n if n else None for key, value in sums.items()}
    report.update(counts)
    report["point_ceiling"] = (counts["represented_points"] / counts["observed_points"]
                               if counts["observed_points"] else None)
    report["voxel_ceiling"] = (counts["represented_voxels"] / counts["observed_voxels"]
                               if counts["observed_voxels"] else None)
    if not training:
        intersection = counts["voxel_tp"]
        union = counts["predicted_voxels"] + counts["target_voxels"] - intersection
        point_tp = counts["point_tp_0p2m"]
        point_total = counts["predicted_points"] + counts["target_points"]
        report.update({
            "voxel_iou": intersection / union if union else 1.0,
            "point_precision_0p2m": (point_tp / counts["predicted_points"]
                                     if counts["predicted_points"] else 0.0),
            "point_recall_0p2m": (point_tp / counts["target_points"]
                                  if counts["target_points"] else 0.0),
            "point_f1_0p2m": 2 * point_tp / point_total if point_total else 1.0,
            "chamfer_squared_m2": (chamfer_sum / counts["chamfer_frames"]
                                   if counts["chamfer_frames"] else None),
        })
        for key in ("rcs", "radial", "compensated_radial", "scan_age"):
            report[f"matched_{key}_mae"] = (
                counts[f"{key}_abs_sum"] / counts["attribute_matches"]
                if counts["attribute_matches"] else None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--slots", type=int, default=SLOTS)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0
            or args.slots < 1 or not 0 < args.threshold < 1):
        parser.error("Invalid training, slot, or threshold setting")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the MinkowskiEngine generative decoder")
    if not args.encoder_checkpoint.is_file():
        parser.error(f"Missing filtered-radar encoder: {args.encoder_checkpoint}")
    train = FilteredRadarCache(args.cache_root, args.raw_root, "train")
    val = FilteredRadarCache(args.cache_root, args.raw_root, "val")
    train.check_complete()
    val.check_complete()
    if args.train_limit is not None:
        train.frame_ids = train.frame_ids[:args.train_limit]
    if args.val_limit is not None:
        val.frame_ids = val.frame_ids[:args.val_limit]
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_filtered_radar,
                            pin_memory=True, persistent_workers=args.num_workers > 0),
        "val": DataLoader(val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_filtered_radar,
                          pin_memory=True, persistent_workers=args.num_workers > 0),
    }
    identity = {
        "version": VERSION,
        "encoder_checkpoint_sha256": _sha256(args.encoder_checkpoint),
        "filtered_manifest_sha256": {
            split: _sha256(args.cache_root / split / "manifest.json")
            for split in ("train", "val")},
        "slots": args.slots, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "threshold": args.threshold,
        "train_limit": args.train_limit, "val_limit": args.val_limit,
    }
    model = RadarSelfAutoencoder(args.encoder_checkpoint, slots=args.slots).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "history.jsonl"
    if history.exists() and args.resume is None:
        parser.error("Existing radar self-reconstruction history requires --resume")
    start, best = 1, -1.0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Radar self-autoencoder resume identity differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_point_f1_0p2m"])
        print(f"Resuming radar self-autoencoder at epoch {start}", flush=True)
    for epoch in range(start, args.epochs + 1):
        training = _epoch(model, loaders["train"], optimizer, args.device,
                          threshold=args.threshold, name=f"Radar self AE train {epoch:03d}")
        with torch.no_grad():
            validation = _epoch(model, loaders["val"], None, args.device,
                                threshold=args.threshold, name=f"Radar self AE val {epoch:03d}")
        score = validation["point_f1_0p2m"]
        improved = score > best
        if improved:
            best = score
        record = {"epoch": epoch, "train": training, "val": validation,
                  "best_point_f1_0p2m": best}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"identity": identity, "epoch": epoch,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "best_point_f1_0p2m": best, "record": record}
        _save(args.output_root / "last.ckpt", checkpoint)
        if improved:
            _save(args.output_root / "best.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: train_loss={training['total_loss']:.4f} "
              f"val_loss={validation['total_loss']:.4f} "
              f"radar_voxel_IoU@{args.threshold:g}={validation['voxel_iou']:.2%} "
              f"radar_point_F1@0.2m={score:.2%} "
              f"P/R={validation['point_precision_0p2m']:.2%}/"
              f"{validation['point_recall_0p2m']:.2%} "
              f"point_ceiling={validation['point_ceiling']:.2%} "
              f"best_F1={best:.2%}", flush=True)
    print(f"Best checkpoint: {args.output_root / 'best.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
