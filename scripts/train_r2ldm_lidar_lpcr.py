"""Train an independent R2LDM-style LiDAR encoder and point decoder on VoD."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.r2ldm_lidar_lpcr import (
    R2LDMGrid, R2LDMLidarLPCR, VoDLidarLPCRDataset, collate_lpcr, lpcr_loss,
)


VERSION = "r2ldm_lidar_lpcr_independent_v1"


def _frame_hash(dataset: VoDLidarLPCRDataset) -> str:
    payload = "\n".join(frame_id for frame_id, _ in dataset.frames).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _epoch(model, loader, optimizer, scaler, device, *, offset_weight, amp, name):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "occupancy": 0.0, "offset": 0.0, "iou": 0.0}
    frames = 0
    progress = tqdm(loader, desc=name, dynamic_ncols=True, unit="batch")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in progress:
            means = batch["means"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda", dtype=torch.float16)
                  if amp else nullcontext()):
                output = model(means, coords, batch["batch_size"])
                metrics = lpcr_loss(model, output, coords, means, offset_weight)
            if training:
                scaler.scale(metrics["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            count = batch["batch_size"]
            frames += count
            for key in totals:
                totals[key] += float(metrics[key].detach()) * count
            progress.set_postfix(loss=f"{totals['loss'] / frames:.4f}",
                                 iou=f"{totals['iou'] / frames:.4f}")
    return {key: value / frames for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        help="Independent LiDAR voxel cache; defaults to OUTPUT_ROOT/voxel_cache")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--offset-weight", type=float, default=0.04)
    parser.add_argument("--offset-activation", choices=("centered_tanh", "legacy_sigmoid"),
                        default="centered_tanh")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if (not torch.cuda.is_available() or args.epochs < 1 or args.batch_size < 1
            or args.num_workers < 0 or args.learning_rate <= 0
            or args.weight_decay < 0 or args.offset_weight < 0):
        parser.error("LPCR needs CUDA and valid positive training settings")
    grid = R2LDMGrid()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root or args.output_root / "voxel_cache"
    train = VoDLidarLPCRDataset(args.vod_root, "train", grid=grid,
                                limit=args.train_limit, cache_root=cache_root)
    val = VoDLidarLPCRDataset(args.vod_root, "val", grid=grid,
                              limit=args.val_limit, cache_root=cache_root)
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_lpcr,
                            pin_memory=True, persistent_workers=args.num_workers > 0),
        "val": DataLoader(val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_lpcr,
                          pin_memory=True, persistent_workers=args.num_workers > 0),
    }
    if (args.output_root / "metrics.jsonl").exists() and args.resume is None:
        parser.error("Existing LPCR metrics require --resume or a new output root")
    model = R2LDMLidarLPCR(grid, offset_activation=args.offset_activation).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    identity = {"version": VERSION, "grid": asdict(grid),
                "offset_activation": args.offset_activation,
                "offset_weight": args.offset_weight,
                "train_frames_sha256": _frame_hash(train),
                "val_frames_sha256": _frame_hash(val)}
    start, best = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("LPCR checkpoint uses different data, grid, or loss settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_val_loss"])
    for epoch in range(start, args.epochs + 1):
        training = _epoch(model, loaders["train"], optimizer, scaler, args.device,
                          offset_weight=args.offset_weight, amp=args.amp,
                          name=f"LPCR train {epoch:03d}")
        validation = _epoch(model, loaders["val"], None, scaler, args.device,
                            offset_weight=args.offset_weight, amp=args.amp,
                            name=f"LPCR val {epoch:03d}")
        scheduler.step()
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
        record = {"epoch": epoch, "train": training, "val": validation,
                  "best_val_loss": best, "learning_rate": scheduler.get_last_lr()[0]}
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {**identity, "epoch": epoch, "best_val_loss": best,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
        _save(args.output_root / "last.ckpt", checkpoint)
        if improved:
            _save(args.output_root / "best.ckpt", checkpoint)
        print(f"LPCR {epoch:03d}/{args.epochs:03d} train={training} val={validation} "
              f"best={best:.5f}", flush=True)
    print(f"LPCR checkpoints and metrics: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
