"""Train a bottleneck-only MinkowskiEngine LiDAR voxel autoencoder on VoD."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lidar_minkowski_ae import SparseVoxelAutoencoder, _inside, _keys, _targets_at
from models.lidar_minkowski_ae_data import CleanVoxelFrames, summarize_counts
from models.radar_lidar_stage2.sparse_geometry import GRID


def _collate(rows: list[dict]) -> dict:
    coordinates = []
    for batch, row in enumerate(rows):
        cells = torch.from_numpy(row["cells"].copy()).int()
        coordinates.append(torch.cat((torch.full((len(cells), 1), batch,
                                                 dtype=torch.int32), cells), dim=1))
    return {"frame_ids": [row["frame_id"] for row in rows],
            "coordinates": torch.cat(coordinates)}


def _counts(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    predicted_keys = torch.unique(_keys(predicted[_inside(predicted)]))
    target_keys = torch.unique(_keys(target))
    return {"tp": int(torch.isin(predicted_keys, target_keys).sum().item()),
            "predicted": len(predicted_keys), "target": len(target_keys)}


def _run(model, loader, device, *, optimizer, threshold, label):
    training = optimizer is not None
    model.train(training)
    frames, loss_sum, latent_sum, oracle_sum = 0, 0.0, 0, 0
    totals = {"tp": 0, "predicted": 0, "target": 0}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for row in tqdm(loader, desc=label, unit="batch", dynamic_ncols=True):
            fine = row["coordinates"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if training:
                output = model(fine, threshold=threshold)
            else:
                # Keep clean targets outside the sparse decoder at evaluation.
                # They are used only after candidate coordinates/logits exist.
                output = model.decode(model.encode(fine), threshold=threshold)
                labels = _targets_at(output.coordinates, fine, 1)
                output.loss = F.binary_cross_entropy_with_logits(
                    output.logits, labels.float())
                output.reachable_targets = int(labels.sum().item())
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite voxel reconstruction loss: {row['frame_ids']}")
            if training:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            nframes = len(row["frame_ids"])
            frames += nframes
            loss_sum += float(output.loss.detach()) * nframes
            latent_sum += output.latent_sites
            if not training:
                oracle_sum += int(output.reachable_targets or 0)
                keep = output.logits.detach() >= math.log(threshold / (1 - threshold))
                predicted = output.coordinates[keep]
                for key, value in _counts(predicted, fine).items():
                    totals[key] += value
    result = {"frames": frames, "loss": loss_sum / frames,
              "mean_latent_sites": latent_sum / frames}
    if not training:
        result.update(summarize_counts(totals))
        result["candidate_recall_ceiling"] = oracle_sum / totals["target"]
    return result


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(model, train: CleanVoxelFrames, val: CleanVoxelFrames) -> dict:
    def digest(dataset):
        names = "\n".join(frame.frame_id for frame in dataset.frames)
        return hashlib.sha256(names.encode("ascii")).hexdigest()
    return {"version": "minkowski_lidar_voxel_ae_v1",
            "grid": asdict(GRID), "model": {"width": model.width,
            "levels": model.levels}, "train_ids_sha256": digest(train),
            "val_ids_sha256": digest(val)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cache", "train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--levels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--seed", type=int, default=3103)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0
            or args.lr <= 0 or not 0 < args.threshold < 1):
        parser.error("Invalid training or threshold setting")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.phase == "cache":
        for split in ("train", "val", "test"):
            data = CleanVoxelFrames(args.vod_root, split, cache_root=args.cache_root)
            for index in tqdm(range(len(data)), desc=f"VoD clean voxels {split}", unit="frame"):
                data[index]
            print(f"Cached {split}: {len(data)} frames", flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("MinkowskiEngine training/evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "evaluate":
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("--phase evaluate requires --checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("grid") != asdict(GRID):
            raise ValueError("Checkpoint has a different LiDAR voxel grid")
        model = SparseVoxelAutoencoder(**saved["model"]).to(device)
        model.load_state_dict(saved["model_state"])
        data = CleanVoxelFrames(args.vod_root, args.split, cache_root=args.cache_root)
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=_collate, shuffle=False, pin_memory=True)
        report = {"epoch": saved["epoch"], "split": args.split,
                  "threshold": args.threshold,
                  "metrics": _run(model, loader, device, optimizer=None,
                                  threshold=args.threshold, label=f"Minkowski {args.split}")}
        path = args.output_root / f"evaluation_{args.split}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        print(f"Report: {path}", flush=True)
        return
    train = CleanVoxelFrames(args.vod_root, "train", cache_root=args.cache_root,
                             limit=args.train_limit)
    val = CleanVoxelFrames(args.vod_root, "val", cache_root=args.cache_root,
                           limit=args.val_limit)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=_collate,
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_collate,
                            pin_memory=True)
    model = SparseVoxelAutoencoder(width=args.width, levels=args.levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    identity = _identity(model, train, val)
    history = args.output_root / "history.jsonl"
    if history.exists() and args.resume is None:
        parser.error("Existing run needs --resume or a new --output-root")
    first, best = 1, -1.0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume checkpoint has different model, grid, or frame split")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        first, best = int(saved["epoch"]) + 1, float(saved["best_val_iou"])
    print(f"Minkowski LiDAR AE: {sum(p.numel() for p in model.parameters()):,} "
          f"parameters, {len(train)} train / {len(val)} val frames, "
          f"width={args.width}, downsampling={2**args.levels}x", flush=True)
    for epoch in range(first, args.epochs + 1):
        training = _run(model, train_loader, device, optimizer=optimizer,
                        threshold=args.threshold, label=f"LiDAR AE train {epoch:03d}")
        validation = _run(model, val_loader, device, optimizer=None,
                          threshold=args.threshold, label=f"LiDAR AE val {epoch:03d}")
        improved = validation["exact_iou"] > best
        best = max(best, validation["exact_iou"])
        record = {"epoch": epoch, "train": training, "val": validation,
                  "best_val_iou": best, "threshold": args.threshold}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        saved = {**identity, "epoch": epoch, "best_val_iou": best,
                 "model_state": model.state_dict(), "optimizer": optimizer.state_dict()}
        _save(args.output_root / "last.ckpt", saved)
        if improved:
            _save(args.output_root / "best.ckpt", saved)
        print(f"Epoch {epoch:03d}: train_BCE={training['loss']:.4f} "
              f"val_BCE={validation['loss']:.4f} "
              f"val_exact_IoU={validation['exact_iou']:.2%} "
              f"val_F1={validation['exact_f1']:.2%} "
              f"candidate_recall_ceiling={validation['candidate_recall_ceiling']:.2%} "
              f"best_IoU={best:.2%}", flush=True)
    print(f"Checkpoints and history: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
