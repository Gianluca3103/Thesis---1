"""Train matched PointNet++ or VoxelNet radar AEs on Stage-II-cropped radar."""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.official_dpcc_radar_data import reconstruction_counts, summarize_counts
from scripts.pointnetpp_stage2_radar_data import (
    Stage2RadarFrames, Stage2RadarScale, VERSION as CACHE_VERSION,
)
from scripts.radar_encoder_ablation import RadarAutoencoder, reconstruction_loss


VERSION = "pointnetpp_stage2_radar20_gate12_large_v1"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if not zipfile.is_zipfile(temporary):
            raise OSError(f"Incomplete checkpoint archive: {temporary}")
        temporary.replace(path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class FixedRadarInput(Dataset):
    def __init__(self, frames: Stage2RadarFrames, scale: Stage2RadarScale,
                 count: int, *, training: bool):
        self.frames, self.scale, self.count, self.training = frames, scale, count, training

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        raw = self.frames.load(index)
        if not len(raw):
            return {"raw": raw, "input": torch.zeros((self.count, 7)), "usable": False}
        if self.training:
            selected = (torch.randperm(len(raw))[:self.count]
                        if len(raw) >= self.count else
                        torch.cat((torch.arange(len(raw)),
                                   torch.randint(len(raw), (self.count - len(raw),)))))
        else:
            selected = (torch.linspace(0, len(raw) - 1, self.count).long()
                        if len(raw) >= self.count else torch.arange(self.count) % len(raw))
        return {"raw": raw, "input": self.scale.encode(raw[selected]), "usable": True}


def collate(rows: list[dict]) -> dict:
    return {"raw": [row["raw"] for row in rows],
            "input": torch.stack([row["input"] for row in rows]),
            "usable": torch.tensor([row["usable"] for row in rows], dtype=torch.bool)}


def make_loader(frames: Stage2RadarFrames, scale: Stage2RadarScale,
                point_count: int, batch_size: int, workers: int, *, training: bool) -> DataLoader:
    return DataLoader(FixedRadarInput(frames, scale, point_count, training=training),
                      batch_size=batch_size, shuffle=training, num_workers=workers,
                      collate_fn=collate, pin_memory=True,
                      persistent_workers=workers > 0)


def evaluate(model: RadarAutoencoder, loader: DataLoader,
             scale: Stage2RadarScale, description: str,
             *, zero_features: bool = False) -> dict:
    model.eval()
    empty = np.empty((0, 7), dtype=np.float64)
    counts = reconstruction_counts(empty, empty, include_voxel_iou_0p2m=True)
    losses = {name: 0.0 for name in ("total", "chamfer", "close", "feature")}
    frames = forwarded = empty_frames = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
            usable = batch["usable"]
            frames += len(usable)
            empty_frames += int((~usable).sum())
            predictions = None
            if usable.any():
                target = batch["input"][usable].cuda(non_blocking=True)
                predicted = model(target, zero_features=zero_features)
                terms = reconstruction_loss(predicted, target,
                                            model.minimum, model.maximum)
                forwarded += len(predicted)
                for name, value in terms.items():
                    losses[name] += float(value) * len(predicted)
                predictions = scale.decode(predicted.cpu()).numpy()
            cursor = 0
            for is_usable, truth in zip(usable.tolist(), batch["raw"]):
                points = predictions[cursor] if is_usable else empty
                cursor += int(is_usable)
                measured = reconstruction_counts(points, truth.numpy(),
                                                 include_voxel_iou_0p2m=True)
                for name in counts:
                    counts[name] += measured[name]
    result = summarize_counts(counts)
    result.update({f"{name}_loss": value / forwarded if forwarded else None
                   for name, value in losses.items()})
    result.update({"frames": frames, "model_forward_frames": forwarded,
                   "empty_frames": empty_frames})
    return result


def build_model(identity: dict) -> RadarAutoencoder:
    scale = Stage2RadarScale.from_json(identity["scale"])
    return RadarAutoencoder(identity.get("variant", "pointnetpp"),
                            scale.minimum, scale.maximum,
                            width=identity["width"], anchors=identity["anchors"],
                            children=identity["children"], columns=7,
                            dropout_p=identity.get("dropout_p", 0.0)).cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("pointnetpp", "voxelnet"),
                        default="pointnetpp")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--children", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            args.input_points < args.anchors or args.anchors < 4 or
            args.width < 16 or args.width % 2 or args.children < 1 or
            not 0.0 <= args.dropout_p < 1.0):
        parser.error("Invalid training or model dimensions")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for radar autoencoder training")
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(3103)
    np.random.seed(3103)
    torch.manual_seed(3103)
    torch.cuda.manual_seed_all(3103)

    if args.phase == "evaluate":
        if args.checkpoint is None:
            parser.error("Evaluation needs --checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = saved["identity"]
        if (identity["version"] != VERSION or
                identity["cache_version"] != CACHE_VERSION or
                identity.get("variant", "pointnetpp") != args.variant):
            raise ValueError("Wrong radar checkpoint")
        model = build_model(identity)
        model.load_state_dict(saved["model"])
        scale = Stage2RadarScale.from_json(identity["scale"])
        frames = Stage2RadarFrames(args.cache_root, args.split)
        if args.split == "val" and frames.sha256() != identity["val_manifest"]:
            raise ValueError("Validation cache differs from checkpoint")
        loader = make_loader(frames, scale, identity["input_points"],
                             identity["batch_size"], args.workers, training=False)
        result = {"checkpoint_epoch": saved["epoch"], "split": args.split,
                  "metrics": evaluate(model, loader, scale, f"Stage-II radar {args.split}"),
                  "zero_feature_ablation": evaluate(
                      model, loader, scale, f"Stage-II radar {args.split} zero-feature",
                      zero_features=True)}
        save_json(args.output_root / f"{args.checkpoint.stem}_{args.split}.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    train = Stage2RadarFrames(args.cache_root, "train", args.train_limit)
    val = Stage2RadarFrames(args.cache_root, "val", args.val_limit)
    last_path = args.output_root / "last.ckpt"
    saved = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
    if saved is None:
        print(f"Fresh {args.variant} run: {len(train)} train / {len(val)} validation frames; "
              f"{args.epochs} epochs, batch size {args.batch_size}, "
              f"{args.input_points} points, {args.anchors} anchors, "
              f"{args.width} channels, dropout {args.dropout_p}. "
              "Fitting train-only radar scale.", flush=True)
    scale = (Stage2RadarScale.from_json(saved["identity"]["scale"])
             if saved else Stage2RadarScale.fit(train))
    identity = {"version": VERSION, "cache_version": CACHE_VERSION,
                "train_manifest": train.sha256(), "val_manifest": val.sha256(),
                "train_frames": len(train), "val_frames": len(val),
                "scale": scale.to_json(), "input_points": args.input_points,
                "anchors": args.anchors, "width": args.width,
                "children": args.children, "batch_size": args.batch_size,
                "loss": "Chamfer + SmoothL1 beta=0.2 + 0.05 attribute MSE",
                "optimizer": "AdamW lr=0.0003 wd=0.0001 ReduceLROnPlateau"}
    if args.variant != "pointnetpp":
        identity["variant"] = args.variant
    if args.dropout_p:
        identity["dropout_p"] = args.dropout_p
    if saved and saved["identity"] != identity:
        raise ValueError("Checkpoint identity differs; use a separate output directory")
    save_json(args.output_root / "run_identity.json", identity)
    model = build_model(identity)
    print(f"{args.variant} width={args.width}: "
          f"{sum(parameter.numel() for parameter in model.parameters()):,} "
          "trainable parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best_iou = best_f1 = -1.0
    start = 1
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_iou, best_f1 = saved["best_iou"], saved["best_f1"]
        start = saved["epoch"] + 1
        print(f"Resuming at epoch {start}", flush=True)
    train_loader = make_loader(train, scale, args.input_points,
                               args.batch_size, args.workers, training=True)
    val_loader = make_loader(val, scale, args.input_points,
                             args.batch_size, args.workers, training=False)
    for epoch in range(start, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("total", "chamfer", "close", "feature")}
        trained = 0
        for batch in tqdm(train_loader, desc=f"{args.variant} Stage-II radar train {epoch:03d}",
                          unit="batch", dynamic_ncols=True):
            usable = batch["usable"]
            if not usable.any():
                continue
            target = batch["input"][usable].cuda(non_blocking=True)
            predicted = model(target)
            terms = reconstruction_loss(predicted, target,
                                        model.minimum, model.maximum)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trained += len(target)
            for name, value in terms.items():
                totals[name] += float(value.detach()) * len(target)
        if not trained:
            raise ValueError("No nonempty training radar frames")
        validation = evaluate(model, val_loader, scale,
                              f"{args.variant} Stage-II radar val {epoch:03d}")
        scheduler.step(validation["total_loss"])
        training = {f"{name}_loss": total / trained for name, total in totals.items()}
        improved_iou = validation["voxel_iou"] > best_iou
        improved_f1 = validation["point_f1_0p2m"] > best_f1
        best_iou = max(best_iou, validation["voxel_iou"])
        best_f1 = max(best_f1, validation["point_f1_0p2m"])
        row = {"epoch": epoch, "train": training, "val": validation,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        checkpoint = {"epoch": epoch, "identity": identity,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "best_iou": best_iou, "best_f1": best_f1}
        save_checkpoint(last_path, checkpoint)
        if improved_iou:
            save_checkpoint(args.output_root / "best_iou.ckpt", checkpoint)
        if improved_f1:
            save_checkpoint(args.output_root / "best_f1.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: train_loss={training['total_loss']:.5f} "
              f"val_loss={validation['total_loss']:.5f} "
              f"voxel_IoU={validation['voxel_iou']:.2%} "
              f"point_F1@0.2m={validation['point_f1_0p2m']:.2%} "
              f"Chamfer={validation['chamfer_squared_m2']:.5f}m2 "
              f"lr={optimizer.param_groups[0]['lr']:.2g}", flush=True)


if __name__ == "__main__":
    main()
