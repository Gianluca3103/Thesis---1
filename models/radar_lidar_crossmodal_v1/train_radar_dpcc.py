"""Train a density-preserving radar -> identical filtered radar point autoencoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import voxelize

from .encoder import GRID
from .radar_dpcc import MAX_CHILDREN, WIDTH, RadarDPCC, reconstruction_loss
from .radar_dpcc_cache import RadarDPCCCache
from .radar_self_autoencoder import reconstruction_counts


VERSION = "radar_dpcc_point_autoencoder_v1"


def collate_rows(rows):
    return rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def epoch(model, loader, optimizer, device, *, threshold, name):
    training = optimizer is not None
    model.train(training)
    sums = {key: 0.0 for key in ("total_loss", "chamfer", "features",
                                      "intermediate", "density")}
    counts = {key: 0 for key in ("frames", "latent_points", "predicted_points",
                                      "target_points", "predicted_voxels", "target_voxels",
                                      "voxel_tp", "point_tp_0p2m", "chamfer_frames",
                                      "attribute_matches", "rcs_abs_sum", "radial_abs_sum",
                                      "compensated_radial_abs_sum", "scan_age_abs_sum")}
    chamfer_sum = 0.0
    for batch in tqdm(loader, desc=name, unit="batch", dynamic_ncols=True):
        if training:
            optimizer.zero_grad(set_to_none=True)
        for row in batch:
            points = row["points"].to(device, non_blocking=True)
            hierarchy = [{key: value.to(device, non_blocking=True) for key, value in stage.items()}
                         for stage in row["hierarchy"]]
            with torch.set_grad_enabled(training):
                result = model(points, hierarchy, threshold=threshold)
                loss, terms = reconstruction_loss(result, points, hierarchy)
                if training and len(points):
                    (loss / len(batch)).backward()
            counts["frames"] += 1
            counts["latent_points"] += result.latent_count
            sums["total_loss"] += float(loss.detach())
            for key, value in terms.items():
                sums[key] += float(value)
            if not training:
                predicted = torch.cat((torch.zeros((len(result.xyz), 1), device=device),
                                       result.xyz, result.attributes), dim=1)
                if len(result.xyz):
                    pred_voxels, _, _, _ = voxelize(
                        torch.cat((result.xyz, result.attributes), dim=1)[None],
                        torch.ones((1, len(result.xyz)), device=device, dtype=torch.bool), GRID)
                else:
                    pred_voxels = torch.empty((0, 4), device=device, dtype=torch.long)
                measured = reconstruction_counts(predicted, pred_voxels, points[None],
                                                 torch.ones((1, len(points)), device=device,
                                                            dtype=torch.bool))
                chamfer_sum += measured.pop("chamfer_squared_sum")
                for key, value in measured.items():
                    counts[key] += value
        if training and any(parameter.grad is not None for parameter in model.parameters()):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    n = counts["frames"]
    report = {key: value / n if n else None for key, value in sums.items()}
    report.update(counts)
    report["mean_latent_points"] = counts["latent_points"] / n if n else None
    report["latent_scalar_ratio"] = (
        counts["latent_points"] * (model.width + 3) / (counts["target_points"] * 7)
        if not training and counts["target_points"] else None)
    if not training:
        tp = counts["voxel_tp"]
        union = counts["predicted_voxels"] + counts["target_voxels"] - tp
        point_tp = counts["point_tp_0p2m"]
        point_total = counts["predicted_points"] + counts["target_points"]
        report.update({
            "voxel_iou": tp / union if union else 1.0,
            "point_precision_0p2m": point_tp / counts["predicted_points"] if counts["predicted_points"] else 0.0,
            "point_recall_0p2m": point_tp / counts["target_points"] if counts["target_points"] else 0.0,
            "point_f1_0p2m": 2 * point_tp / point_total if point_total else 1.0,
            "chamfer_squared_m2": chamfer_sum / counts["chamfer_frames"] if counts["chamfer_frames"] else None,
        })
        for key in ("rcs", "radial", "compensated_radial", "scan_age"):
            report[f"matched_{key}_mae"] = (
                counts[f"{key}_abs_sum"] / counts["attribute_matches"]
                if counts["attribute_matches"] else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--max-children", type=int, default=MAX_CHILDREN)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0
            or args.max_children < 2 or args.width < 8 or not 0 < args.threshold < 1):
        parser.error("Invalid model or training setting")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(3103)
    datasets = {split: RadarDPCCCache(args.filtered_root, args.raw_root,
                                      args.geometry_root, split)
                for split in ("train", "val")}
    for cache in datasets.values():
        cache.check_complete()
    if args.train_limit is not None:
        datasets["train"].frame_ids = datasets["train"].frame_ids[:args.train_limit]
    if args.val_limit is not None:
        datasets["val"].frame_ids = datasets["val"].frame_ids[:args.val_limit]
    loaders = {split: DataLoader(cache, batch_size=args.batch_size,
                                 shuffle=split == "train", num_workers=args.workers,
                                 collate_fn=collate_rows, pin_memory=True,
                                 persistent_workers=args.workers > 0)
               for split, cache in datasets.items()}
    identity = {"version": VERSION, "width": args.width,
                "max_children": args.max_children, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "threshold": args.threshold,
                "train_limit": args.train_limit, "val_limit": args.val_limit,
                "geometry_manifest_sha256": {
                    split: _sha256(args.geometry_root / split / "manifest.json")
                    for split in ("train", "val")}}
    model = RadarDPCC(args.width, args.max_children).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "history.jsonl"
    if history.exists() and args.resume is None:
        parser.error("Existing D-PCC history requires --resume or a new output directory")
    start, best = 1, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("D-PCC resume identity differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["epoch"] + 1, saved["best_point_f1_0p2m"]
        print(f"Resuming radar D-PCC at epoch {start}", flush=True)
    for number in range(start, args.epochs + 1):
        training = epoch(model, loaders["train"], optimizer, args.device,
                         threshold=args.threshold, name=f"Radar D-PCC train {number:03d}")
        with torch.no_grad():
            validation = epoch(model, loaders["val"], None, args.device,
                               threshold=args.threshold, name=f"Radar D-PCC val {number:03d}")
        score = validation["point_f1_0p2m"]
        improved = score > best
        best = max(best, score)
        record = {"epoch": number, "train": training, "val": validation,
                  "best_point_f1_0p2m": best}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {"identity": identity, "epoch": number, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "best_point_f1_0p2m": best,
                   "record": record}
        _save(args.output_root / "last.ckpt", payload)
        if improved:
            _save(args.output_root / "best.ckpt", payload)
        print(f"Epoch {number:03d}: train_loss={training['total_loss']:.4f} "
              f"val_loss={validation['total_loss']:.4f} "
              f"radar_voxel_IoU@{args.threshold:g}={validation['voxel_iou']:.2%} "
              f"radar_point_F1@0.2m={score:.2%} "
              f"P/R={validation['point_precision_0p2m']:.2%}/"
              f"{validation['point_recall_0p2m']:.2%} "
              f"chamfer={validation['chamfer_squared_m2']:.4f}m2 "
              f"latent_points/frame={validation['mean_latent_points']:.0f} "
              f"latent_scalar_ratio={validation['latent_scalar_ratio']:.2f} "
              f"best_F1={best:.2%}", flush=True)
    print(f"Best checkpoint: {args.output_root / 'best.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
