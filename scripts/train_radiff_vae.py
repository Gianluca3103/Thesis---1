"""Train a paper-guided 4D-RaDiff radar VAE on five-sweep VoD frames.

This trains the radar VAE only, not the paper's conditioned diffusion models.
The paper's unpublished fixed point count and layer widths are recorded as
explicit choices in each checkpoint and run_identity.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.official_dpcc_radar_data import reconstruction_counts, summarize_counts
from scripts.radiff_radar_data import (GlobalScale, RadarFiveFrames,
                                       RANGE_MAX, RANGE_MIN)
from scripts.radiff_vae import RaDiffRadarVAE, RaDiffVAEConfig


SEED = 3103
VERSION = "radiff_v2_paper_guided_fullscene_vae_v1"
COUNT_KEYS = ("predicted_points", "target_points", "predicted_voxels",
              "target_voxels", "voxel_tp", "point_tp_0p2m",
              "chamfer_squared_sum", "chamfer_frames", "attribute_matches",
              "rcs_abs_sum", "compensated_radial_abs_sum",
              "outside_predicted_points", "nearest_attribute_points",
              "nearest_rcs_abs_sum", "nearest_compensated_doppler_abs_sum")
LOSS_KEYS = ("chamfer", "density", "cardinality", "feature", "kl", "total")


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _save(path: Path, row: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(row, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _to_metric_columns(points: np.ndarray) -> np.ndarray:
    """Use the previous radar metrics on XYZ/Doppler/RCS without fake scores."""
    expanded = np.zeros((len(points), 7), dtype=np.float64)
    expanded[:, :3] = points[:, :3]
    expanded[:, 3] = points[:, 4]  # RCS
    expanded[:, 5] = points[:, 3]  # compensated Doppler
    return expanded


def _paper_voxel_keys(xyz: np.ndarray, *, predicted: bool) -> tuple[np.ndarray, int]:
    """Voxelize within the paper's VoD crop; out-of-crop outputs are false positives."""
    inside = ((xyz >= RANGE_MIN) & (xyz < RANGE_MAX)).all(axis=1)
    index = np.floor((xyz[inside] - RANGE_MIN) /
                     np.array((0.2, 0.2, 0.25))).astype(np.int64)
    keys = np.unique(index[:, 2] * 256 * 256 + index[:, 1] * 256 + index[:, 0])
    outside = int((~inside).sum())
    if predicted and outside:
        keys = np.concatenate((keys, -np.arange(1, outside + 1, dtype=np.int64)))
    return keys, outside


def matched_counts(predicted: np.ndarray, target: np.ndarray) -> dict:
    measured = reconstruction_counts(_to_metric_columns(predicted),
                                     _to_metric_columns(target))
    pred_keys, outside = _paper_voxel_keys(predicted[:, :3], predicted=True)
    target_keys, _ = _paper_voxel_keys(target[:, :3], predicted=False)
    measured.update({"predicted_voxels": len(pred_keys),
                     "target_voxels": len(target_keys),
                     "voxel_tp": int(np.intersect1d(pred_keys, target_keys,
                                                    assume_unique=True).size),
                     "outside_predicted_points": outside})
    measured.update({"nearest_attribute_points": 0, "nearest_rcs_abs_sum": 0.0,
                     "nearest_compensated_doppler_abs_sum": 0.0})
    if len(predicted) and len(target):
        closest = cKDTree(target[:, :3]).query(predicted[:, :3])[1]
        measured["nearest_attribute_points"] = len(predicted)
        measured["nearest_rcs_abs_sum"] = float(np.abs(
            predicted[:, 4] - target[closest, 4]).sum())
        measured["nearest_compensated_doppler_abs_sum"] = float(np.abs(
            predicted[:, 3] - target[closest, 3]).sum())
    return {name: measured[name] for name in COUNT_KEYS}


def metric_summary(counts: dict) -> dict:
    full = dict(counts)
    full.update({name: 0.0 for name in (
        "radial_abs_sum", "scan_age_abs_sum")})
    summary = summarize_counts(full)
    summary.update({
        "paper_nearest_rcs_mae": counts["nearest_rcs_abs_sum"] /
        counts["nearest_attribute_points"] if counts["nearest_attribute_points"] else None,
        "paper_nearest_compensated_doppler_mae":
        counts["nearest_compensated_doppler_abs_sum"] /
        counts["nearest_attribute_points"] if counts["nearest_attribute_points"] else None,
    })
    return {name: summary[name] for name in (
        *COUNT_KEYS, "voxel_iou", "point_precision_0p2m", "point_recall_0p2m",
        "point_f1_0p2m", "chamfer_squared_m2", "matched_rcs_mae",
        "matched_compensated_radial_mae", "paper_nearest_rcs_mae",
        "paper_nearest_compensated_doppler_mae")}


class FixedInput(Dataset):
    def __init__(self, frames: RadarFiveFrames, scale: GlobalScale,
                 point_count: int, *, training: bool):
        self.frames, self.scale = frames, scale
        self.point_count, self.training = point_count, training
        if point_count < 14:
            raise ValueError("At least 14 points are required for three downsamplings")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        raw = self.frames.load(index)
        n = len(raw)
        if not n:
            return {"frame_id": self.frames.ids[index], "raw": raw,
                    "input": torch.zeros((self.point_count, 5)), "usable": False}
        if self.training:
            if n >= self.point_count:
                selection = torch.randperm(n)[:self.point_count]
            else:
                selection = torch.cat((torch.arange(n),
                                       torch.randint(n, (self.point_count - n,))))
        else:
            if n >= self.point_count:
                selection = torch.linspace(0, n - 1, self.point_count).long()
            else:
                selection = torch.arange(self.point_count) % n
        return {"frame_id": self.frames.ids[index], "raw": raw,
                "input": self.scale.encode(raw[selection]), "usable": True}


def collate(rows: list[dict]) -> dict:
    return {"frame_id": [r["frame_id"] for r in rows],
            "raw": [r["raw"] for r in rows],
            "usable": torch.tensor([r["usable"] for r in rows], dtype=torch.bool),
            "input": torch.stack([r["input"] for r in rows])}


def _loss_terms(output: dict) -> dict:
    return {**{key: float(output["terms"][key].detach())
               for key in LOSS_KEYS if key != "total"},
            "total": float(output["loss"].detach())}


def validate(model: RaDiffRadarVAE, loader: DataLoader,
             scale: GlobalScale, device: torch.device, *, title: str) -> dict:
    model.eval()
    random.seed(SEED)
    counts = {key: 0 for key in COUNT_KEYS}
    losses = {key: 0.0 for key in LOSS_KEYS}
    seen = forwarded = empty = 0
    for batch in tqdm(loader, desc=title, unit="batch", dynamic_ncols=True):
        usable = batch["usable"]
        raw = batch["raw"]
        seen += len(raw)
        empty += int((~usable).sum())
        output = None
        if usable.any():
            with torch.no_grad():
                output = model(batch["input"][usable].transpose(1, 2).contiguous().to(device),
                               sample_latent=False)
            terms = _loss_terms(output)
            count = int(usable.sum())
            forwarded += count
            for name in LOSS_KEYS:
                losses[name] += terms[name] * count
            decoded = output["points"].transpose(1, 2).detach().cpu()
            decoded = scale.decode(decoded).numpy()
        cursor = 0
        for index, target in enumerate(raw):
            if usable[index]:
                predicted = decoded[cursor]
                cursor += 1
            else:
                predicted = np.empty((0, 5), dtype=np.float32)
            measured = matched_counts(predicted, target.numpy())
            for name in COUNT_KEYS:
                counts[name] += measured[name]
    result = metric_summary(counts)
    result.update({f"{name}_loss": losses[name] / forwarded if forwarded else None
                   for name in LOSS_KEYS})
    result.update({"frames": seen, "model_forward_frames": forwarded,
                   "empty_input_frames": empty})
    return result


def _identity(train: RadarFiveFrames, val: RadarFiveFrames,
              scale: GlobalScale, config: RaDiffVAEConfig,
              point_count: int, batch_size: int) -> dict:
    return {"version": VERSION, "paper": "arXiv:2512.14235v2",
            "upstream_dpcc_revision": "d738f59284e505166d26127dbe21cc71e2fa62c9",
            "train_manifest_sha256": train.sha256(),
            "val_manifest_sha256": val.sha256(), "train_frames": len(train),
            "val_frames": len(val), "model": asdict(config),
            "global_scale": scale.to_json(), "fixed_input_points": point_count,
            "batch_size": batch_size,
            "data": "VoD native radar_5frames, paper ROI, XYZ/compensated-Doppler/RCS",
            "objective": "Chamfer+intermediate-Chamfer+density+cardinality+feature-MSE+KL",
            "publication_deviations": [
                "Full-scene VAE rather than separate annotated foreground/background VAEs",
                "Fixed input point count and unpublished MLP widths are educated guesses",
                "Global normalization bounds fitted on training data",
                "Uses pinned D-PCC point blocks as the paper describes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-points", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.input_points < 14:
        parser.error("epochs and batch size must be positive; input points must be >=14")
    if not torch.cuda.is_available():
        parser.error("The pinned D-PCC CUDA pointops and Chamfer operators require CUDA")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.phase == "evaluate":
        if not args.checkpoint or not args.report_json:
            parser.error("Evaluation requires --checkpoint and --report-json")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = saved["identity"]
        if identity["version"] != VERSION:
            raise ValueError("Checkpoint is from another experiment")
        config = RaDiffVAEConfig(**identity["model"])
        scale = GlobalScale.from_json(identity["global_scale"])
        frames = RadarFiveFrames(args.cache_root, args.split, args.val_limit)
        if (args.split == "val" and args.val_limit is None and
                frames.sha256() != identity["val_manifest_sha256"]):
            raise ValueError("Validation data changed since training")
        model = RaDiffRadarVAE(config).to(device)
        model.load_state_dict(saved["model"])
        loader = DataLoader(FixedInput(frames, scale, identity["fixed_input_points"],
                                       training=False), batch_size=identity["batch_size"],
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        result = {"split": args.split, "checkpoint_epoch": saved["epoch"],
                  "manifest_sha256": frames.sha256(), "identity": identity,
                  "metrics": validate(model, loader, scale, device,
                                      title=f"4D-RaDiff VAE {args.split}")}
        _write_json(args.report_json, result)
        print(json.dumps(result, indent=2), flush=True)
        return

    train = RadarFiveFrames(args.cache_root, "train", args.train_limit)
    val = RadarFiveFrames(args.cache_root, "val", args.val_limit)
    last_path = args.output_root / "last.ckpt"
    saved = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.is_file() else None
    if saved is None and (args.output_root / "history.jsonl").exists():
        raise RuntimeError("History exists without checkpoint; use a new output directory")
    scale = (GlobalScale.from_json(saved["identity"]["global_scale"])
             if saved is not None else GlobalScale.fit(train))
    config = RaDiffVAEConfig()
    identity = _identity(train, val, scale, config, args.input_points, args.batch_size)
    if saved is not None and saved["identity"] != identity:
        raise ValueError("Checkpoint identity differs; refusing incompatible resume")
    _write_json(args.output_root / "run_identity.json", identity)
    model = RaDiffRadarVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.5)
    best_iou = -1.0
    start = 1
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_iou = saved["best_voxel_iou"]
        start = saved["epoch"] + 1
        print(f"Resuming 4D-RaDiff VAE from epoch {start}", flush=True)
    train_loader = DataLoader(FixedInput(train, scale, args.input_points, training=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(FixedInput(val, scale, args.input_points, training=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    for epoch in range(start, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in LOSS_KEYS}
        trained = 0
        for batch in tqdm(train_loader, desc=f"4D-RaDiff VAE train {epoch:03d}",
                          unit="batch", dynamic_ncols=True):
            usable = batch["usable"]
            if not usable.any():
                continue
            points = batch["input"][usable].transpose(1, 2).contiguous().to(device)
            output = model(points)
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            count = len(points)
            trained += count
            terms = _loss_terms(output)
            for name in LOSS_KEYS:
                totals[name] += terms[name] * count
        if not trained:
            raise ValueError("No nonempty train frames in the paper ROI")
        scheduler.step()
        training = {f"{name}_loss": totals[name] / trained for name in LOSS_KEYS}
        training.update({"frames": len(train), "model_forward_frames": trained})
        validation = validate(model, val_loader, scale, device,
                              title=f"4D-RaDiff VAE val {epoch:03d}")
        row = {"epoch": epoch, "train": training, "val": validation}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        improved = validation["voxel_iou"] > best_iou
        best_iou = max(best_iou, validation["voxel_iou"])
        checkpoint = {"epoch": epoch, "identity": identity,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "best_voxel_iou": best_iou}
        _save(last_path, checkpoint)
        if improved:
            _save(args.output_root / "best_voxel_iou.ckpt", checkpoint)
        print(f"Epoch {epoch:03d}: train_loss={training['total_loss']:.6f} "
              f"val_loss={validation['total_loss']:.6f} "
              f"val_voxel_IoU={validation['voxel_iou']:.2%} "
              f"val_point_F1@0.2m={validation['point_f1_0p2m']:.2%} "
              f"val_Chamfer={validation['chamfer_squared_m2']:.6f}m2 "
              f"best_IoU={best_iou:.2%}", flush=True)


if __name__ == "__main__":
    main()
