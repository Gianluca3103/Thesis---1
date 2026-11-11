"""Train/evaluate upstream D-PCC on whole, density-filtered VoD radar frames.

Run with PYTHONPATH=<patched upstream>:<this repo>. The imported AutoEncoder,
encoder, decoder, sampling, entropy bottlenecks, and losses are upstream code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from scripts.official_dpcc_radar_data import (ATTR_NAMES, FilteredFrames,
                                              RadarScale, reconstruction_counts,
                                              summarize_counts)


UPSTREAM_REV = "d738f59284e505166d26127dbe21cc71e2fa62c9"
VERSION = "official_dpcc_radar_4attribute_wholeframe_v1"
COUNT_KEYS = ("predicted_points", "target_points", "predicted_voxels",
              "target_voxels", "voxel_tp", "point_tp_0p2m",
              "chamfer_squared_sum", "chamfer_frames", "attribute_matches")
LOSS_KEYS = ("chamfer_loss", "density_loss", "pts_num_loss",
             "latent_xyzs_loss", "normal_loss", "bpp_loss")


@dataclass(frozen=True)
class ModelConfig:
    # Official `python train.py --dataset semantickitti --compress_normal True`
    # defaults, with only 7 input channels and a 4-channel radar attribute
    # output in place of XYZ+normals. train.py overrides the args module's
    # bpp_lambda=5e-4 to 1e-3 through reset_model_args.
    in_fdim: int = 7
    radar_attribute_dim: int = 4
    compress_normal: bool = True
    k: int = 16
    downsample_rate: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    max_upsample_num: tuple[int, ...] = (8, 8, 8)
    layer_num: int = 3
    dim: int = 8
    hidden_dim: int = 64
    ngroups: int = 1
    quantize_latent_xyzs: bool = True
    latent_xyzs_conv_mode: str = "mlp"
    sub_point_conv_mode: str = "mlp"
    chamfer_coe: float = 0.1
    pts_num_coe: float = 5e-7
    normal_coe: float = 1e-2
    bpp_lambda: float = 1e-3
    mean_distance_coe: float = 50.0
    density_coe: float = 1e-4
    latent_xyzs_coe: float = 1e-2
    lr: float = 1e-3
    aux_lr: float = 1e-3
    lr_decay_step: int = 15
    gamma: float = 0.5

    def upstream_args(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self))


def _summarize_frames(model, frames: FilteredFrames, scale: RadarScale,
                      device: torch.device, title: str) -> dict:
    model.eval()
    counts = {key: 0 for key in COUNT_KEYS}
    counts.update({f"{name}_abs_sum": 0.0 for name in ATTR_NAMES})
    losses = {key: 0.0 for key in LOSS_KEYS}
    losses["total_loss"] = 0.0
    for i in tqdm(range(len(frames)), desc=title, unit="frame", dynamic_ncols=True):
        raw = frames.load(i)
        inp = scale.encode(raw).T.contiguous()[None].to(device)
        with torch.no_grad():
            reset_encoder_neighbors(model)
            xyz, attrs, loss, terms, _ = model(inp)
        predicted = scale.decode(torch.cat((xyz[0].T, attrs[0].T), 1)).cpu().numpy()
        measured = reconstruction_counts(predicted, raw.numpy())
        for key in counts:
            counts[key] += measured[key]
        losses["total_loss"] += float(loss)
        for key in LOSS_KEYS:
            losses[key] += terms[key]
    result = summarize_counts(counts)
    result.update({key: value / len(frames) for key, value in losses.items()})
    result["frames"] = len(frames)
    return result


def reset_encoder_neighbors(model) -> None:
    """Undo upstream's persistent k reduction after processing a small frame.

    DownsampleLayer.forward assigns `self.k = min(self.k, N)`. Without this
    reset, one sparse frame permanently changes all later frames' receptive
    fields. Each frame still uses the upstream adaptive k=min(16,N).
    """
    for layer in model.encoder.encoder_layers:
        layer.k = model.args.k


def _atomic_save(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json(path: Path, row: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _identity(train: FilteredFrames, val: FilteredFrames,
              scale: RadarScale, config: ModelConfig) -> dict:
    return {"version": VERSION, "upstream_revision": UPSTREAM_REV,
            "train_manifest_sha256": train.sha256(), "val_manifest_sha256": val.sha256(),
            "train_frames": len(train), "val_frames": len(val),
            "radar_scale": scale.to_json(), "model": asdict(config),
            "batch_size": 1, "loss": "original_semantickitti_D-PCC_rate_distortion"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not torch.cuda.is_available():
        parser.error("Original D-PCC CUDA pointops and Chamfer3D are required")
    torch.manual_seed(3103)
    np.random.seed(3103)
    random.seed(3103)
    device = torch.device("cuda")
    # Imports are intentionally delayed: they load the official custom CUDA ops.
    from models.autoencoder import AutoEncoder

    if args.phase == "evaluate":
        if not args.checkpoint or not args.report_json:
            parser.error("Evaluation requires --checkpoint and --report-json")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"]["upstream_revision"] != UPSTREAM_REV:
            raise ValueError("Checkpoint is not from pinned official D-PCC")
        scale = RadarScale.from_json(saved["identity"]["radar_scale"])
        config = ModelConfig(**saved["identity"]["model"])
        model = AutoEncoder(config.upstream_args()).to(device)
        model.load_state_dict(saved["model"])
        frames = FilteredFrames(args.filtered_root, args.split, args.val_limit)
        if (args.split == "val" and args.val_limit is None
                and frames.sha256() != saved["identity"]["val_manifest_sha256"]):
            raise ValueError("Validation cache differs from checkpoint")
        result = {"checkpoint_epoch": saved["epoch"], "split": args.split,
                  "filtered_manifest_sha256": frames.sha256(),
                  "metrics": _summarize_frames(model, frames, scale, device,
                                               f"Official D-PCC {args.split}")}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.report_json, result)
        print(json.dumps(result, indent=2), flush=True)
        return

    train = FilteredFrames(args.filtered_root, "train", args.train_limit)
    val = FilteredFrames(args.filtered_root, "val", args.val_limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    last_path = args.output_root / "last.ckpt"
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        scale = RadarScale.from_json(saved["identity"]["radar_scale"])
    else:
        if (args.output_root / "history.jsonl").exists():
            raise RuntimeError("Existing history without checkpoint: choose a new output directory")
        scale = RadarScale.fit(train)
    config = ModelConfig()
    identity = _identity(train, val, scale, config)
    if last_path.is_file() and saved["identity"] != identity:
        raise ValueError("Run configuration or filtered cache changed; cannot resume")
    _write_json(args.output_root / "run_identity.json", identity)
    model = AutoEncoder(config.upstream_args()).to(device)
    parameters = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=config.aux_lr)
    scheduler = StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.gamma)
    start, best_iou, best_chamfer = 1, -1.0, float("inf")
    if last_path.is_file():
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        aux_optimizer.load_state_dict(saved["aux_optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
        best_iou = saved["best_voxel_iou"]
        best_chamfer = saved["best_chamfer_squared_m2"]
        print(f"Resuming official D-PCC from epoch {start}", flush=True)
    else:
        print(f"Starting official D-PCC from epoch 1; {len(train)} train, {len(val)} val frames",
              flush=True)
    for number in range(start, args.epochs + 1):
        model.train()
        training = {key: 0.0 for key in LOSS_KEYS}
        training["total_loss"] = training["aux_loss"] = 0.0
        indices = torch.randperm(len(train)).tolist()
        for i in tqdm(indices, desc=f"Official D-PCC train {number:03d}",
                      unit="frame", dynamic_ncols=True):
            inp = scale.encode(train.load(i)).T.contiguous()[None].to(device)
            reset_encoder_neighbors(model)
            _, _, loss, terms, _ = model(inp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            aux_loss = model.feats_eblock.loss() + model.xyzs_eblock.loss()
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss.backward()
            aux_optimizer.step()
            training["total_loss"] += float(loss.detach())
            training["aux_loss"] += float(aux_loss.detach())
            for key in LOSS_KEYS:
                training[key] += terms[key]
        scheduler.step()
        training = {key: value / len(train) for key, value in training.items()}
        validation = _summarize_frames(model, val, scale, device,
                                       f"Official D-PCC val {number:03d}")
        row = {"epoch": number, "train": training, "val": validation}
        with (args.output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        new_best_iou = validation["voxel_iou"] > best_iou
        new_best_chamfer = (validation["chamfer_squared_m2"] is not None
                            and validation["chamfer_squared_m2"] < best_chamfer)
        best_iou = max(best_iou, validation["voxel_iou"])
        if new_best_chamfer:
            best_chamfer = validation["chamfer_squared_m2"]
        saved = {"epoch": number, "identity": identity, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "aux_optimizer": aux_optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_voxel_iou": best_iou,
                 "best_chamfer_squared_m2": best_chamfer}
        _atomic_save(last_path, saved)
        if new_best_iou:
            _atomic_save(args.output_root / "best_voxel_iou.ckpt", saved)
        if new_best_chamfer:
            _atomic_save(args.output_root / "best_chamfer.ckpt", saved)
        print(f"Epoch {number:03d}: train_loss={training['total_loss']:.6f} "
              f"val_loss={validation['total_loss']:.6f} "
              f"val_voxel_IoU={validation['voxel_iou']:.2%} "
              f"val_point_F1@0.2m={validation['point_f1_0p2m']:.2%} "
              f"val_Chamfer={validation['chamfer_squared_m2']:.6f}m2 "
              f"best_IoU={best_iou:.2%}", flush=True)


if __name__ == "__main__":
    main()
