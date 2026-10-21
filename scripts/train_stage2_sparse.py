"""Train Stage-II sparse LPCR, then radar/faulty-conditioned latent diffusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import load_faulty, synthetic_fault
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    VERSION, SparseResidualStage2, cosine_alpha_bars,
)
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import load_stage1, sha256, stage1_frame


def _one(rows: list[dict]) -> dict:
    if len(rows) != 1:
        raise ValueError("Sparse Stage II currently trains one frame per step")
    return rows[0]


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _epoch(model, stage1, fixed_cache, loader, optimizer, args, epoch, split,
           alpha_bars, radius_m: float) -> dict:
    training = optimizer is not None
    model.train(training)
    if stage1 is not None:
        stage1.eval()
    totals: dict[str, float] = {}
    processed = skipped = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for row in tqdm(loader, desc=f"Stage II {args.phase} {split} {epoch:03d}",
                        unit="frame", dynamic_ncols=True):
            clean = row["clean_lidar"][:, :3].to(args.device, non_blocking=True)
            if fixed_cache is not None:
                output, targets = fixed_cache.load(row["frame_id"], args.device)
            else:
                radar = row["radar"].to(args.device, non_blocking=True)
                with torch.no_grad():
                    output = stage1_frame(stage1, radar, radius_m)
                targets = None
            conditions = model.conditions(output, args.gate_threshold)
            conditions.cached_targets = targets
            if not len(conditions.hierarchy[0]):
                skipped += 1
                continue
            if args.phase == "lpcr":
                latent, _ = model.encode_lidar(clean, conditions)
                result = model.lpcr_loss(latent, conditions, clean)
                values = {key: result[key] for key in
                          ("loss", "occupancy", "offset", "iou_known",
                           "iou_full", "gate_coverage", "known_sites",
                           "positive_sites")}
            else:
                faulty = (load_faulty(args.faulty_root, split, row["frame_id"])
                          if args.faulty_root else synthetic_fault(
                              row["clean_lidar"], row["frame_id"],
                              epoch if training else 0, seed=args.fault_seed,
                              dropout=args.fault_dropout,
                              sector_deg=args.fault_sector_deg))
                faulty = faulty[:, :3].to(args.device, non_blocking=True)
                generator = None
                if not training:
                    digest = hashlib.sha256(row["frame_id"].encode()).digest()
                    generator = torch.Generator(device=args.device)
                    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
                result = model.diffusion_loss(faulty, clean, conditions, alpha_bars,
                                              residual_weight=args.residual_weight,
                                              reconstruction_weight=args.reconstruction_weight,
                                              generator=generator)
                reconstruction = result["reconstruction"]
                values = {"loss": result["loss"], "noise": result["noise"],
                          "residual": result["residual"],
                          "reconstruction": reconstruction["loss"],
                          "iou_known": reconstruction["iou_known"],
                          "iou_full": reconstruction["iou_full"],
                          "gate_coverage": reconstruction["gate_coverage"],
                          "known_sites": reconstruction["known_sites"],
                          "positive_sites": reconstruction["positive_sites"]}
            if training:
                optimizer.zero_grad(set_to_none=True)
                values["loss"].backward()
                trainable = [parameter for parameter in model.parameters()
                             if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable,
                                               1.0 if args.phase == "lpcr" else 5.0)
                optimizer.step()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()
                    if isinstance(value, torch.Tensor) else value)
            totals["allowed_sites"] = totals.get("allowed_sites", 0.0) + len(
                conditions.hierarchy[0])
            processed += 1
    if not processed:
        raise RuntimeError("Stage I selected zero sites across this split")
    return {key: value / processed for key, value in totals.items()} | {
        "frames": processed, "skipped_empty_gate": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lpcr", "diffusion"), required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--stage2-cache-root", type=Path,
                        help="Compressed frozen Stage-I and LiDAR-label cache")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path,
                        help="Required for diffusion; best LPCR checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--gate-threshold", type=float, default=0.25)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--faulty-root", type=Path)
    parser.add_argument("--fault-seed", type=int, default=3103)
    parser.add_argument("--fault-dropout", type=float, default=0.25)
    parser.add_argument("--fault-sector-deg", type=float, default=35.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 30 if args.phase == "lpcr" else 50
    if args.lr is None:
        args.lr = 1e-3 if args.phase == "lpcr" else 2e-4
    if (args.device != "cuda" or not torch.cuda.is_available() or
            not 0 < args.gate_threshold < 1 or args.epochs < 1 or
            args.num_workers < 0 or args.lr <= 0 or
            args.phase == "diffusion" and args.lpcr_checkpoint is None or
            args.phase == "lpcr" and args.lpcr_checkpoint is not None or
            args.train_limit is not None and args.train_limit < 1 or
            args.val_limit is not None and args.val_limit < 1):
        parser.error("Stage-II arguments require CUDA, valid limits, and an LPCR checkpoint for diffusion")
    run_seed = 3407 if args.phase == "lpcr" else 3103
    torch.manual_seed(run_seed)
    if args.stage2_cache_root:
        stage1_saved = torch.load(args.stage1_checkpoint, map_location="cpu",
                                  weights_only=False)
        if stage1_saved.get("radar_variant") != args.radar_variant:
            raise ValueError("Radar variant differs from Stage-I checkpoint")
        stage1 = None
    else:
        stage1, stage1_saved = load_stage1(args.stage1_checkpoint, args.device,
                                            args.radar_variant)
    radius_m = float(stage1_saved.get("radius_m", 3.2))
    model = SparseResidualStage2().to(args.device)
    identity = {"version": VERSION, "phase": args.phase, "grid": asdict(GRID),
                "stage1_sha256": sha256(args.stage1_checkpoint),
                "radar_variant": args.radar_variant, "gate_threshold": args.gate_threshold,
                "radius_m": radius_m,
                "lpcr_sha256": (sha256(args.lpcr_checkpoint)
                                if args.lpcr_checkpoint else None),
                "faulty_root": str(args.faulty_root.resolve()) if args.faulty_root else None,
                "fault_seed": args.fault_seed, "fault_dropout": args.fault_dropout,
                "fault_sector_deg": args.fault_sector_deg,
                "run_seed": run_seed,
                "initial_lr": args.lr,
                "residual_weight": args.residual_weight,
                "reconstruction_weight": args.reconstruction_weight}
    if args.phase == "diffusion":
        saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu", weights_only=False)
        if (saved_lpcr.get("version") != VERSION or saved_lpcr.get("phase") != "lpcr"
                or saved_lpcr.get("stage1_sha256") != identity["stage1_sha256"]
                or saved_lpcr.get("gate_threshold") != args.gate_threshold):
            raise ValueError("LPCR checkpoint differs from this Stage-I checkpoint or gate")
        model.load_state_dict(saved_lpcr["model"], strict=True)
        model.lidar_encoder.requires_grad_(False)
        model.lpcr.requires_grad_(False)
        trainable = list(model.radar_adapter.parameters()) + list(model.diffusion.parameters())
    else:
        model.radar_adapter.requires_grad_(False)
        model.diffusion.requires_grad_(False)
        trainable = list(model.lidar_encoder.parameters()) + list(model.lpcr.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr,
        betas=(0.9, 0.99) if args.phase == "lpcr" else (0.9, 0.999),
        weight_decay=0.0 if args.phase == "lpcr" else 0.01)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
                 if args.phase == "lpcr" else None)
    alpha_bars = cosine_alpha_bars(device=args.device)
    full_datasets = {split: paired_dataset(args.vod_root, split, args.radar_variant,
                                           include_clean=True,
                                           data_cache_root=args.data_cache_root)
                     for split in ("train", "val")}
    fixed_caches = {}
    for split, dataset in full_datasets.items():
        if args.stage2_cache_root:
            expected = cache_identity(
                args.stage1_checkpoint, args.vod_root, args.radar_variant,
                radius_m, args.gate_threshold, split,
                [frame.frame_id for frame in dataset.frames])
            fixed_caches[split] = Stage2FixedCache(
                args.stage2_cache_root / split, expected)
            fixed_caches[split].check_complete()
        else:
            fixed_caches[split] = None
    datasets = {split: (Subset(dataset, range(min(limit, len(dataset))))
                        if limit is not None else dataset)
                for split, dataset in full_datasets.items()
                for limit in (getattr(args, f"{split}_limit"),)}
    loaders = {split: DataLoader(dataset, batch_size=1,
                                 shuffle=(split == "train"),
                                 num_workers=args.num_workers, collate_fn=_one,
                                 pin_memory=True, persistent_workers=args.num_workers > 0)
               for split, dataset in datasets.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    if history.exists() and not args.resume:
        parser.error("Output metrics exist; use --resume or a new output directory")
    start, best = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume checkpoint configuration differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_val_loss"])
    for epoch in range(start, args.epochs + 1):
        train = _epoch(model, stage1, fixed_caches["train"], loaders["train"],
                       optimizer, args, epoch,
                       "train", alpha_bars, radius_m)
        val = _epoch(model, stage1, fixed_caches["val"], loaders["val"],
                     None, args, epoch,
                     "val", alpha_bars, radius_m)
        if scheduler is not None:
            scheduler.step()
        improved = val["loss"] < best
        if improved:
            best = val["loss"]
        record = {"epoch": epoch, "train": train, "val": val,
                  "best_val_loss": best}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {**identity, "epoch": epoch, "best_val_loss": best,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict() if scheduler else None,
                   "record": record}
        _atomic_save(args.output_root / "last.ckpt", payload)
        if improved:
            _atomic_save(args.output_root / "best.ckpt", payload)
        print(f"Epoch {epoch:03d}: val loss={val['loss']:.5f}, "
              f"Known/Full IoU={val['iou_known']:.4f}/{val['iou_full']:.4f}, "
              f"gate coverage={val['gate_coverage']:.4f}", flush=True)


if __name__ == "__main__":
    main()
