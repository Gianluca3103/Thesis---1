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
from models.radar_lidar_stage2.fault_data import (
    FAULT_PLAN, FAULT_POLICY_VERSION, SevereFogCache, load_faulty, severe_fault,
)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    MAX_POINTS_PER_VOXEL, MODEL_REVISION, VERSION,
    SparseResidualStage2, Stage2Conditions,
    cosine_alpha_bars,
)
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import load_stage1, sha256, stage1_frame


def _rows(rows: list[dict]) -> list[dict]:
    return rows


def _combine_conditions(items: list[Stage2Conditions]) -> Stage2Conditions:
    """Join independent frames without allowing sparse kernels to cross frames."""
    if len(items) == 1:
        return items[0]
    levels = []
    for level in range(5):
        frames = []
        for batch_id, item in enumerate(items):
            coordinates = item.hierarchy[level].clone()
            coordinates[:, 0] = batch_id
            frames.append(coordinates)
        levels.append(torch.cat(frames))
    if any(item.cached_targets is None for item in items):
        raise ValueError("Batched Stage II requires fixed cached targets")
    targets = tuple(torch.cat([item.cached_targets[index] for item in items])
                    for index in range(3))
    coverage = sum(item.cached_targets[3] for item in items) / len(items)
    return Stage2Conditions(tuple(levels),
                            torch.cat([item.fine_probabilities for item in items]),
                            torch.cat([item.latent_probabilities for item in items]),
                            torch.cat([item.radar_features for item in items]),
                            (*targets, coverage))


def _combine_points(clouds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    points = torch.cat(clouds)
    if len(clouds) == 1:
        return points, None
    batch_ids = torch.cat([torch.full((len(cloud),), index, dtype=torch.long,
                                      device=cloud.device)
                           for index, cloud in enumerate(clouds)])
    return points, batch_ids


class _CachedRows:
    """Decompress Stage-II geometry in DataLoader workers, before CUDA transfer."""

    def __init__(self, dataset, cache: Stage2FixedCache):
        self.dataset = dataset
        self.cache = cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        row = self.dataset[index]
        output, targets = self.cache.load(row["frame_id"], "cpu")
        row["stage2_output"] = output
        row["stage2_targets"] = targets
        return row


def _atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _epoch(model, stage1, fixed_cache, fog_cache, loader, optimizer, args, epoch, split,
           alpha_bars, radius_m: float) -> dict:
    training = optimizer is not None
    model.train(training)
    if stage1 is not None:
        stage1.eval()
    totals: dict[str, float] = {}
    fault_counts = {name: 0 for name, _ in FAULT_PLAN}
    processed = skipped = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for rows in tqdm(loader, desc=f"Stage II {args.phase} {split} {epoch:03d}",
                         unit="batch", dynamic_ncols=True):
            active_rows, items, clean_clouds = [], [], []
            for row in rows:
                clean = row["clean_lidar"][:, :3].to(args.device, non_blocking=True)
                if fixed_cache is not None:
                    output = {key: value.to(args.device, non_blocking=True)
                              for key, value in row["stage2_output"].items()}
                    stored = row["stage2_targets"]
                    targets = tuple(value.to(args.device, non_blocking=True)
                                    for value in stored[:3]) + (stored[3],)
                else:
                    radar = row["radar"].to(args.device, non_blocking=True)
                    with torch.no_grad():
                        output = stage1_frame(stage1, radar, radius_m)
                    targets = None
                item = model.conditions(output, args.gate_threshold)
                item.cached_targets = targets
                if not len(item.hierarchy[0]):
                    skipped += 1
                    continue
                active_rows.append(row)
                items.append(item)
                clean_clouds.append(clean)
            if not active_rows:
                continue
            conditions = _combine_conditions(items)
            clean, clean_batch_ids = _combine_points(clean_clouds)
            batch_size = len(active_rows)
            if args.phase == "lpcr":
                latent, _ = model.encode_lidar(clean, conditions, clean_batch_ids)
                result = model.lpcr_loss(latent, conditions, clean,
                                         batch_size=batch_size,
                                         clean_batch_ids=clean_batch_ids)
                values = {key: result[key] for key in
                          ("loss", "occupancy", "count", "offset", "iou_known",
                           "iou_full", "gate_coverage", "known_sites",
                           "positive_sites")}
            else:
                faulty_clouds = []
                for row in active_rows:
                    if args.faulty_root:
                        faulty = load_faulty(args.faulty_root, split,
                                             row["frame_id"])
                        fault_name = "real"
                    else:
                        faulty, fault_name = severe_fault(
                            row["clean_lidar"], row["frame_id"],
                            epoch if training else 0, fog_cache,
                            seed=args.fault_seed)
                    fault_counts[fault_name] = fault_counts.get(fault_name, 0) + 1
                    faulty_clouds.append(faulty.to(args.device,
                                                   non_blocking=True))
                faulty, faulty_batch_ids = _combine_points(faulty_clouds)
                generator = None
                if not training:
                    frame_key = ":".join(row["frame_id"] for row in active_rows)
                    digest = hashlib.sha256(frame_key.encode()).digest()
                    generator = torch.Generator(device=args.device)
                    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
                result = model.diffusion_loss(faulty, clean, conditions, alpha_bars,
                                              residual_weight=args.residual_weight,
                                              reconstruction_weight=args.reconstruction_weight,
                                              generator=generator,
                                              faulty_batch_ids=faulty_batch_ids,
                                              clean_batch_ids=clean_batch_ids,
                                              batch_size=batch_size)
                reconstruction = result["reconstruction"]
                values = {"loss": result["loss"], "noise": result["noise"],
                          "residual": result["residual"],
                          "reconstruction": reconstruction["loss"],
                          "occupancy": reconstruction["occupancy"],
                          "count": reconstruction["count"],
                          "offset": reconstruction["offset"],
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
                totals[key] = totals.get(key, 0.0) + batch_size * float(
                    value.detach() if isinstance(value, torch.Tensor) else value)
            totals["allowed_sites"] = totals.get("allowed_sites", 0.0) + len(
                conditions.hierarchy[0])
            processed += batch_size
    if not processed:
        raise RuntimeError("Stage I selected zero sites across this split")
    return {key: value / processed for key, value in totals.items()} | {
        "frames": processed, "skipped_empty_gate": skipped,
        "fault_counts": fault_counts if args.phase == "diffusion" else {}}


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--faulty-root", type=Path)
    parser.add_argument("--fault-cache-root", type=Path,
                        help="Required for synthetic severe faults; contains severity-5 fog")
    parser.add_argument("--fault-seed", type=int, default=3103)
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
            args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0 or
            args.batch_size > 1 and args.stage2_cache_root is None or
            args.phase == "diffusion" and args.lpcr_checkpoint is None or
            (args.phase == "diffusion" and args.faulty_root is None
             and args.fault_cache_root is None) or
            args.phase == "lpcr" and args.lpcr_checkpoint is not None or
            args.train_limit is not None and args.train_limit < 1 or
            args.val_limit is not None and args.val_limit < 1):
        parser.error("Stage II requires CUDA, valid limits, an LPCR checkpoint, and a severe fog cache for synthetic diffusion faults")
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
    identity = {"version": VERSION, "model_revision": MODEL_REVISION,
                "phase": args.phase, "grid": asdict(GRID),
                "max_points_per_voxel": MAX_POINTS_PER_VOXEL,
                "count_weight": 0.1, "offset_weight": 0.1,
                "stage1_sha256": sha256(args.stage1_checkpoint),
                "radar_variant": args.radar_variant, "gate_threshold": args.gate_threshold,
                "radius_m": radius_m,
                "lpcr_sha256": (sha256(args.lpcr_checkpoint)
                                if args.lpcr_checkpoint else None),
                "faulty_root": str(args.faulty_root.resolve()) if args.faulty_root else None,
                "fault_seed": args.fault_seed,
                "fault_policy_version": FAULT_POLICY_VERSION,
                "fault_plan": FAULT_PLAN,
                "fault_cache_root": (str(args.fault_cache_root.resolve())
                                     if args.fault_cache_root else None),
                "run_seed": run_seed,
                "initial_lr": args.lr,
                "residual_weight": args.residual_weight,
                "reconstruction_weight": args.reconstruction_weight}
    if args.phase == "diffusion":
        saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu", weights_only=False)
        if (saved_lpcr.get("version") != VERSION
                or saved_lpcr.get("model_revision") != MODEL_REVISION
                or saved_lpcr.get("phase") != "lpcr"
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
    fog_caches = {}
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
        fog_caches[split] = (SevereFogCache(
            args.fault_cache_root, split,
            [frame.frame_id for frame in dataset.frames], seed=args.fault_seed,
            check_complete=getattr(args, f"{split}_limit") is None)
            if args.phase == "diffusion" and args.faulty_root is None else None)
    datasets = {split: (Subset(dataset, range(min(limit, len(dataset))))
                        if limit is not None else dataset)
                for split, dataset in full_datasets.items()
                for limit in (getattr(args, f"{split}_limit"),)}
    datasets = {split: (_CachedRows(dataset, fixed_caches[split])
                        if fixed_caches[split] is not None else dataset)
                for split, dataset in datasets.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=(split == "train"),
                                 num_workers=args.num_workers, collate_fn=_rows,
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
        train = _epoch(model, stage1, fixed_caches["train"], fog_caches["train"], loaders["train"],
                       optimizer, args, epoch,
                       "train", alpha_bars, radius_m)
        val = _epoch(model, stage1, fixed_caches["val"], fog_caches["val"], loaders["val"],
                     None, args, epoch,
                     "val", alpha_bars, radius_m)
        if scheduler is not None:
            scheduler.step()
        improved = val["loss"] < best
        if improved:
            best = val["loss"]
        record = {"epoch": epoch, "batch_size": args.batch_size,
                  "num_workers": args.num_workers, "train": train, "val": val,
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
              f"count CE={val['count']:.5f}, "
              f"offset L1={val['offset']:.5f}, "
              f"gate coverage={val['gate_coverage']:.4f}, "
              f"train faults={train['fault_counts']}", flush=True)


if __name__ == "__main__":
    main()
