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
from models.radar_lidar_stage2.addition import (
    HEALTHY_RETAINED_FRACTION, MIN_CLEAN_CELLS_IN_REGION,
    REGION_STRIDE_ZYX,
)
from models.radar_lidar_stage2.fault_data import (
    FAULT_PLAN, FAULT_POLICY_VERSION, SevereFogCache, load_faulty, severe_fault,
)
from models.radar_lidar_stage2.full_lpcr import full_region_training_conditions
from models.radar_lidar_stage2.lpcr_eval import evaluate_full_region
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, MAX_POINTS_PER_VOXEL, MODEL_REVISION, VERSION,
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
                if args.phase == "lpcr":
                    item = full_region_training_conditions(
                        clean, row["frame_id"], epoch if training else 1,
                        args.lpcr_sites_per_frame)
                else:
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
                          ("loss", "occupancy", "offset", "positive_sites")}
                values["sample_iou"] = result["iou_known"]
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
                                              addition_weight=args.addition_weight,
                                              healthy_weight=args.healthy_weight,
                                              generator=generator,
                                              faulty_batch_ids=faulty_batch_ids,
                                              clean_batch_ids=clean_batch_ids,
                                              batch_size=batch_size)
                reconstruction = result["reconstruction"]
                addition = result["addition"]
                values = {"loss": result["loss"],
                          "clean_latent": result["clean_latent"],
                          "residual": result["residual"],
                          "reconstruction": reconstruction["loss"],
                          "occupancy": reconstruction["occupancy"],
                          "offset": reconstruction["offset"],
                          "iou_known": reconstruction["iou_known"],
                          "iou_full": reconstruction["iou_full"],
                          "gate_coverage": reconstruction["gate_coverage"],
                          "known_sites": reconstruction["known_sites"],
                          "positive_sites": reconstruction["positive_sites"],
                          "addition_final_loss": addition["final"],
                          "addition_healthy_loss": addition["healthy"],
                          "addition_precision": addition["addition_precision"],
                          "addition_recall": addition["addition_recall"],
                          "healthy_addition_rate": addition["healthy_addition_rate"],
                          "final_gate_iou": addition["final_gate_iou"],
                          "missing_sites": addition["missing_sites"],
                          "healthy_negative_sites": addition["healthy_negative_sites"]}
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
        raise RuntimeError("Stage II found no usable frames across this split")
    return {key: value / processed for key, value in totals.items()} | {
        "frames": processed,
        "skipped_empty_gate": skipped if args.phase == "diffusion" else None,
        "fault_counts": fault_counts if args.phase == "diffusion" else {}}


@torch.no_grad()
def _paper_metrics(model, dataset, args) -> dict:
    """Evaluate complete LiDAR ROI, independently of radar and target sites."""
    model.eval()
    count = min(args.paper_metrics_frames, len(dataset))
    indices = [round((i + 0.5) * len(dataset) / count - 0.5)
               for i in range(count)]
    frames = []
    for index in tqdm(indices, desc="LPCR full-region point metrics", unit="frame"):
        row = dataset[index]
        clean = row["clean_lidar"][:, :3].to(args.device)
        frames.append(evaluate_full_region(
            model, clean, threshold=args.paper_metrics_threshold,
            tile_size=16, halo=3))
    fields = ("chamfer_squared_l2", "hausdorff_squared_l2", "precision",
              "recall", "fscore_5cm", "predicted_points", "target_points")
    means = {field: (sum(values) / len(values) if values else None)
             for field in fields
             for values in ([frame[field] for frame in frames
                             if frame[field] is not None],)}
    return {"frames": count, "empty_predictions": sum(
        frame["predicted_points"] == 0 for frame in frames),
        "candidate_cells_per_frame": frames[0]["candidate_cells"],
        "occupancy_threshold": args.paper_metrics_threshold,
        "fscore_radius_m": 0.05, **means}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lpcr", "diffusion"), required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--stage2-cache-root", type=Path,
                        help="Compressed frozen Stage-I and LiDAR-label cache")
    parser.add_argument("--stage1-checkpoint", type=Path,
                        help="Required only for radar-conditioned diffusion")
    parser.add_argument("--lpcr-checkpoint", type=Path,
                        help="Required for diffusion; best LPCR checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--gate-threshold", type=float, default=0.25)
    parser.add_argument("--lpcr-sites-per-frame", type=int, default=8192,
                        help="Minimum 7744; LPCR uses a 16x16 core and 3-cell halo")
    parser.add_argument("--paper-metrics-every", type=int, default=5)
    parser.add_argument("--paper-metrics-frames", type=int, default=4)
    parser.add_argument("--paper-metrics-threshold", type=float, default=0.5)
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
    parser.add_argument("--addition-weight", type=float, default=1.0)
    parser.add_argument("--healthy-weight", type=float, default=1.0)
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
            args.lpcr_sites_per_frame < 7744 or
            args.paper_metrics_every < 1 or args.paper_metrics_frames < 1 or
            not 0 < args.paper_metrics_threshold < 1 or
            min(args.addition_weight, args.healthy_weight) < 0 or
            args.phase == "diffusion" and args.stage1_checkpoint is None or
            args.phase == "diffusion" and args.batch_size > 1
            and args.stage2_cache_root is None or
            args.phase == "diffusion" and args.lpcr_checkpoint is None or
            (args.phase == "diffusion" and args.faulty_root is None
             and args.fault_cache_root is None) or
            args.phase == "lpcr" and args.lpcr_checkpoint is not None or
            args.train_limit is not None and args.train_limit < 1 or
            args.val_limit is not None and args.val_limit < 1):
        parser.error("Stage II requires CUDA and valid phase-specific paths and limits")
    run_seed = 3407 if args.phase == "lpcr" else 3103
    torch.manual_seed(run_seed)
    if args.phase == "lpcr":
        stage1 = None
        radius_m = None
    elif args.stage2_cache_root:
        stage1_saved = torch.load(args.stage1_checkpoint, map_location="cpu",
                                  weights_only=False)
        if stage1_saved.get("radar_variant") != args.radar_variant:
            raise ValueError("Radar variant differs from Stage-I checkpoint")
        stage1 = None
        radius_m = float(stage1_saved.get("radius_m", 3.2))
    else:
        stage1, stage1_saved = load_stage1(args.stage1_checkpoint, args.device,
                                            args.radar_variant)
        radius_m = float(stage1_saved.get("radius_m", 3.2))
    model = SparseResidualStage2().to(args.device)
    identity = {"version": VERSION,
                "model_revision": (LPCR_REVISION if args.phase == "lpcr"
                                   else MODEL_REVISION),
                "phase": args.phase, "grid": asdict(GRID),
                "max_points_per_voxel": MAX_POINTS_PER_VOXEL,
                "occupancy_weight": 0.9, "offset_weight": 0.1,
                "diffusion_objective": (
                    "clean_latent_mse_residual_l1_lpcr_addition_healthy"
                    if args.phase == "diffusion" else
                    "clean_latent_mse_residual_l1_lpcr"),
                "stage1_sha256": (sha256(args.stage1_checkpoint)
                                  if args.phase == "diffusion" else None),
                "radar_variant": (args.radar_variant if args.phase == "diffusion" else None),
                "gate_threshold": (args.gate_threshold if args.phase == "diffusion" else None),
                "radius_m": radius_m,
                "lpcr_sites_per_frame": (args.lpcr_sites_per_frame
                                          if args.phase == "lpcr" else None),
                "lpcr_full_region": True,
                "lpcr_query_patch": "16x16_xy_core_3cell_halo_all_z",
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
        identity.update({"addition_weight": args.addition_weight,
                         "healthy_weight": args.healthy_weight,
                         "addition_region_stride_zyx": REGION_STRIDE_ZYX,
                         "healthy_min_clean_cells": MIN_CLEAN_CELLS_IN_REGION,
                         "healthy_retained_fraction": HEALTHY_RETAINED_FRACTION})
    if args.phase == "diffusion":
        saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu", weights_only=False)
        if (saved_lpcr.get("version") != VERSION
                or saved_lpcr.get("model_revision") != LPCR_REVISION
                or saved_lpcr.get("phase") != "lpcr"
                or not saved_lpcr.get("lpcr_full_region")):
            raise ValueError("Diffusion requires a full-region LPCR checkpoint")
        missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
        if unexpected or any(not key.startswith("addition_head.") for key in missing):
            raise ValueError("LPCR checkpoint differs from the Stage-II architecture")
        model.lidar_encoder.requires_grad_(False)
        model.lpcr.requires_grad_(False)
        trainable = (list(model.radar_adapter.parameters())
                     + list(model.diffusion.parameters())
                     + list(model.addition_head.parameters()))
    else:
        model.radar_adapter.requires_grad_(False)
        model.diffusion.requires_grad_(False)
        model.addition_head.requires_grad_(False)
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
        if args.phase == "diffusion" and args.stage2_cache_root:
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
        paper = (_paper_metrics(model, datasets["val"], args)
                 if args.phase == "lpcr" and epoch % args.paper_metrics_every == 0
                 else None)
        if scheduler is not None:
            scheduler.step()
        improved = val["loss"] < best
        if improved:
            best = val["loss"]
        record = {"epoch": epoch, "batch_size": args.batch_size,
                  "num_workers": args.num_workers, "train": train, "val": val,
                  "best_val_loss": best, "point_cloud_metrics": paper}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {**identity, "epoch": epoch, "best_val_loss": best,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict() if scheduler else None,
                   "record": record}
        _atomic_save(args.output_root / "last.ckpt", payload)
        if improved:
            _atomic_save(args.output_root / "best.ckpt", payload)
        diffusion_terms = (f"clean latent MSE={val['clean_latent']:.5f}, "
                           f"residual L1={val['residual']:.5f}, "
                           if args.phase == "diffusion" else "")
        if args.phase == "lpcr":
            print(f"Epoch {epoch:03d}: full-region patch val loss={val['loss']:.5f}, "
                  f"patch IoU={val['sample_iou']:.4f}, "
                  f"occupancy BCE={val['occupancy']:.5f}, "
                  f"offset L1={val['offset']:.5f}", flush=True)
            if paper is not None:
                print(f"  Full ROI: squared-L2 CD={paper['chamfer_squared_l2']}, "
                      f"HD={paper['hausdorff_squared_l2']}, "
                      f"F-score@5cm={paper['fscore_5cm']}, "
                      f"mean predicted/target points="
                      f"{paper['predicted_points']:.0f}/{paper['target_points']:.0f}, "
                      f"empty predictions={paper['empty_predictions']}",
                      flush=True)
        else:
            print(f"Epoch {epoch:03d}: val loss={val['loss']:.5f}, "
                  f"Known/Full gate IoU={val['iou_known']:.4f}/{val['iou_full']:.4f}, "
                  f"{diffusion_terms}occupancy BCE={val['occupancy']:.5f}, "
                  f"offset L1={val['offset']:.5f}, "
                  f"addition/healthy loss={val['addition_final_loss']:.5f}/"
                  f"{val['addition_healthy_loss']:.5f}, "
                  f"addition precision/recall={val['addition_precision']:.4f}/"
                  f"{val['addition_recall']:.4f}, "
                  f"healthy addition rate={val['healthy_addition_rate']:.4f}, "
                  f"gate coverage={val['gate_coverage']:.4f}, "
                  f"train faults={train['fault_counts']}", flush=True)


if __name__ == "__main__":
    main()
