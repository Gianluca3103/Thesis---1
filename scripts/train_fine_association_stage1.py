"""Fine radar-voxel to LiDAR-voxel association, initialized from current Stage I."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.correspondence import local_neighbors_xyz
from models.radar_lidar_stage1.sparse import voxelize
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.fine_association import (
    FineAssociationStudent, VERSION, checkpoint_payload)
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    scan_overlap_counts_observed, scan_overlap_rates)
from models.radar_lidar_crossmodal_v1.pipeline_data import (
    collate_stage1, paired_dataset)
from models.radar_lidar_crossmodal_v1.surface_student import SurfaceStudentConfig
from models.radar_lidar_crossmodal_v1.surface_student_loss import (
    clean_surface_sites, surface_student_objective)
from models.radar_lidar_crossmodal_v1.train_teacher import _hash, _move


def association_loss(output, clean, clean_valid, radius_m: float):
    """Nearest clean voxel centers within the trained radius supervise slots.

    Correspondences exist during training only. Inference emits learned
    radar-only pointers and cannot inspect clean LiDAR coordinates.
    """
    proposals = output.surface
    clean_sites = clean_surface_sites(clean, clean_valid)
    if not len(clean_sites.coords) or not len(proposals.anchor_xyz):
        return proposals.xyz.sum() * 0, 0
    nearby = local_neighbors_xyz(proposals.anchor_xyz, proposals.anchor_batch,
                                 clean_sites, GRID, radius_m,
                                 proposals.xyz.shape[1])
    valid = nearby.valid
    if not bool(valid.any()):
        return proposals.xyz.sum() * 0, 0
    target = clean_sites.centers_xyz(GRID)[nearby.indices.clamp_min(0)]
    return F.smooth_l1_loss(proposals.xyz[valid], target[valid], beta=0.2), int(valid.sum())


def anchor_support(coords, anchors, anchor_batch, radius_m: float):
    """Use active fine radar voxel centers for the validation denominator."""
    centers = clean_surface_sites_from_coords(coords).detach().cpu().numpy()
    frame = coords[:, 0].detach().cpu().numpy()
    xyz = anchors.detach().cpu().numpy()
    radar_frame = anchor_batch.detach().cpu().numpy()
    supported = np.zeros(len(coords), dtype=np.bool_)
    for batch_id in np.unique(frame):
        target = np.flatnonzero(frame == batch_id)
        source = xyz[radar_frame == batch_id]
        if len(source):
            distance, _ = cKDTree(source).query(centers[target], k=1)
            supported[target] = distance <= radius_m + 1e-6
    return torch.from_numpy(supported).to(coords.device)


def clean_surface_sites_from_coords(coords):
    size = coords.new_tensor(GRID.size_xyz, dtype=torch.float32)
    minimum = coords.new_tensor(GRID.minimum_xyz, dtype=torch.float32)
    return minimum + (coords[:, [3, 2, 1]].float() + 0.5) * size


def run_epoch(model, loader, device, *, optimizer=None, association_weight=0.2,
              threshold=0.25):
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in ("total", "pointer", "association", "bce", "dice")}
    counts = {name: 0 for name in ("frames", "batches", "anchors", "slots",
                                  "matched_slots", "candidates", "selected",
                                  "predicted", "observed", "tp_exact", "tp_at_0_2m")}
    for raw in tqdm(loader, desc="Fine association train" if training else
                    "Fine association val", unit="batch", dynamic_ncols=True):
        batch = _move(raw, device)
        with torch.set_grad_enabled(training):
            output = model(batch["radar"], batch["radar_valid"])
            pointer, parts, field, _ = surface_student_objective(
                output, batch["clean_lidar"], batch["clean_lidar_valid"], model.config)
            association, matched = association_loss(
                output, batch["clean_lidar"], batch["clean_lidar_valid"],
                model.config.target_radius_m)
            loss = pointer + association_weight * association
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        counts["frames"] += len(batch["frame_id"])
        counts["batches"] += 1
        counts["anchors"] += len(output.surface.anchor_xyz)
        counts["slots"] += output.surface.logits.numel()
        counts["matched_slots"] += matched
        counts["candidates"] += len(field.coords)
        counts["selected"] += int((field.confidence >= threshold).sum())
        for name, value in (("total", loss), ("pointer", pointer),
                            ("association", association), ("bce", parts["bce"]),
                            ("dice", parts["dice"])):
            sums[name] += float(value.detach())
        if not training:
            selected = field.coords[field.confidence >= threshold]
            clean_coords, _, _, _ = voxelize(
                batch["clean_lidar"], batch["clean_lidar_valid"], GRID)
            support = anchor_support(clean_coords, output.surface.anchor_xyz,
                                     output.surface.anchor_batch,
                                     model.config.target_radius_m)
            overlap = scan_overlap_counts_observed(selected, clean_coords[support])
            for name in ("predicted", "observed", "tp_exact", "tp_at_0_2m"):
                counts[name] += overlap[name]
    report = {"loss_" + key: value / max(counts["batches"], 1)
              for key, value in sums.items()}
    report.update(counts)
    if not training:
        report.update(scan_overlap_rates(counts))
        report["clean_voxel_recall_exact"] = (
            counts["tp_exact"] / counts["observed"] if counts["observed"] else None)
    return report


def save(path, model, optimizer, scheduler, source, source_path, source_hash,
         epoch, validation, association_weight, warm_start_hash, *, resumable=True):
    payload = checkpoint_payload(model, source=source, source_path=source_path,
                                 epoch=epoch, validation=validation)
    payload["source_student_sha256"] = source_hash
    payload["association_weight"] = association_weight
    payload["initial_eight_slot_sha256"] = warm_start_hash
    if resumable:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--source-student", type=Path, required=True)
    parser.add_argument("--initialize-from-association", type=Path,
                        help="Optional trained eight-slot checkpoint for a 32-slot warm start")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/crossmodal_fine_association_3m32.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--feature-lr", type=float, default=2e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--association-weight", type=float, default=0.2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_workers + 1,
           args.head_lr, args.feature_lr, args.encoder_lr) <= 0:
        parser.error("Epochs, batch size and learning rates must be positive")
    if args.association_weight < 0:
        parser.error("Association weight cannot be negative")
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_config["context_radii_m"] = tuple(raw_config["context_radii_m"])
    config = SurfaceStudentConfig(**raw_config)
    source = torch.load(args.source_student, map_location="cpu", weights_only=False)
    if source.get("version") != "crossmodal_student_v5_multiscale_refined_raw64":
        parser.error("Source checkpoint must be the trained cross-modal radar student")
    variant = source["radar_variant"]
    source_hash = _hash(args.source_student)
    model = FineAssociationStudent(config)
    model.initialize_from_student(source)
    warm_start_hash = None
    if args.initialize_from_association and not args.resume:
        prior = torch.load(args.initialize_from_association, map_location="cpu",
                           weights_only=False)
        model.initialize_from_eight_slot_checkpoint(prior, source_hash)
        warm_start_hash = _hash(args.initialize_from_association)
        del prior
        print(f"Initialized first eight slots from {args.initialize_from_association}",
              flush=True)
    source_meta = {"radar_variant": variant, "lidar_fov": source.get("lidar_fov")}
    del source  # Source checkpoint may also contain large optimizer states.
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": model.radar_encoder.parameters(), "lr": args.encoder_lr},
        {"params": list(model.shared.parameters()) +
                   list(model.feature_head.parameters()) +
                   list(model.feature_norm.parameters()), "lr": args.feature_lr},
        {"params": model.surface_head.parameters(), "lr": args.head_lr},
    ], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train = paired_dataset(args.vod_root, "train", variant,
                           data_cache_root=args.data_cache_root, limit=args.train_limit)
    val = paired_dataset(args.vod_root, "val", variant,
                         data_cache_root=args.data_cache_root, limit=args.val_limit)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_stage1,
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    start, best = 1, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != VERSION or saved["config"] != asdict(config)
                or saved["source_student_sha256"] != source_hash
                or saved.get("association_weight") != args.association_weight):
            raise ValueError("Association resume checkpoint has a different source or config")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        warm_start_hash = saved.get("initial_eight_slot_sha256")
        start = saved["epoch"] + 1
    elif history.exists():
        parser.error("Existing metrics require --resume or a new output directory")
    if (args.output_root / "best.ckpt").exists():
        prior = torch.load(args.output_root / "best.ckpt", map_location="cpu",
                           weights_only=False)
        best = prior["validation"].get("iou_exact") or -1.0
    for epoch in range(start, args.epochs + 1):
        train.set_epoch(epoch)
        training = run_epoch(model, train_loader, args.device,
                             optimizer=optimizer,
                             association_weight=args.association_weight,
                             threshold=config.threshold)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, args.device,
                                   association_weight=args.association_weight,
                                   threshold=config.threshold)
        scheduler.step()
        record = {"epoch": epoch, "train": training, "val": validation,
                  "threshold": config.threshold,
                  "radius_m": config.target_radius_m,
                  "slots_per_radar_voxel": config.proposals_per_anchor}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        save(args.output_root / "last.ckpt", model, optimizer, scheduler,
             source_meta, args.source_student, source_hash, epoch, validation,
             args.association_weight, warm_start_hash)
        score = validation["iou_exact"] or 0.0
        if score > best:
            best = score
            save(args.output_root / "best.ckpt", model, optimizer, scheduler,
                 source_meta, args.source_student, source_hash, epoch, validation,
                 args.association_weight, warm_start_hash, resumable=False)
        print(f"Epoch {epoch:03d}: train {training['loss_total']:.4f}, "
              f"val {validation['loss_total']:.4f}, IoU exact "
              f"{validation['iou_exact']}, recall "
              f"{validation['clean_voxel_recall_exact']}, selected "
              f"{validation['selected']}", flush=True)


if __name__ == "__main__":
    main()
