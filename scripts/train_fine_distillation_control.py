"""Matched Stage-I 64-slot heads for raw, distilled, and teacher features.

All three heads start with identical weights and see the same radar pyramid,
clean-LiDAR targets, optimizer, and frame order. Frozen conditioning differs:
raw radar bottleneck; radar-only source-student bottleneck; privileged
clean-LiDAR-plus-radar teacher bottleneck. The teacher is a reference ceiling,
never a radar-only inference path. Source-student features were trained with
distillation and occupancy losses, so raw versus distilled is a source-training
ablation rather than a pure distillation-loss ablation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import (
    collate_stage1, paired_dataset)
from models.radar_lidar_crossmodal_v1.fine_association import FineAssociationStudent
from models.radar_lidar_crossmodal_v1.surface_student import (
    RadarSurfaceHead, SurfaceStudentConfig, SurfaceStudentOutput)
from models.radar_lidar_crossmodal_v1.surface_student_loss import (
    surface_student_objective)
from models.radar_lidar_crossmodal_v1.train_teacher import load_teacher
from scripts.train_fine_association_stage1 import association_loss


ARMS = ("raw_radar", "distilled_radar", "teacher_conditioned")
VERSION = "fine_distillation_control_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MatchedHeads(nn.Module):
    def __init__(self, config: SurfaceStudentConfig, source: dict):
        super().__init__()
        # Reuse the exact architecture and source-student radar representation.
        self.representation = FineAssociationStudent(config)
        self.representation.initialize_from_student(source)
        self.representation.surface_head = nn.Identity()
        self.representation.requires_grad_(False)
        self.representation.eval()
        base = RadarSurfaceHead(config)
        self.heads = nn.ModuleDict({name: copy.deepcopy(base) for name in ARMS})
        self.config = config

    def train(self, mode: bool = True):
        super().train(mode)
        self.representation.eval()
        return self

    def conditions(self, radar, radar_valid, clean, clean_valid, teacher):
        with torch.no_grad():
            # Hold only the small teacher bottleneck while building the radar
            # pyramid, avoiding simultaneous LiDAR and radar encoder peaks.
            teacher_sites = teacher(radar, radar_valid, clean, clean_valid).fused
            coarse, levels = self.representation.radar_encoder.forward_pyramid(
                radar, radar_valid)
            shared = coarse.replace_features(
                self.representation.shared(coarse.features))
            distilled = shared.replace_features(
                self.representation.feature_norm(
                    self.representation.feature_head(shared.features)))
            if (teacher_sites.stride != coarse.stride or
                    teacher_sites.shape_zyx != coarse.shape_zyx or
                    not torch.equal(teacher_sites.coords, coarse.coords)):
                # Both architectures use the same radar grid, but their site
                # ordering need not be identical after independent encoders.
                from models.radar_lidar_stage1.sparse import encode_keys
                teacher_keys = encode_keys(teacher_sites.coords, coarse.shape_zyx)
                coarse_keys = encode_keys(coarse.coords, coarse.shape_zyx)
                sorted_keys, order = torch.sort(teacher_keys)
                if not torch.equal(sorted_keys, torch.sort(coarse_keys).values):
                    raise ValueError("Teacher and student radar sites differ")
                position = torch.searchsorted(sorted_keys, coarse_keys)
                teacher_features = teacher_sites.features[order[position]]
            else:
                teacher_features = teacher_sites.features
            conditions = {
                "raw_radar": coarse,
                "distilled_radar": distilled,
                "teacher_conditioned": coarse.replace_features(teacher_features),
            }
        return levels, conditions

    def forward(self, radar, radar_valid, clean, clean_valid, teacher):
        levels, conditions = self.conditions(
            radar, radar_valid, clean, clean_valid, teacher)
        return {
            name: SurfaceStudentOutput(
                conditions[name], self.heads[name]((*levels, conditions[name])),
                full_confidence_field=True)
            for name in ARMS
        }


def _run_epoch(model, teacher, loader, device, *, optimizer=None,
               association_weight: float, epoch: int):
    training = optimizer is not None
    model.train(training)
    teacher.eval()
    sums = {name: {"total": 0.0, "pointer": 0.0,
                   "association": 0.0} for name in ARMS}
    batches = 0
    for batch in tqdm(loader, desc=f"Matched heads {'train' if training else 'val'} {epoch:03d}",
                      unit="batch", dynamic_ncols=True):
        radar = batch["radar"].to(device, non_blocking=True)
        radar_valid = batch["radar_valid"].to(device, non_blocking=True)
        clean = batch["clean_lidar"].to(device, non_blocking=True)
        clean_valid = batch["clean_lidar_valid"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            levels, conditions = model.conditions(
                radar, radar_valid, clean, clean_valid, teacher)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for name in ARMS:
                output = SurfaceStudentOutput(
                    conditions[name], model.heads[name]((*levels, conditions[name])),
                    full_confidence_field=True)
                pointer, _, _, _ = surface_student_objective(
                    output, clean, clean_valid, model.config)
                association, _ = association_loss(
                    output, clean, clean_valid,
                    model.config.target_radius_m)
                loss = pointer + association_weight * association
                sums[name]["total"] += float(loss.detach())
                sums[name]["pointer"] += float(pointer.detach())
                sums[name]["association"] += float(association.detach())
                if training:
                    loss.backward()
            if training:
                torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 5.0)
                optimizer.step()
        batches += 1
    return {name: {key: value / max(batches, 1)
                   for key, value in sums[name].items()} for name in ARMS}


def _save(path: Path, model, optimizer, scheduler, args, source_sha,
          teacher_sha, epoch, train_metrics, val_metrics):
    payload = {
        "version": VERSION, "epoch": epoch,
        "config": asdict(model.config),
        "heads": model.heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "source_student": str(args.source_student.resolve()),
        "source_sha256": source_sha,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_sha256": teacher_sha,
        "association_weight": args.association_weight,
        "seed": args.seed,
        "train": train_metrics, "val": val_metrics,
    }
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--source-student", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Defaults to the path recorded in the source student")
    parser.add_argument("--config", type=Path, default=Path(
        "configs/crossmodal_fine_association_1p6m64.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--quick-val-frames", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--association-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.num_workers + 1,
            args.quick_val_frames, args.head_lr) <= 0 or
            args.association_weight < 0):
        parser.error("Invalid epoch, batch, worker, validation, or loss setting")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    raw["context_radii_m"] = tuple(raw["context_radii_m"])
    config = SurfaceStudentConfig(**raw)
    source = torch.load(args.source_student, map_location="cpu", weights_only=False)
    if source.get("version") != "crossmodal_student_v5_multiscale_refined_raw64":
        parser.error("Expected a source radar-only crossmodal student checkpoint")
    if args.teacher_checkpoint is None:
        args.teacher_checkpoint = Path(source["teacher_checkpoint"])
    source_sha = sha256(args.source_student)
    teacher_sha = sha256(args.teacher_checkpoint)
    if source.get("teacher_sha256") != teacher_sha:
        parser.error("Source student and teacher checkpoint hash differ")
    teacher, teacher_saved = load_teacher(args.teacher_checkpoint, args.device)
    if (teacher_saved["radar_variant"] != source["radar_variant"] or
            teacher_saved.get("lidar_fov") != source.get("lidar_fov")):
        parser.error("Teacher and student use different radar/LiDAR data")
    teacher.requires_grad_(False)
    teacher.eval()
    torch.manual_seed(args.seed)
    model = MatchedHeads(config, source).to(args.device)
    del source
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=args.head_lr,
                                   weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    train = paired_dataset(args.vod_root, "train", teacher_saved["radar_variant"],
                           data_cache_root=args.data_cache_root,
                           limit=args.train_limit)
    val = paired_dataset(args.vod_root, "val", teacher_saved["radar_variant"],
                         data_cache_root=args.data_cache_root,
                         limit=args.val_limit)
    quick_indices = random.Random(args.seed).sample(
        range(len(val)), min(args.quick_val_frames, len(val)))
    quick_val = Subset(val, quick_indices)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_stage1,
        pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(
        quick_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_stage1,
        pin_memory=args.device.startswith("cuda"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    start = 1
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != VERSION or saved["config"] != asdict(config)
                or saved["source_sha256"] != source_sha
                or saved["teacher_sha256"] != teacher_sha
                or saved["association_weight"] != args.association_weight
                or saved["seed"] != args.seed):
            parser.error("Resume checkpoint differs from this experiment")
        model.heads.load_state_dict(saved["heads"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
    elif history.exists():
        parser.error("Existing metrics require --resume or a new output directory")
    for epoch in range(start, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        train.set_epoch(epoch)
        train_metrics = _run_epoch(model, teacher, train_loader, args.device,
                                   optimizer=optimizer,
                                   association_weight=args.association_weight,
                                   epoch=epoch)
        with torch.no_grad():
            val_metrics = _run_epoch(model, teacher, val_loader, args.device,
                                     association_weight=args.association_weight,
                                     epoch=epoch)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _save(args.output_root / "last.ckpt", model, optimizer, scheduler,
              args, source_sha, teacher_sha, epoch, train_metrics, val_metrics)
        print("Epoch %03d: " % epoch + " | ".join(
            f"{name}: train {train_metrics[name]['total']:.4f}, "
            f"val {val_metrics[name]['total']:.4f}" for name in ARMS),
            flush=True)
    print(f"Checkpoint: {args.output_root / 'last.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
