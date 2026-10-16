"""Distill teacher bottleneck features into a radar-only student before decoding."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig
from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC
from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import GRID
from .feature_student import (FeatureStudentConfig, RadarOnlyFeatureStudent,
                              feature_distillation_loss)
from .history import log_record, open_writer, overfitting_warning
from .pipeline_data import collate_stage1, paired_dataset
from .student_compact_cache import open_student_cache
from .train_teacher import (_assert_frozen, _encoder_snapshot, _hash, _move,
                            _prior_full_records, load_teacher)


VERSION = "crossmodal_feature_student_v1"


def cached_teacher_sites(cache, frame_ids: list[str], student: SparseSites,
                         device: str) -> SparseSites:
    """Read only frozen teacher bottleneck targets from an existing student cache."""
    entries = [cache._read(frame_id) for frame_id in frame_ids]
    if any(entry["bottleneck_coords"][:, 0].ne(0).any() for entry in entries):
        raise ValueError("Cached teacher features must be stored one frame at a time")
    coords = torch.cat([
        torch.cat((entry["bottleneck_coords"][:, :1] + batch,
                   entry["bottleneck_coords"][:, 1:]), dim=1)
        for batch, entry in enumerate(entries)
    ]).to(device=device, dtype=torch.long)
    if not torch.equal(coords, student.coords):
        raise ValueError("Cached teacher coordinates differ from radar-only bottleneck sites")
    features = torch.cat([entry["teacher_features"] for entry in entries]).to(
        device=device, dtype=student.features.dtype)
    if features.shape != student.features.shape:
        raise ValueError("Cached teacher features differ from radar-only bottleneck features")
    return SparseSites._from_validated(coords, features, GRID.scale_shape(8), 8)


def _epoch(student, teacher, dataset, loader, device, *, cache=None,
           optimizer=None, epoch=1):
    training = optimizer is not None
    student.train(training)
    if teacher is not None:
        teacher.eval()
    if training:
        dataset.set_epoch(epoch)
    weighted_loss, elements, frames, sites = 0.0, 0, 0, 0
    for raw in tqdm(loader, desc=f"Feature {'train' if training else 'val'} {epoch:03d}",
                    unit="batch", dynamic_ncols=True, leave=training):
        batch = _move(raw, device)
        with torch.set_grad_enabled(training):
            output = student(batch["radar"], batch["radar_valid"])
            if cache is not None:
                target = cached_teacher_sites(cache, batch["frame_id"], output, device)
            else:
                with torch.no_grad():
                    target = teacher(batch["radar"], batch["radar_valid"],
                                     batch["clean_lidar"], batch["clean_lidar_valid"]).fused
            loss = feature_distillation_loss(output, target, student.config)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad), 5.0)
            optimizer.step()
        count = output.features.numel()
        weighted_loss += float(loss.detach()) * count
        elements += count
        sites += len(output.coords)
        frames += len(batch["frame_id"])
    if elements == 0:
        raise ValueError("No radar bottleneck features were available for distillation")
    value = weighted_loss / elements
    return {"loss_total": value, "loss_distillation": value,
            "feature_elements": elements, "radar_sites": sites, "frames": frames}


def _save(path, student, optimizer, scheduler, teacher_path, teacher_hash,
          epoch, args, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({"version": VERSION, "config": asdict(student.config),
                    "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "teacher_checkpoint": str(teacher_path.resolve()),
                    "teacher_sha256": teacher_hash,
                    "radar_checkpoint": student.radar_encoder.source_checkpoint,
                    "radar_sha256": _hash(student.radar_encoder.source_checkpoint),
                    "radar_variant": args.radar_variant,
                    "lidar_fov": RADAR_FOV_SPEC,
                    "augmentation": args.augmentation.to_dict(),
                    "validation": validation}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_feature_student(path: str | Path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("Expected a feature-only radar student checkpoint")
    if _hash(saved["radar_checkpoint"]) != saved["radar_sha256"]:
        raise ValueError("Frozen radar encoder source changed")
    model = RadarOnlyFeatureStudent(saved["radar_checkpoint"],
                                    FeatureStudentConfig(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--training-cache-root", type=Path)
    parser.add_argument("--validation-cache-root", type=Path)
    parser.add_argument("--online-augmentation-config", type=Path)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--quick-val-frames", type=int, default=32)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validation_batch_size,
           args.full_val_every, args.quick_val_frames) < 1:
        parser.error("Epochs, batch sizes, and validation intervals must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    if args.training_cache_root and args.augmentation.enabled:
        parser.error("Training cache requires geometric augmentation to be disabled")
    teacher, source = load_teacher(args.teacher_checkpoint, args.device)
    if source.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("Teacher and student LiDAR field of view differ")
    if source["radar_variant"] != args.radar_variant:
        raise ValueError("Teacher and student radar variants differ")
    teacher_hash = _hash(args.teacher_checkpoint)
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           augmentation=args.augmentation, seed=args.augmentation_seed,
                           limit=args.train_limit, include_clean=args.training_cache_root is None,
                           data_cache_root=args.data_cache_root)
    val = paired_dataset(args.vod_root, "val", args.radar_variant,
                         limit=args.val_limit, include_clean=args.validation_cache_root is None,
                         data_cache_root=args.data_cache_root)
    caches = {}
    for split, root, dataset in (("train", args.training_cache_root, train),
                                 ("val", args.validation_cache_root, val)):
        cache = open_student_cache(root, teacher_hash, args.radar_variant,
                                   split=split) if root else None
        if cache is not None:
            missing = [frame.frame_id for frame in dataset.frames
                       if not cache.path(frame.frame_id).is_file()]
            if missing:
                raise FileNotFoundError(f"{split} teacher-feature cache misses "
                                        f"{len(missing)} frames; first: {missing[0]}")
        caches[split] = cache
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1,
                            pin_memory=True),
        "quick": DataLoader(Subset(val, range(min(args.quick_val_frames, len(val)))),
                            batch_size=args.validation_batch_size,
                            num_workers=args.num_workers, collate_fn=collate_stage1,
                            pin_memory=True),
        "full": DataLoader(val, batch_size=args.validation_batch_size,
                           num_workers=args.num_workers, collate_fn=collate_stage1,
                           pin_memory=True),
    }
    student = RadarOnlyFeatureStudent(source["radar_checkpoint"]).to(args.device)
    frozen = _encoder_snapshot(student)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "metrics.jsonl").exists() and not args.resume:
        parser.error("Existing feature-student metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    start = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != VERSION or saved["config"] != asdict(student.config)
                or saved["teacher_sha256"] != teacher_hash
                or saved["augmentation"] != args.augmentation.to_dict()
                or saved.get("lidar_fov") != RADAR_FOV_SPEC):
            raise ValueError("Feature-student resume configuration or teacher changed")
        student.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
    best = min((row["val"]["loss_distillation"]
                for row in _prior_full_records(args.output_root)), default=float("inf"))
    for epoch in range(start, args.epochs + 1):
        training = _epoch(student, teacher if caches["train"] is None else None,
                          train, loaders["train"], args.device,
                          cache=caches["train"], optimizer=optimizer, epoch=epoch)
        scheduler.step()
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation = _epoch(student, teacher if caches["val"] is None else None,
                            val, loaders["full" if full else "quick"], args.device,
                            cache=caches["val"], epoch=epoch)
        _assert_frozen(student, frozen)
        gap = validation["loss_total"] - training["loss_total"]
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick", "loss_gap": gap,
                  "relative_loss_gap": gap / (abs(training["loss_total"]) + 1e-8)}
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        if warning := overfitting_warning(args.output_root / "metrics.jsonl"):
            print(f"Overfitting warning: {warning}", flush=True)
        _save(args.output_root / "last.ckpt", student, optimizer, scheduler,
              args.teacher_checkpoint, teacher_hash, epoch, args, record)
        if full and validation["loss_distillation"] < best:
            best = validation["loss_distillation"]
            _save(args.output_root / "best_distillation.ckpt", student, optimizer,
                  scheduler, args.teacher_checkpoint, teacher_hash, epoch, args, record)
        print(f"Feature student {epoch:03d}/{args.epochs:03d} "
              f"train {training['loss_total']:.4f} val {validation['loss_total']:.4f} "
              f"scope {'full' if full else 'quick'}", flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
