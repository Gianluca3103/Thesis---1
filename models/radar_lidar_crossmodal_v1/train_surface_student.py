"""Train the surface decoder after freezing a trained radar-only student."""

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

from .history import log_record, open_writer, overfitting_warning
from .pipeline_data import collate_stage1, paired_dataset
from .surface_student import RadarOnlySurfaceStudent, SurfaceStudentConfig
from .surface_student_loss import (SurfaceMetricCounts, clean_surface_sites,
                                   surface_metric_counts, surface_student_objective)
from .train_teacher import _assert_frozen, _encoder_snapshot, _hash, _move


VERSION = "crossmodal_surface_student_v3_feature_first"


def _epoch(student, dataset, loader, device, *, optimizer=None, epoch=1):
    training = optimizer is not None
    student.train(training)
    if training:
        dataset.set_epoch(epoch)
    sums = {key: 0.0 for key in ("distillation", "geometry", "confidence")}
    counts = {"frames": 0, "anchors": 0, "proposals": 0,
              "anchors_with_targets": 0, "feature_elements": 0,
              "geometry_elements": 0}
    metrics = SurfaceMetricCounts()
    for raw in tqdm(loader, desc=f"Surface {'train' if training else 'val'} {epoch:03d}",
                    unit="batch", dynamic_ncols=True, leave=training):
        batch = _move(raw, device)
        with torch.no_grad():
            clean = clean_surface_sites(batch["clean_lidar"], batch["clean_lidar_valid"])
        with torch.set_grad_enabled(training):
            output = student(batch["radar"], batch["radar_valid"])
            loss, pieces = surface_student_objective(
                output, None, clean, student.config)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad), 5.0)
            optimizer.step()
        feature_count = output.conditioning.features.numel()
        counts["frames"] += len(batch["frame_id"])
        counts["anchors"] += pieces["anchors"]
        counts["proposals"] += pieces["proposals"]
        counts["anchors_with_targets"] += pieces["anchors_with_targets"]
        counts["feature_elements"] += feature_count
        counts["geometry_elements"] += pieces["geometry_elements"]
        sums["distillation"] += float(pieces["distillation"].detach()) * feature_count
        sums["geometry"] += float(pieces["geometry"].detach()) * pieces["geometry_elements"]
        sums["confidence"] += float(pieces["confidence"].detach()) * pieces["proposals"]
        if not training:
            metrics.add(surface_metric_counts(output, clean, student.config))
    distill = sums["distillation"] / max(counts["feature_elements"], 1)
    geometry = sums["geometry"] / max(counts["geometry_elements"], 1)
    confidence = sums["confidence"] / max(counts["proposals"], 1)
    report = {"loss_distillation": distill, "loss_geometry": geometry,
              "loss_confidence": confidence,
              "loss_total": (student.config.distillation_weight * distill
                             + student.config.geometry_weight * geometry
                             + student.config.confidence_weight * confidence), **counts}
    if not training:
        report.update(metrics.report())
    return report


def _save(path, student, optimizer, scheduler, feature_student_path, feature_student_hash,
          epoch, args, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({"version": VERSION, "config": asdict(student.config),
                    "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "feature_student_checkpoint": str(feature_student_path.resolve()),
                    "feature_student_sha256": feature_student_hash,
                    "radar_checkpoint": student.radar_encoder.source_checkpoint,
                    "radar_variant": args.radar_variant, "lidar_fov": RADAR_FOV_SPEC,
                    "augmentation": args.augmentation.to_dict(),
                    "validation": validation}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_surface_student(path: str | Path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") not in (VERSION, "crossmodal_surface_student_v2_sequential",
                                    "crossmodal_surface_student_v1"):
        raise ValueError("Expected a trained raw surface-proposal student checkpoint")
    config = dict(saved["config"])
    config["context_radii_m"] = tuple(config["context_radii_m"])
    model = RadarOnlySurfaceStudent(saved["radar_checkpoint"],
                                    SurfaceStudentConfig(**config))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--feature-student-checkpoint", type=Path, required=True,
                        help="Trained feature-only radar student checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/crossmodal_surface_student.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--online-augmentation-config", type=Path)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
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
    if args.batch_size < 1 or args.validation_batch_size < 1 or args.full_val_every < 1:
        parser.error("Batch sizes and validation interval must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_config["context_radii_m"] = tuple(raw_config["context_radii_m"])
    config = SurfaceStudentConfig(**raw_config)
    if not config.decoder_from_conditioned or config.distillation_weight != 0:
        parser.error("Sequential decoding requires decoder_from_conditioned=true and distillation_weight=0")
    args.augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    source = torch.load(args.feature_student_checkpoint, map_location="cpu", weights_only=False)
    if source.get("version") != "crossmodal_feature_student_v1":
        raise ValueError("Expected a completed feature-only radar student checkpoint")
    if source.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("Student checkpoint used a different LiDAR field of view")
    if source["radar_variant"] != args.radar_variant:
        raise ValueError("Radar variants differ")
    feature_student_hash = _hash(args.feature_student_checkpoint)
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           augmentation=args.augmentation, seed=args.augmentation_seed,
                           limit=args.train_limit, data_cache_root=args.data_cache_root)
    val = paired_dataset(args.vod_root, "val", args.radar_variant,
                         limit=args.val_limit, data_cache_root=args.data_cache_root)
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
    student = RadarOnlySurfaceStudent(source["radar_checkpoint"], config).to(args.device)
    student.load_feature_student(source)
    frozen_student = _encoder_snapshot(student)
    frozen_features = {key: value.detach().cpu().clone()
                       for key, value in student.state_dict().items()
                       if key.startswith(("shared.", "feature_head.", "feature_norm."))}
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "metrics.jsonl").exists() and not args.resume:
        parser.error("Existing metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    start = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != VERSION or saved["config"] != asdict(config)
                or saved["feature_student_sha256"] != feature_student_hash
                or saved["augmentation"] != args.augmentation.to_dict()
                or saved.get("lidar_fov") != RADAR_FOV_SPEC):
            raise ValueError("Surface-decoder resume configuration or trained student changed")
        student.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
    best = float("-inf")
    if args.resume and (args.output_root / "metrics.jsonl").is_file():
        for line in (args.output_root / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("validation_scope") == "full":
                best = max(best, row["val"].get("surface_f1_20cm") or float("-inf"))
    for epoch in range(start, args.epochs + 1):
        training = _epoch(student, train, loaders["train"], args.device,
                          optimizer=optimizer, epoch=epoch)
        scheduler.step()
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation = _epoch(student, val, loaders["full" if full else "quick"],
                            args.device, epoch=epoch)
        _assert_frozen(student, frozen_student)
        if any(not torch.equal(value, student.state_dict()[key].detach().cpu())
               for key, value in frozen_features.items()):
            raise RuntimeError("Frozen trained student changed during surface training")
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "loss_gap": validation["loss_total"] - training["loss_total"]}
        record["relative_loss_gap"] = record["loss_gap"] / (abs(training["loss_total"]) + 1e-8)
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        if warning := overfitting_warning(args.output_root / "metrics.jsonl"):
            print(f"Overfitting warning: {warning}", flush=True)
        (args.output_root / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        _save(args.output_root / "last.ckpt", student, optimizer, scheduler,
              args.feature_student_checkpoint, feature_student_hash, epoch, args, record)
        if full and (score := validation.get("surface_f1_20cm")) is not None and score > best:
            best = score
            _save(args.output_root / "best_surface_f1.ckpt", student, optimizer,
                  scheduler, args.feature_student_checkpoint, feature_student_hash, epoch, args, record)
        print(f"Surface student {epoch:03d}/{args.epochs:03d} train {training['loss_total']:.4f} "
              f"val {validation['loss_total']:.4f} raw F1@20cm "
              f"{validation.get('surface_f1_20cm')} selected "
              f"{validation['selected_proposals']} scope {'full' if full else 'quick'}",
              flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
