"""Separate frozen-teacher training for the raw radar-only sparse student."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig

from .pipeline_data import collate_stage1, paired_dataset
from .history import log_record, open_writer, overfitting_warning
from .plots import update_plots
from .student import RadarOnlyStudent, StudentConfig
from .student_loss import (radar_supported_iou_sweep, student_objective, student_targets,
                           teacher_correspondence_evidence)
from .student_validation_cache import StudentValidationCache
from .train_teacher import _assert_frozen, _encoder_snapshot, _hash, _move, _prior_full_records, load_teacher


class Summary:
    def __init__(self, config):
        self.config = config
        self.feature_elements = self.known_sites = 0
        self.distill_sum = self.bce_sum = 0.
        self.bce_weight_sum = 0.
        self.dice_inter = self.dice_prob = self.dice_true = 0.
        self.counts = {name: 0 for name in ("occupied_sites", "free_sites", "unknown_sites")}
        self.sweep_counts = None
        self.clean_represented = self.clean_in_grid = 0
        self.frames = 0
        self.candidate_sites = 0

    def add(self, pieces, output, targets, batch_count, *, teacher_evidence=None,
            evaluate=False):
        features, known = pieces["feature_elements"], pieces["known_sites"]
        self.feature_elements += features
        self.known_sites += known
        self.distill_sum += pieces["distillation"].detach() * features
        bce_weight = pieces["bce_weight_sum"]
        self.bce_sum += pieces["bce"].detach() * bce_weight
        self.bce_weight_sum += bce_weight
        self.dice_inter += pieces["dice_intersection"]
        self.dice_prob += pieces["dice_probability_sum"]
        self.dice_true += pieces["dice_target_sum"]
        for key in self.counts:
            self.counts[key] += pieces[key]
        self.frames += batch_count
        self.candidate_sites += len(output.confidence_logits.coords)
        if evaluate:
            if teacher_evidence is None:
                raise ValueError("Validation requires frozen teacher correspondence evidence")
            self.clean_represented += targets.clean_points_in_candidates
            self.clean_in_grid += targets.clean_points_in_grid
            counts = radar_supported_iou_sweep(
                output, targets, self.config.thresholds, teacher_evidence,
                self.config.teacher_evidence_threshold)
            self.sweep_counts = counts if self.sweep_counts is None else self.sweep_counts + counts

    def finish(self, *, select_threshold=False):
        # All per-batch scalar reductions remain on the GPU. Transfer them
        # together only after the epoch's validation has finished.
        scalar_values = torch.stack([torch.as_tensor(value, dtype=torch.float64,
                                                      device=self.distill_sum.device)
                                     for value in (self.known_sites, self.distill_sum,
                                                   self.bce_sum, self.bce_weight_sum,
                                                   self.dice_inter, self.dice_prob, self.dice_true,
                                                   *(self.counts.values()))]).cpu().tolist()
        known_sites, distill_sum, bce_sum, bce_weight_sum, dice_inter, dice_prob, dice_true, *count_values = scalar_values
        distill = distill_sum / self.feature_elements if self.feature_elements else None
        bce = bce_sum / bce_weight_sum if bce_weight_sum else None
        dice = (1 - (2 * dice_inter + 1e-6) /
                (dice_prob + dice_true + 1e-6)) if dice_true else 0.
        spatial = (bce or 0.) + self.config.dice_weight * dice
        total = self.config.distillation_weight * (distill or 0.) + self.config.spatial_weight * spatial
        report = {"loss_total": total, "loss_distillation": distill,
                  "loss_spatial": spatial, "loss_bce": bce, "loss_dice": dice,
                  "feature_elements": self.feature_elements, "known_sites": int(known_sites),
                  "candidate_sites": self.candidate_sites, "frames": self.frames,
                  **{key: int(value) for key, value in zip(self.counts, count_values)}}
        if select_threshold:
            if self.sweep_counts is None:
                raise ValueError("Validation had no frames")
            sweep_counts = self.sweep_counts.cpu().tolist()
            rows = {}
            for threshold, values in zip(self.config.thresholds, sweep_counts):
                rows[str(threshold)] = {
                    name: {"intersection": pair[0], "union": pair[1],
                           "iou": pair[0] / pair[1] if pair[1] else None}
                    for name, pair in zip(("known", "conventional", "teacher"), values)}
            # Known-space IoU is the primary unknown-safe metric. Tie-break
            # toward a higher threshold to avoid gratuitous additions.
            best = max(self.config.thresholds,
                       key=lambda t: ((rows[str(t)]["known"]["iou"] or -1), t))
            report.update({"selected_threshold": best,
                           "radar_supported_iou_known": rows[str(best)]["known"]["iou"],
                           "radar_supported_iou_conventional": rows[str(best)]["conventional"]["iou"],
                           "teacher_correspondence_supported_iou": rows[str(best)]["teacher"]["iou"],
                           "teacher_evidence_threshold": self.config.teacher_evidence_threshold,
                           "threshold_sweep": rows,
                           "candidate_coverage": self.clean_represented / max(self.clean_in_grid, 1),
                           "clean_points_in_candidates": self.clean_represented,
                           "clean_points_in_grid": self.clean_in_grid})
        return report


def _epoch(student, teacher, dataset, loader, device, optimizer=None, epoch=0,
           validation_cache=None):
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    if training:
        dataset.set_epoch(epoch)
    aggregate = Summary(student.config)
    for batch_index, raw in enumerate(tqdm(loader, desc=("Student train" if training else "Student val")
                                            + f" {epoch:03d}", unit="batch", dynamic_ncols=True,
                                            leave=training), 1):
        batch = _move(raw, device)
        if not training and validation_cache is not None:
            with torch.no_grad():
                output = student(batch["radar"], batch["radar_valid"])
                teacher_sites, targets, evidence = validation_cache.load_batch(
                    batch["frame_id"], output, device)
                loss, pieces = student_objective(output, teacher_sites, targets,
                                                 student.config, evidence)
        else:
            with torch.no_grad():
                teacher_output = teacher(batch["radar"], batch["radar_valid"],
                                         batch["clean_lidar"], batch["clean_lidar_valid"])
            with torch.set_grad_enabled(training):
                output = student(batch["radar"], batch["radar_valid"])
                targets = student_targets(output, batch["clean_lidar"], batch["clean_lidar_valid"])
                evidence = teacher_correspondence_evidence(
                    teacher_output, output.confidence_logits)
                loss, pieces = student_objective(output, teacher_output.fused, targets,
                                                 student.config, evidence)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in student.parameters() if parameter.requires_grad), 5.)
            optimizer.step()
        aggregate.add(pieces, output, targets, len(batch["frame_id"]),
                      teacher_evidence=evidence, evaluate=not training)
    return aggregate.finish(select_threshold=not training)


def _save(path, student, optimizer, scheduler, teacher_path, teacher_hash, epoch, step,
          args, validation):
    torch.save({"version": "crossmodal_student_v4_stage1_native_sparse_raw64", "config": asdict(student.config),
                "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
                "teacher_checkpoint": str(teacher_path.resolve()), "teacher_sha256": teacher_hash,
                "radar_checkpoint": student.radar_encoder.source_checkpoint,
                "radar_variant": args.radar_variant,
                "augmentation": args.augmentation.to_dict(),
                "augmentation_seed": args.augmentation_seed,
                "selected_threshold": validation.get("val", {}).get("selected_threshold"),
                "validation": validation}, path)


def load_student(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != "crossmodal_student_v4_stage1_native_sparse_raw64":
        raise ValueError("Expected versioned raw-confidence student checkpoint")
    model = RadarOnlyStudent(saved["radar_checkpoint"], StudentConfig(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/crossmodal_student_raw64.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--online-augmentation-config", type=Path)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-cache-root", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--quick-val-frames", type=int, default=32)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--minimum-candidate-coverage", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.validation_batch_size < 1:
        parser.error("--validation-batch-size must be at least 1")
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("The unmodified Downloads encoder and sparse student need CUDA")
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_config["decoder_channels"] = tuple(raw_config["decoder_channels"])
    if "thresholds" in raw_config:
        raw_config["thresholds"] = tuple(raw_config["thresholds"])
    config = StudentConfig(**raw_config)
    args.augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    teacher, source = load_teacher(args.teacher_checkpoint, args.device)
    if source["radar_variant"] != args.radar_variant:
        raise ValueError("Teacher/student radar variants differ")
    teacher_hash = _hash(args.teacher_checkpoint)
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           augmentation=args.augmentation, seed=args.augmentation_seed,
                           limit=args.train_limit)
    val = paired_dataset(args.vod_root, "val", args.radar_variant, limit=args.val_limit,
                         include_clean=args.validation_cache_root is None)
    validation_cache = (StudentValidationCache(args.validation_cache_root, teacher_hash,
                                               args.radar_variant)
                        if args.validation_cache_root else None)
    if validation_cache is not None:
        missing = [frame.frame_id for frame in val.frames
                   if not validation_cache.path(frame.frame_id).is_file()]
        if missing:
            raise FileNotFoundError(f"Validation cache misses {len(missing)} frames, first: {missing[0]}")
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1, pin_memory=True),
        "quick": DataLoader(Subset(val, list(range(min(args.quick_val_frames, len(val))))),
                            batch_size=args.validation_batch_size, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True),
        "full": DataLoader(val, batch_size=args.validation_batch_size, num_workers=args.num_workers,
                           collate_fn=collate_stage1, pin_memory=True),
    }
    student = RadarOnlyStudent(source["radar_checkpoint"], config).to(args.device)
    frozen_student = _encoder_snapshot(student)
    frozen_teacher = {key: value.detach().cpu().clone() for key, value in teacher.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "metrics.jsonl").exists() and not args.resume:
        parser.error("Existing student metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    start, step = 1, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved["config"] != asdict(config) or saved["teacher_sha256"] != teacher_hash
                or saved["augmentation"] != args.augmentation.to_dict()):
            raise ValueError("Student resume configuration or frozen teacher changed")
        student.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, step = saved["epoch"] + 1, saved["step"]
    best_iou, best_distill = float("-inf"), float("inf")
    for row in (_prior_full_records(args.output_root) if args.resume else []):
        iou = row["val"].get("radar_supported_iou_known")
        distill = row["val"].get("loss_distillation")
        if distill is not None:
            best_distill = min(best_distill, distill)
        if (iou is not None and row["val"].get("candidate_coverage", 0)
                >= args.minimum_candidate_coverage):
            best_iou = max(best_iou, iou)
    for epoch in range(start, args.epochs + 1):
        training = _epoch(student, teacher, train, loaders["train"], args.device, optimizer, epoch)
        step += len(loaders["train"])
        scheduler.step()
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation = _epoch(student, teacher, val, loaders["full" if full else "quick"],
                            args.device, epoch=epoch, validation_cache=validation_cache)
        _assert_frozen(student, frozen_student)
        if any(not torch.equal(value, teacher.state_dict()[key].detach().cpu())
               for key, value in frozen_teacher.items()):
            raise RuntimeError("Frozen teacher state changed during student training")
        gap = validation["loss_total"] - training["loss_total"]
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "loss_gap": gap,
                  "relative_loss_gap": gap / (abs(training["loss_total"]) + 1e-8)}
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        update_plots(args.output_root, "student")
        warning = overfitting_warning(args.output_root / "metrics.jsonl")
        if warning:
            print(f"Overfitting warning: {warning}", flush=True)
        (args.output_root / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        _save(args.output_root / "last.ckpt", student, optimizer, scheduler,
              args.teacher_checkpoint, teacher_hash, epoch, step, args, record)
        if full:
            iou = validation.get("radar_supported_iou_known")
            distill = validation.get("loss_distillation")
            if distill is not None and distill < best_distill:
                best_distill = distill
                _save(args.output_root / "best_distillation.ckpt", student, optimizer,
                      scheduler, args.teacher_checkpoint, teacher_hash, epoch, step, args, record)
            if (iou is not None and iou > best_iou
                    and validation["candidate_coverage"] >= args.minimum_candidate_coverage):
                best_iou = iou
                _save(args.output_root / "best_radar_iou.ckpt", student, optimizer,
                      scheduler, args.teacher_checkpoint, teacher_hash, epoch, step, args, record)
        print(f"Student {epoch:03d}/{args.epochs:03d} train {training['loss_total']:.4f} "
              f"val {validation['loss_total']:.4f} known IoU {validation.get('radar_supported_iou_known')} "
              f"coverage {validation.get('candidate_coverage')} scope {'full' if full else 'quick'}",
              flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
