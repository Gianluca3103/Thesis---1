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
from .student_loss import radar_supported_iou, student_objective, student_targets
from .train_teacher import _assert_frozen, _encoder_snapshot, _hash, _move, _prior_full_records, load_teacher


class Summary:
    def __init__(self, config):
        self.config = config
        self.feature_elements = self.known_sites = 0
        self.distill_sum = self.bce_sum = 0.
        self.dice_inter = self.dice_prob = self.dice_true = 0.
        self.counts = {name: 0 for name in ("occupied_sites", "free_sites", "unknown_sites")}
        self.sweep = {threshold: {"known": [0, 0], "conventional": [0, 0]}
                      for threshold in config.thresholds}
        self.clean_represented = self.clean_in_grid = 0
        self.frames = 0
        self.candidate_sites = 0

    def add(self, pieces, output, targets, batch_count, *, evaluate=False):
        features, known = pieces["feature_elements"], pieces["known_sites"]
        self.feature_elements += features
        self.known_sites += known
        self.distill_sum += float(pieces["distillation"].detach()) * features
        self.bce_sum += float(pieces["bce"].detach()) * known
        self.dice_inter += float(pieces["dice_intersection"])
        self.dice_prob += float(pieces["dice_probability_sum"])
        self.dice_true += float(pieces["dice_target_sum"])
        for key in self.counts:
            self.counts[key] += pieces[key]
        self.frames += batch_count
        self.candidate_sites += len(output.confidence_logits.coords)
        if evaluate:
            self.clean_represented += targets.clean_points_in_candidates
            self.clean_in_grid += targets.clean_points_in_grid
            for threshold in self.config.thresholds:
                row = radar_supported_iou(output, targets, threshold)
                for name, target_key in (("known", "known_space_masked"),
                                         ("conventional", "conventional_clean_scan")):
                    self.sweep[threshold][name][0] += row[target_key]["intersection"]
                    self.sweep[threshold][name][1] += row[target_key]["union"]

    def finish(self, *, select_threshold=False):
        distill = self.distill_sum / self.feature_elements if self.feature_elements else None
        bce = self.bce_sum / self.known_sites if self.known_sites else None
        dice = (1 - (2 * self.dice_inter + 1e-6) /
                (self.dice_prob + self.dice_true + 1e-6)) if self.dice_true else 0.
        spatial = (bce or 0.) + self.config.dice_weight * dice
        total = self.config.distillation_weight * (distill or 0.) + self.config.spatial_weight * spatial
        report = {"loss_total": total, "loss_distillation": distill,
                  "loss_spatial": spatial, "loss_bce": bce, "loss_dice": dice,
                  "feature_elements": self.feature_elements, "known_sites": self.known_sites,
                  "candidate_sites": self.candidate_sites, "frames": self.frames, **self.counts}
        if select_threshold:
            rows = {}
            for threshold, pair in self.sweep.items():
                rows[str(threshold)] = {
                    name: {"intersection": values[0], "union": values[1],
                           "iou": values[0] / values[1] if values[1] else None}
                    for name, values in pair.items()}
            # Known-space IoU is the primary unknown-safe metric. Tie-break
            # toward a higher threshold to avoid gratuitous additions.
            best = max(self.config.thresholds,
                       key=lambda t: ((rows[str(t)]["known"]["iou"] or -1), t))
            report.update({"selected_threshold": best,
                           "radar_supported_iou_known": rows[str(best)]["known"]["iou"],
                           "radar_supported_iou_conventional": rows[str(best)]["conventional"]["iou"],
                           "threshold_sweep": rows,
                           "candidate_coverage": self.clean_represented / max(self.clean_in_grid, 1),
                           "clean_points_in_candidates": self.clean_represented,
                           "clean_points_in_grid": self.clean_in_grid})
        return report


def _epoch(student, teacher, dataset, loader, device, optimizer=None, epoch=0):
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    if training:
        dataset.set_epoch(epoch)
    aggregate = Summary(student.config)
    for batch_index, raw in enumerate(tqdm(loader, desc=("Student train" if training else "Student val")
                                            + f" {epoch:03d}", unit="batch", dynamic_ncols=True), 1):
        batch = _move(raw, device)
        with torch.no_grad():
            teacher_output = teacher(batch["radar"], batch["radar_valid"],
                                     batch["clean_lidar"], batch["clean_lidar_valid"])
        with torch.set_grad_enabled(training):
            output = student(batch["radar"], batch["radar_valid"])
            targets = student_targets(output, batch["clean_lidar"], batch["clean_lidar_valid"])
            loss, pieces = student_objective(output, teacher_output.fused, targets, student.config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in student.parameters() if parameter.requires_grad), 5.)
                optimizer.step()
        aggregate.add(pieces, output, targets, len(batch["frame_id"]), evaluate=not training)
        if training and batch_index % 50 == 0:
            print(json.dumps({"epoch": epoch, "step": batch_index,
                              "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "loss_components": {key: float(pieces[key].detach())
                                                  for key in ("distillation", "spatial", "bce", "dice")},
                              "feature_elements": pieces["feature_elements"],
                              "known_sites": pieces["known_sites"],
                              "candidate_sites": len(output.confidence_logits.coords),
                              "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30}),
                  flush=True)
    return aggregate.finish(select_threshold=not training)


def _save(path, student, optimizer, scheduler, teacher_path, teacher_hash, epoch, step,
          args, validation):
    torch.save({"version": "crossmodal_student_v2_doppler_radar_raw64", "config": asdict(student.config),
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
    if saved.get("version") != "crossmodal_student_v2_doppler_radar_raw64":
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
    val = paired_dataset(args.vod_root, "val", args.radar_variant, limit=args.val_limit)
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1, pin_memory=True),
        "quick": DataLoader(Subset(val, list(range(min(args.quick_val_frames, len(val))))),
                            batch_size=1, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True),
        "full": DataLoader(val, batch_size=1, num_workers=args.num_workers,
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
                            args.device, epoch=epoch)
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
