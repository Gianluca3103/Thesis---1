"""Ablate joint feature-student/pointer fine-tuning against head-only fine-tuning.

Both arms start from the same trained pointer checkpoint. The radar encoder and
cross-modal teacher stay frozen; only the joint arm updates the distilled radar
feature layers. The teacher is supervision only, never a student input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC

from .feature_student import FeatureStudentConfig, feature_distillation_loss
from .history import log_record, open_writer, overfitting_warning
from .pipeline_data import collate_stage1, paired_dataset
from .student_compact_cache import open_student_cache
from .surface_student_loss import (SurfaceMetricCounts, surface_metric_counts,
                                   surface_student_objective)
from .train_feature_student import cached_teacher_sites
from .train_surface_student import (JOINT_VERSION, VERSION as POINTER_VERSION,
                                    load_surface_student)
from .train_teacher import _hash, _move, load_teacher


def configure_trainable(model, *, freeze_feature_student: bool) -> None:
    model.radar_encoder.requires_grad_(False)
    for name in ("shared", "feature_head", "feature_norm"):
        getattr(model, name).requires_grad_(not freeze_feature_student)
    model.surface_head.requires_grad_(True)


def joint_losses(output, teacher_sites, clean_points, clean_valid,
                 surface_config, feature_config, *, distillation_weight: float,
                 pointer_weight: float):
    pointer, pieces, field, targets = surface_student_objective(
        output, clean_points, clean_valid, surface_config)
    distillation = feature_distillation_loss(
        output.conditioning, teacher_sites, feature_config)
    total = pointer_weight * pointer + distillation_weight * distillation
    return total, pointer, distillation, pieces, field, targets


def _epoch(model, teacher, cache, feature_config, dataset, loader, device, *,
           distillation_weight: float, pointer_weight: float,
           optimizer=None, epoch: int = 1):
    training = optimizer is not None
    model.train(training)
    if training:
        dataset.set_epoch(epoch)
    sums = {key: 0.0 for key in ("bce", "dice", "hit_reward")}
    total_sum = pointer_sum = distillation_sum = 0.0
    counts = {"frames": 0, "batches": 0, "feature_elements": 0,
              "pointer_slots": 0, "in_grid_pointers": 0,
              "candidate_sites": 0, "known_sites": 0,
              "occupied_sites": 0, "known_free_sites": 0}
    metrics = SurfaceMetricCounts()
    for raw in tqdm(loader, desc=f"Joint {'train' if training else 'val'} {epoch:03d}",
                    unit="batch", dynamic_ncols=True, leave=training):
        batch = _move(raw, device)
        with torch.set_grad_enabled(training):
            output = model(batch["radar"], batch["radar_valid"])
            if cache is None:
                with torch.no_grad():
                    teacher_sites = teacher(batch["radar"], batch["radar_valid"],
                                            batch["clean_lidar"],
                                            batch["clean_lidar_valid"]).fused
            else:
                teacher_sites = cached_teacher_sites(
                    cache, batch["frame_id"], output.conditioning, device)
            total, pointer, distillation, pieces, field, targets = joint_losses(
                output, teacher_sites, batch["clean_lidar"],
                batch["clean_lidar_valid"], model.config, feature_config,
                distillation_weight=distillation_weight,
                pointer_weight=pointer_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters()
                 if parameter.requires_grad), 5.0)
            optimizer.step()
        counts["frames"] += len(batch["frame_id"])
        counts["batches"] += 1
        counts["feature_elements"] += output.conditioning.features.numel()
        for key in ("pointer_slots", "in_grid_pointers", "candidate_sites",
                    "known_sites", "occupied_sites", "known_free_sites"):
            counts[key] += pieces[key]
        for key in sums:
            sums[key] += float(pieces[key].detach())
        total_sum += float(total.detach())
        pointer_sum += float(pointer.detach())
        distillation_sum += float(distillation.detach())
        if not training:
            metrics.add(surface_metric_counts(field, targets, model.config,
                                              pointer_slots=pieces["pointer_slots"]))
    batches = max(counts["batches"], 1)
    report = {"loss_total": total_sum / batches,
              "loss_pointer": pointer_sum / batches,
              "loss_distillation": distillation_sum / batches,
              "loss_bce": sums["bce"] / batches,
              "loss_dice": sums["dice"] / batches,
              "distinct_hit_reward": sums["hit_reward"] / batches,
              **counts}
    if not training:
        report.update(metrics.report())
    return report


def _save(path, model, optimizer, scheduler, epoch, args, source, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({"version": JOINT_VERSION, "config": asdict(model.config),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "radar_checkpoint": model.radar_encoder.source_checkpoint,
                    "radar_variant": source["radar_variant"],
                    "lidar_fov": RADAR_FOV_SPEC,
                    "initial_surface_checkpoint": str(args.surface_checkpoint.resolve()),
                    "initial_surface_sha256": args.surface_sha256,
                    "initial_feature_sha256": args.feature_sha256,
                    "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                    "teacher_sha256": args.teacher_sha256,
                    "freeze_feature_student": args.freeze_feature_student,
                    "distillation_weight": args.distillation_weight,
                    "pointer_weight": args.pointer_weight,
                    "seed": args.seed, "validation": validation}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--surface-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-student-checkpoint", type=Path,
                        help="Override the source path stored in the surface checkpoint")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Override the teacher path stored in the feature checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--training-cache-root", type=Path,
                        help="Existing teacher-feature cache for the train split")
    parser.add_argument("--validation-cache-root", type=Path,
                        help="Existing teacher-feature cache for the val split")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--quick-val-frames", type=int, default=32)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--feature-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--pointer-weight", type=float, default=1.0)
    parser.add_argument("--freeze-feature-student", action="store_true",
                        help="Matched head-only control when distillation-weight is zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validation_batch_size,
           args.quick_val_frames, args.full_val_every) < 1:
        parser.error("Epochs, batch sizes, and validation intervals must be positive")
    if min(args.head_learning_rate, args.feature_learning_rate) <= 0:
        parser.error("Learning rates must be positive")
    if min(args.distillation_weight, args.pointer_weight, args.weight_decay) < 0:
        parser.error("Loss weights and weight decay must be nonnegative")
    if args.freeze_feature_student and args.distillation_weight != 0:
        parser.error("The head-only control requires distillation-weight=0")
    if not args.freeze_feature_student and args.distillation_weight == 0:
        parser.error("Joint training requires a positive distillation weight")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)

    model, surface_source = load_surface_student(args.surface_checkpoint, args.device)
    if surface_source["version"] != POINTER_VERSION:
        parser.error("Start the ablation from a completed v4 pointer checkpoint")
    if surface_source.get("lidar_fov") != RADAR_FOV_SPEC:
        parser.error("The pointer checkpoint used a different LiDAR field of view")
    if (not model.config.pointer_voxel_mode or
            not model.config.decoder_from_conditioned):
        parser.error("Expected the trained radar-conditioned pointer decoder")
    feature_path = args.feature_student_checkpoint or Path(
        surface_source["feature_student_checkpoint"])
    if _hash(feature_path) != surface_source["feature_student_sha256"]:
        parser.error("Feature checkpoint differs from pointer training source")
    feature_source = torch.load(feature_path, map_location="cpu", weights_only=False)
    teacher_path = args.teacher_checkpoint or Path(feature_source["teacher_checkpoint"])
    if _hash(teacher_path) != feature_source["teacher_sha256"]:
        parser.error("Teacher checkpoint differs from feature distillation source")
    if feature_source.get("lidar_fov") != RADAR_FOV_SPEC:
        parser.error("The teacher/student LiDAR field of view differs")
    if (feature_source["radar_variant"] != surface_source["radar_variant"] or
            feature_source["radar_checkpoint"] != surface_source["radar_checkpoint"]):
        parser.error("Pointer and feature checkpoints used different radar inputs")
    args.teacher_checkpoint = teacher_path
    args.surface_sha256 = _hash(args.surface_checkpoint)
    args.feature_sha256 = _hash(feature_path)
    args.teacher_sha256 = _hash(teacher_path)
    feature_config = FeatureStudentConfig(**feature_source["config"])
    configure_trainable(model, freeze_feature_student=args.freeze_feature_student)
    radar_initial = {key: value.detach().cpu().clone()
                     for key, value in model.radar_encoder.state_dict().items()}
    teacher = None
    if not (args.training_cache_root and args.validation_cache_root):
        teacher, teacher_source = load_teacher(teacher_path, args.device)
        if (teacher_source["radar_variant"] != surface_source["radar_variant"] or
                teacher_source.get("lidar_fov") != RADAR_FOV_SPEC):
            parser.error("Teacher checkpoint used different radar inputs or LiDAR FOV")
        teacher.requires_grad_(False)
        teacher.eval()

    train = paired_dataset(args.vod_root, "train", surface_source["radar_variant"],
                           limit=args.train_limit, data_cache_root=args.data_cache_root)
    val = paired_dataset(args.vod_root, "val", surface_source["radar_variant"],
                         limit=args.val_limit, data_cache_root=args.data_cache_root)
    caches = {}
    for split, root, dataset in (("train", args.training_cache_root, train),
                                 ("val", args.validation_cache_root, val)):
        cache = (open_student_cache(root, args.teacher_sha256,
                                    surface_source["radar_variant"], split=split)
                 if root else None)
        if cache is not None:
            missing = [frame.frame_id for frame in dataset.frames
                       if not cache.path(frame.frame_id).is_file()]
            if missing:
                raise FileNotFoundError(f"{split} teacher cache misses {len(missing)} "
                                        f"frames; first: {missing[0]}")
        caches[split] = cache
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            generator=generator, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True),
        "quick": DataLoader(Subset(val, range(min(args.quick_val_frames, len(val)))),
                            batch_size=args.validation_batch_size,
                            num_workers=args.num_workers, collate_fn=collate_stage1,
                            pin_memory=True),
        "full": DataLoader(val, batch_size=args.validation_batch_size,
                           num_workers=args.num_workers, collate_fn=collate_stage1,
                           pin_memory=True),
    }
    groups = [{"params": list(model.surface_head.parameters()),
               "lr": args.head_learning_rate}]
    if not args.freeze_feature_student:
        groups.append({"params": [parameter for name, parameter in
                                   model.named_parameters()
                                   if name.startswith(("shared.", "feature_head.",
                                                       "feature_norm."))],
                       "lr": args.feature_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    if history.exists() and not args.resume:
        parser.error("Existing metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    start = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != JOINT_VERSION or
                saved["config"] != asdict(model.config) or
                saved["initial_surface_sha256"] != args.surface_sha256 or
                saved["teacher_sha256"] != args.teacher_sha256 or
                saved["freeze_feature_student"] != args.freeze_feature_student or
                saved["distillation_weight"] != args.distillation_weight or
                saved["pointer_weight"] != args.pointer_weight or
                saved["seed"] != args.seed):
            parser.error("Joint checkpoint or ablation settings differ")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
    best = float("-inf")
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("validation_scope") == "full":
                best = max(best, row["val"].get("proposal_utility") or 0.0)
    for epoch in range(start, args.epochs + 1):
        # Each arm sees the same frame order, including after an interrupted run.
        generator.manual_seed(args.seed + epoch)
        training = _epoch(
            model, teacher, caches["train"], feature_config, train,
            loaders["train"], args.device,
            distillation_weight=args.distillation_weight,
            pointer_weight=args.pointer_weight, optimizer=optimizer, epoch=epoch)
        if epoch == start and training["occupied_sites"] == 0 and len(train) > 16:
            raise RuntimeError("No pointed training voxels overlap clean LiDAR")
        scheduler.step()
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation = _epoch(
            model, teacher, caches["val"], feature_config, val,
            loaders["full" if full else "quick"], args.device,
            distillation_weight=args.distillation_weight,
            pointer_weight=args.pointer_weight, epoch=epoch)
        if any(not torch.equal(value, model.radar_encoder.state_dict()[key].detach().cpu())
               for key, value in radar_initial.items()):
            raise RuntimeError("Frozen radar encoder changed during ablation")
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "loss_gap": validation["loss_total"] - training["loss_total"],
                  "arm": "control" if args.freeze_feature_student else "joint"}
        record["relative_loss_gap"] = record["loss_gap"] / (
            abs(training["loss_total"]) + 1e-8)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        if warning := overfitting_warning(history):
            print(f"Overfitting warning: {warning}", flush=True)
        (args.output_root / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        _save(args.output_root / "last.ckpt", model, optimizer, scheduler,
              epoch, args, surface_source, record)
        if full and (score := validation.get("proposal_utility")) is not None and score > best:
            best = score
            _save(args.output_root / "best_proposal_utility.ckpt", model,
                  optimizer, scheduler, epoch, args, surface_source, record)
        print(f"{'Control' if args.freeze_feature_student else 'Joint'} "
              f"{epoch:03d}/{args.epochs:03d} "
              f"train {training['loss_total']:.4f} val {validation['loss_total']:.4f} "
              f"distill {validation['loss_distillation']:.4f} "
              f"unique hits {validation['distinct_correct_voxels']} "
              f"verified precision {validation['verified_proposal_precision']} "
              f"utility {validation['proposal_utility']:.3f} "
              f"scope {'full' if full else 'quick'}", flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
