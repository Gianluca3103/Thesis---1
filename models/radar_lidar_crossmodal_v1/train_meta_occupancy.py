"""Joint bilevel training: clean-LiDAR relationship teacher, radar-only map.

Run with ``python -m models.radar_lidar_crossmodal_v1.train_meta_occupancy --help``.
The teacher is training-only; ``load_meta_student`` builds the inference model.
"""

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

from .meta_occupancy import (MetaOccupancyConfig, candidate_support,
                             feature_distillation, meta_teacher_step,
                             occupancy_loss, scan_overlap_counts,
                             scan_overlap_counts_observed, scan_overlap_rates,
                             supported_clean_points,
                             supported_clean_voxels)
from .meta_cache import (MetaOccupancyCache, MetaOccupancyDataset,
                         cache_manifest, collate_meta_cache,
                         prepare_meta_batch)
from .pipeline_data import collate_stage1, paired_dataset
from .student import RadarOnlyStudent, StudentConfig
from .student_loss import student_targets
from .teacher import CrossModalTeacher, TeacherConfig
from .train_teacher import (_hash, _move,
                            load_teacher, validate_lidar_pretraining_fov)


VERSION = "crossmodal_meta_occupancy_trainable_encoders_v2"


def _prepare_batch(raw, device):
    return (prepare_meta_batch(raw, device) if "radar_coords" in raw
            else _move(raw, device))


def _student_output(student, batch):
    output = student(batch["radar"], batch["radar_valid"])
    if "candidate_coords" in batch and not torch.equal(
            output.confidence_logits.coords, batch["candidate_coords"]):
        raise ValueError("Meta cache candidate coordinates differ from decoder output")
    return output


def _teacher_output(teacher, batch):
    output = teacher(batch["radar"], batch["radar_valid"],
                     batch["clean_lidar"], batch["clean_lidar_valid"],
                     neighbors=batch.get("neighbors"))
    if "radar_coords" in batch and (
            not torch.equal(output.radar_frozen.coords, batch["radar_coords"])
            or not torch.equal(output.lidar_frozen.coords, batch["lidar_coords"])):
        raise ValueError("Cached teacher geometry differs from live encoder coordinates")
    return output


def _targets_and_support(output, batch, config):
    if "targets" in batch:
        return batch["targets"], batch["support"]
    targets = student_targets(output, batch["clean_lidar"], batch["clean_lidar_valid"])
    support = candidate_support(output.confidence_logits.coords, batch["radar"],
                                batch["radar_valid"], config.radius_m)
    return targets, support


class EpochMetrics:
    def __init__(self):
        self.frames = self.batches = self.meta_steps = 0
        self.bce_sum = self.brier_sum = self.distill_sum = 0.0
        self.distill_count = self.clean_supported = self.clean_voxels_supported = 0
        self.quiz_bce_sum = 0.0
        self.scan_counts = None
        self.counts = {key: 0 for key in (
            "known", "occupied", "free", "candidate_sites", "supported_candidate_sites",
            "represented_clean_points", "intersection", "union")}

    def add(self, stats, batch, clean_supported, clean_voxels_supported,
            distill=None, feature_elements=0, meta=None):
        self.frames += len(batch["frame_id"])
        self.batches += 1
        self.bce_sum += stats["bce_sum"]
        self.brier_sum += stats["brier_sum"]
        self.clean_supported += clean_supported
        self.clean_voxels_supported += clean_voxels_supported
        for key in self.counts:
            self.counts[key] += stats[key]
        if distill is not None:
            self.distill_sum += distill * feature_elements
            self.distill_count += feature_elements
        if meta is not None and not meta.get("skipped"):
            self.meta_steps += 1
            self.quiz_bce_sum += meta["quiz_bce"]

    def add_scan(self, counts):
        if self.scan_counts is None:
            self.scan_counts = {key: 0 for key in counts}
        for key, value in counts.items():
            self.scan_counts[key] += value

    def finish(self):
        known, union = self.counts["known"], self.counts["union"]
        missed_occupied = max(self.clean_voxels_supported - self.counts["occupied"], 0)
        full_union = union + missed_occupied
        report = {"frames": self.frames, "batches": self.batches,
                "map_bce": self.bce_sum / known if known else None,
                "brier": self.brier_sum / known if known else None,
                "iou_known_supported": self.counts["intersection"] / union if union else None,
                "iou_full_support": (self.counts["intersection"] / full_union
                                     if full_union else None),
                "missed_occupied_voxels": missed_occupied,
                "candidate_coverage": (self.counts["represented_clean_points"] /
                                       self.clean_supported if self.clean_supported else None),
                "clean_points_in_support": self.clean_supported,
                "clean_voxels_in_support": self.clean_voxels_supported,
                "distillation": (self.distill_sum / self.distill_count
                                 if self.distill_count else None),
                "meta_steps": self.meta_steps,
                "quiz_bce": self.quiz_bce_sum / self.meta_steps if self.meta_steps else None,
                **self.counts}
        if self.scan_counts is not None:
            report.update(self.scan_counts)
            report.update(scan_overlap_rates(self.scan_counts))
        return report


def _student_step(student, teacher, batch, optimizer, config, *, use_teacher):
    output = _student_output(student, batch)
    targets, support = _targets_and_support(output, batch, config)
    map_loss, stats = occupancy_loss(output, targets, support, config.threshold)
    distill = None
    feature_elements = 0
    if use_teacher:
        with torch.no_grad():
            teacher_output = _teacher_output(teacher, batch)
        eligible = teacher_output.neighbors.valid.any(-1)
        feature_elements = int(eligible.sum()) * output.conditioning.features.shape[-1]
        if feature_elements:
            distill = feature_distillation(output.conditioning, teacher_output.fused,
                                           beta=config.smooth_l1_beta,
                                           detach_teacher=True, eligible=eligible)
    loss = map_loss + (config.distillation_weight * distill if distill is not None else 0)
    optimizer.zero_grad(set_to_none=True)
    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in student.parameters() if parameter.requires_grad), 5.0)
        optimizer.step()
    return stats, float(distill.detach()) if distill is not None else None, feature_elements


def _train_epoch(student, teacher, dataset, loader, student_optimizer,
                 teacher_optimizer, config, epoch, warmup_epochs, device):
    dataset.set_epoch(epoch)
    student.train()
    teacher.train()
    summary = EpochMetrics()
    iterator = iter(loader)
    try:
        previous = _prepare_batch(next(iterator), device)
    except StopIteration:
        raise ValueError("The training split has no frames") from None
    progress = tqdm(range(len(loader)), desc=f"Meta train {epoch:03d}", unit="batch")
    for index in progress:
        raw_next = next(iterator, None)
        following = (_prepare_batch(raw_next, device)
                     if raw_next is not None and epoch > warmup_epochs else None)
        meta = None
        if epoch > warmup_epochs and following is not None:
            if set(previous["frame_id"]) & set(following["frame_id"]):
                raise ValueError("Paired and quiz batches must contain different frames")
            meta = meta_teacher_step(teacher, student, previous, following,
                                     teacher_optimizer, config)
        stats, distill, feature_elements = _student_step(
            student, teacher, previous, student_optimizer, config,
            use_teacher=epoch > warmup_epochs)
        if "targets" in previous:
            clean_supported = previous["clean_supported_points"]
            clean_voxels = previous["clean_supported_voxels"]
        else:
            clean_supported = supported_clean_points(
                previous["clean_lidar"], previous["clean_lidar_valid"],
                previous["radar"], previous["radar_valid"], config.radius_m)
            clean_voxels = supported_clean_voxels(
                previous["clean_lidar"], previous["clean_lidar_valid"],
                previous["radar"], previous["radar_valid"], config.radius_m)
        summary.add(stats, previous, clean_supported, clean_voxels,
                    distill, feature_elements, meta)
        if raw_next is not None:
            previous = following if following is not None else _prepare_batch(raw_next, device)
    return summary.finish()


@torch.no_grad()
def _validate(student, loader, config, device, *, evaluate_scan_metrics=False):
    student.eval()
    summary = EpochMetrics()
    for raw in tqdm(loader, desc="Radar-only val", unit="batch"):
        batch = _prepare_batch(raw, device)
        output = _student_output(student, batch)
        targets, support = _targets_and_support(output, batch, config)
        _, stats = occupancy_loss(output, targets, support, config.threshold)
        if "targets" in batch:
            clean_supported = batch["clean_supported_points"]
            clean_voxels = batch["clean_supported_voxels"]
            if evaluate_scan_metrics:
                predicted = output.confidence_logits.coords[
                    support & (torch.sigmoid(output.confidence_logits.features[:, 0])
                               >= config.threshold)]
                scan = scan_overlap_counts_observed(predicted, batch["observed_coords"])
            else:
                scan = None
        else:
            clean_supported = supported_clean_points(
                batch["clean_lidar"], batch["clean_lidar_valid"], batch["radar"],
                batch["radar_valid"], config.radius_m)
            scan = (scan_overlap_counts(
                output, batch["clean_lidar"], batch["clean_lidar_valid"],
                batch["radar"], batch["radar_valid"], support,
                config.threshold, config.radius_m) if evaluate_scan_metrics else None)
            clean_voxels = (scan["observed"] if scan is not None else supported_clean_voxels(
                batch["clean_lidar"], batch["clean_lidar_valid"], batch["radar"],
                batch["radar_valid"], config.radius_m))
        summary.add(stats, batch, clean_supported, clean_voxels)
        if scan is not None:
            summary.add_scan(scan)
    return summary.finish()


def _save(path, teacher, student, teacher_optimizer, student_optimizer, epoch,
          config, args, record):
    torch.save({
        "version": VERSION, "epoch": epoch, "meta_config": asdict(config),
        "teacher_config": asdict(teacher.config),
        "student_config": asdict(student.config),
        "teacher": teacher.state_dict(), "student": student.state_dict(),
        "teacher_optimizer": teacher_optimizer.state_dict(),
        "student_optimizer": student_optimizer.state_dict(),
        "radar_checkpoint": str(args.radar_encoder.resolve()),
        "lidar_checkpoint": str(args.lidar_encoder.resolve()),
        "radar_sha256": _hash(args.radar_encoder),
        "lidar_sha256": _hash(args.lidar_encoder),
        "radar_variant": args.radar_variant,
        "lidar_fov": RADAR_FOV_SPEC,
        "augmentation": args.augmentation.to_dict(),
        "record": record,
    }, path)


def _save_student_only(path, student, epoch, config, args):
    torch.save({
        "version": VERSION + "_student", "epoch": epoch,
        "student_config": asdict(student.config), "student": student.state_dict(),
        "radar_checkpoint": str(args.radar_encoder.resolve()),
        "radar_sha256": _hash(args.radar_encoder),
        "radius_m": config.radius_m, "threshold": config.threshold,
        "radar_variant": args.radar_variant, "lidar_fov": RADAR_FOV_SPEC,
    }, path)


def load_meta_student(path, device="cuda"):
    """Inference requires only radar and the trained student checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") not in (VERSION, VERSION + "_student"):
        raise ValueError("Expected a meta-occupancy student checkpoint")
    radar_path = Path(saved["radar_checkpoint"])
    if not radar_path.is_file() or _hash(radar_path) != saved["radar_sha256"]:
        raise FileNotFoundError("Initial radar encoder checkpoint missing or changed")
    config = StudentConfig(**saved["student_config"])
    if not config.decoder_from_conditioned:
        raise ValueError("Meta student must decode its distilled feature")
    model = RadarOnlyStudent(str(radar_path), config, trainable_encoder=True)
    model.load_state_dict(saved["student"], strict=True)
    return model.to(device).eval(), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--lidar-encoder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/crossmodal_meta_occupancy.json"))
    parser.add_argument("--teacher-init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--meta-cache-root", type=Path,
                        help="Version 3 losslessly compressed geometry and labels; raw points come from --data-cache-root")
    parser.add_argument("--online-augmentation-config", type=Path)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--student-lr", type=float, default=2e-4)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--scan-metrics-every", type=int, default=5)
    parser.add_argument("--quick-val-frames", type=int, default=32)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available()
            or args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs
            or args.batch_size < 1 or args.scan_metrics_every < 1
            or args.quick_val_frames < 1 or args.full_val_every < 1
            or min(args.student_lr, args.teacher_lr) <= 0):
        parser.error("Training needs CUDA, positive rates/batch size/metric period, and warmup < epochs")
    if args.resume and args.teacher_init:
        parser.error("Use either --resume or --teacher-init")
    config = MetaOccupancyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    validate_lidar_pretraining_fov(args.lidar_encoder)
    args.augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    if args.meta_cache_root:
        if args.augmentation.enabled:
            parser.error("Meta cache requires disabled geometric augmentation")
        if args.data_cache_root is None:
            parser.error("Trainable encoders require --data-cache-root with --meta-cache-root")

        def open_split(split, limit):
            points = paired_dataset(args.vod_root, split, args.radar_variant,
                                    data_cache_root=args.data_cache_root)
            expected = cache_manifest(
                split=split, frame_ids=[frame.frame_id for frame in points.frames],
                vod_root=args.vod_root, radar_variant=args.radar_variant,
                radar_encoder=args.radar_encoder, lidar_encoder=args.lidar_encoder,
                radius_m=config.radius_m, max_candidates=config.max_candidates)
            cache = MetaOccupancyCache(args.meta_cache_root / split, expected)
            cache.check_complete()
            joined = MetaOccupancyDataset(points, cache)
            if limit is not None:
                joined.frames = joined.frames[:limit]
            return joined

        train = open_split("train", args.train_limit)
        val = open_split("val", args.val_limit)
        collate = collate_meta_cache
    else:
        train = paired_dataset(args.vod_root, "train", args.radar_variant,
                               augmentation=args.augmentation, seed=args.augmentation_seed,
                               limit=args.train_limit, data_cache_root=args.data_cache_root)
        val = paired_dataset(args.vod_root, "val", args.radar_variant,
                             limit=args.val_limit, data_cache_root=args.data_cache_root)
        collate = collate_stage1
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True),
        "val": DataLoader(val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate,
                          pin_memory=True),
        "quick": DataLoader(Subset(val, list(range(min(args.quick_val_frames, len(val))))),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True),
    }
    if len(loaders["train"]) < 2 or len(loaders["val"]) < 1:
        parser.error("Meta training needs at least two train batches and one val batch")
    teacher_config = TeacherConfig(radius_m=config.radius_m,
                                   max_candidates=config.max_candidates,
                                   sigma_m=config.teacher_sigma_m,
                                   object_weight=0, geometry_weight=0,
                                   contrastive_weight=0, fusion_weight=0)
    student_config = StudentConfig(positive_weight=1, dice_weight=0,
                                   lambda_teacher_evidence=0,
                                   decoder_from_conditioned=True)
    teacher = CrossModalTeacher(str(args.radar_encoder), str(args.lidar_encoder),
                                teacher_config, trainable_encoders=True).to(args.device)
    student = RadarOnlyStudent(str(args.radar_encoder), student_config,
                               trainable_encoder=True).to(args.device)
    if args.teacher_init:
        pretrained, source = load_teacher(args.teacher_init, args.device)
        if (source["radar_sha256"] != _hash(args.radar_encoder)
                or source["lidar_sha256"] != _hash(args.lidar_encoder)
                or source["radar_variant"] != args.radar_variant
                or source.get("lidar_fov") != RADAR_FOV_SPEC):
            raise ValueError("Teacher initialization uses different encoder or radar data")
        teacher.load_state_dict(pretrained.state_dict(), strict=True)
        del pretrained
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)
    student_optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.student_lr)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    if history.exists() and not args.resume:
        parser.error("Existing metrics require --resume or a new output directory")
    start = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("version") != VERSION or saved["meta_config"] != asdict(config)
                or saved["teacher_config"] != asdict(teacher_config)
                or saved["student_config"] != asdict(student_config)
                or saved["radar_sha256"] != _hash(args.radar_encoder)
                or saved["lidar_sha256"] != _hash(args.lidar_encoder)
                or saved["radar_variant"] != args.radar_variant
                or saved["augmentation"] != args.augmentation.to_dict()):
            raise ValueError("Resume configuration, source weights, or data changed")
        teacher.load_state_dict(saved["teacher"], strict=True)
        student.load_state_dict(saved["student"], strict=True)
        teacher_optimizer.load_state_dict(saved["teacher_optimizer"])
        student_optimizer.load_state_dict(saved["student_optimizer"])
        start = saved["epoch"] + 1
    best = float("-inf")
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            previous = json.loads(line)
            if previous.get("validation_scope", "full") != "full":
                continue
            value = previous["val"].get("iou_full_support")
            if value is not None:
                best = max(best, value)
    for epoch in range(start, args.epochs + 1):
        training = _train_epoch(student, teacher, train, loaders["train"],
                                student_optimizer, teacher_optimizer, config,
                                epoch, args.warmup_epochs, args.device)
        evaluate_scan = epoch % args.scan_metrics_every == 0
        full = (epoch % args.full_val_every == 0 or evaluate_scan
                or epoch == args.epochs)
        validation = _validate(student, loaders["val" if full else "quick"],
                               config, args.device,
                               evaluate_scan_metrics=evaluate_scan)
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "radius_m": config.radius_m, "threshold": config.threshold,
                  "scan_metrics_every": args.scan_metrics_every}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _save(args.output_root / "last.ckpt", teacher, student, teacher_optimizer,
              student_optimizer, epoch, config, args, record)
        _save_student_only(args.output_root / "last_student.ckpt", student,
                           epoch, config, args)
        score = validation["iou_full_support"]
        if full and score is not None and score > best:
            best = score
            _save(args.output_root / "best_iou.ckpt", teacher, student,
                  teacher_optimizer, student_optimizer, epoch, config, args, record)
            _save_student_only(args.output_root / "best_student.ckpt", student,
                               epoch, config, args)
        scan_text = (f" IoU exact={validation['iou_exact']}"
                     f" IoU@0.2m={validation['iou_at_0_2m']}"
                     f" F1 exact={validation['f1_exact']}"
                     f" F1@0.2m={validation['f1_at_0_2m']}"
                     if evaluate_scan else "")
        print(f"Epoch {epoch:03d}: train BCE={training['map_bce']} "
              f"val BCE={validation['map_bce']} full-support IoU={score} "
              f"coverage={validation['candidate_coverage']} "
              f"meta_steps={training['meta_steps']}{scan_text}", flush=True)


if __name__ == "__main__":
    main()
