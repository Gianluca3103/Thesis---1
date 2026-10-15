"""Independent frozen-encoder teacher training on aligned VoD pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig
from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC

from .encoder import GRID
from .pipeline_data import annotate_batch, collate_stage1, paired_dataset
from .history import log_record, open_writer, overfitting_warning
from .plots import update_plots
from .teacher import CrossModalTeacher, TeacherConfig
from .teacher_static_cache import TeacherStaticCache
from .teacher_loss import teacher_objective


LOSS_KEYS = ("object", "geometry", "contrastive", "fusion")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _encoder_snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key.startswith(("radar_encoder.", "lidar_encoder."))}


def _assert_frozen(model, snapshot):
    current = model.state_dict()
    changed = [key for key, value in snapshot.items()
               if not torch.equal(value, current[key].detach().cpu())]
    if changed:
        raise RuntimeError(f"Frozen original encoder state changed: {changed[:3]}")


def _prior_full_records(root):
    history = root / "metrics.jsonl"
    if not history.is_file():
        return []
    return [row for line in history.read_text(encoding="utf-8").splitlines()
            if line and (row := json.loads(line)).get("validation_scope") == "full"]


class Summary:
    def __init__(self, config):
        self.config = config
        self.sums = {}
        self.counts = {}
        self.samples = 0
        self.aux = {name: 0 for name in (
            "object_queries", "excluded_object_queries", "geometry_eligible",
            "radar_active_sites", "lidar_active_sites")}

    def add(self, terms, metrics, batch_count):
        self.samples += batch_count
        for key in self.aux:
            self.aux[key] += metrics[key]
        for key, counted in {**terms, **metrics}.items():
            if isinstance(counted, int):
                continue
            self.sums[key] = self.sums.get(key, 0.) + float(counted.numerator.detach())
            self.counts[key] = self.counts.get(key, 0) + counted.denominator

    def finish(self):
        means = {key: self.sums[key] / self.counts[key] if self.counts[key] else None
                 for key in self.sums}
        total = sum(getattr(self.config, key + "_weight") * (means[key] or 0.)
                    for key in LOSS_KEYS)
        cosine_gap = (means.get("positive_cosine") - means.get("negative_cosine")
                      if means.get("positive_cosine") is not None and
                      means.get("negative_cosine") is not None else None)
        return {"loss_total": total, "cosine_gap": cosine_gap,
                **{f"loss_{key}": means.get(key) for key in LOSS_KEYS},
                **{f"weighted_loss_{key}": getattr(self.config, key + "_weight") * (means.get(key) or 0.)
                   for key in LOSS_KEYS},
                **{key: value for key, value in means.items() if key not in LOSS_KEYS},
                "counts": self.counts, "frames": self.samples, **self.aux}


def _epoch(model, dataset, loader, device, optimizer=None, epoch=0,
           static_cache=None):
    training = optimizer is not None
    model.train(training)
    if training:
        dataset.set_epoch(epoch)
    summary = Summary(model.config)
    progress = tqdm(loader, desc=("Teacher train" if training else "Teacher val") + f" {epoch:03d}",
                    unit="batch", dynamic_ncols=True, leave=training)
    for batch_index, raw in enumerate(progress, 1):
        batch = _move(raw, device)
        with torch.set_grad_enabled(training):
            if static_cache is None:
                output = model(batch["radar"], batch["radar_valid"],
                               batch["clean_lidar"], batch["clean_lidar_valid"])
                radar_ids, lidar_ids, classes = annotate_batch(
                    dataset, batch, output.radar_frozen, output.lidar_frozen)
            else:
                radar, lidar, neighbors, radar_ids, lidar_ids, classes = static_cache.load_batch(
                    batch["frame_id"], device)
                output = model.core(radar, lidar, neighbors)
            total, terms, metrics = teacher_objective(output, radar_ids, lidar_ids,
                                                      model.config, class_by_id=classes)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.core.parameters(), 5.)
                optimizer.step()
        summary.add(terms, metrics, len(batch["frame_id"]))
    return summary.finish()


def _save(path, model, optimizer, scheduler, epoch, step, args, validation):
    state = {"version": "crossmodal_teacher_v4_stage1_native_sparse_raw64", "config": asdict(model.config),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
             "radar_checkpoint": str(args.radar_encoder.resolve()),
             "lidar_checkpoint": str(args.lidar_encoder.resolve()),
             "radar_sha256": _hash(args.radar_encoder),
             "lidar_sha256": _hash(args.lidar_encoder),
             "radar_variant": args.radar_variant,
             "lidar_fov": RADAR_FOV_SPEC,
             "augmentation": args.augmentation.to_dict(),
             "augmentation_seed": args.augmentation_seed,
             "validation": validation}
    torch.save(state, path)


def load_teacher(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != "crossmodal_teacher_v4_stage1_native_sparse_raw64":
        raise ValueError("Expected versioned native Stage-I sparse teacher checkpoint")
    for key in ("radar", "lidar"):
        source = saved[f"{key}_checkpoint"]
        if not Path(source).is_file() or _hash(source) != saved[f"{key}_sha256"]:
            raise FileNotFoundError(f"Frozen {key} encoder checkpoint missing or changed: {source}")
    model = CrossModalTeacher(saved["radar_checkpoint"], saved["lidar_checkpoint"],
                              TeacherConfig(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def validate_lidar_pretraining_fov(encoder_path: Path) -> None:
    """Prevent training the cropped teacher with old 360° LiDAR weights."""
    record = encoder_path.with_name("best_pretrain.ckpt")
    if not record.is_file():
        raise FileNotFoundError(
            f"LiDAR encoder training metadata is missing: {record}")
    saved = torch.load(record, map_location="cpu", weights_only=False)
    if saved.get("modality") != "lidar" or saved.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("LiDAR encoder was not pretrained with the configured radar field of view")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--lidar-encoder", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--static-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/crossmodal_teacher_raw64.json"))
    parser.add_argument("--online-augmentation-config", type=Path)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--quick-val-frames", type=int, default=64)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("The unmodified Downloads encoder requires CUDA")
    config = TeacherConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    validate_lidar_pretraining_fov(args.lidar_encoder)
    args.augmentation = GeometricAugmentationConfig.from_dict(
        json.loads(args.online_augmentation_config.read_text(encoding="utf-8"))
        if args.online_augmentation_config else None)
    if args.static_cache_root and args.augmentation.enabled:
        parser.error("Teacher static cache requires geometric augmentation to be disabled")
    train = paired_dataset(args.vod_root, "train", args.radar_variant,
                           augmentation=args.augmentation, seed=args.augmentation_seed,
                           limit=args.train_limit, data_cache_root=args.data_cache_root,
                           id_only=args.static_cache_root is not None)
    val = paired_dataset(args.vod_root, "val", args.radar_variant,
                         limit=args.val_limit, data_cache_root=args.data_cache_root,
                         id_only=args.static_cache_root is not None)
    static_caches = None
    if args.static_cache_root:
        static_caches = {
            split: TeacherStaticCache(args.static_cache_root, split=split,
                                      radar_variant=args.radar_variant,
                                      radar_sha256=_hash(args.radar_encoder),
                                      lidar_sha256=_hash(args.lidar_encoder),
                                      radius_m=config.radius_m,
                                      max_candidates=config.max_candidates)
            for split in ("train", "val")
        }
        for split, dataset in (("train", train), ("val", val)):
            missing = [frame.frame_id for frame in dataset.frames
                       if not static_caches[split].path(frame.frame_id).is_file()]
            if missing:
                raise FileNotFoundError(f"Teacher {split} cache misses {len(missing)} frames; "
                                        f"first: {missing[0]}")
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_stage1, pin_memory=True),
        "quick": DataLoader(Subset(val, list(range(min(args.quick_val_frames, len(val))))),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_stage1, pin_memory=True),
        "full": DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers,
                           collate_fn=collate_stage1, pin_memory=True),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "metrics.jsonl").exists() and not args.resume:
        parser.error("Existing teacher metrics require --resume or a new output directory")
    writer = open_writer(args.output_root, args.tensorboard)
    model = CrossModalTeacher(str(args.radar_encoder), str(args.lidar_encoder), config).to(args.device)
    frozen = _encoder_snapshot(model)
    optimizer = torch.optim.AdamW(model.core.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, step = 1, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved["config"] != asdict(config) or saved["radar_sha256"] != _hash(args.radar_encoder)
                or saved["lidar_sha256"] != _hash(args.lidar_encoder)
                or saved["augmentation"] != args.augmentation.to_dict()
                or saved.get("lidar_fov") != RADAR_FOV_SPEC):
            raise ValueError("Teacher resume configuration or frozen encoder sources changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, step = saved["epoch"] + 1, saved["step"]
    best_r1, best_fusion = float("-inf"), float("inf")
    for row in (_prior_full_records(args.output_root) if args.resume else []):
        r1, mae = row["val"].get("object_r1"), row["val"].get("fusion_mae")
        if mae is not None:
            best_fusion = min(best_fusion, mae)
        if r1 is not None and mae is not None and mae <= best_fusion * 1.05:
            best_r1 = max(best_r1, r1)
    for epoch in range(start, args.epochs + 1):
        training = _epoch(model, train, loaders["train"], args.device, optimizer,
                          epoch, static_cache=static_caches["train"] if static_caches else None)
        step += len(loaders["train"])
        scheduler.step()
        full = epoch % args.full_val_every == 0 or epoch == args.epochs
        validation = _epoch(model, val, loaders["full" if full else "quick"], args.device,
                            epoch=epoch, static_cache=static_caches["val"] if static_caches else None)
        _assert_frozen(model, frozen)
        gap = validation["loss_total"] - training["loss_total"]
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "loss_gap": gap,
                  "relative_loss_gap": gap / (abs(training["loss_total"]) + 1e-8)}
        with (args.output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record)
        update_plots(args.output_root, "teacher")
        warning = overfitting_warning(args.output_root / "metrics.jsonl")
        if warning:
            print(f"Overfitting warning: {warning}", flush=True)
        (args.output_root / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        _save(args.output_root / "last.ckpt", model, optimizer, scheduler,
              epoch, step, args, record)
        if full:
            r1, mae = validation.get("object_r1"), validation.get("fusion_mae")
            if mae is not None and mae < best_fusion:
                best_fusion = mae
                _save(args.output_root / "best_fusion.ckpt", model, optimizer, scheduler,
                      epoch, step, args, record)
            if (r1 is not None and mae is not None and r1 > best_r1
                    and mae <= best_fusion * 1.05):
                best_r1 = r1
                _save(args.output_root / "best_object_r1.ckpt", model, optimizer, scheduler,
                      epoch, step, args, record)
        print(f"Teacher {epoch:03d}/{args.epochs:03d} train {training['loss_total']:.4f} "
              f"val {validation['loss_total']:.4f} object R@1 {validation.get('object_r1')} "
              f"fusion MAE {validation.get('fusion_mae')} scope {'full' if full else 'quick'}",
              flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
