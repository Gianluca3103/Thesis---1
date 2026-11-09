"""Train direct radar encoder/decoder with the existing LiDAR occupancy labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC

from .meta_cache import (
    VERSION as CACHE_VERSION, CachedOccupancyTargets, MetaOccupancyCache,
    MetaOccupancyDataset, _sha256, _source_hash,
)
from .meta_occupancy import (
    MetaOccupancyConfig, occupancy_loss, scan_overlap_counts_observed,
)
from .pipeline_data import paired_dataset
from .radar_autoencoder import RadarOccupancyAutoencoder
from .student import StudentConfig
from .train_meta_occupancy import EpochMetrics
from .radar_voxel_gate import SUPPORT_POLICY as FULL_RADAR_SUPPORT_POLICY


VERSION = "radar_direct_occupancy_autoencoder_v1"


def collate_radar_autoencoder(rows: list[dict]) -> dict:
    """Read only radar input and cached LiDAR labels; omit teacher neighbors."""
    if not rows or any("geometry" not in row for row in rows):
        raise ValueError("Radar autoencoder requires cached geometry rows")
    maximum = max(len(row["radar"]) for row in rows)
    radar = torch.zeros((len(rows), maximum, 7), dtype=torch.float32)
    radar_valid = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for i, row in enumerate(rows):
        count = len(row["radar"])
        radar[i, :count] = row["radar"]
        radar_valid[i, :count] = True

    def coordinates(name):
        return torch.cat([
            torch.cat((torch.full_like(row["geometry"][name][:, :1], i),
                       row["geometry"][name][:, 1:]), dim=1)
            for i, row in enumerate(rows)
        ]).long()

    return {"frame_id": [row["frame_id"] for row in rows],
            "radar": radar, "radar_valid": radar_valid,
            "candidate_coords": coordinates("candidate_coords"),
            "observed_coords": coordinates("observed_coords"),
            "occupied": torch.cat([row["geometry"]["occupied"] for row in rows]),
            "known_free": torch.cat([row["geometry"]["known_free"] for row in rows]),
            "support": torch.cat([row["geometry"]["support"] for row in rows]),
            "clean_supported_points": sum(row["geometry"]["clean_supported_points"]
                                          for row in rows),
            "represented_clean_points_supported": sum(
                row["geometry"]["represented_clean_points_supported"] for row in rows),
            "clean_supported_voxels": sum(len(row["geometry"]["observed_coords"])
                                          for row in rows)}


def _move(raw: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor)
            else value for key, value in raw.items()}


def _epoch(model, loader, optimizer, config, device, *, name, scan=False):
    training = optimizer is not None
    model.train(training)
    summary = EpochMetrics()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw in tqdm(loader, desc=name, unit="batch", dynamic_ncols=True):
            batch = _move(raw, device)
            output = model(batch["radar"], batch["radar_valid"])
            if not torch.equal(output.confidence_logits.coords,
                               batch["candidate_coords"]):
                raise ValueError("Radar decoder sites differ from cached LiDAR labels")
            targets = CachedOccupancyTargets(batch["occupied"], batch["known_free"],
                                              batch["represented_clean_points_supported"])
            loss, stats = occupancy_loss(output, targets, batch["support"],
                                         config.threshold)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            summary.add(stats, batch, batch["clean_supported_points"],
                        batch["clean_supported_voxels"])
            if scan:
                predicted = output.confidence_logits.coords[
                    batch["support"] &
                    (torch.sigmoid(output.confidence_logits.features[:, 0])
                     >= config.threshold)]
                counts = scan_overlap_counts_observed(
                    predicted, batch["observed_coords"])
                summary.add_scan(counts)
    return summary.finish()


def _atomic_save(path: Path, payload: dict):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_score(validation: dict, metric: str) -> float | None:
    if metric == "exact":
        if "iou_exact" not in validation:
            raise ValueError("Exact-IoU checkpoint selection requires scan validation")
        return validation["iou_exact"]
    if metric == "full_support":
        return validation["iou_full_support"]
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def load_radar_autoencoder(path: Path, device: str = "cuda"):
    """Inference uses only the trained radar model checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") not in (VERSION, VERSION + "_student"):
        raise ValueError("Expected direct radar occupancy autoencoder checkpoint")
    model = RadarOccupancyAutoencoder(
        config=StudentConfig(**saved["student_config"]))
    model.load_state_dict(saved["student"], strict=True)
    return model.to(device).eval(), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--meta-cache-root", type=Path, required=True)
    parser.add_argument("--gate-support-root", type=Path,
                        help="Require cached >=12 radar voxels within 1.6 m")
    parser.add_argument("--full-radar-gate", action="store_true",
                        help="Train only in the full encoder-grid >=12/1.6 m gate")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/crossmodal_meta_occupancy.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--student-lr", type=float, default=2e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--scan-metrics-every", type=int, default=5)
    parser.add_argument("--checkpoint-metric", choices=("full_support", "exact"),
                        default="full_support",
                        help="Select the best-IoU checkpoint using this validation metric")
    parser.add_argument("--quick-val-frames", type=int, default=32)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.full_radar_gate and args.gate_support_root is not None:
        parser.error("Full-grid and Stage-II ROI radar gates are exclusive")
    if args.checkpoint_metric == "exact" and args.scan_metrics_every != 1:
        parser.error("Exact-IoU checkpoint selection needs scan metrics every epoch")
    if (args.device != "cuda" or not torch.cuda.is_available()
            or args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0
            or args.student_lr <= 0 or args.scan_metrics_every < 1
            or args.quick_val_frames < 1 or args.full_val_every < 1
            or args.train_limit is not None and args.train_limit < 1
            or args.val_limit is not None and args.val_limit < 1):
        parser.error("Radar-only training requires CUDA and valid settings")
    config = MetaOccupancyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    def open_split(split: str, limit: int | None):
        points = paired_dataset(args.vod_root, split, args.radar_variant,
                                include_clean=False,
                                data_cache_root=args.data_cache_root,
                                full_lidar=args.full_radar_gate,
                                check_complete=limit is None)
        root = args.meta_cache_root / split
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"version": CACHE_VERSION, "split": split,
                    "frame_ids": [frame.frame_id for frame in points.frames],
                    "vod_root": str(args.vod_root.resolve()),
                    "radar_variant": args.radar_variant,
                    "radar_initial_sha256": _sha256(args.radar_encoder),
                    "radius_m": config.radius_m,
                    "max_candidates": config.max_candidates,
                    "source_hash": _source_hash()}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing {split} meta geometry cache differs: {manifest_path}")
        fixed_hash = manifest.get("fixed_frame_manifest_sha256")
        if ((args.full_radar_gate and fixed_hash is None) or
                fixed_hash is not None and fixed_hash !=
                _sha256(args.data_cache_root / split / "manifest.json")):
            raise ValueError(f"{split} geometry uses a different clean LiDAR cache")
        if args.full_radar_gate:
            if manifest.get("support_policy") != FULL_RADAR_SUPPORT_POLICY:
                raise ValueError(f"{split} cache is not the full-grid 12-voxel gate")
        elif args.gate_support_root is not None:
            support_manifest = args.gate_support_root / split / "manifest.json"
            if (manifest.get("support_policy") != "radar12vox_1p6m_stage2_roi_v1"
                    or manifest.get("gate_support_manifest_sha256") !=
                    _sha256(support_manifest)):
                raise ValueError(f"{split} cache is not bound to the 12-voxel gate")
        elif manifest.get("support_policy") is not None:
            raise ValueError("Gated geometry requires --gate-support-root")
        cache = MetaOccupancyCache(root, manifest)
        joined = MetaOccupancyDataset(points, cache)
        selected = (Subset(joined, range(min(limit, len(joined))))
                    if limit is not None else joined)
        if limit is None:
            cache.check_complete()
        else:
            missing = [points.frames[index].frame_id for index in range(len(selected))
                       if not cache.path(points.frames[index].frame_id).is_file()]
            if missing:
                raise FileNotFoundError(f"{split} geometry lacks a selected frame: {missing[0]}")
        return selected, _sha256(manifest_path)
    train, train_manifest = open_split("train", args.train_limit)
    val, val_manifest = open_split("val", args.val_limit)
    if not len(train) or not len(val):
        parser.error("Radar-only training needs nonempty splits")
    loaders = {
        "train": DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=collate_radar_autoencoder, pin_memory=True),
        "val": DataLoader(val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=collate_radar_autoencoder, pin_memory=True),
        "quick": DataLoader(Subset(val, range(min(args.quick_val_frames, len(val)))),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_radar_autoencoder, pin_memory=True),
    }
    student = RadarOccupancyAutoencoder(str(args.radar_encoder)).to(args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr)
    identity = {"meta_config": asdict(config),
                "student_config": asdict(student.config),
                "radar_sha256": _sha256(args.radar_encoder),
                "radar_variant": args.radar_variant,
                "lidar_fov": None if args.full_radar_gate else RADAR_FOV_SPEC,
                "clean_lidar_scope": ("uncropped" if args.full_radar_gate
                                      else "forward_radar_sector"),
                "radius_m": config.radius_m, "threshold": config.threshold,
                "train_manifest_sha256": train_manifest,
                "val_manifest_sha256": val_manifest,
                "student_lr": args.student_lr, "batch_size": args.batch_size}
    if args.gate_support_root is not None:
        identity["support_policy"] = "radar12vox_1p6m_stage2_roi_v1"
        identity["gate_support_manifest_sha256"] = {
            split: _sha256(args.gate_support_root / split / "manifest.json")
            for split in ("train", "val")}
    if args.full_radar_gate:
        identity["support_policy"] = FULL_RADAR_SUPPORT_POLICY
    if args.checkpoint_metric == "exact":
        identity["checkpoint_metric"] = "exact"
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = args.output_root / "metrics.jsonl"
    if history.exists() and args.resume is None:
        parser.error("Existing metrics require --resume or a new output root")
    best_key = ("best_iou_exact" if args.checkpoint_metric == "exact"
                else "best_iou_full_support")
    start, best = 1, float("-inf")
    best_val_bce = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("version") != VERSION or any(saved.get(key) != value
                                                 for key, value in identity.items()):
            raise ValueError("Resume checkpoint differs from this radar-only experiment")
        student.load_state_dict(saved["student"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved[best_key])
        best_val_bce = float(saved.get("best_val_bce", float("inf")))
    for epoch in range(start, args.epochs + 1):
        training = _epoch(student, loaders["train"], optimizer, config, args.device,
                          name=f"Radar AE train {epoch:03d}")
        evaluate_scan = epoch % args.scan_metrics_every == 0
        full = epoch % args.full_val_every == 0 or evaluate_scan or epoch == args.epochs
        validation = _epoch(student, loaders["val" if full else "quick"], None,
                            config, args.device, name=f"Radar AE val {epoch:03d}",
                            scan=evaluate_scan)
        score = _checkpoint_score(validation, args.checkpoint_metric)
        improved = full and score is not None and score > best
        if improved:
            best = score
        val_bce = validation["map_bce"]
        bce_improved = full and val_bce is not None and val_bce < best_val_bce
        if bce_improved:
            best_val_bce = val_bce
        record = {"epoch": epoch, "train": training, "val": validation,
                  "validation_scope": "full" if full else "quick",
                  "scan_metrics": evaluate_scan,
                  best_key: best,
                  "best_val_bce": best_val_bce,
                  "ablation": "radar_encoder_decoder_lidar_occupancy_supervision"}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        base = {"version": VERSION, **identity, "epoch": epoch,
                best_key: best,
                "best_val_bce": best_val_bce,
                "student": student.state_dict(), "optimizer": optimizer.state_dict(),
                "record": record}
        only = {key: value for key, value in base.items() if key != "optimizer"}
        only["version"] = VERSION + "_student"
        _atomic_save(args.output_root / "last.ckpt", base)
        _atomic_save(args.output_root / "last_student.ckpt", only)
        if improved:
            _atomic_save(args.output_root / "best_iou.ckpt", base)
            _atomic_save(args.output_root / "best_student.ckpt", only)
        if bce_improved:
            _atomic_save(args.output_root / "best_bce.ckpt", base)
            _atomic_save(args.output_root / "best_bce_student.ckpt", only)
        if evaluate_scan:
            print(f"Epoch {epoch:03d}: train_BCE={training['map_bce']:.4f} "
                  f"val_BCE={validation['map_bce']:.4f} "
                  f"eligible_exact_IoU@{config.threshold:g}="
                  f"{validation['iou_exact']:.2%} "
                  f"eligible_IoU@0.2m@{config.threshold:g}="
                  f"{validation['iou_at_0_2m']:.2%} "
                  f"val_frames={validation['frames']} "
                  f"best_eligible_{args.checkpoint_metric}_IoU="
                  f"{best:.2%}", flush=True)
        else:
            print(f"Epoch {epoch:03d}: train_BCE={training['map_bce']:.4f} "
                  f"val_BCE={validation['map_bce']:.4f} "
                  f"known/full_IoU@{config.threshold:g}="
                  f"{validation['iou_known_supported']:.2%}/"
                  f"{validation['iou_full_support']:.2%} "
                  f"val_frames={validation['frames']}", flush=True)


if __name__ == "__main__":
    main()
