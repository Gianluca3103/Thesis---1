"""Train Stage I to prune, never expand, the exact radar-supported domain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    missing_target, target_mask)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import (
    RadarPruner, RadarPrunerConfig, SUPPORT_MIN_VOXELS,
    checkpoint_payload, version_for_radius)
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FOG2_TYPES, MILD_FOG2_VERSION)


def _one(rows: list[dict]) -> dict:
    return rows[0]


def _loss(logits: torch.Tensor, positive: torch.Tensor,
          config: RadarPrunerConfig) -> torch.Tensor:
    if not len(logits):
        return logits.sum() * 0
    pos = F.softplus(-logits[positive]).mean() if bool(positive.any()) else logits.sum() * 0
    negatives = logits[~positive]
    neg = F.softplus(negatives).mean() if len(negatives) else logits.sum() * 0
    hard = (F.softplus(torch.topk(negatives, min(10000, len(negatives))).values).mean()
            if len(negatives) else logits.sum() * 0)
    return pos + config.negative_loss_weight * neg + 0.15 * hard


def _operating_point(positive_bins: np.ndarray, negative_bins: np.ndarray,
                     *, desired_recall: float = 0.95) -> dict:
    tp = positive_bins[::-1].cumsum()[::-1]
    fp = negative_bins[::-1].cumsum()[::-1]
    total = int(tp[0])
    if not total:
        return {"threshold": None, "precision": 0.0, "eligible_recall": 0.0,
                "selected": 0, "tp": 0}
    feasible = np.flatnonzero(tp / total >= desired_recall)
    if not len(feasible):
        raise RuntimeError("Threshold sweep omitted its zero threshold")
    precision = tp / np.maximum(tp + fp, 1)
    best = int(max(feasible, key=lambda i: (precision[i], i)))
    return {"threshold": best / 1000, "precision": float(precision[best]),
            "eligible_recall": float(tp[best] / total),
            "selected": int(tp[best] + fp[best]), "tp": int(tp[best])}


def _epoch(model: RadarPruner, dataset, indices: list[int] | None,
           support_cache: RadarSupportCache, faulty_root: Path, split: str,
           device: str, optimizer, *, workers: int, epoch: int,
           frame_faults: dict[str, str], desired_recall: float = 0.95) -> dict:
    training = optimizer is not None
    model.train(training)
    selected = Subset(dataset, indices) if indices is not None else dataset
    loader = DataLoader(selected, batch_size=1, shuffle=training,
                        collate_fn=_one, num_workers=workers,
                        pin_memory=device.startswith("cuda"))
    totals = {"frames": 0, "loss": 0.0, "candidate_cells": 0,
              "all_missing": 0, "eligible_missing": 0}
    bins_pos = np.zeros(1001, dtype=np.int64)
    bins_neg = np.zeros(1001, dtype=np.int64)
    by_fault = {name: {"frames": 0, "all_missing": 0, "eligible_missing": 0}
                for name in MILD_FOG2_TYPES}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for row in tqdm(loader, desc=f"Radar pruner {split} {epoch:03d}", unit="frame"):
            frame_id = row["frame_id"]
            radar = row["radar"].to(device, non_blocking=True)
            faulty = load_faulty(faulty_root, split, frame_id).to(device, non_blocking=True)
            clean = row["clean_lidar"][:, :3].to(device, non_blocking=True)
            support = support_cache.load(frame_id)
            target = missing_target(clean, faulty)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                output = model(radar, faulty, support)
                positive = target_mask(output["coords"], target, 1)
                loss = _loss(output["logits"], positive, model.config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            count = int(positive.sum())
            totals["frames"] += 1
            totals["loss"] += float(loss.detach())
            totals["candidate_cells"] += len(output["coords"])
            totals["all_missing"] += len(target)
            totals["eligible_missing"] += count
            fault = frame_faults[frame_id]
            by_fault[fault]["frames"] += 1
            by_fault[fault]["all_missing"] += len(target)
            by_fault[fault]["eligible_missing"] += count
            if not training and len(output["coords"]):
                bin_index = torch.clamp(
                    (output["logits"].float().sigmoid() * 1000).long(), 0, 1000)
                bins_pos += torch.bincount(bin_index[positive],
                                            minlength=1001).cpu().numpy()
                bins_neg += torch.bincount(bin_index[~positive],
                                            minlength=1001).cpu().numpy()
    frames = max(totals["frames"], 1)
    report = {"frames": totals["frames"], "mean_loss": totals["loss"] / frames,
              "mean_candidate_cells": totals["candidate_cells"] / frames,
              "all_missing": totals["all_missing"],
              "eligible_missing": totals["eligible_missing"],
              "geometry_full_recall_ceiling": (totals["eligible_missing"] /
                                                totals["all_missing"]
                                                if totals["all_missing"] else 1.0),
              "by_fault": by_fault}
    if not training:
        point = _operating_point(bins_pos, bins_neg,
                                 desired_recall=desired_recall)
        report["target_eligible_recall"] = desired_recall
        report["precision_at_target_eligible_recall"] = point
        report["full_recall_at_selected_threshold"] = (
            point["tp"] / totals["all_missing"] if totals["all_missing"] else 1.0)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--radius-m", type=float, choices=(1.6, 3.0), default=3.0)
    parser.add_argument("--target-eligible-recall", type=float,
                        choices=(0.90, 0.95), default=0.95)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--dropout-p", type=float, default=0.0,
                        help="Channel dropout at the deepest 3D feature map")
    parser.add_argument("--negative-loss-weight", type=float, default=0.35)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (args.epochs < 1 or args.learning_rate <= 0 or args.val_limit < 1
            or args.num_workers < 0 or args.negative_loss_weight < 0 or
            not 0 <= args.dropout_p < 1 or
            not 0 < args.target_eligible_recall <= 1):
        parser.error("Invalid training settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    datasets = {split: paired_dataset(
        args.vod_root, split, args.radar_variant, include_clean=True,
        data_cache_root=args.data_cache_root) for split in ("train", "val")}
    support = {split: RadarSupportCache(
        args.support_cache_root / split,
        identity(split, [f.frame_id for f in dataset.frames], args.radar_variant,
                 data_cache_root=args.data_cache_root,
                 radius_m=args.radius_m))
        for split, dataset in datasets.items()}
    for cache in support.values():
        cache.check_complete()
    fault_maps = {}
    data_policy = {}
    for split, dataset in datasets.items():
        manifest_path = args.faulty_root / split / "manifest.json"
        manifest = json.loads(manifest_path.read_text(
            encoding="utf-8"))
        if (manifest.get("version") != MILD_FOG2_VERSION or
                manifest.get("radar_variant") != args.radar_variant or
                manifest.get("frame_ids") != [frame.frame_id for frame in dataset.frames] or
                manifest.get("fault_types") != list(MILD_FOG2_TYPES)):
            raise ValueError(f"Mild-fog2 manifest differs from pruner data: {split}")
        fault_maps[split] = manifest["frame_faults"]
        data_policy[f"faulty_{split}_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()).hexdigest()
        data_policy[f"support_{split}_sha256"] = hashlib.sha256(
            (args.support_cache_root / split / "manifest.json").read_bytes()).hexdigest()
        if set(f.frame_id for f in dataset.frames) - set(fault_maps[split]):
            raise ValueError(f"Mild faulty dataset is incomplete for {split}")
    config = RadarPrunerConfig(channels=args.channels,
                               negative_loss_weight=args.negative_loss_weight,
                               support_radius_m=args.radius_m,
                               dropout_p=args.dropout_p)
    model = RadarPruner(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    best = -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = RadarPrunerConfig(**saved.get("config", {}))
        if (saved.get("version") != version_for_radius(args.radius_m) or
                saved.get("support_radius_m") != args.radius_m or
                saved.get("support_min_distinct_radar_voxels") != SUPPORT_MIN_VOXELS or
                saved.get("data_policy") != data_policy or
                saved.get("target_eligible_recall", 0.95) != args.target_eligible_recall or
                asdict(saved_config) != asdict(config) or
                saved.get("radar_variant") != args.radar_variant):
            raise ValueError("Resume checkpoint uses another pruner policy")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get("best_score", -1))
    args.output_root.mkdir(parents=True, exist_ok=True)
    val_indices = random.Random(args.seed).sample(
        range(len(datasets["val"])), min(args.val_limit, len(datasets["val"])))
    for epoch in range(start_epoch, args.epochs + 1):
        train = _epoch(model, datasets["train"], None, support["train"],
                       args.faulty_root, "train", args.device, optimizer,
                       workers=args.num_workers, epoch=epoch,
                       frame_faults=fault_maps["train"],
                       desired_recall=args.target_eligible_recall)
        val = _epoch(model, datasets["val"], val_indices, support["val"],
                     args.faulty_root, "val", args.device, None,
                     workers=args.num_workers, epoch=epoch,
                     frame_faults=fault_maps["val"],
                     desired_recall=args.target_eligible_recall)
        score = val["precision_at_target_eligible_recall"]["precision"]
        best = max(best, score)
        payload = checkpoint_payload(
            model, epoch=epoch, radar_variant=args.radar_variant,
            optimizer=optimizer.state_dict(), best_score=best,
            data_policy=data_policy,
            target_eligible_recall=args.target_eligible_recall,
            validation=val, validation_frame_ids=[
                datasets["val"].frames[i].frame_id for i in val_indices])
        temporary = args.output_root / "last.ckpt.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output_root / "last.ckpt")
        if score >= best:
            torch.save(payload, args.output_root / "best.ckpt.tmp")
            (args.output_root / "best.ckpt.tmp").replace(
                args.output_root / "best.ckpt")
        print(json.dumps({"epoch": epoch, "train": train, "val": val},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
