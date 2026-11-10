"""Sweep radar self-reconstruction thresholds on validation or score test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.radar_lidar_crossmodal_v1.filtered_radar_cache import FilteredRadarCache
from models.radar_lidar_crossmodal_v1.radar_self_autoencoder import (
    RadarSelfAutoencoder, collate_filtered_radar)
from models.radar_lidar_crossmodal_v1.train_radar_self_autoencoder import _epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.thresholds or any(not 0 < t < 1 for t in args.thresholds):
        parser.error("Thresholds must lie strictly between zero and one")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity", {})
    if identity.get("version") != "radar_self_point_autoencoder_v1":
        parser.error("Checkpoint is not the radar-to-radar point autoencoder")
    cache = FilteredRadarCache(args.cache_root, args.raw_root, args.split)
    cache.check_complete()
    loader = DataLoader(cache, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_filtered_radar,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    model = RadarSelfAutoencoder(None, slots=identity["slots"]).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for threshold in args.thresholds:
            metrics = _epoch(model, loader, None, args.device,
                             threshold=threshold,
                             name=f"Radar self {args.split} @{threshold:g}")
            rows.append({"threshold": threshold, **metrics})
            print(f"{args.split} threshold={threshold:g}: "
                  f"voxel_IoU={metrics['voxel_iou']:.2%} "
                  f"point_F1@0.2m={metrics['point_f1_0p2m']:.2%} "
                  f"P/R={metrics['point_precision_0p2m']:.2%}/"
                  f"{metrics['point_recall_0p2m']:.2%} "
                  f"point_ceiling={metrics['point_ceiling']:.2%}", flush=True)
    best_f1 = max(rows, key=lambda row: row["point_f1_0p2m"])
    best_iou = max(rows, key=lambda row: row["voxel_iou"])
    report = {"checkpoint_epoch": saved["epoch"], "split": args.split,
              "frames": len(cache), "input_equals_target_identity_baseline": 1.0,
              "best_point_f1_threshold": best_f1["threshold"],
              "best_voxel_iou_threshold": best_iou["threshold"], "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
