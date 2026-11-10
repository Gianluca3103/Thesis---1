"""Matched filtered-radar metrics for the D-PCC adaptation on validation or test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.radar_lidar_crossmodal_v1.radar_dpcc import RadarDPCC
from models.radar_lidar_crossmodal_v1.radar_dpcc_cache import RadarDPCCCache
from models.radar_lidar_crossmodal_v1.train_radar_dpcc import (
    VERSION, collate_rows, epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    parser.add_argument("--selected-validation", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        parser.error("Thresholds must be strictly between 0 and 1")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_sha256 = digest.hexdigest()
    identity = saved.get("identity", {})
    if identity.get("version") != VERSION:
        parser.error("Not a radar D-PCC checkpoint")
    if args.split == "test" and args.selected_validation is None:
        parser.error("Test needs thresholds selected on validation")
    if args.split == "val" and args.selected_validation is not None:
        parser.error("Validation selects its own thresholds")
    if args.selected_validation:
        previous = json.loads(args.selected_validation.read_text(encoding="utf-8"))
        if (previous["split"] != "val"
                or previous["checkpoint_sha256"] != checkpoint_sha256):
            parser.error("Selected validation report differs from checkpoint")
        args.thresholds = sorted(set((previous["best_point_f1_threshold"],
                                      previous["best_voxel_iou_threshold"])))
    cache = RadarDPCCCache(args.filtered_root, args.raw_root,
                           args.geometry_root, args.split)
    cache.check_complete()
    loader = DataLoader(cache, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_rows,
                        pin_memory=True, persistent_workers=args.workers > 0)
    model = RadarDPCC(identity["width"], identity["max_children"]).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    rows = []
    with torch.no_grad():
        for threshold in args.thresholds:
            metrics = epoch(model, loader, None, args.device, threshold=threshold,
                            name=f"Radar D-PCC {args.split} @{threshold:g}")
            rows.append({"threshold": threshold, **metrics})
            print(f"{args.split} threshold={threshold:g}: "
                  f"voxel_IoU={metrics['voxel_iou']:.2%} "
                  f"point_F1@0.2m={metrics['point_f1_0p2m']:.2%} "
                  f"P/R={metrics['point_precision_0p2m']:.2%}/"
                  f"{metrics['point_recall_0p2m']:.2%} "
                  f"Chamfer={metrics['chamfer_squared_m2']:.4f}m2", flush=True)
    best_f1 = max(rows, key=lambda row: row["point_f1_0p2m"])
    best_iou = max(rows, key=lambda row: row["voxel_iou"])
    report = {"model": VERSION, "checkpoint_epoch": saved["epoch"],
              "checkpoint_sha256": checkpoint_sha256,
              "split": args.split, "frames": len(cache),
              "input_equals_target_identity_baseline": 1.0,
              "best_point_f1_threshold": best_f1["threshold"],
              "best_voxel_iou_threshold": best_iou["threshold"], "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
