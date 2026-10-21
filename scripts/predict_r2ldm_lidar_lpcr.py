"""Reconstruct XYZ point clouds with an independently trained LiDAR LPCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.r2ldm_lidar_lpcr import (
    R2LDMGrid, R2LDMLidarLPCR, VoDLidarLPCRDataset, collate_lpcr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        help="Optional independently built LiDAR voxel cache")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema",
                        help="R2LDM inference uses EMA weights by default")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 < args.threshold < 1:
        parser.error("Reconstruction needs CUDA and a threshold in (0,1)")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("version") != "r2ldm_lidar_lpcr_independent_v2":
        raise ValueError("Checkpoint is not from the independent LiDAR LPCR trainer")
    grid = R2LDMGrid(**saved["grid"])
    grid.validate()
    model = R2LDMLidarLPCR(grid, offset_activation=saved["offset_activation"])
    model.load_state_dict(saved["model"])
    if args.weights == "ema":
        from ema_pytorch import EMA
        training = saved["training"]
        ema = EMA(model, beta=training["ema_decay"],
                  update_every=training["ema_update_every"],
                  update_after_step=training["ema_update_after_step"])
        ema.load_state_dict(saved["ema"])
        model = ema.ema_model
    model.to(args.device).eval()
    dataset = VoDLidarLPCRDataset(args.vod_root, args.split, grid=grid,
                                  limit=args.limit, cache_root=args.cache_root)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=collate_lpcr)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="LPCR reconstruct", unit="frame", dynamic_ncols=True):
            means = batch["means"].to(args.device)
            coords = batch["coords"].to(args.device)
            output = model(means, coords, batch_size=1)
            points = model.reconstruct(output, args.threshold)[0]
            frame_id = batch["frame_id"][0]
            path = args.output_root / f"{frame_id}.bin"
            points.cpu().numpy().astype("<f4", copy=False).tofile(path)
            rows.append({"frame_id": frame_id, "input_voxels": len(means),
                         "reconstructed_points": len(points), "path": str(path)})
            print(f"{frame_id}: {len(points)} reconstructed XYZ points | {path}", flush=True)
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "weights": args.weights,
                "split": args.split,
                "threshold": args.threshold, "grid": saved["grid"], "frames": rows}
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
