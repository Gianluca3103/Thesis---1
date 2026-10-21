"""Cache frozen Stage-I gate/features and observed LiDAR labels once per frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.stage2_cache import (
    Stage2FixedCache, cache_identity,
)
from models.radar_lidar_stage2.stage2_io import load_stage1, stage1_frame


def _one(rows: list[dict]) -> dict:
    return rows[0]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--gate-threshold", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            not 0 < args.gate_threshold < 1 or args.num_workers < 0 or
            args.limit is not None and args.limit < 1):
        parser.error("Cache creation requires CUDA, a valid gate and workers")
    stage1, saved = load_stage1(args.stage1_checkpoint, args.device,
                                 args.radar_variant)
    radius_m = float(saved.get("radius_m", 3.2))
    for split in (("train", "val") if args.split == "both" else (args.split,)):
        dataset = paired_dataset(args.vod_root, split, args.radar_variant,
                                 include_clean=True,
                                 data_cache_root=args.data_cache_root,
                                 limit=args.limit)
        identity = cache_identity(args.stage1_checkpoint, args.vod_root,
                                  args.radar_variant, radius_m,
                                  args.gate_threshold, split,
                                  [frame.frame_id for frame in dataset.frames])
        cache = Stage2FixedCache(args.cache_root / split, identity, create=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_one,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
        written = reused = 0
        for row in tqdm(loader, desc=f"Stage II cache {split}", unit="frame",
                        dynamic_ncols=True):
            frame_id = row["frame_id"]
            if cache.path(frame_id).is_file():
                reused += 1
                continue
            output = stage1_frame(stage1, row["radar"].to(args.device), radius_m)
            cache.write(frame_id, output,
                        row["clean_lidar"][:, :3].to(args.device),
                        args.gate_threshold)
            written += 1
        cache.check_complete()
        print(f"{split}: wrote {written}, reused {reused} cached frames", flush=True)


if __name__ == "__main__":
    main()
