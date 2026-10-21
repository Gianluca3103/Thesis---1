"""Build the independent R2LDM LiDAR voxel cache before GPU training."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.r2ldm_lidar_lpcr import R2LDMGrid, VoDLidarLPCRDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("Worker count cannot be negative")
    for split in (("train", "val") if args.split == "both" else (args.split,)):
        dataset = VoDLidarLPCRDataset(args.vod_root, split, grid=R2LDMGrid(),
                                      limit=args.limit, cache_root=args.cache_root)
        loader = DataLoader(dataset, batch_size=None, shuffle=False,
                            num_workers=args.num_workers)
        for _ in tqdm(loader, total=len(dataset), desc=f"LPCR cache {split}",
                      unit="frame", dynamic_ncols=True):
            pass
        dataset.check_complete()
        print(f"{split}: {len(dataset)} LiDAR voxel entries complete", flush=True)


if __name__ == "__main__":
    main()
