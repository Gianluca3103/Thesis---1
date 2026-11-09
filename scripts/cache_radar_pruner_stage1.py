"""Precompute the exact 3-distinct-radar-voxel/3 m Stage-I domain."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import radar_support_mask
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity)


def _one(rows: list[dict]) -> dict:
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--radius-m", type=float, choices=(1.6, 3.0), default=3.0)
    parser.add_argument("--min-distinct-radar-voxels", type=int, default=3)
    parser.add_argument("--split", choices=("train", "val", "test", "both"), default="both")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--build-limit", type=int,
                        help="Cache only the first N frames, retaining a full-split manifest for a smoke test")
    args = parser.parse_args()
    if (args.num_workers < 0 or args.min_distinct_radar_voxels < 1 or
            args.limit is not None and args.limit < 1 or
            args.build_limit is not None and args.build_limit < 1 or
            args.limit is not None and args.build_limit is not None):
        parser.error("Workers and optional limit must be nonnegative/positive")
    for split in (("train", "val") if args.split == "both" else (args.split,)):
        dataset = paired_dataset(args.vod_root, split, args.radar_variant,
                                 include_clean=True,
                                 data_cache_root=args.data_cache_root,
                                 limit=args.limit)
        ids = [frame.frame_id for frame in dataset.frames]
        cache = RadarSupportCache(args.output_root / split,
                                  identity(split, ids, args.radar_variant,
                                           data_cache_root=args.data_cache_root,
                                           radius_m=args.radius_m,
                                           min_radar_voxels=args.min_distinct_radar_voxels),
                                  create=True)
        selected = (Subset(dataset, range(min(args.build_limit, len(dataset))))
                    if args.build_limit is not None else dataset)
        loader = DataLoader(selected, batch_size=1, shuffle=False,
                            collate_fn=_one,
                            num_workers=args.num_workers)
        reused = written = supported = 0
        for row in tqdm(loader, desc=f"Radar support {split}", unit="frame"):
            frame_id = row["frame_id"]
            if cache.path(frame_id).is_file():
                mask = cache.load(frame_id)
                reused += 1
            else:
                mask = radar_support_mask(row["radar"], args.radius_m,
                                          args.min_distinct_radar_voxels)
                cache.write(frame_id, mask)
                written += 1
            supported += int(mask.sum())
        if args.build_limit is None:
            cache.check_complete()
        print(f"{split}: {len(selected)}/{len(ids)} frames, {written} written, {reused} reused, "
              f"mean supported cells/frame={supported / max(len(selected), 1):,.0f}",
              flush=True)


if __name__ == "__main__":
    main()
