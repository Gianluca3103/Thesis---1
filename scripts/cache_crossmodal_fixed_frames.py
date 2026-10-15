"""Cache unaugmented aligned radar and clean LiDAR before full training."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.fixed_frame_cache import FixedFrameCache
from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage1.radar_fov import RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--minimum-free-gib", type=float, default=10.)
    args = parser.parse_args()
    dataset = VoDStage1Dataset(
        args.vod_root, args.split, radar_variant=args.radar_variant,
        lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
        lidar_radar_max_range_m=RADAR_MAX_RANGE_M)
    cache = FixedFrameCache(args.cache_root / args.split, dataset, create=True)
    frames = dataset.frames[:args.limit] if args.limit is not None else dataset.frames
    missing = [i for i, frame in enumerate(frames)
               if not all(cache.path(frame.frame_id, modality).is_file()
                          for modality in ("radar", "clean_lidar"))]
    loader = DataLoader(Subset(dataset, missing), batch_size=None,
                        num_workers=args.num_workers)
    sizes = []
    for row in tqdm(loader, desc=f"Aligned {args.split} frames", unit="frame",
                    dynamic_ncols=True):
        sizes.append(cache.write(row))
        if len(sizes) == min(8, len(missing)):
            average = sum(sizes) / len(sizes)
            free = shutil.disk_usage(cache.root).free
            remaining = len(missing) - len(sizes)
            print(f"Raw cache pilot: {average / 1024**2:.1f} MiB/frame, "
                  f"estimated remaining {average * remaining / 1024**3:.1f} GiB, "
                  f"free {free / 1024**3:.1f} GiB", flush=True)
            if average * remaining * 1.2 + args.minimum_free_gib * 1024**3 > free:
                raise OSError("Insufficient disk space for the fixed frame cache; "
                              "partial entries remain reusable")
    if args.limit is None:
        cache.check_complete()
    print(f"Fixed {args.split} frames ready: {len(frames)} requested, "
          f"{len(sizes)} built, {len(frames) - len(sizes)} reused", flush=True)


if __name__ == "__main__":
    main()
