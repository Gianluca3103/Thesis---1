"""Precompute D-PCC farthest-point hierarchy on the existing filtered radar."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.radar_dpcc_cache import RadarDPCCCache


class Missing(Dataset):
    def __init__(self, cache, indices):
        self.cache, self.indices = cache, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.cache.build(self.indices[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cache = RadarDPCCCache(args.filtered_root, args.raw_root, args.geometry_root,
                           args.split, create=True)
    cache.filtered.check_complete()
    missing = [i for i, name in enumerate(cache.frame_ids) if not cache.path(name).is_file()]
    loader = DataLoader(Missing(cache, missing), batch_size=None,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    written_bytes = 0
    for number, row in enumerate(tqdm(loader, total=len(missing),
                                      desc=f"D-PCC FPS {args.split}", unit="frame"), 1):
        cache.write(row)
        written_bytes += cache.path(row["frame_id"]).stat().st_size
        if number == min(8, len(missing)):
            estimated_remaining = written_bytes / number * (len(missing) - number)
            free = shutil.disk_usage(cache.root).free
            print(f"FPS cache pilot: projected remaining {estimated_remaining/1024**3:.2f} GiB; "
                  f"free {free/1024**3:.2f} GiB", flush=True)
            if estimated_remaining * 1.3 + 1024**3 > free:
                raise OSError("Insufficient space for the D-PCC FPS cache; "
                              "written frames remain reusable")
    cache.check_complete()
    print(f"{args.split}: {len(cache)} frames, {len(missing)} new, "
          f"{len(cache)-len(missing)} reused", flush=True)


if __name__ == "__main__":
    main()
