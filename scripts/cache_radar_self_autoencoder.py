"""Cache the filtered radar input and identical radar target once per frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.filtered_radar_cache import (
    FilteredRadarCache, filtered_frame)


class _RawFrames(Dataset):
    def __init__(self, raw_root: Path, split: str, frame_ids: list[str]):
        self.raw_root, self.split, self.frame_ids = raw_root, split, frame_ids

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        return filtered_frame(self.raw_root, self.split, self.frame_ids[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    cache = FilteredRadarCache(args.cache_root, args.raw_root, args.split, create=True)
    missing = [frame_id for frame_id in cache.frame_ids if not cache.path(frame_id).is_file()]
    loader = DataLoader(_RawFrames(args.raw_root, args.split, missing), batch_size=None,
                        num_workers=args.num_workers)
    written = retained = raw = size_bytes = 0
    for row in tqdm(loader, desc=f"Filter radar {args.split}", unit="frame"):
        written += cache.write(row)
        size_bytes += cache.path(row["frame_id"]).stat().st_size
        retained += len(row["points"])
        raw += row["raw_returns"]
        if written == min(8, len(missing)):
            remaining = len(missing) - written
            projected = size_bytes / max(written, 1) * remaining
            free = shutil.disk_usage(cache.root).free
            print(f"Filtered radar cache pilot: projected remaining "
                  f"{projected / 1024**3:.2f} GiB; free {free / 1024**3:.2f} GiB",
                  flush=True)
            if projected * 1.3 + 2 * 1024**3 > free:
                raise OSError("Insufficient disk space for the filtered radar cache; "
                              "written frames remain reusable")
    cache.check_complete()
    print(f"{args.split}: {len(cache)} frames, {written} new, "
          f"{len(cache) - written} reused; new-frame returns {retained:,}/{raw:,}", flush=True)


if __name__ == "__main__":
    main()
