"""Convert existing teacher .pt entries to verified lossless .ptz entries.

The source cache is never changed or removed. Interrupted conversions resume
from verified files in the destination directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.teacher_static_cache import TeacherStaticCache
from models.radar_lidar_crossmodal_v1.teacher_compact_cache import CompactTeacherCache


def convert_split(source_root: Path, destination_root: Path, split: str,
                  codec: str, level: int, limit: int | None,
                  minimum_free_gib: float) -> dict:
    manifest = source_root / split / "manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("version") != 1 or saved.get("split") != split:
        raise ValueError(f"Expected a legacy teacher cache manifest for {split}")
    common = dict(split=split, radar_variant=saved["radar_variant"],
                  radar_sha256=saved["radar_sha256"],
                  lidar_sha256=saved["lidar_sha256"],
                  radius_m=saved["radius_m"],
                  max_candidates=saved["max_candidates"])
    source = TeacherStaticCache(source_root, **common)
    destination = CompactTeacherCache(destination_root, **common,
                                      codec=codec, level=level, create=True)
    paths = sorted(source.root.glob("*.pt"))
    if limit is not None:
        paths = paths[:limit]
    old_bytes = new_bytes = 0
    for index, path in enumerate(tqdm(paths, desc=f"Compact teacher {split}",
                                      unit="frame", dynamic_ncols=True), 1):
        if not path.stem.isdigit():
            continue
        entry = source._read(path.stem)
        new_size = destination.write_entry(path.stem, entry)
        old_bytes += path.stat().st_size
        new_bytes += new_size
        if index == min(8, len(paths)):
            remaining = len(paths) - index
            estimated = new_bytes / index * remaining
            free = shutil.disk_usage(destination.root).free
            print(f"{split} pilot: {old_bytes / 1024**2:.1f} MiB original, "
                  f"{new_bytes / 1024**2:.1f} MiB compact across {index} frames; "
                  f"estimated remaining {estimated / 1024**3:.1f} GiB, "
                  f"free {free / 1024**3:.1f} GiB", flush=True)
            if estimated * 1.2 + minimum_free_gib * 1024**3 > free:
                raise OSError("Insufficient space for the estimated compact teacher cache. "
                              "Verified partial entries remain reusable.")
    return {"split": split, "frames": len(paths), "original_bytes": old_bytes,
            "compact_bytes": new_bytes,
            "ratio": old_bytes / new_bytes if new_bytes else None,
            "source": str(source.root), "destination": str(destination.root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--split", nargs="+", choices=("train", "val"),
                        default=("train", "val"))
    parser.add_argument("--codec", choices=("zlib", "zstd"), default="zlib")
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-free-gib", type=float, default=5.)
    args = parser.parse_args()
    source = args.source_root.resolve()
    destination = args.destination_root.resolve()
    if (source == destination or source in destination.parents
            or destination in source.parents):
        parser.error("Source and destination must be separate, non-nested directories")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be nonnegative")
    for split in args.split:
        result = convert_split(source, destination, split, args.codec,
                               args.level, args.limit, args.minimum_free_gib)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
