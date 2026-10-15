"""Convert a v2 student supervision cache to the lossless v3 format.

Source files are left intact. The command is resumable and verifies every
decoded tensor before reporting an entry as converted. It needs no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.student_compact_cache import CompactStudentCache
from models.radar_lidar_crossmodal_v1.student_validation_cache import StudentValidationCache


FIELDS = ("candidate_coords", "bottleneck_coords", "teacher_features",
          "evidence", "occupied", "known_free")


def _verify(original: dict, restored: dict, frame_id: str) -> None:
    for field in FIELDS:
        if not torch.equal(original[field], restored[field]):
            raise ValueError(f"Lossless verification failed for {frame_id}: {field}")
    for field in ("clean_points_in_candidates", "clean_points_in_grid"):
        if original[field] != restored[field]:
            raise ValueError(f"Lossless verification failed for {frame_id}: {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Convert only this many frames for a pilot")
    parser.add_argument("--codec", choices=("zlib", "zstd"), default="zlib")
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--benchmark-reads", action="store_true",
                        help="Time warm reads of up to 32 converted frames")
    args = parser.parse_args()
    if args.source_root.resolve() == args.output_root.resolve():
        parser.error("Source and output roots must differ")
    metadata = json.loads((args.source_root / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("version") != 2:
        parser.error("Source must be a version 2 student supervision cache")
    common = (metadata["teacher_sha256"], metadata["radar_variant"])
    source = StudentValidationCache(args.source_root, *common, split=metadata["split"])
    compact = CompactStudentCache(args.output_root, *common, split=metadata["split"],
                                  create=True, codec=args.codec, level=args.level)
    paths = sorted(args.source_root.glob("*.pt"))
    if args.limit is not None:
        paths = paths[:args.limit]
    legacy_bytes = compressed_bytes = 0
    raw_field_bytes = {field: 0 for field in FIELDS}
    packed_field_bytes = {field: 0 for field in FIELDS}
    packed_names = {"candidate_coords": "candidate_delta",
                    "bottleneck_coords": "bottleneck_delta",
                    "teacher_features": "teacher_features", "evidence": "evidence",
                    "occupied": "occupied_bits", "known_free": "known_free_bits"}
    started = time.perf_counter()
    for path in tqdm(paths, desc=f"Compress {metadata['split']}", unit="frame", dynamic_ncols=True):
        frame_id = path.stem
        original = source._read(frame_id)
        compact.write_entry(frame_id, original)
        _verify(original, compact._read(frame_id), frame_id)
        packed = torch.load(compact.path(frame_id), map_location="cpu", weights_only=True)
        for field in FIELDS:
            value = original[field]
            raw_field_bytes[field] += value.numel() * value.element_size()
            packed_field_bytes[field] += len(packed[packed_names[field]])
        legacy_bytes += path.stat().st_size
        compressed_bytes += compact.path(frame_id).stat().st_size
    summary = {"frames_verified": len(paths), "codec": args.codec, "level": args.level,
               "legacy_gib": legacy_bytes / 1024**3,
               "compact_gib": compressed_bytes / 1024**3,
               "size_percent": 100 * compressed_bytes / legacy_bytes if legacy_bytes else None,
               "seconds_per_frame": (time.perf_counter() - started) / len(paths) if paths else None,
               "raw_field_mib": {key: value / 1024**2 for key, value in raw_field_bytes.items()},
               "packed_field_mib": {key: value / 1024**2 for key, value in packed_field_bytes.items()},
               "output_root": str(args.output_root)}
    if args.benchmark_reads and paths:
        sample = paths[:32]
        for cache, key in ((source, "legacy_read_seconds_per_frame"),
                           (compact, "compact_read_seconds_per_frame")):
            for path in sample:
                cache._read(path.stem)
            started = time.perf_counter()
            for path in sample:
                cache._read(path.stem)
            summary[key] = (time.perf_counter() - started) / len(sample)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
