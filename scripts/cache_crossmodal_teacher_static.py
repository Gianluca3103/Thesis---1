"""Build fixed encoder, neighbor, and object-label supervision for the teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import (
    annotate_batch, collate_stage1, paired_dataset,
)
from models.radar_lidar_crossmodal_v1.teacher import CrossModalTeacher, TeacherConfig
from models.radar_lidar_crossmodal_v1.teacher_compact_cache import open_teacher_cache
from models.radar_lidar_crossmodal_v1.train_teacher import (
    _hash, _move, validate_lidar_pretraining_fov,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--radar-encoder", type=Path, required=True)
    parser.add_argument("--lidar-encoder", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=10.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-format", choices=("auto", "compact", "legacy"),
                        default="auto")
    parser.add_argument("--compression-codec", choices=("zlib", "zstd"), default="zlib")
    parser.add_argument("--compression-level", type=int, default=9)
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("Teacher static cache requires the CUDA sparse runtime")
    config = TeacherConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    validate_lidar_pretraining_fov(args.lidar_encoder)
    model = CrossModalTeacher(str(args.radar_encoder), str(args.lidar_encoder),
                              config).to(args.device).eval()
    cache = open_teacher_cache(
        args.cache_root, split=args.split, radar_variant=args.radar_variant,
        radar_sha256=_hash(args.radar_encoder),
        lidar_sha256=_hash(args.lidar_encoder), radius_m=config.radius_m,
        max_candidates=config.max_candidates, create=True,
        cache_format=args.cache_format, codec=args.compression_codec,
        level=args.compression_level)
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             limit=args.limit, data_cache_root=args.data_cache_root)
    missing = [i for i, frame in enumerate(dataset.frames)
               if not cache.path(frame.frame_id).is_file()]
    loader = DataLoader(Subset(dataset, missing), batch_size=1,
                        num_workers=args.num_workers,
                        collate_fn=collate_stage1, pin_memory=True)
    sizes = []
    for raw in tqdm(loader, desc=f"Teacher {args.split} cache", unit="frame",
                    dynamic_ncols=True):
        batch = _move(raw, args.device)
        with torch.no_grad():
            output = model(batch["radar"], batch["radar_valid"],
                           batch["clean_lidar"], batch["clean_lidar_valid"])
            radar_ids, lidar_ids, classes = annotate_batch(
                dataset, batch, output.radar_frozen, output.lidar_frozen)
        sizes.append(cache.write(batch["frame_id"][0], output.radar_frozen,
                                 output.lidar_frozen, output.neighbors,
                                 radar_ids, lidar_ids, classes))
        if len(sizes) == min(8, len(missing)):
            average = sum(sizes) / len(sizes)
            free = shutil.disk_usage(cache.root).free
            remaining = len(missing) - len(sizes)
            print(f"Teacher cache pilot: {average / 1024**2:.1f} MiB/frame, "
                  f"estimated remaining {average * remaining / 1024**3:.1f} GiB, "
                  f"free {free / 1024**3:.1f} GiB", flush=True)
            if average * remaining * 1.2 + args.minimum_free_gib * 1024**3 > free:
                raise OSError("Insufficient disk space for teacher static cache; "
                              "partial entries remain reusable")
    print(f"Teacher {args.split} cache ready: {len(dataset)} requested, "
          f"{len(sizes)} built, {len(dataset) - len(sizes)} reused", flush=True)


if __name__ == "__main__":
    main()
