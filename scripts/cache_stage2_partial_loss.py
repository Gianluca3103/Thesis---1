"""Cache only 30%, 40%, and 50% missing-sector LiDAR faults for Stage II."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FAULT_ENCODING, PARTIAL_ONLY_TYPES, PARTIAL_ONLY_VERSION,
    balanced_fault_assignment, decode_mild_xyz, encode_mild_xyz, frame_seed,
    partial_azimuth_loss,
)


def _build_frame(job: tuple) -> tuple[str, str, bool, int]:
    frame_id, split, fault, seed, clean_path, destination, source_hash = job
    expected = {
        "version": PARTIAL_ONLY_VERSION,
        "frame_id": frame_id, "split": split, "fault_type": fault,
        "seed": seed, "encoding": MILD_FAULT_ENCODING,
        "source_manifest_sha256": source_hash,
    }
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing partial-loss entry differs: {destination}")
            decode_mild_xyz(archive["points_bytes"], int(metadata["retained_points"]))
        return frame_id, fault, True, destination.stat().st_size

    saved = torch.load(clean_path, map_location="cpu", weights_only=True)
    points = saved.get("points")
    if (saved.get("frame_id") != frame_id or not isinstance(points, torch.Tensor)
            or points.ndim != 2 or points.shape[1] != 4 or
            not torch.isfinite(points).all()):
        raise ValueError(f"Invalid fixed clean LiDAR frame: {clean_path}")
    clean = points[:, :3].numpy()
    fraction = int(fault.rsplit("_", 1)[1]) / 100
    faulty = partial_azimuth_loss(
        clean, fraction, frame_seed(seed, split, frame_id))
    metadata = {
        **expected, "clean_points": len(clean),
        "retained_points": len(faulty),
        "removed_fraction_requested": fraction,
        "retained_fraction_actual": len(faulty) / len(clean) if len(clean) else 1.0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, points_bytes=encode_mild_xyz(faulty),
                metadata_json=np.asarray(json.dumps(metadata)))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return frame_id, fault, False, destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "both", "all"),
                        default="all")
    parser.add_argument("--radar-variant",
                        default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cache-gib", type=float, default=5.0)
    args = parser.parse_args()
    if (args.num_workers < 1 or args.max_cache_gib <= 0 or
            args.limit is not None and args.limit < 1):
        parser.error("Workers, cache limit, and optional frame limit must be positive")
    if args.output_root.resolve() == args.data_cache_root.resolve():
        parser.error("Fault output must differ from the clean-frame cache")
    splits = (("train", "val", "test") if args.split == "all" else
              ("train", "val") if args.split == "both" else (args.split,))
    datasets = {
        split: paired_dataset(
            args.vod_root, split, args.radar_variant,
            include_clean=True, data_cache_root=args.data_cache_root)
        for split in splits
    }
    total_frames = sum(min(len(dataset), args.limit or len(dataset))
                       for dataset in datasets.values())
    processed = total_bytes = 0
    for split, dataset in datasets.items():
        ids = [frame.frame_id for frame in dataset.frames]
        assignment = balanced_fault_assignment(
            ids, split, args.seed, fault_types=PARTIAL_ONLY_TYPES)
        source_hash = hashlib.sha256(
            (args.data_cache_root / split / "manifest.json").read_bytes()
        ).hexdigest()
        manifest = {
            "version": PARTIAL_ONLY_VERSION, "split": split,
            "seed": args.seed, "radar_variant": args.radar_variant,
            "vod_root": str(args.vod_root.resolve()),
            "data_cache_root": str(args.data_cache_root.resolve()),
            "source_manifest_sha256": source_hash,
            "assignment_policy": "balanced_one_fault_per_frame_v1",
            "partial_loss_policy": "contiguous_azimuth_exact_point_fraction_v1",
            "fault_types": list(PARTIAL_ONLY_TYPES),
            "frame_ids": ids, "frame_faults": assignment,
        }
        folder = args.output_root / split
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Existing partial-loss manifest differs: {manifest_path}")
        else:
            temporary = manifest_path.with_name("manifest.json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
            temporary.replace(manifest_path)
        selected = ids[:args.limit] if args.limit is not None else ids
        jobs = iter((frame_id, split, assignment[frame_id], args.seed,
                     dataset.path(frame_id, "clean_lidar"),
                     folder / f"{frame_id}.npz", source_hash)
                    for frame_id in selected)
        counts = {fault: 0 for fault in PARTIAL_ONLY_TYPES}
        reused = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            pending = set()

            def submit_next() -> bool:
                job = next(jobs, None)
                if job is None:
                    return False
                pending.add(pool.submit(_build_frame, job))
                return True

            for _ in range(min(2 * args.num_workers, len(selected))):
                submit_next()
            with tqdm(total=len(selected), desc=f"Partial loss {split}",
                      unit="frame") as progress:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        _, fault, cached, size = future.result()
                        counts[fault] += 1
                        reused += cached
                        processed += 1
                        total_bytes += size
                        progress.update()
                        if total_bytes > args.max_cache_gib * 2**30:
                            raise RuntimeError("Partial-loss cache exceeded size limit")
                        if processed >= 16 and processed % 16 == 0:
                            projected = total_bytes / processed * total_frames / 2**30
                            progress.set_postfix(projected_gib=f"{projected:.2f}")
                            if projected > args.max_cache_gib:
                                raise RuntimeError(
                                    f"Projected cache {projected:.2f} GiB exceeds "
                                    f"--max-cache-gib {args.max_cache_gib:g}")
                        submit_next()
        print(f"{split}: {len(selected)} frames, {reused} reused, "
              f"fault counts={counts}", flush=True)
    print(f"Partial-loss cache: {args.output_root} | "
          f"{total_bytes / 2**30:.2f} GiB for {processed} frames", flush=True)


if __name__ == "__main__":
    main()
