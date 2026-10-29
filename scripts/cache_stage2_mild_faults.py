"""Build a compact Stage-II dataset: fog severity 5 or 30/40/50% partial loss.

One deterministic fault is assigned to each VoD frame. Partial loss removes a
contiguous azimuth sector and always retains at least half the clean returns.
Fog reuses the established severity-5 simulation and original-return filter.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import (
    FOG_CACHE_VERSION, SevereFogCache, retain_original_fog_returns,
)
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FAULT_ENCODING, MILD_FAULT_TYPES, MILD_FAULT_VERSION,
    balanced_fault_assignment, decode_mild_xyz, encode_mild_xyz, frame_seed,
    partial_azimuth_loss,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, points: np.ndarray, metadata: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,
                                points_bytes=encode_mild_xyz(points),
                                metadata_json=np.asarray(json.dumps(metadata)))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


@lru_cache(maxsize=2)
def _fog_cache(root: Path, split: str, seed: int) -> SevereFogCache:
    return SevereFogCache(root, split, None, seed=seed, check_complete=False)


def _build_frame(job: tuple) -> tuple[str, str, str, int, int, int]:
    (frame_id, split, fault_type, seed, clean_path, fog_root, destination,
     source_hash) = job
    expected = {"version": MILD_FAULT_VERSION, "frame_id": frame_id,
                "split": split, "fault_type": fault_type, "seed": seed,
                "encoding": MILD_FAULT_ENCODING,
                "source_manifest_sha256": source_hash}
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing mild-fault entry differs: {destination}")
            count = int(metadata["retained_points"])
            decode_mild_xyz(archive["points_bytes"], count)
            return frame_id, fault_type, "cached", int(metadata["clean_points"]), count, destination.stat().st_size

    saved = torch.load(clean_path, map_location="cpu", weights_only=True)
    clean_tensor = saved.get("points")
    if (saved.get("frame_id") != frame_id or not isinstance(clean_tensor, torch.Tensor)
            or clean_tensor.ndim != 2 or clean_tensor.shape[1] != 4):
        raise ValueError(f"Invalid fixed clean-LiDAR cache: {clean_path}")
    clean = np.asarray(clean_tensor.numpy(), dtype=np.float32)
    if not np.isfinite(clean).all():
        raise ValueError(f"Nonfinite clean LiDAR: {frame_id}")
    if fault_type == "fog_sim_5":
        points = retain_original_fog_returns(
            clean, _fog_cache(fog_root, split, seed).load(frame_id).numpy())
        remove_fraction = None
    else:
        remove_fraction = int(fault_type.rsplit("_", 1)[1]) / 100
        points = partial_azimuth_loss(
            clean[:, :3], remove_fraction, frame_seed(seed, split, frame_id))
        if len(points) < (len(clean) + 1) // 2:
            raise AssertionError("Partial loss retained fewer than half the points")
    metadata = {**expected, "clean_points": len(clean),
                "retained_points": len(points),
                "removed_fraction_requested": remove_fraction,
                "retained_fraction_actual": len(points) / len(clean) if len(clean) else 1.0}
    size = _write(destination, points, metadata)
    return frame_id, fault_type, "written", len(clean), len(points), size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True,
                        help="Existing radar_5frames fixed-frame cache")
    parser.add_argument("--fog-cache-root", type=Path, required=True,
                        help="Existing complete severity-5 fog cache")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-cache-gib", type=float, default=5.0)
    parser.add_argument("--limit", type=int,
                        help="Build only the first N assigned frames per split")
    args = parser.parse_args()
    if (args.num_workers < 1 or args.max_cache_gib <= 0
            or args.limit is not None and args.limit < 1):
        parser.error("Workers, size limit and optional frame limit must be positive")
    if args.output_root.resolve() in (args.data_cache_root.resolve(),
                                      args.fog_cache_root.resolve()):
        parser.error("Output root must differ from source caches")

    splits = ("train", "val") if args.split == "both" else (args.split,)
    datasets = {split: paired_dataset(args.vod_root, split, args.radar_variant,
                                      include_clean=True,
                                      data_cache_root=args.data_cache_root)
                for split in splits}
    total_frames = sum(len(dataset) for dataset in datasets.values())
    total_bytes = processed = 0
    for split, dataset in datasets.items():
        frame_ids = [frame.frame_id for frame in dataset.frames]
        fog_manifest_path = args.fog_cache_root / split / "manifest.json"
        fog_manifest = json.loads(fog_manifest_path.read_text(encoding="utf-8"))
        if (fog_manifest.get("version") != FOG_CACHE_VERSION
                or fog_manifest.get("split") != split
                or fog_manifest.get("seed") != args.seed
                or fog_manifest.get("severity") != 5
                or fog_manifest.get("frame_ids") != frame_ids):
            raise ValueError(f"Fog cache is not the expected severity-5 split: {fog_manifest_path}")
        assignment = balanced_fault_assignment(frame_ids, split, args.seed)
        source_hash = hashlib.sha256((
            _sha256(args.data_cache_root / split / "manifest.json")
            + _sha256(fog_manifest_path)).encode()).hexdigest()
        manifest = {"version": MILD_FAULT_VERSION, "split": split,
                    "seed": args.seed, "radar_variant": args.radar_variant,
                    "vod_root": str(args.vod_root.resolve()),
                    "data_cache_root": str(args.data_cache_root.resolve()),
                    "fog_cache_root": str(args.fog_cache_root.resolve()),
                    "source_manifest_sha256": source_hash,
                    "assignment_policy": "balanced_one_fault_per_frame_v1",
                    "partial_loss_policy": "contiguous_azimuth_exact_point_fraction_v1",
                    "fault_types": list(MILD_FAULT_TYPES),
                    "frame_ids": frame_ids, "frame_faults": assignment}
        folder = args.output_root / split
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Mild-fault manifest differs: {manifest_path}")
        else:
            temporary = manifest_path.with_name("manifest.json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
        selected = frame_ids[:args.limit] if args.limit is not None else frame_ids
        jobs = iter((frame_id, split, assignment[frame_id], args.seed,
                     dataset.path(frame_id, "clean_lidar"), args.fog_cache_root,
                     folder / f"{frame_id}.npz", source_hash)
                    for frame_id in selected)
        counts = {name: 0 for name in MILD_FAULT_TYPES}
        saved = 0
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
            with tqdm(total=len(selected), desc=f"Mild faults {split}",
                      unit="frame") as progress:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        frame_id, fault_type, status, clean_count, retained, size = future.result()
                        counts[fault_type] += 1
                        saved += status == "cached"
                        total_bytes += size
                        processed += 1
                        progress.update()
                        if total_bytes > args.max_cache_gib * 2**30:
                            raise RuntimeError("Mild-fault cache exceeded --max-cache-gib")
                        if processed >= 16 and processed % 16 == 0:
                            projected = total_bytes / processed * total_frames / 2**30
                            progress.set_postfix(projected_gib=f"{projected:.2f}")
                            if projected > args.max_cache_gib:
                                raise RuntimeError(
                                    f"Projected cache {projected:.2f} GiB exceeds "
                                    f"--max-cache-gib {args.max_cache_gib:g}")
                        submit_next()
        print(f"{split}: {len(selected)} frames, {saved} reused, fault counts={counts}",
              flush=True)
    if processed:
        print(f"Dataset: {args.output_root} | {total_bytes / 2**30:.2f} GiB "
              f"for {processed} frames", flush=True)


if __name__ == "__main__":
    main()
