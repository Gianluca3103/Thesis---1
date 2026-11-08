"""Build a separate mixed-fault dataset with severity-2 fog and 30/40/50% loss.

One balanced deterministic fault is assigned per VoD frame. Fog is simulated
on the raw scan before the Stage-II radar-sector crop; only original XYZ
returns still present are retained, regardless of intensity. Output is
compatible with --faulty-root and the hierarchical Stage-I mild_dataset mode.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar, load_vod_radar_to_lidar,
)
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FAULT_ENCODING, MILD_FOG2_TYPES, MILD_FOG2_VERSION,
    balanced_fault_assignment, decode_mild_xyz, frame_seed, partial_azimuth_loss,
)
from scripts.cache_stage2_mild_faults import _write
from scripts.preview_stage2_milder_fog import _simulate_retained


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_frame(job: tuple) -> tuple[str, str, str, int, int, int]:
    (frame, fault_type, clean_path, fog_root, destination, seed, source_hash) = job
    expected = {"version": MILD_FOG2_VERSION,
                "fog_return_policy": "original_xyz_presence_ignore_intensity_v2",
                "frame_id": frame.frame_id,
                "split": frame.split, "fault_type": fault_type, "seed": seed,
                "encoding": MILD_FAULT_ENCODING,
                "source_manifest_sha256": source_hash}
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing mild-fog2 entry has another policy: {destination}")
            count = int(metadata["retained_points"])
            decode_mild_xyz(archive["points_bytes"], count)
            return (frame.frame_id, fault_type, "cached", int(metadata["clean_points"]),
                    count, destination.stat().st_size)
    saved = torch.load(clean_path, map_location="cpu", weights_only=True)
    tensor = saved.get("points")
    if (saved.get("frame_id") != frame.frame_id or not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2 or tensor.shape[1] != 4):
        raise ValueError(f"Invalid fixed clean scan: {clean_path}")
    clean = np.asarray(tensor.numpy(), dtype=np.float32)
    if not np.isfinite(clean).all():
        raise ValueError(f"Nonfinite clean scan: {frame.frame_id}")
    if fault_type == "fog_sim_2":
        raw = np.asarray(load_vod_lidar(frame.lidar_path), dtype=np.float32)
        transform = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path)
        points = _simulate_retained(raw, clean, transform, fog_root,
                                    frame.frame_id, 2, seed)
        removed_fraction = None
    else:
        removed_fraction = int(fault_type.rsplit("_", 1)[1]) / 100
        points = partial_azimuth_loss(
            clean[:, :3], removed_fraction,
            frame_seed(seed, frame.split, frame.frame_id))
    metadata = {**expected, "clean_points": len(clean),
                "retained_points": len(points),
                "removed_fraction_requested": removed_fraction,
                "retained_fraction_actual": len(points) / len(clean) if len(clean) else 1.0,
                "fog_severity": 2 if fault_type == "fog_sim_2" else None,
                "fog_alpha": 0.01 if fault_type == "fog_sim_2" else None}
    size = _write(destination, points, metadata)
    return frame.frame_id, fault_type, "written", len(clean), len(points), size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True,
                        help="Fixed-frame clean LiDAR and radar cache")
    parser.add_argument("--fog-simulator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "both"),
                        default="both")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cache-gib", type=float, default=5.0)
    args = parser.parse_args()
    if (args.num_workers < 1 or args.max_cache_gib <= 0
            or args.limit is not None and args.limit < 1):
        parser.error("Workers, size limit, and optional frame limit must be positive")
    if args.output_root.resolve() == args.data_cache_root.resolve():
        parser.error("Output root must differ from the fixed-frame cache")
    source = args.fog_simulator_root / "fog_simulation.py"
    lookup = (args.fog_simulator_root / "integral_lookup_tables" / "original"
              / "integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.01.pickle")
    if not source.is_file() or not lookup.is_file():
        parser.error(f"Missing severity-2 fog simulator or lookup: {source}, {lookup}")
    splits = ("train", "val") if args.split == "both" else (args.split,)
    datasets = {split: paired_dataset(args.vod_root, split, args.radar_variant,
                                      include_clean=True,
                                      data_cache_root=args.data_cache_root)
                for split in splits}
    total_frames = sum(len(dataset) for dataset in datasets.values())
    total_bytes = processed = 0
    for split, dataset in datasets.items():
        frame_ids = [frame.frame_id for frame in dataset.frames]
        assignment = balanced_fault_assignment(
            frame_ids, split, args.seed, fault_types=MILD_FOG2_TYPES)
        source_hash = hashlib.sha256((
            _sha256(args.data_cache_root / split / "manifest.json")
            + _sha256(source) + _sha256(lookup)).encode()).hexdigest()
        manifest = {"version": MILD_FOG2_VERSION,
                    "fog_return_policy": "original_xyz_presence_ignore_intensity_v2",
                    "split": split,
                    "seed": args.seed, "radar_variant": args.radar_variant,
                    "vod_root": str(args.vod_root.resolve()),
                    "data_cache_root": str(args.data_cache_root.resolve()),
                    "fog_simulator_root": str(args.fog_simulator_root.resolve()),
                    "fog_severity": 2, "fog_alpha": 0.01, "fog_noise": 10,
                    "source_manifest_sha256": source_hash,
                    "assignment_policy": "balanced_one_fault_per_frame_v1",
                    "partial_loss_policy": "contiguous_azimuth_exact_point_fraction_v1",
                    "fault_types": list(MILD_FOG2_TYPES), "frame_ids": frame_ids,
                    "frame_faults": assignment}
        folder = args.output_root / split
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Existing mild-fog2 manifest differs: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        selected = dataset.frames[:args.limit] if args.limit is not None else dataset.frames
        jobs = iter((frame, assignment[frame.frame_id],
                     dataset.path(frame.frame_id, "clean_lidar"),
                     args.fog_simulator_root, folder / f"{frame.frame_id}.npz",
                     args.seed, source_hash) for frame in selected)
        counts = {name: 0 for name in MILD_FOG2_TYPES}
        cached = 0
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
            with tqdm(total=len(selected), desc=f"Mild fog2 {split}", unit="frame") as progress:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        _, fault_type, status, _, _, size = future.result()
                        counts[fault_type] += 1
                        cached += status == "cached"
                        total_bytes += size
                        processed += 1
                        progress.update()
                        if total_bytes > args.max_cache_gib * 2**30:
                            raise RuntimeError("Mild-fog2 cache exceeded --max-cache-gib")
                        if processed >= 16 and processed % 16 == 0:
                            projected = total_bytes / processed * total_frames / 2**30
                            progress.set_postfix(projected_gib=f"{projected:.2f}")
                            if projected > args.max_cache_gib:
                                raise RuntimeError(
                                    f"Projected cache {projected:.2f} GiB exceeds "
                                    f"--max-cache-gib {args.max_cache_gib:g}")
                        submit_next()
        print(f"{split}: {len(selected)} frames, {cached} reused, "
              f"fault counts={counts}", flush=True)
    if processed:
        print(f"Dataset: {args.output_root} | {total_bytes / 2**30:.2f} GiB "
              f"for {processed} frames", flush=True)


if __name__ == "__main__":
    main()
