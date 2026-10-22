"""Cache the established severity-5 fog injector on full raw VoD LiDAR scans."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import (
    discover_vod_frames, load_vod_lidar, load_vod_radar_to_lidar,
)
from models.radar_lidar_stage1.radar_fov import (
    RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M, crop_lidar_to_radar_fov,
)
from models.radar_lidar_stage2.fault_data import (
    FAULT_POLICY_VERSION, FOG_ALPHA, FOG_NOISE,
)


@lru_cache(maxsize=1)
def _fog_module(path: str):
    source = Path(path) / "fog_simulation.py"
    spec = importlib.util.spec_from_file_location("stage2_original_fog_simulation", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original fog simulator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Severity is fixed, so keep the large lookup table in each worker once.
    lookup = module.get_integral_dict(module.ParameterSet(
        alpha=FOG_ALPHA, gamma=0.000001))
    module.get_integral_dict = lambda _parameters: lookup
    return module


def _fog_frame(job: tuple) -> tuple[str, str, int, int]:
    frame, root, fog_root, seed = job
    destination = root / frame.split / f"{frame.frame_id}.npz"
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("version") != FAULT_POLICY_VERSION
                    or metadata.get("frame_id") != frame.frame_id
                    or metadata.get("fault") != "fog_sim"
                    or metadata.get("severity") != 5
                    or metadata.get("seed") != seed):
                raise ValueError(f"Existing severe fog cache has another policy: {destination}")
        return frame.frame_id, "cached", 0, destination.stat().st_size
    clean = np.asarray(load_vod_lidar(frame.lidar_path), dtype=np.float32)
    fog = _fog_module(str(fog_root))
    injection_seed = int(np.random.SeedSequence(
        [seed, int(frame.frame_id), 5]).generate_state(1)[0])
    # The upstream simulator uses a module-level Generator. Reset it per frame
    # so output is independent of worker count and processing order.
    fog.RNG = np.random.default_rng(injection_seed)
    parameters = fog.ParameterSet(alpha=FOG_ALPHA, gamma=0.000001)
    hard = fog.P_R_fog_hard(parameters, clean.copy())
    simulated, _, _ = fog.P_R_fog_soft(
        parameters, hard.copy(), clean[:, 3].copy(), noise=FOG_NOISE,
        gain=False, noise_variant="v1")
    if simulated.shape != clean.shape or not np.isfinite(simulated).all():
        raise ValueError(f"Original fog simulator returned invalid points: {frame.frame_id}")
    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path, frame.radar_calibration_path)
    cropped = crop_lidar_to_radar_fov(
        np.asarray(simulated[:, :4], dtype=np.float32), lidar_from_radar,
        half_fov_deg=RADAR_HALF_FOV_DEG, max_range_m=RADAR_MAX_RANGE_M)
    metadata = {"version": FAULT_POLICY_VERSION, "frame_id": frame.frame_id,
                "fault": "fog_sim", "severity": 5, "seed": seed,
                "injection_seed": injection_seed, "alpha": FOG_ALPHA,
                "noise": FOG_NOISE, "raw_points": len(clean),
                "cropped_points": len(cropped)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, points=cropped,
                                metadata_json=np.asarray(json.dumps(metadata)))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return frame.frame_id, "written", len(cropped), destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--fog-simulator-root", type=Path, required=True,
                        help="Existing Weather_Injector/LiDAR_fog_sim directory")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int,
                        help="Build only the first N frames for a smoke test")
    args = parser.parse_args()
    source = args.fog_simulator_root / "fog_simulation.py"
    lookup = (args.fog_simulator_root / "integral_lookup_tables" / "original"
              / "integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.06.pickle")
    if not source.is_file() or not lookup.is_file():
        parser.error(f"Original fog simulator or severity-5 lookup is missing: {source}, {lookup}")
    if args.num_workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("Workers and optional limit must be positive")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    lookup_sha = hashlib.sha256(lookup.read_bytes()).hexdigest()
    for split in (("train", "val") if args.split == "both" else (args.split,)):
        frames = discover_vod_frames(args.vod_root, split,
                                     radar_variant=args.radar_variant)
        manifest = {"version": FAULT_POLICY_VERSION, "fault": "fog_sim",
                    "severity": 5, "alpha": FOG_ALPHA, "noise": FOG_NOISE,
                    "split": split, "seed": args.seed,
                    "vod_root": str(args.vod_root.resolve()),
                    "radar_variant": args.radar_variant,
                    "fog_simulator_sha256": source_sha,
                    "fog_lookup_sha256": lookup_sha,
                    "frame_ids": [frame.frame_id for frame in frames]}
        folder = args.cache_root / split
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"Fog cache manifest differs: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")
        selected = frames[:args.limit] if args.limit is not None else frames
        jobs = ((frame, args.cache_root, args.fog_simulator_root, args.seed)
                for frame in selected)
        counts = {"written": 0, "cached": 0}
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for frame_id, status, points, size in tqdm(
                    pool.map(_fog_frame, jobs), total=len(selected),
                    desc=f"Severe fog cache {split}", unit="frame"):
                counts[status] += 1
        print(f"{split}: {counts['written']} written, {counts['cached']} cached; "
              f"{len(selected)}/{len(frames)} frames processed", flush=True)


if __name__ == "__main__":
    main()
