"""Create a separate DoppDrive-inspired VoD 20-scan radar variant.

The released seventh radar field is an ordinal scan age. ``--frame-period-s``
is therefore an explicit timing approximation, not an original radar timestamp.
The radial shift follows DoppDrive; the optional displacement window uses a
constant lateral-speed ratio instead of the paper's dataset-specific heading
prior. Neither variant changes the raw source scans or verified ego stack.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

from Fault_Localization_Model.vod_dataset.doppler_accumulation import (
    radial_compensate_verified_stack,
)
from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_radar, load_vod_split_ids, resolve_vod_public_root,
    vod_partition_for_split,
)


VARIANTS = (
    "radar_20frames_verified_doppler_radial",
    "radar_20frames_verified_doppler_window",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-variant", choices=VARIANTS, required=True)
    parser.add_argument("--frame-period-s", type=float, required=True,
                        help="Approximate interval between VoD synchronized frames")
    parser.add_argument("--doppler-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--dispersion-tolerance-m", type=float, default=2.0)
    parser.add_argument("--lateral-speed-ratio", type=float, default=1.0)
    parser.add_argument("--frame-ids", nargs="+", help="Subset of official split IDs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_tofile(path: Path, points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        points.tofile(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_one(task: tuple) -> tuple[str, int, int, int]:
    (public_text, partition, frame_id, output_variant, frame_period_s,
     doppler_sign, tolerance, lateral_ratio, overwrite) = task
    public = Path(public_text)
    source_path = (public / "radar_20frames_verified" / partition /
                   "velodyne" / f"{frame_id}.bin")
    destination = public / output_variant / partition / "velodyne" / f"{frame_id}.bin"
    if destination.exists() and not overwrite:
        if destination.stat().st_size == 0 or destination.stat().st_size % 28:
            raise ValueError(f"Existing Doppler stack is empty or malformed: {destination}")
        return frame_id, 0, 0, 1
    stack = load_vod_radar(source_path)
    ages = sorted(set(stack[:, 6].astype(int)))
    raw = {
        age: load_vod_radar(
            public / "radar" / partition / "velodyne" /
            f"{int(frame_id) + age:05d}.bin"
        )
        for age in ages
    }
    points, stats = radial_compensate_verified_stack(
        stack, raw, frame_period_s=frame_period_s,
        doppler_sign=doppler_sign,
        dispersion_tolerance_m=tolerance,
        lateral_speed_ratio=lateral_ratio,
    )
    _atomic_tofile(destination, points)
    return frame_id, len(points), int(stats["window_rejected"]), 0


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("num-workers must be at least one")
    is_radial_only = args.output_variant.endswith("_radial")
    tolerance = None if is_radial_only else args.dispersion_tolerance_m
    public = resolve_vod_public_root(args.vod_root)
    official_ids = load_vod_split_ids(public, args.split)
    if args.frame_ids:
        unknown = sorted(set(args.frame_ids) - set(official_ids))
        if unknown:
            raise ValueError(f"IDs outside the official {args.split} split: {unknown[:5]}")
        frame_ids = list(dict.fromkeys(args.frame_ids))
    else:
        frame_ids = official_ids
    partition = vod_partition_for_split(public, args.split, frame_ids)
    source_root = public / "radar_20frames_verified" / partition / "velodyne"
    missing = [frame_id for frame_id in frame_ids
               if not (source_root / f"{frame_id}.bin").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Verified 20-scan stack is missing for {len(missing)} IDs: {missing[:5]}"
        )
    destination_root = public / args.output_variant / partition / "velodyne"
    manifest_path = public / args.output_variant / f"doppler_manifest_{args.split}.json"
    settings = {
        "source_variant": "radar_20frames_verified",
        "output_variant": args.output_variant,
        "frame_period_s": args.frame_period_s,
        "timestamp_source": "explicit_approximate_frame_period_not_radar_timestamps",
        "doppler_sign": args.doppler_sign,
        "dispersion_tolerance_m": tolerance,
        "lateral_speed_ratio": args.lateral_speed_ratio,
        "window": "abs(compensated_doppler)*elapsed*lateral_speed_ratio <= tolerance"
                  if tolerance is not None else None,
        "angle_dependent_DoppDrive_prior": False,
        "ego_alignment": "verified_official_5",
    }
    if destination_root.exists() and any(destination_root.glob("*.bin")):
        existing_ids = {path.stem for path in destination_root.glob("*.bin")}
        overlapping_ids = existing_ids & set(frame_ids)
        if overlapping_ids and not manifest_path.is_file() and not args.overwrite:
            raise ValueError(
                f"Existing {args.split} Doppler files have no manifest: {destination_root}"
            )
        for existing_manifest in (public / args.output_variant).glob(
            "doppler_manifest_*.json"
        ):
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if any(existing.get(key) != value for key, value in settings.items()):
                raise ValueError(
                    f"Existing Doppler variant uses different settings: {existing_manifest}"
                )
    tasks = [
        (str(public), partition, frame_id, args.output_variant,
         args.frame_period_s, args.doppler_sign, tolerance,
         args.lateral_speed_ratio, args.overwrite)
        for frame_id in frame_ids
    ]
    created = cached = points_total = rejected_total = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for index, (frame_id, point_count, rejected, was_cached) in enumerate(
            executor.map(_generate_one, tasks, chunksize=8), 1
        ):
            created += int(not was_cached)
            cached += was_cached
            points_total += point_count
            rejected_total += rejected
            if index % 100 == 0 or index == len(tasks):
                print(f"{args.output_variant}: {index}/{len(tasks)} "
                      f"created={created} cached={cached} latest={frame_id}",
                      flush=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **settings,
        "split": args.split,
        "target_frames_requested": len(frame_ids),
        "frames_in_directory": sum(1 for _ in destination_root.glob("*.bin")),
        "created_this_run": created,
        "cached_this_run": cached,
        "output_points_this_run": points_total,
        "window_rejected_this_run": rejected_total,
        "warning": "Radial compensation does not resolve tangential motion. "
                   "The frame period is approximate and the angle-dependent "
                   "DoppDrive prior was not reproduced.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
