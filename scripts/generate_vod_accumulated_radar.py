"""Generate ego-motion-aligned VoD radar stacks for selected official splits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from Fault_Localization_Model.vod_dataset import (
    RadarTemporalFilterConfig,
    accumulate_vod_radar_scans,
    load_vod_split_ids,
    load_vod_odom_from_camera,
)
from Fault_Localization_Model.vod_dataset.radar_accumulation import (
    radar_current_from_official_previous,
)
from Fault_Localization_Model.vod_dataset.vod_io import vod_partition_for_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--stack-sizes", nargs="+", type=int, default=(10, 20))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "train_val", "full"),
        help="Generate only target frames from this official VoD split.",
    )
    parser.add_argument(
        "--max-step-translation-m",
        type=float,
        default=5.0,
        help="Break history at recording boundaries or implausible pose jumps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--alignment-source", choices=("pose", "official-5"), default="pose",
        help="Use the released VoD five-scan geometry to align longer stacks",
    )
    parser.add_argument("--isolate-split-history", action="store_true",
                        help="Stack only radar frames from the requested split")
    parser.add_argument(
        "--output-suffix",
        default="",
        choices=("", "temporal_filtered", "rangeview", "verified",
                 "verified_motion_aware"),
        help=(
            "Optional dataset-directory suffix, for example temporal_filtered "
            "writes radar_20frames_temporal_filtered."
        ),
    )
    parser.add_argument(
        "--basic-validity-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--temporal-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--radar-min-range-m", type=float, default=1.0)
    parser.add_argument("--radar-max-range-m", type=float, default=80.0)
    parser.add_argument("--radar-min-height-m", type=float, default=-5.0)
    parser.add_argument("--radar-max-height-m", type=float, default=5.0)
    parser.add_argument("--radar-min-rcs", type=float)
    parser.add_argument("--radar-max-abs-velocity-mps", type=float)
    parser.add_argument(
        "--motion-aware", action="store_true",
        help="Keep likely moving returns only from the most recent scans; preserve static history.",
    )
    parser.add_argument("--moving-velocity-threshold-mps", type=float, default=1.0)
    parser.add_argument("--moving-max-age-scans", type=int, default=1)
    parser.add_argument("--temporal-support-radius-m", type=float, default=0.75)
    parser.add_argument("--temporal-min-support-scans", type=int, default=2)
    parser.add_argument(
        "--preserve-current-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _public_root(vod_root: Path) -> Path:
    public = vod_root / "view_of_delft_PUBLIC"
    return public if public.is_dir() else vod_root


def _calibration_path(public: Path, frame_id: int, partition: str) -> Path:
    return public / "radar" / partition / "calib" / f"{frame_id:05d}.txt"


def _pose_path(public: Path, frame_id: int, partition: str) -> Path:
    return public / "lidar" / partition / "pose" / f"{frame_id:05d}.json"


def _radar_path(public: Path, frame_id: int, partition: str) -> Path:
    return public / "radar" / partition / "velodyne" / f"{frame_id:05d}.bin"


def _same_recording(public: Path, previous: int, current: int, maximum: float,
                    partition: str) -> bool:
    if current != previous + 1:
        return False
    try:
        previous_pose = load_vod_odom_from_camera(_pose_path(public, previous, partition))
        current_pose = load_vod_odom_from_camera(_pose_path(public, current, partition))
    except (OSError, ValueError):
        return False
    relative = np.linalg.inv(current_pose) @ previous_pose
    return float(np.linalg.norm(relative[:3, 3])) <= maximum


def _histories(
    public: Path,
    frame_ids: list[int],
    maximum_stack: int,
    max_step_translation_m: float,
    partition: str,
    official_steps: dict[int, np.ndarray] | None = None,
) -> dict[int, list[int]]:
    histories: dict[int, list[int]] = {}
    active: list[int] = []
    for frame_id in frame_ids:
        if not active:
            same_recording = True
        elif official_steps is not None:
            same_recording = frame_id == active[-1] + 1 and frame_id in official_steps
        else:
            same_recording = _same_recording(
                public, active[-1], frame_id, max_step_translation_m, partition
            )
        if active and not same_recording:
            active = []
        active.append(frame_id)
        active = active[-maximum_stack:]
        histories[frame_id] = active.copy()
    return histories


def _official_step_transforms(
    public: Path, frame_ids: list[int], partition: str
) -> dict[int, np.ndarray]:
    """Read each adjacent transform once; absent -1 rows mark scene starts."""
    steps: dict[int, np.ndarray] = {}
    for previous, current in zip(frame_ids, frame_ids[1:]):
        if current != previous + 1:
            continue
        official_path = (
            public / "radar_5frames" / partition / "velodyne" / f"{current:05d}.bin"
        )
        if not official_path.is_file():
            raise FileNotFoundError(f"Official VoD five-frame scan missing: {official_path}")
        official = np.memmap(official_path, dtype=np.float32).reshape(-1, 7)
        if not np.any(official[:, 6] == -1):
            continue
        steps[current] = radar_current_from_official_previous(
            _radar_path(public, previous, partition), official_path
        )
    return steps


def _source_to_current(
    history: list[int], steps: dict[int, np.ndarray]
) -> list[np.ndarray]:
    transforms = [np.eye(4, dtype=np.float64) for _ in history]
    for index in range(len(history) - 2, -1, -1):
        transforms[index] = transforms[index + 1] @ steps[history[index + 1]]
    return transforms


def _valid_existing(path: Path) -> bool:
    row_bytes = 7 * np.dtype(np.float32).itemsize
    return (
        path.is_file()
        and path.stat().st_size >= row_bytes
        and path.stat().st_size % row_bytes == 0
    )


def _atomic_tofile(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        points.astype(np.float32, copy=False).tofile(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_one(task: tuple) -> tuple[int, int, int, bool]:
    (public_text, partition, frame_id, history, stack_size, overwrite,
     suffix, filter_values, transforms) = task
    public = Path(public_text)
    variant = f"radar_{stack_size}frames"
    if suffix:
        variant = f"{variant}_{suffix}"
    destination = (
        public
        / variant
        / partition
        / "velodyne"
        / f"{frame_id:05d}.bin"
    )
    if not overwrite and _valid_existing(destination):
        existing = np.fromfile(destination, dtype=np.float32).reshape(-1, 7)
        current = np.fromfile(
            _radar_path(public, frame_id, partition), dtype=np.float32
        ).reshape(-1, 7)
        if filter_values is None:
            if len(existing) < len(current) or not np.array_equal(
                existing[-len(current):, :6], current[:, :6]
            ) or not np.all(existing[-len(current):, 6] == 0):
                raise ValueError(f"Existing radar stack does not end in its raw current scan: {destination}")
            selected = history[-stack_size:]
            expected_rows = sum(
                _radar_path(public, item, partition).stat().st_size // 28
                for item in selected
            )
            if len(existing) != expected_rows:
                raise ValueError(f"Existing radar stack has the wrong scan history: {destination}")
        rows = len(existing)
        return stack_size, frame_id, rows, True

    selected = history[-stack_size:]
    points = accumulate_vod_radar_scans(
        [_radar_path(public, item, partition) for item in selected],
        [_pose_path(public, item, partition) for item in selected],
        [_calibration_path(public, item, partition) for item in selected],
        filter_config=(
            RadarTemporalFilterConfig(**filter_values)
            if filter_values is not None
            else None
        ),
        source_to_current=transforms,
    )
    _atomic_tofile(destination, points)
    return stack_size, frame_id, len(points), False


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("num-workers must be at least one")
    if not args.stack_sizes or any(size < 1 for size in args.stack_sizes):
        raise ValueError("stack sizes must be positive")
    if args.max_step_translation_m <= 0.0:
        raise ValueError("max-step-translation-m must be positive")
    suffix = args.output_suffix.strip("_")
    if args.motion_aware and suffix != "verified_motion_aware":
        raise ValueError("Motion-aware stacks require --output-suffix verified_motion_aware")
    if suffix == "verified_motion_aware" and not args.motion_aware:
        raise ValueError("The verified_motion_aware suffix requires --motion-aware")

    filter_values = None
    if args.basic_validity_filter or args.temporal_filter or args.motion_aware:
        filter_config = RadarTemporalFilterConfig(
            min_range_m=(
                args.radar_min_range_m if args.basic_validity_filter else 0.0
            ),
            max_range_m=(
                args.radar_max_range_m if args.basic_validity_filter else 1.0e9
            ),
            min_height_m=(
                args.radar_min_height_m if args.basic_validity_filter else -1.0e9
            ),
            max_height_m=(
                args.radar_max_height_m if args.basic_validity_filter else 1.0e9
            ),
            min_rcs=(args.radar_min_rcs if args.basic_validity_filter else None),
            max_abs_compensated_velocity_mps=(
                args.radar_max_abs_velocity_mps
                if args.basic_validity_filter
                else None
            ),
            moving_velocity_threshold_mps=(
                args.moving_velocity_threshold_mps if args.motion_aware else None
            ),
            moving_max_age_scans=args.moving_max_age_scans,
            temporal_radius_m=(
                args.temporal_support_radius_m if args.temporal_filter else None
            ),
            temporal_min_scans=args.temporal_min_support_scans,
            preserve_current_scan=args.preserve_current_scan,
        )
        filter_config.validate()
        filter_values = filter_config.__dict__

    public = _public_root(args.vod_root)
    partition = vod_partition_for_split(public, args.split) if args.split else "training"
    radar_root = public / "radar" / partition / "velodyne"
    all_frame_ids = sorted(int(path.stem) for path in radar_root.glob("*.bin"))
    if not all_frame_ids:
        raise FileNotFoundError(f"No single-frame VoD radar files found in {radar_root}")

    stack_sizes = sorted(set(args.stack_sizes))
    if args.alignment_source == "official-5" and suffix not in {"verified", "verified_motion_aware"}:
        raise ValueError("Official five-frame alignment requires a verified output suffix")
    if args.motion_aware and args.alignment_source != "official-5":
        raise ValueError("Motion-aware stacks require --alignment-source official-5")
    if args.split is None:
        frame_ids = all_frame_ids
    else:
        requested_ids = set(load_vod_split_ids(public, args.split))
        frame_ids = [
            frame_id
            for frame_id in all_frame_ids
            if f"{frame_id:05d}" in requested_ids
        ]
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]
    if not frame_ids:
        raise FileNotFoundError("No requested VoD radar target frames were found")
    official_steps = (
        _official_step_transforms(public, all_frame_ids, partition)
        if args.alignment_source == "official-5"
        else None
    )
    histories = _histories(
        public,
        frame_ids if args.isolate_split_history else all_frame_ids,
        max(stack_sizes),
        args.max_step_translation_m,
        partition,
        official_steps,
    )
    for size in stack_sizes:
        variant = f"radar_{size}frames" + (f"_{suffix}" if suffix else "")
        destination = public / variant / partition / "velodyne"
        manifest_name = f"filter_manifest_{args.split}.json" if args.split else "filter_manifest.json"
        manifest_path = destination.parent.parent / manifest_name
        if args.overwrite or not any(
            (destination / f"{frame_id:05d}.bin").is_file()
            for frame_id in frame_ids
        ):
            continue
        if not manifest_path.is_file():
            raise ValueError(f"Existing stack has no provenance manifest: {destination}; use --overwrite")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "alignment_source": args.alignment_source,
            "stack_size": size,
            "basic_validity_filter": args.basic_validity_filter,
            "temporal_filter": args.temporal_filter,
            "filter": filter_values,
            "isolate_split_history": args.isolate_split_history,
        }
        if any(existing.get(key, "pose" if key == "alignment_source" else None) != value
               for key, value in expected.items()):
            raise ValueError(f"Existing stack settings differ: {destination}; use --overwrite")
    tasks = [
        (
            str(public),
            partition,
            frame_id,
            histories[frame_id],
            size,
            args.overwrite,
            suffix,
            filter_values,
            (
                _source_to_current(histories[frame_id][-size:], official_steps)
                if official_steps is not None else None
            ),
        )
        for size in stack_sizes
        for frame_id in frame_ids
    ]

    created = cached = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for index, (size, frame_id, rows, was_cached) in enumerate(
            executor.map(_generate_one, tasks, chunksize=8), 1
        ):
            cached += int(was_cached)
            created += int(not was_cached)
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"Processed {index}/{len(tasks)}; created={created}; "
                    f"cached={cached}; latest={size} scans/{frame_id:05d} "
                    f"({rows} points)",
                    flush=True,
                )

    for size in stack_sizes:
        variant = f"radar_{size}frames" + (f"_{suffix}" if suffix else "")
        destination = public / variant / partition / "velodyne"
        count = sum(1 for _ in destination.glob("*.bin"))
        print(f"{variant}: {count} files in {destination}")
        manifest = {
            "variant": variant,
            "source": "radar",
            "stack_size": size,
            "ego_motion_compensated": True,
            "alignment_source": args.alignment_source,
            "basic_validity_filter": args.basic_validity_filter,
            "temporal_filter": args.temporal_filter,
            "motion_aware": args.motion_aware,
            "filter": filter_values,
            "frames": count,
            "target_frames": len(frame_ids),
            "frames_in_directory": count,
            "target_split": args.split,
            "isolate_split_history": args.isolate_split_history,
            "scene_boundary_source": (
                "official_5frames" if official_steps is not None else "pose_and_id"
            ),
        }
        manifest_name = f"filter_manifest_{args.split}.json" if args.split else "filter_manifest.json"
        (destination.parent.parent / manifest_name).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
