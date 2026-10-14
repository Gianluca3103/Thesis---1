"""Ego-motion compensated accumulation of View-of-Delft radar scans."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    cKDTree = None

from Fault_Localization_Model.vod_dataset.vod_io import (
    VOD_RADAR_FIELDS,
    _named_transform,
    load_vod_radar,
)


@dataclass(frozen=True)
class RadarTemporalFilterConfig:
    """Validity and temporal-consistency gates for an aligned radar stack."""

    min_range_m: float = 1.0
    max_range_m: float = 80.0
    min_height_m: float = -5.0
    max_height_m: float = 5.0
    min_rcs: float | None = None
    max_abs_compensated_velocity_mps: float | None = None
    moving_velocity_threshold_mps: float | None = None
    moving_max_age_scans: int = 1
    temporal_radius_m: float | None = None
    temporal_min_scans: int = 2
    preserve_current_scan: bool = True

    def validate(self) -> None:
        if self.min_range_m < 0.0:
            raise ValueError("min_range_m must be non-negative")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if self.max_height_m <= self.min_height_m:
            raise ValueError("max_height_m must exceed min_height_m")
        if self.temporal_radius_m is not None and self.temporal_radius_m <= 0.0:
            raise ValueError("temporal_radius_m must be positive when enabled")
        if self.temporal_min_scans < 2:
            raise ValueError("temporal_min_scans must be at least two")
        if (
            self.max_abs_compensated_velocity_mps is not None
            and self.max_abs_compensated_velocity_mps <= 0.0
        ):
            raise ValueError(
                "max_abs_compensated_velocity_mps must be positive when enabled"
            )
        if (self.moving_velocity_threshold_mps is not None
                and self.moving_velocity_threshold_mps <= 0.0):
            raise ValueError("moving_velocity_threshold_mps must be positive")
        if self.moving_max_age_scans < 0:
            raise ValueError("moving_max_age_scans must be non-negative")


def filter_accumulated_radar_points(
    radar_points: np.ndarray,
    config: RadarTemporalFilterConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter an ego-motion-aligned stack without changing its seven fields.

    Temporal support is measured in XY and must come from distinct scans. The
    current scan is retained by default so newly visible or moving objects are
    not erased merely because ego-motion compensation cannot align their motion.
    """

    config.validate()
    points = np.asarray(radar_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != len(VOD_RADAR_FIELDS):
        raise ValueError(
            f"radar_points must have shape [N,{len(VOD_RADAR_FIELDS)}], got "
            f"{points.shape}"
        )

    initial_count = len(points)
    finite = np.isfinite(points).all(axis=1)
    ranges = np.linalg.norm(points[:, :2], axis=1)
    valid = (
        finite
        & (ranges >= config.min_range_m)
        & (ranges <= config.max_range_m)
        & (points[:, 2] >= config.min_height_m)
        & (points[:, 2] <= config.max_height_m)
    )
    # Accumulated releases use integer-valued relative scan indices.
    valid &= np.abs(points[:, 6] - np.rint(points[:, 6])) <= 1.0e-4
    if config.min_rcs is not None:
        valid &= points[:, 3] >= config.min_rcs
    if config.max_abs_compensated_velocity_mps is not None:
        valid &= (
            np.abs(points[:, 5]) <= config.max_abs_compensated_velocity_mps
        )

    points = points[valid]
    after_validity = len(points)
    after_motion = after_validity
    if config.moving_velocity_threshold_mps is not None:
        # Doppler gives only radial motion. Do not extrapolate a 3D position
        # from it: retain recent moving returns and remove older trails.
        moving = np.abs(points[:, 5]) >= config.moving_velocity_threshold_mps
        stale = points[:, 6] < -config.moving_max_age_scans
        points = points[~(moving & stale)]
        after_motion = len(points)
    temporal_keep = np.ones(after_motion, dtype=bool)
    if config.temporal_radius_m is not None and after_motion:
        time_indices = np.rint(points[:, 6]).astype(np.int32)
        radius = config.temporal_radius_m
        if cKDTree is not None:
            neighborhoods = cKDTree(points[:, :2]).query_ball_point(
                points[:, :2],
                r=radius,
            )

            def has_support(point_index: int) -> bool:
                return (
                    len(np.unique(time_indices[neighborhoods[point_index]]))
                    >= config.temporal_min_scans
                )

        else:
            cells = np.floor(points[:, :2] / radius).astype(np.int64)
            cell_members: dict[tuple[int, int], list[int]] = {}
            for point_index, cell in enumerate(cells):
                cell_members.setdefault((int(cell[0]), int(cell[1])), []).append(
                    point_index
                )
            radius_squared = radius * radius

            def has_support(point_index: int) -> bool:
                cell_x, cell_y = cells[point_index]
                candidate_indices: list[int] = []
                for offset_x in (-1, 0, 1):
                    for offset_y in (-1, 0, 1):
                        candidate_indices.extend(
                            cell_members.get(
                                (int(cell_x + offset_x), int(cell_y + offset_y)),
                                (),
                            )
                        )
                candidate_array = np.asarray(candidate_indices, dtype=np.int64)
                offsets = points[candidate_array, :2] - points[point_index, :2]
                nearby = candidate_array[
                    np.einsum("ij,ij->i", offsets, offsets) <= radius_squared
                ]
                return (
                    len(np.unique(time_indices[nearby]))
                    >= config.temporal_min_scans
                )

        temporal_keep = np.fromiter(
            (has_support(index) for index in range(after_motion)),
            dtype=bool,
            count=after_motion,
        )
        if config.preserve_current_scan:
            temporal_keep |= time_indices == 0
        points = points[temporal_keep]

    return points, {
        "input_points": initial_count,
        "validity_rejected": initial_count - after_validity,
        "motion_rejected": after_validity - after_motion,
        "temporal_rejected": after_motion - len(points),
        "output_points": len(points),
    }


def load_vod_odom_from_camera(path: str | Path) -> np.ndarray:
    """Load VoD's camera-to-odometry pose from one per-frame JSON file."""

    path = Path(path)
    values = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "odomToCamera" in payload:
            values = payload["odomToCamera"]
            break
    if values is None:
        raise ValueError(f"odomToCamera was not found in {path}")
    transform = np.asarray(values, dtype=np.float64)
    if transform.size != 16 or not np.isfinite(transform).all():
        raise ValueError(f"Malformed odomToCamera in {path}")
    transform = transform.reshape(4, 4)
    if abs(np.linalg.det(transform[:3, :3])) < 1e-10:
        raise ValueError(f"Singular odomToCamera rotation in {path}")
    return transform


def radar_current_from_source(
    source_pose_path: str | Path,
    current_pose_path: str | Path,
    source_calibration_path: str | Path,
    current_calibration_path: str | Path,
) -> np.ndarray:
    """Return the rigid transform from a source radar scan to current radar."""

    odom_from_source_camera = load_vod_odom_from_camera(source_pose_path)
    odom_from_current_camera = load_vod_odom_from_camera(current_pose_path)
    source_camera_from_radar = _named_transform(source_calibration_path)
    current_camera_from_radar = _named_transform(current_calibration_path)
    return (
        np.linalg.inv(current_camera_from_radar)
        @ np.linalg.inv(odom_from_current_camera)
        @ odom_from_source_camera
        @ source_camera_from_radar
    )


def radar_current_from_official_previous(
    previous_raw_path: str | Path,
    current_official_five_path: str | Path,
    *,
    maximum_residual_m: float = 0.001,
) -> np.ndarray:
    """Recover VoD's preceding-scan transform from its released five-scan stack.

    The official stack retains each source scan's row order and measured fields.
    Fitting one rigid transform to those corresponding rows reproduces the
    released coordinates without relying on a different odometry estimate.
    Missing ``-1`` rows mean the current frame begins a new sequence.
    """

    source = load_vod_radar(previous_raw_path)
    official = load_vod_radar(current_official_five_path)
    previous = official[official[:, 6] == -1]
    if not len(previous):
        raise ValueError(f"No preceding scan in {current_official_five_path}")
    if len(source) != len(previous):
        raise ValueError("Official preceding scan does not match raw point count")
    if not np.array_equal(source[:, 3:6], previous[:, 3:6]):
        raise ValueError("Official preceding scan changed measured radar fields or row order")
    source_xyz = source[:, :3].astype(np.float64)
    target_xyz = previous[:, :3].astype(np.float64)
    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source_xyz - source_center).T @ (target_xyz - target_center)
    )
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    row_rotation = left @ correction @ right
    fitted = (source_xyz - source_center) @ row_rotation + target_center
    residual = np.linalg.norm(fitted - target_xyz, axis=1)
    if residual.max() > maximum_residual_m:
        raise ValueError(
            f"Official five-scan alignment is not rigid: max residual "
            f"{residual.max():.6f} m in {current_official_five_path}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = row_rotation.T
    transform[:3, 3] = target_center - source_center @ row_rotation
    return transform


def transform_radar_scan(
    radar_points: np.ndarray,
    current_from_source: np.ndarray,
    time_index: int,
) -> np.ndarray:
    """Transform radar XYZ and label the scan while retaining measured fields."""

    radar_points = np.asarray(radar_points, dtype=np.float32)
    if radar_points.ndim != 2 or radar_points.shape[1] != len(VOD_RADAR_FIELDS):
        raise ValueError(
            f"radar_points must have shape [N,{len(VOD_RADAR_FIELDS)}], got "
            f"{radar_points.shape}"
        )
    transform = np.asarray(current_from_source, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("current_from_source must be a finite 4x4 transform")
    homogeneous = np.column_stack(
        (radar_points[:, :3], np.ones(len(radar_points), dtype=np.float64))
    )
    output = radar_points.copy()
    output[:, :3] = (homogeneous @ transform.T)[:, :3].astype(np.float32)
    output[:, 6] = float(time_index)
    return output


def accumulate_vod_radar_scans(
    source_paths: list[str | Path],
    pose_paths: list[str | Path],
    calibration_paths: list[str | Path],
    *,
    filter_config: RadarTemporalFilterConfig | None = None,
    source_to_current: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Align chronological source scans into the final scan's radar frame."""

    if not source_paths:
        raise ValueError("At least one source radar scan is required")
    if not (len(source_paths) == len(pose_paths) == len(calibration_paths)):
        raise ValueError("Radar, pose, and calibration path counts must match")
    if source_to_current is not None and len(source_to_current) != len(source_paths):
        raise ValueError("A source-to-current transform is required for each scan")

    current_pose = pose_paths[-1]
    current_calibration = calibration_paths[-1]
    count = len(source_paths)
    scans = []
    for index, (radar_path, pose_path, calibration_path) in enumerate(
        zip(source_paths, pose_paths, calibration_paths)
    ):
        time_index = index - count + 1
        points = load_vod_radar(
            radar_path,
            allow_nonfinite=filter_config is not None,
        )
        if source_to_current is not None:
            transform = source_to_current[index]
        elif time_index == 0:
            transform = np.eye(4, dtype=np.float64)
        else:
            transform = radar_current_from_source(
                pose_path,
                current_pose,
                calibration_path,
                current_calibration,
            )
        scans.append(transform_radar_scan(points, transform, time_index))
    accumulated = np.concatenate(scans, axis=0)
    if filter_config is not None:
        accumulated, _ = filter_accumulated_radar_points(
            accumulated,
            filter_config,
        )
    return accumulated
