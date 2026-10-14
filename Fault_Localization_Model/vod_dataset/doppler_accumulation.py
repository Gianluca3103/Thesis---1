"""Doppler-based radial alignment of an already ego-aligned VoD radar stack.

VoD's seventh radar field is scan age, not elapsed seconds. The caller must
explicitly supply a frame period; a period inferred from nominal sensor rates
is an approximation until original radar timestamps are available.
"""

from __future__ import annotations

import numpy as np

from .vod_io import VOD_RADAR_FIELDS


def _corresponding_rigid_transform(
    source: np.ndarray, aligned: np.ndarray, *, maximum_residual_m: float = 0.001
) -> np.ndarray:
    """Recover the official ego transform from corresponding radar rows."""
    if source.shape != aligned.shape or source.ndim != 2 or source.shape[1] != 7:
        raise ValueError("Source and aligned scans must have matching [N,7] shapes")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(aligned).all():
        raise ValueError("At least three finite corresponding radar points are required")
    if not np.array_equal(source[:, 3:6], aligned[:, 3:6]):
        raise ValueError("Verified stack changed source RCS or Doppler row order")
    source_xyz = source[:, :3].astype(np.float64)
    aligned_xyz = aligned[:, :3].astype(np.float64)
    source_mean = source_xyz.mean(axis=0)
    aligned_mean = aligned_xyz.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source_xyz - source_mean).T @ (aligned_xyz - aligned_mean)
    )
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    row_rotation = left @ correction @ right
    offset = aligned_mean - source_mean @ row_rotation
    residual = np.linalg.norm(source_xyz @ row_rotation + offset - aligned_xyz, axis=1)
    if float(residual.max()) > maximum_residual_m:
        raise ValueError(f"Verified ego alignment is not rigid: {residual.max():.6f} m")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = row_rotation.T
    transform[:3, 3] = offset
    return transform


def radial_compensate_verified_stack(
    aligned_stack: np.ndarray,
    source_scans_by_age: dict[int, np.ndarray],
    *,
    frame_period_s: float,
    doppler_sign: int = 1,
    dispersion_tolerance_m: float | None = 2.0,
    lateral_speed_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Shift old radar returns in their source line of sight, then ego-align.

    ``dispersion_tolerance_m`` provides a per-point age budget
    ``abs(v_r_comp) * age_seconds * lateral_speed_ratio <= tolerance``.
    The constant lateral-speed ratio is a VoD-oriented approximation to
    DoppDrive's angle-dependent heading prior, not a reproduction of it.
    The current scan is copied exactly and all seven fields are preserved.
    """
    points = np.asarray(aligned_stack, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != len(VOD_RADAR_FIELDS):
        raise ValueError("Aligned stack must have shape [N,7]")
    if not np.isfinite(points).all():
        raise ValueError("Aligned stack contains non-finite radar measurements")
    if not np.isfinite(frame_period_s) or frame_period_s <= 0:
        raise ValueError("frame_period_s must be finite and positive")
    if doppler_sign not in (-1, 1):
        raise ValueError("doppler_sign must be +1 or -1")
    if dispersion_tolerance_m is not None and (
        not np.isfinite(dispersion_tolerance_m) or dispersion_tolerance_m <= 0
    ):
        raise ValueError("dispersion_tolerance_m must be positive or None")
    if not np.isfinite(lateral_speed_ratio) or lateral_speed_ratio < 0:
        raise ValueError("lateral_speed_ratio must be finite and non-negative")
    ages = np.rint(points[:, 6]).astype(np.int32)
    if not np.all(np.abs(points[:, 6] - ages) < 1e-4) or np.any(ages > 0):
        raise ValueError("Stack must contain non-positive integer scan ages")
    output = points.copy()
    keep = np.ones(len(points), dtype=bool)
    shifted = rejected = 0
    for age in sorted(set(ages)):
        positions = np.flatnonzero(ages == age)
        source = source_scans_by_age.get(int(age))
        if source is None:
            raise ValueError(f"Missing source scan for age {age}")
        source = np.asarray(source, dtype=np.float32)
        if source.shape != (len(positions), len(VOD_RADAR_FIELDS)):
            raise ValueError(f"Source scan for age {age} does not match stack rows")
        aligned = points[positions]
        if age == 0:
            if not np.array_equal(source[:, :6], aligned[:, :6]):
                raise ValueError("Verified current scan differs from raw radar")
            continue
        transform = _corresponding_rigid_transform(source, aligned)
        elapsed = -int(age) * frame_period_s
        planar_range = np.linalg.norm(source[:, :2].astype(np.float64), axis=1)
        radial_unit = np.divide(
            source[:, :2], planar_range[:, None],
            out=np.zeros((len(source), 2), dtype=np.float64),
            where=planar_range[:, None] > 1e-6,
        )
        shifted_source = source[:, :3].astype(np.float64).copy()
        shifted_source[:, :2] += (
            doppler_sign * source[:, 5].astype(np.float64) * elapsed
        )[:, None] * radial_unit
        output[positions, :3] = (
            shifted_source @ transform[:3, :3].T + transform[:3, 3]
        ).astype(np.float32)
        shifted += len(positions)
        if dispersion_tolerance_m is not None:
            within_budget = (
                np.abs(source[:, 5].astype(np.float64))
                * elapsed * lateral_speed_ratio <= dispersion_tolerance_m
            )
            keep[positions] = within_budget
            rejected += int((~within_budget).sum())
    return output[keep], {
        "input_points": len(points),
        "shifted_historical_points": shifted,
        "window_rejected": rejected,
        "output_points": int(keep.sum()),
        "frame_period_s": frame_period_s,
    }
