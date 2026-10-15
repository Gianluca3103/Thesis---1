"""A fixed, calibration-aware forward radar region for paired VoD LiDAR."""

from __future__ import annotations

import math

import numpy as np


# The 20-scan radar input is forward facing. Keep this fixed across frames so
# missing radar returns cannot shrink the clean-LiDAR training target.
RADAR_HALF_FOV_DEG = 70.0
RADAR_MAX_RANGE_M = 80.0
RADAR_FOV_SPEC = {
    "coordinate_frame": "radar",
    "half_fov_deg": RADAR_HALF_FOV_DEG,
    "max_range_m": RADAR_MAX_RANGE_M,
}


def crop_lidar_to_radar_fov(
    lidar: np.ndarray,
    lidar_from_radar: np.ndarray,
    *,
    half_fov_deg: float = RADAR_HALF_FOV_DEG,
    max_range_m: float = RADAR_MAX_RANGE_M,
) -> np.ndarray:
    """Keep LiDAR points in the fixed forward radar sector, in radar coordinates.

    The input and output remain in LiDAR coordinates. All original point
    features are preserved; this is a geometric crop, not a point-count cap.
    """
    points = np.asarray(lidar)
    transform = np.asarray(lidar_from_radar, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR points must have at least XYZ columns")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("lidar_from_radar must be a finite 4x4 transform")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("LiDAR XYZ must be finite")
    if not 0 < half_fov_deg < 90 or not math.isfinite(max_range_m) or max_range_m <= 0:
        raise ValueError("Radar sector must have a positive range and a forward half-angle")
    if len(points) == 0:
        return points.copy()
    radar_from_lidar = np.linalg.inv(transform)
    xyz = points[:, :3].astype(np.float64, copy=False)
    radar_xyz = xyz @ radar_from_lidar[:3, :3].T + radar_from_lidar[:3, 3]
    x, y = radar_xyz[:, 0], radar_xyz[:, 1]
    inside = ((x > 0) & (np.abs(np.arctan2(y, x)) <= math.radians(half_fov_deg))
              & (np.linalg.norm(radar_xyz, axis=1) <= max_range_m))
    return points[inside]
