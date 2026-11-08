"""Oracle diagnostic gate: radar-supported voxels wholly inside the injected fault.

The injector footprint is known for synthetic faults only. This mask is not
available for an unknown fault at deployment time.
"""

from __future__ import annotations

import math

import numpy as np

from models.radar_lidar_crossmodal_v1.radar_pruner import SHAPE
from .sparse_geometry import GRID


POLICY = "radar_support_intersect_injected_fault_inward_voxel_margin_v1"


def fault_only_support(mask: np.ndarray, region: dict) -> np.ndarray:
    """Keep only radar-supported cells with no XY overlap with the reliable sector.

    For partial azimuth loss, a voxel's center must be inside the removed
    sector by at least its conservative angular half diagonal. Fog touched the
    full scan, so its radar gate is unchanged.
    """
    if mask.shape != SHAPE or mask.dtype != np.bool_:
        raise ValueError(f"Radar support mask must be bool {SHAPE}")
    kind = region.get("kind")
    if kind == "full_scan":
        return mask.copy()
    if kind == "empty":
        return np.zeros_like(mask)
    if kind != "azimuth_sector":
        raise ValueError(f"Unknown injector footprint: {kind}")
    start = float(region["start_rad"])
    end = float(region["end_rad"])
    if not all(math.isfinite(v) and 0 <= v < math.tau for v in (start, end)):
        raise ValueError("Invalid injector azimuth bounds")
    out = np.zeros_like(mask)
    z, y, x = np.nonzero(mask)
    if not len(x):
        return out
    px = GRID.minimum_xyz[0] + (x.astype(np.float64) + 0.5) * GRID.voxel_xyz[0]
    py = GRID.minimum_xyz[1] + (y.astype(np.float64) + 0.5) * GRID.voxel_xyz[1]
    radius = np.hypot(px, py)
    half_diagonal = math.hypot(GRID.voxel_xyz[0], GRID.voxel_xyz[1]) * 0.5
    margin = np.arcsin(np.minimum(1.0, half_diagonal / np.maximum(radius, half_diagonal)))
    relative = (np.arctan2(py, px) - start) % math.tau
    width = (end - start) % math.tau
    inside = (relative >= margin) & (relative <= width - margin)
    out[z[inside], y[inside], x[inside]] = True
    return out
