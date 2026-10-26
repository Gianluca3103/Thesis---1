"""Lossless bookkeeping for a full-scan detector comparison."""

from __future__ import annotations

import numpy as np

from models.radar_lidar_stage1.radar_fov import crop_lidar_to_radar_fov


def radar_sector_indices(clean_xyzi: np.ndarray,
                         lidar_from_radar: np.ndarray) -> np.ndarray:
    """Return raw row indices in the exact sector used by Stage I/II."""
    clean = np.asarray(clean_xyzi, dtype=np.float32)
    if clean.ndim != 2 or clean.shape[1] != 4:
        raise ValueError("Expected clean LiDAR [N,4]")
    # float32 represents all practical VoD row indices exactly.
    if len(clean) >= 2**24:
        raise ValueError("Too many LiDAR rows for exact float32 row tags")
    tagged = np.column_stack((clean, np.arange(len(clean), dtype=np.float32)))
    return crop_lidar_to_radar_fov(tagged, lidar_from_radar)[:, 4].astype(np.int64)


def retained_clean_mask(clean_xyz: np.ndarray, retained_xyz: np.ndarray) -> np.ndarray:
    """Match original XYZ rows, preserving duplicate multiplicity and intensity."""
    clean = np.ascontiguousarray(clean_xyz, dtype=np.float32)
    retained = np.ascontiguousarray(retained_xyz, dtype=np.float32)
    if (clean.ndim != 2 or retained.ndim != 2 or clean.shape[1] != 3
            or retained.shape[1] != 3 or not np.isfinite(clean).all()
            or not np.isfinite(retained).all()):
        raise ValueError("Expected finite XYZ arrays")
    if not len(clean):
        if len(retained):
            raise ValueError("Fault contains points absent from clean LiDAR")
        return np.zeros(0, dtype=bool)
    row_type = np.dtype((np.void, 12))
    clean_rows = clean.view(row_type).reshape(-1)
    retained_rows = retained.view(row_type).reshape(-1)
    keys, inverse, clean_counts = np.unique(
        clean_rows, return_inverse=True, return_counts=True)
    requested = np.zeros(len(keys), dtype=np.int64)
    if len(retained):
        wanted, counts = np.unique(retained_rows, return_counts=True)
        locations = np.searchsorted(keys, wanted)
        if (np.any(locations >= len(keys))
                or not np.array_equal(keys[locations], wanted)):
            raise ValueError("Fault contains points absent from clean LiDAR")
        requested[locations] = counts
    if np.any(requested > clean_counts):
        raise ValueError("Fault duplicates original LiDAR returns")
    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    starts = np.maximum.accumulate(np.where(
        np.r_[True, sorted_groups[1:] != sorted_groups[:-1]],
        np.arange(len(clean)), 0))
    occurrence = np.empty(len(clean), dtype=np.int64)
    occurrence[order] = np.arange(len(clean)) - starts
    keep = occurrence < requested[inverse]
    if int(keep.sum()) != len(retained):
        raise AssertionError("Fault matching lost LiDAR rows")
    return keep


def detector_clouds(clean_xyzi: np.ndarray, sector_indices: np.ndarray,
                    retained_sector_xyz: np.ndarray,
                    additions_xyz: np.ndarray, *,
                    retained_full_xyz: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Export a full-scan fault and add Stage II XYZ with zero intensity.

    Fog has only a radar-sector simulation cache, so its outside returns stay
    unchanged. Total-loss and FoV faults supply their full-scan retained XYZ.
    """
    clean = np.asarray(clean_xyzi, dtype=np.float32)
    sector = np.asarray(sector_indices, dtype=np.int64)
    if len(np.unique(sector)) != len(sector) or np.any(sector < 0) or np.any(sector >= len(clean)):
        raise ValueError("Invalid radar-sector row indices")
    additions = np.asarray(additions_xyz, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(additions).all():
        raise ValueError("Non-finite reconstructed point")
    sector_keep = retained_clean_mask(clean[sector, :3], retained_sector_xyz)
    if retained_full_xyz is None:
        keep = np.ones(len(clean), dtype=bool)
        keep[sector] = sector_keep
    else:
        keep = retained_clean_mask(clean[:, :3], retained_full_xyz)
        if not np.array_equal(keep[sector], sector_keep):
            raise ValueError("Full-scan fault differs from Stage-II sector fault")
    faulty = np.ascontiguousarray(clean[keep])
    generated = np.column_stack((additions, np.zeros(len(additions), dtype=np.float32)))
    completed = np.ascontiguousarray(np.concatenate((faulty, generated), axis=0))
    return faulty, completed
