"""Deterministic, spatial partial LiDAR loss for the mild-fault dataset."""

from __future__ import annotations

import hashlib

import numpy as np


MILD_FAULT_VERSION = "stage2_mild_fog5_partial_v1"
MILD_FAULT_TYPES = ("fog_sim_5", "partial_loss_30", "partial_loss_40",
                    "partial_loss_50")
MILD_FAULT_ENCODING = "xyz_float32_byte_shuffle_v1"


def frame_seed(seed: int, split: str, frame_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{frame_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def balanced_fault_assignment(frame_ids: list[str], split: str,
                              seed: int) -> dict[str, str]:
    """Assign one of four faults per frame, with counts differing by at most one."""
    shuffled = list(frame_ids)
    rng = np.random.default_rng(frame_seed(seed, split, "assignment"))
    rng.shuffle(shuffled)
    return {frame_id: MILD_FAULT_TYPES[index % len(MILD_FAULT_TYPES)]
            for index, frame_id in enumerate(shuffled)}


def partial_azimuth_loss(clean_xyz: np.ndarray, remove_fraction: float,
                         seed: int) -> np.ndarray:
    """Remove a contiguous angular sector containing exactly 30/40/50% of returns.

    This creates a missing region instead of uniformly thinning the whole cloud.
    It keeps the surviving XYZ coordinates and their original point order.
    """
    points = np.asarray(clean_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Partial loss expects finite LiDAR XYZ [N,3]")
    if remove_fraction not in (0.30, 0.40, 0.50):
        raise ValueError("Partial loss must remove 30%, 40%, or 50%")
    count = len(points)
    remove_count = min(int(np.floor(remove_fraction * count + 0.5)), count // 2)
    if not remove_count:
        return points.copy()
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(azimuth, kind="stable")
    start = int(np.random.default_rng(seed).integers(count))
    removed = order[(start + np.arange(remove_count)) % count]
    keep = np.ones(count, dtype=bool)
    keep[removed] = False
    return np.ascontiguousarray(points[keep])


def encode_mild_xyz(points: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Mild-fault cache expects finite XYZ [N,3]")
    return np.frombuffer(values.tobytes(), dtype=np.uint8).reshape(-1, 4).T.copy()


def decode_mild_xyz(encoded: np.ndarray, count: int) -> np.ndarray:
    if (count < 0 or encoded.dtype != np.uint8
            or encoded.shape != (4, count * 3)):
        raise ValueError("Invalid mild-fault byte-shuffled XYZ")
    points = np.ascontiguousarray(encoded.T).reshape(-1).view(np.float32).reshape(
        count, 3).copy()
    if not np.isfinite(points).all():
        raise ValueError("Mild-fault XYZ contains NaN or Inf")
    return points
