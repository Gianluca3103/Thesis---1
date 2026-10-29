"""Severe VoD LiDAR faults used to condition Stage-II diffusion.

The fault plan matches the established reconstruction injector: total_loss:1,
fov_filter:5, and fog_sim:5. Fog is generated from the full raw scan once and
cached after the same calibrated radar-FOV crop used by clean Stage-II data.
Only detected original XYZ returns are exposed to the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FAULT_POLICY_VERSION = "stage2_fog_intensity_detection_xyz_v3"
# Raw simulator archives retain their old on-disk format so existing caches work.
FOG_CACHE_VERSION = "stage2_severe_faults_v1"
FAULT_PLAN = (("total_loss", 1), ("fov_filter", 5), ("fog_sim", 5))
FAULT_NAMES = tuple(name for name, _ in FAULT_PLAN)
FOG_ALPHA = 0.06
FOG_NOISE = 10
FOG_CACHE_ENCODING = "float32_byte_shuffle_v1"
# The established laser-degradation injector uses the clean scan's 10th
# intensity percentile as its detection threshold. The fog simulator rounds
# attenuated intensity to integer units, so a zero response is never a return.
FOG_DETECTION_QUANTILE = 0.10
FOG_MIN_DETECTABLE_INTENSITY = 1.0


def encode_fog_points(points: np.ndarray) -> np.ndarray:
    """Rearrange float32 bytes for lossless, fast ZIP compression."""
    values = np.ascontiguousarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Fog cache expects XYZ and reflectivity [N,4]")
    return np.frombuffer(values.tobytes(), dtype=np.uint8).reshape(-1, 4).T.copy()


def decode_fog_points(encoded: np.ndarray, count: int) -> np.ndarray:
    if (encoded.dtype != np.uint8 or encoded.shape != (4, count * 4)
            or count < 0):
        raise ValueError("Invalid byte-shuffled fog scan")
    return np.ascontiguousarray(encoded.T).reshape(-1).view(np.float32).reshape(
        count, 4).copy()


def selected_fault(frame_id: str, epoch: int, seed: int = 3103) -> str:
    """Rotate each frame through all three faults in successive train epochs."""
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode()).digest()
    base = int.from_bytes(digest[:8], "little") % len(FAULT_NAMES)
    return FAULT_NAMES[(base + int(epoch)) % len(FAULT_NAMES)]


def severe_fov_filter(points: np.ndarray) -> np.ndarray:
    """Exact severity-5 filter from 3D_Corruptions_AD/LiDAR_corruptions.py."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("FOV filter expects LiDAR XYZ rows")
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = (np.arctan(points[:, 0] / points[:, 1])
                 + (points[:, 1] < 0) * np.pi + 2 * np.pi) % (2 * np.pi)
    angle[angle > np.pi] -= 2 * np.pi
    degrees = angle / np.pi * 180
    return points[(degrees >= -45) & (degrees <= 45)]


class SevereFogCache:
    def __init__(self, root: Path, split: str, frame_ids: list[str] | None,
                 *, seed: int = 3103, check_complete: bool = True):
        self.root = Path(root) / split
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("version") != FOG_CACHE_VERSION
                or manifest.get("split") != split
                or manifest.get("seed") != seed
                or manifest.get("fault") != "fog_sim"
                or manifest.get("severity") != 5
                or manifest.get("alpha") != FOG_ALPHA
                or manifest.get("noise") != FOG_NOISE
                or (frame_ids is not None
                    and manifest.get("frame_ids") != frame_ids)):
            raise ValueError(f"Severe fog cache manifest differs: {manifest_path}")
        self.seed = seed
        if check_complete:
            if frame_ids is None:
                raise ValueError("Complete fog-cache validation requires frame IDs")
            missing = [frame for frame in frame_ids if not self.path(frame).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Severe fog cache missing {len(missing)} frames, first {missing[0]}")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("VoD frame ID must be numeric")
        return self.root / f"{frame_id}.npz"

    def load(self, frame_id: str) -> torch.Tensor:
        with np.load(self.path(frame_id), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("encoding") == FOG_CACHE_ENCODING:
                points = decode_fog_points(
                    archive["points_bytes"], int(metadata["cropped_points"]))
            elif metadata.get("encoding") is None:
                # Read caches made before byte shuffling so the builder can
                # compact them in place without regenerating fog.
                points = np.asarray(archive["points"], dtype=np.float32)
            else:
                raise ValueError(f"Unknown severe fog encoding: {self.path(frame_id)}")
        if (metadata.get("version") != FOG_CACHE_VERSION
                or metadata.get("frame_id") != frame_id
                or metadata.get("fault") != "fog_sim"
                or metadata.get("severity") != 5
                or metadata.get("seed") != self.seed
                or metadata.get("alpha") != FOG_ALPHA
                or metadata.get("noise") != FOG_NOISE
                or points.ndim != 2 or points.shape[1] != 4
                or len(points) != metadata.get("cropped_points")
                or not np.isfinite(points).all()):
            raise ValueError(f"Invalid severe fog cache entry: {self.path(frame_id)}")
        # Keep the simulated intensity until the return-detection decision.
        # severe_fault exposes XYZ only after filtering.
        return torch.from_numpy(points.copy())


def fog_detection_floor(clean_lidar: np.ndarray) -> float:
    """Per-frame receiver floor, in the simulator's original intensity units."""
    clean = np.asarray(clean_lidar, dtype=np.float32)
    if clean.ndim != 2 or clean.shape[1] < 4:
        raise ValueError("Clean LiDAR must contain XYZ and original intensity")
    if not np.isfinite(clean[:, :4]).all():
        raise ValueError("Clean LiDAR contains nonfinite XYZ or intensity")
    if not len(clean):
        return FOG_MIN_DETECTABLE_INTENSITY
    return max(FOG_MIN_DETECTABLE_INTENSITY,
               float(np.quantile(clean[:, 3], FOG_DETECTION_QUANTILE)))


def retain_original_fog_returns(clean_lidar: np.ndarray,
                                simulated_fog: np.ndarray) -> np.ndarray:
    """Keep detectable original XYZ returns, respecting duplicate multiplicity.

    Hard fog attenuates and rounds intensity without moving most XYZ points.
    Those below the detection floor must disappear. Soft fog also replaces
    some original rays with shifted echoes; exact XYZ matching removes those
    synthetic returns without exposing intensity to the model.
    """
    clean_lidar = np.asarray(clean_lidar, dtype=np.float32)
    simulated_fog = np.asarray(simulated_fog, dtype=np.float32)
    floor = fog_detection_floor(clean_lidar)
    if simulated_fog.ndim != 2 or simulated_fog.shape[1] < 4:
        raise ValueError("Simulated fog must contain XYZ and attenuated intensity")
    if not np.isfinite(simulated_fog[:, :4]).all():
        raise ValueError("Simulated fog contains nonfinite XYZ or intensity")
    clean = np.ascontiguousarray(clean_lidar[:, :3])
    detected = simulated_fog[simulated_fog[:, 3] >= floor]
    fog = np.ascontiguousarray(detected[:, :3])
    if not len(clean) or not len(fog):
        return fog[:0].copy()
    row_type = np.dtype((np.void, 3 * np.dtype(np.float32).itemsize))
    clean_rows = clean.view(row_type).reshape(-1)
    fog_rows = fog.view(row_type).reshape(-1)
    unique, clean_counts = np.unique(clean_rows, return_counts=True)
    positions = np.searchsorted(unique, fog_rows)
    safe = np.minimum(positions, len(unique) - 1)
    present = (positions < len(unique)) & (unique[safe] == fog_rows)
    order = np.argsort(fog_rows, kind="stable")
    ordered = fog_rows[order]
    group_start = np.maximum.accumulate(np.where(
        np.r_[True, ordered[1:] != ordered[:-1]], np.arange(len(fog)), 0))
    occurrence = np.empty(len(fog), dtype=np.int64)
    occurrence[order] = np.arange(len(fog)) - group_start
    return fog[present & (occurrence < clean_counts[safe])].copy()


def severe_fault(clean: torch.Tensor, frame_id: str, epoch: int,
                 fog_cache: SevereFogCache, *, seed: int = 3103,
                 fault_type: str | None = None) -> tuple[torch.Tensor, str]:
    """Use one established severe fault; clean must be the cropped clean scan."""
    name = fault_type or selected_fault(frame_id, epoch, seed)
    clean_xyz = clean[:, :3]
    if name == "total_loss":
        return clean_xyz[:0], name
    if name == "fov_filter":
        filtered = severe_fov_filter(clean_xyz.cpu().numpy())
        return torch.from_numpy(filtered.copy()), name
    if name == "fog_sim":
        filtered = retain_original_fog_returns(
            clean.cpu().numpy(), fog_cache.load(frame_id).numpy())
        return torch.from_numpy(filtered), name
    raise ValueError(f"Unsupported Stage-II fault: {name}")


def load_faulty(root: Path, split: str, frame_id: str) -> torch.Tensor:
    """Read legacy four-column BIN or compressed mild-fault XYZ for Stage II."""
    paths = (root / split / f"{frame_id}.bin", root / f"{frame_id}.bin",
             root / split / f"{frame_id}.npz", root / f"{frame_id}.npz")
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one calibrated faulty LiDAR file for {split}/{frame_id} "
            f"under {root}; found {len(matches)}")
    if matches[0].suffix == ".npz":
        from .mild_fault_data import (
            MILD_FAULT_ENCODING, MILD_FAULT_TYPES, MILD_FAULT_VERSION,
            decode_mild_xyz,
        )
        with np.load(matches[0], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("version") != MILD_FAULT_VERSION
                    or metadata.get("frame_id") != frame_id
                    or metadata.get("split") != split
                    or metadata.get("fault_type") not in MILD_FAULT_TYPES
                    or metadata.get("encoding") != MILD_FAULT_ENCODING):
                raise ValueError(f"Invalid mild-fault metadata: {matches[0]}")
            points = decode_mild_xyz(
                archive["points_bytes"], int(metadata["retained_points"]))
        return torch.from_numpy(points)
    raw = np.fromfile(matches[0], dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Faulty LiDAR must have four float32 columns: {matches[0]}")
    points = torch.from_numpy(raw.reshape(-1, 4).copy())
    if not torch.isfinite(points).all():
        raise ValueError(f"Faulty LiDAR contains NaN or Inf: {matches[0]}")
    return points[:, :3].contiguous()
