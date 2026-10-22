"""Severe VoD LiDAR faults used to condition Stage-II diffusion.

The fault plan matches the established reconstruction injector: total_loss:1,
fov_filter:5, and fog_sim:5. Fog is generated from the full raw scan once and
cached after the same calibrated radar-FOV crop used by clean Stage-II data.
Only original XYZ returns are exposed to the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FAULT_POLICY_VERSION = "stage2_fog_return_dropout_xyz_v2"
# Raw simulator archives retain their old on-disk format so existing caches work.
FOG_CACHE_VERSION = "stage2_severe_faults_v1"
FAULT_PLAN = (("total_loss", 1), ("fov_filter", 5), ("fog_sim", 5))
FAULT_NAMES = tuple(name for name, _ in FAULT_PLAN)
FOG_ALPHA = 0.06
FOG_NOISE = 10
FOG_CACHE_ENCODING = "float32_byte_shuffle_v1"


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
    def __init__(self, root: Path, split: str, frame_ids: list[str],
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
                or manifest.get("frame_ids") != frame_ids):
            raise ValueError(f"Severe fog cache manifest differs: {manifest_path}")
        self.seed = seed
        if check_complete:
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
        return torch.from_numpy(points[:, :3].copy())


def retain_original_fog_returns(clean_xyz: np.ndarray,
                                simulated_fog: np.ndarray) -> np.ndarray:
    """Keep only original XYZ returns, respecting duplicate multiplicity.

    The simulator replaces some original rays with shifted fog echoes. Its
    cached output preserves no source indices, so exact XYZ matching against
    the clean scan removes those synthetic returns without regenerating fog.
    """
    clean = np.ascontiguousarray(clean_xyz[:, :3], dtype=np.float32)
    fog = np.ascontiguousarray(simulated_fog[:, :3], dtype=np.float32)
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
            clean_xyz.cpu().numpy(), fog_cache.load(frame_id).numpy())
        return torch.from_numpy(filtered), name
    raise ValueError(f"Unsupported Stage-II fault: {name}")


def load_faulty(root: Path, split: str, frame_id: str) -> torch.Tensor:
    """Read the legacy four-column file and expose only XYZ to Stage II."""
    paths = (root / split / f"{frame_id}.bin", root / f"{frame_id}.bin")
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one calibrated four-float LiDAR file for {split}/{frame_id} "
            f"under {root}; found {len(matches)}")
    raw = np.fromfile(matches[0], dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Faulty LiDAR must have four float32 columns: {matches[0]}")
    points = torch.from_numpy(raw.reshape(-1, 4).copy())
    if not torch.isfinite(points).all():
        raise ValueError(f"Faulty LiDAR contains NaN or Inf: {matches[0]}")
    return points[:, :3].contiguous()
