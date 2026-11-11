"""Whole-frame radar adapter and matched metrics for the original D-PCC code.

The filtered cache is the existing >=12 distinct radar voxels within 1.6 m
cache. No LiDAR is loaded. D-PCC itself never sees the original unfiltered cloud.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


GRID_MIN = np.array((-51.2, -51.2, -2.0), dtype=np.float64)
GRID_MAX = np.array((51.2, 51.2, 2.0), dtype=np.float64)
GRID_SIZE = np.array((0.2, 0.2, 0.25), dtype=np.float64)
# The original DownsampleLayer reduces N to round(N/3) three times. It also
# lowers its own k when fewer than 16 neighbors exist. N=14 -> 5 -> 2 -> 1;
# N=13 -> 4 -> 1 -> 0, which the CUDA model cannot process.
MIN_INPUT_POINTS = 14
ATTR_NAMES = ("rcs", "radial", "compensated_radial", "scan_age")
FILTER_VERSION = "filtered_radar_self_reconstruction_v1"


class FilteredFrames:
    def __init__(self, root: Path, split: str, limit: int | None = None):
        self.root = Path(root) / split
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("version") != FILTER_VERSION
                or self.manifest.get("split") != split
                or self.manifest.get("min_distinct_radar_voxels") != 12
                or self.manifest.get("radius_m") != 1.6
                or tuple(self.manifest.get("grid_minimum_xyz", ())) != tuple(GRID_MIN)
                or tuple(self.manifest.get("grid_maximum_xyz", ())) != tuple(GRID_MAX)
                or tuple(self.manifest.get("grid_size_xyz", ())) != tuple(GRID_SIZE)):
            raise ValueError(f"Wrong radar filter or grid: {self.manifest_path}")
        self.ids = self.manifest["frame_ids"][:limit]
        missing = [item for item in self.ids if not (self.root / f"{item}.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} filtered radar frames missing; first {missing[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, index: int) -> torch.Tensor:
        frame_id = self.ids[index]
        row = torch.load(self.root / f"{frame_id}.pt", map_location="cpu", weights_only=True)
        points = row.get("points")
        if (row.get("version") != FILTER_VERSION or row.get("frame_id") != frame_id
                or not isinstance(points, torch.Tensor) or points.ndim != 2
                or points.shape[1] != 7 or not torch.isfinite(points).all()):
            raise ValueError(f"Invalid filtered radar frame {frame_id}")
        if len(points) < MIN_INPUT_POINTS:
            raise ValueError(f"Three upstream D-PCC downsamplings require >= {MIN_INPUT_POINTS} input points; "
                             f"{frame_id} has {len(points)}. No points are synthesized or skipped.")
        return points.float()

    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()


class RadarScale:
    """Invertible train-only mapping of metric XYZ and 4 attributes to [-1,1]."""

    def __init__(self, minimum: list[float], maximum: list[float]):
        self.minimum = torch.tensor(minimum, dtype=torch.float32)
        self.maximum = torch.tensor(maximum, dtype=torch.float32)
        if self.minimum.shape != (4,) or self.maximum.shape != (4,):
            raise ValueError("Expected four radar attribute bounds")
        if not torch.all(self.maximum > self.minimum):
            raise ValueError("Radar attribute bounds are degenerate")

    @classmethod
    def fit(cls, frames: FilteredFrames) -> "RadarScale":
        minimum = torch.full((4,), float("inf"))
        maximum = torch.full((4,), float("-inf"))
        for i in range(len(frames)):
            attrs = frames.load(i)[:, 3:]
            minimum = torch.minimum(minimum, attrs.amin(0))
            maximum = torch.maximum(maximum, attrs.amax(0))
        maximum = torch.maximum(maximum, minimum + 1e-6)
        return cls(minimum.tolist(), maximum.tolist())

    def to_json(self) -> dict:
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                "source": "training split only"}

    @classmethod
    def from_json(cls, row: dict) -> "RadarScale":
        return cls(row["minimum"], row["maximum"])

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        # The original SemanticKITTI adapter supplies normalized 3-D positions
        # and, for normal compression, attributes within [-1,1].
        xyz_half_range = torch.tensor(GRID_MAX, dtype=points.dtype, device=points.device)
        minimum = self.minimum.to(points.device)
        maximum = self.maximum.to(points.device)
        attrs = ((points[:, 3:] - minimum) * 2 / (maximum - minimum) - 1).clamp(-1, 1)
        return torch.cat((points[:, :3] / xyz_half_range, attrs), 1)

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        xyz_half_range = torch.tensor(GRID_MAX, dtype=normalized.dtype, device=normalized.device)
        minimum = self.minimum.to(normalized.device)
        maximum = self.maximum.to(normalized.device)
        attrs = (normalized[:, 3:] + 1) * (maximum - minimum) / 2 + minimum
        return torch.cat((normalized[:, :3] * xyz_half_range, attrs), 1)


def _voxel_keys(xyz: np.ndarray) -> np.ndarray:
    inside = ((xyz >= GRID_MIN) & (xyz < GRID_MAX)).all(1)
    indices = np.floor((xyz[inside] - GRID_MIN) / GRID_SIZE).astype(np.int64)
    # This is the same unique fine-grid occupancy definition as the earlier AE.
    return np.unique(indices[:, 2] * 512 * 512 + indices[:, 1] * 512 + indices[:, 0])


def reconstruction_counts(predicted: np.ndarray, target: np.ndarray) -> dict:
    """Same exact voxel IoU, one-to-one 0.2m F1, Chamfer and attribute MAE."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 7 or target.ndim != 2 or target.shape[1] != 7:
        raise ValueError("Expected N x 7 predicted and target radar returns")
    pred_keys, true_keys = _voxel_keys(predicted[:, :3]), _voxel_keys(target[:, :3])
    counts = {"predicted_points": len(predicted), "target_points": len(target),
              "predicted_voxels": len(pred_keys), "target_voxels": len(true_keys),
              "voxel_tp": int(np.intersect1d(pred_keys, true_keys, assume_unique=True).size),
              "point_tp_0p2m": 0, "chamfer_squared_sum": 0.0,
              "chamfer_frames": 0, "attribute_matches": 0}
    counts.update({f"{name}_abs_sum": 0.0 for name in ATTR_NAMES})
    if not len(predicted) or not len(target):
        return counts
    left_tree, right_tree = cKDTree(predicted[:, :3]), cKDTree(target[:, :3])
    left_dist, _ = right_tree.query(predicted[:, :3])
    right_dist, _ = left_tree.query(target[:, :3])
    counts["chamfer_squared_sum"] = float(np.square(left_dist).mean() + np.square(right_dist).mean())
    counts["chamfer_frames"] = 1
    edges = left_tree.sparse_distance_matrix(right_tree, max_distance=0.2 + 1e-6,
                                             output_type="coo_matrix")
    if edges.nnz:
        graph = csr_matrix((np.ones(edges.nnz, dtype=np.int8), (edges.row, edges.col)),
                           shape=(len(predicted), len(target)))
        matches = maximum_bipartite_matching(graph, perm_type="column")
        selected = np.flatnonzero(matches >= 0)
        counts["point_tp_0p2m"] = counts["attribute_matches"] = len(selected)
        differences = np.abs(predicted[selected, 3:7] - target[matches[selected], 3:7]).sum(0)
        for name, value in zip(ATTR_NAMES, differences):
            counts[f"{name}_abs_sum"] = float(value)
    return counts


def summarize_counts(counts: dict) -> dict:
    voxel_union = counts["predicted_voxels"] + counts["target_voxels"] - counts["voxel_tp"]
    point_total = counts["predicted_points"] + counts["target_points"]
    result = dict(counts)
    result.update({
        "voxel_iou": counts["voxel_tp"] / voxel_union if voxel_union else 1.0,
        "point_precision_0p2m": counts["point_tp_0p2m"] / counts["predicted_points"]
        if counts["predicted_points"] else 0.0,
        "point_recall_0p2m": counts["point_tp_0p2m"] / counts["target_points"]
        if counts["target_points"] else 0.0,
        "point_f1_0p2m": 2 * counts["point_tp_0p2m"] / point_total if point_total else 1.0,
        "chamfer_squared_m2": counts["chamfer_squared_sum"] / counts["chamfer_frames"]
        if counts["chamfer_frames"] else None,
    })
    for name in ATTR_NAMES:
        result[f"matched_{name}_mae"] = (counts[f"{name}_abs_sum"] / counts["attribute_matches"]
                                         if counts["attribute_matches"] else None)
    return result
