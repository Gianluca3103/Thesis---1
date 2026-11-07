"""Occupancy-only exact/tolerant scores and voxel-center Chamfer distance."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch

from .eligible_occupancy_metrics import occupancy_counts
from .sparse_geometry import GRID


def voxel_centers(coords: torch.Tensor) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Expected [N,4] BZYX occupancy coordinates")
    ijk = coords[:, [3, 2, 1]].detach().cpu().numpy().astype(np.float64)
    return (ijk + 0.5) * np.asarray(GRID.voxel_xyz) + np.asarray(GRID.minimum_xyz)


def voxel_center_chamfer(predicted: torch.Tensor,
                         target: torch.Tensor) -> dict | None:
    """Symmetric mean nearest-center distances; undefined for an empty side."""
    if not len(predicted) or not len(target):
        return None
    pred = voxel_centers(predicted)
    truth = voxel_centers(target)
    pred_to_target = cKDTree(truth).query(pred, workers=1)[0]
    target_to_pred = cKDTree(pred).query(truth, workers=1)[0]
    return {"chamfer_l2_m": float(pred_to_target.mean() + target_to_pred.mean()),
            "chamfer_squared_l2_m2": float(
                np.square(pred_to_target).mean() +
                np.square(target_to_pred).mean())}


def empty_accumulator() -> dict:
    return {"frames": 0, "predicted": 0, "eligible_target": 0,
            "exact_tp": 0, "tolerant_tp_0p2m": 0,
            "chamfer_paired_frames": 0, "empty_prediction_frames": 0,
            "empty_target_frames": 0, "chamfer_l2_m_sum": 0.0,
            "chamfer_squared_l2_m2_sum": 0.0}


def add_frame(accumulator: dict, predicted: torch.Tensor,
              target: torch.Tensor) -> None:
    counts = occupancy_counts(predicted, target, tolerance_m=0.2)
    accumulator["frames"] += 1
    for key, value in counts.items():
        accumulator[key] += value
    accumulator["empty_prediction_frames"] += int(len(predicted) == 0)
    accumulator["empty_target_frames"] += int(len(target) == 0)
    chamfer = voxel_center_chamfer(predicted, target)
    if chamfer is not None:
        accumulator["chamfer_paired_frames"] += 1
        for key, value in chamfer.items():
            accumulator[key + "_sum"] += value


def summarize(accumulator: dict) -> dict:
    predicted = accumulator["predicted"]
    target = accumulator["eligible_target"]
    exact = accumulator["exact_tp"]
    tolerant = accumulator["tolerant_tp_0p2m"]
    paired = accumulator["chamfer_paired_frames"]
    return {"frames": accumulator["frames"],
            "predicted": predicted, "eligible_target": target,
            "exact_tp": exact, "tolerant_tp_0p2m": tolerant,
            "precision": exact / predicted if predicted else 0.0,
            "recall": exact / target if target else 0.0,
            "iou": exact / (predicted + target - exact)
            if predicted + target - exact else 0.0,
            "f1": 2 * exact / (predicted + target)
            if predicted + target else 0.0,
            "precision_0p2m": tolerant / predicted if predicted else 0.0,
            "recall_0p2m": tolerant / target if target else 0.0,
            "iou_0p2m": tolerant / (predicted + target - tolerant)
            if predicted + target - tolerant else 0.0,
            "f1_0p2m": 2 * tolerant / (predicted + target)
            if predicted + target else 0.0,
            "chamfer_l2_m_paired_frame_mean": (
                accumulator["chamfer_l2_m_sum"] / paired if paired else None),
            "chamfer_squared_l2_m2_paired_frame_mean": (
                accumulator["chamfer_squared_l2_m2_sum"] / paired
                if paired else None),
            "chamfer_paired_frames": paired,
            "empty_prediction_frames": accumulator["empty_prediction_frames"],
            "empty_target_frames": accumulator["empty_target_frames"]}
