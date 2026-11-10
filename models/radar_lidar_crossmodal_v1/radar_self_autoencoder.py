"""Radar-only point-set autoencoder on the filtered 12-voxel/1.6 m cloud.

The decoder predicts up to four returns per generated fine voxel. Each return
has a presence logit, a within-voxel XYZ offset, and the four remaining VoD
radar attributes (RCS, two Doppler velocities, and scan age). It uses only the
stride-8 radar latent: no fine-resolution skip bypasses the bottleneck.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import encode_keys, voxelize

from .encoder import GRID
from .previous_encoder import PreviousStage1Sparse
from .student import RawSparseConfidenceDecoder, StudentConfig


SLOTS = 4
CHANNELS_PER_SLOT = 8
ATTRIBUTE_SCALES = (20.0, 10.0, 10.0, 1.0)
POINT_TOLERANCE_M = 0.2


class RadarSelfAutoencoder(nn.Module):
    def __init__(self, encoder_checkpoint=None, *, slots: int = SLOTS):
        super().__init__()
        if slots < 1:
            raise ValueError("At least one radar-return slot is required")
        self.slots = slots
        self.encoder = PreviousStage1Sparse(encoder_checkpoint, modality="radar")
        self.decoder = RawSparseConfidenceDecoder(
            StudentConfig(positive_weight=1, dice_weight=0,
                          lambda_teacher_evidence=0),
            output_channels=slots * CHANNELS_PER_SLOT)
        with torch.no_grad():
            self.decoder.logits.bias[0::CHANNELS_PER_SLOT].fill_(-4.0)

    def forward(self, radar: torch.Tensor, valid: torch.Tensor):
        latent = self.encoder(radar, valid)
        return self.decoder(latent)


def collate_filtered_radar(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty radar batch")
    maximum = max(len(row["points"]) for row in rows)
    points = torch.zeros((len(rows), maximum, 7), dtype=torch.float32)
    valid = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for batch, row in enumerate(rows):
        count = len(row["points"])
        points[batch, :count] = row["points"]
        valid[batch, :count] = True
    return {"frame_id": [row["frame_id"] for row in rows],
            "radar": points, "valid": valid}


def _canonical_order(voxel_index: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Give multi-return voxels a deterministic XYZ-sorted slot assignment."""
    order = torch.arange(len(points), device=points.device)
    for column in (2, 1, 0):
        order = order[torch.argsort(points[order, column], stable=True)]
    return order[torch.argsort(voxel_index[order], stable=True)]


def target_slots(prediction, radar: torch.Tensor, valid: torch.Tensor,
                 slots: int = SLOTS) -> dict:
    """Align every filtered radar return to its generated fine voxel and slot."""
    coords, inverse, points, _ = voxelize(radar, valid, GRID)
    n = len(prediction.coords)
    labels = torch.zeros((n, slots), device=radar.device, dtype=radar.dtype)
    continuous = torch.zeros((n, slots, 7), device=radar.device, dtype=radar.dtype)
    if not len(points) or not n:
        return {"labels": labels, "continuous": continuous, "observed_coords": coords,
                "observed_points": len(points), "represented_points": 0,
                "represented_voxels": 0}

    source_keys = encode_keys(coords, GRID.shape_zyx)
    candidate_keys = prediction.keys
    positions = torch.searchsorted(candidate_keys, source_keys)
    present = ((positions < n) &
               (candidate_keys[positions.clamp(max=n - 1)] == source_keys))
    order = _canonical_order(inverse, points)
    ordered_inverse = inverse[order]
    counts = torch.bincount(ordered_inverse, minlength=len(coords))
    beginnings = torch.cumsum(counts, 0) - counts
    ranks = torch.arange(len(points), device=radar.device) - beginnings[ordered_inverse]
    represented = present[ordered_inverse] & (ranks < slots)
    source_points = points[order[represented]]
    source_voxels = ordered_inverse[represented]
    candidate_positions = positions[source_voxels]
    slot_positions = ranks[represented]
    labels[candidate_positions, slot_positions] = 1.0

    centers = GRID.centers_xyz(coords[source_voxels])
    voxel_size = source_points.new_tensor(GRID.size_xyz)
    continuous[candidate_positions, slot_positions, :3] = (
        (source_points[:, :3] - centers) / voxel_size)
    attribute_scales = source_points.new_tensor(ATTRIBUTE_SCALES)
    continuous[candidate_positions, slot_positions, 3:] = torch.asinh(
        source_points[:, 3:] / attribute_scales)
    return {"labels": labels, "continuous": continuous,
            "observed_coords": coords, "observed_points": len(points),
            "represented_points": int(represented.sum()),
            "represented_voxels": int(present.sum())}


def reconstruction_loss(prediction, targets: dict, *, positive_weight: float = 10.0):
    if prediction.features.shape[1] != targets["labels"].shape[1] * CHANNELS_PER_SLOT:
        raise ValueError("Radar decoder output and target slots differ")
    values = prediction.features.reshape(len(prediction.coords), -1, CHANNELS_PER_SLOT)
    labels = targets["labels"]
    if not values.numel():
        zero = prediction.features.sum() * 0
        return zero, {"bce": zero.detach(), "offset": zero.detach(),
                      "attributes": zero.detach()}
    bce = F.binary_cross_entropy_with_logits(
        values[..., 0], labels,
        pos_weight=values.new_tensor(positive_weight))
    positive = labels.bool()
    if positive.any():
        offsets = 0.5 * torch.tanh(values[..., 1:4][positive])
        target_values = targets["continuous"][positive]
        offset = F.smooth_l1_loss(offsets, target_values[:, :3])
        attributes = F.smooth_l1_loss(values[..., 4:8][positive],
                                      target_values[:, 3:])
    else:
        offset = attributes = values.sum() * 0
    return bce + offset + 0.1 * attributes, {
        "bce": bce.detach(), "offset": offset.detach(),
        "attributes": attributes.detach()}


def decoded_points(prediction, *, threshold: float = 0.5,
                   slots: int = SLOTS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [batch, x, y, z, RCS, radial, compensated, age] and occupied voxels."""
    values = prediction.features.reshape(len(prediction.coords), slots, CHANNELS_PER_SLOT)
    selected = torch.sigmoid(values[..., 0]) >= threshold
    voxel_selected = selected.any(1)
    if not selected.any():
        return values.new_empty((0, 8)), prediction.coords[voxel_selected]
    voxel_index, slot_index = selected.nonzero(as_tuple=True)
    coords = prediction.coords[voxel_index]
    output = values[voxel_index, slot_index]
    centers = GRID.centers_xyz(coords)
    xyz = centers + 0.5 * torch.tanh(output[:, 1:4]) * output.new_tensor(GRID.size_xyz)
    attrs = torch.sinh(output[:, 4:8].clamp(-8, 8)) * output.new_tensor(ATTRIBUTE_SCALES)
    points = torch.cat((coords[:, :1].to(xyz.dtype), xyz, attrs), 1)
    return points, prediction.coords[voxel_selected]


def reconstruction_counts(predicted: torch.Tensor, predicted_voxels: torch.Tensor,
                          radar: torch.Tensor, valid: torch.Tensor) -> dict:
    """One-to-one 0.2 m point matching and exact fine-voxel occupancy."""
    target_voxels, _, _, _ = voxelize(radar, valid, GRID)
    pred_keys = encode_keys(predicted_voxels, GRID.shape_zyx).cpu().numpy()
    true_keys = encode_keys(target_voxels, GRID.shape_zyx).cpu().numpy()
    result = {"predicted_points": len(predicted), "target_points": int(valid.sum()),
              "predicted_voxels": len(predicted_voxels),
              "target_voxels": len(target_voxels),
              "voxel_tp": int(np.intersect1d(pred_keys, true_keys,
                                              assume_unique=True).size),
              "point_tp_0p2m": 0, "chamfer_squared_sum": 0.0,
              "chamfer_frames": 0, "attribute_matches": 0,
              "rcs_abs_sum": 0.0, "radial_abs_sum": 0.0,
              "compensated_radial_abs_sum": 0.0, "scan_age_abs_sum": 0.0}
    pred = predicted.detach().cpu().numpy()
    truth = radar.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy()
    for batch in range(len(radar)):
        predicted_frame = pred[pred[:, 0] == batch]
        true_frame = truth[batch, mask[batch]]
        left = predicted_frame[:, 1:4]
        right = true_frame[:, :3]
        if not len(left) or not len(right):
            continue
        left_tree, right_tree = cKDTree(left), cKDTree(right)
        left_dist, _ = right_tree.query(left)
        right_dist, _ = left_tree.query(right)
        result["chamfer_squared_sum"] += float(
            np.square(left_dist).mean() + np.square(right_dist).mean())
        result["chamfer_frames"] += 1
        edges = left_tree.sparse_distance_matrix(
            right_tree, max_distance=POINT_TOLERANCE_M + 1e-6,
            output_type="coo_matrix")
        if edges.nnz:
            graph = csr_matrix((np.ones(edges.nnz, dtype=np.int8),
                                (edges.row, edges.col)), shape=(len(left), len(right)))
            # Column permutation maps each predicted row to its matched target.
            matches = maximum_bipartite_matching(graph, perm_type="column")
            selected = np.flatnonzero(matches >= 0)
            result["point_tp_0p2m"] += len(selected)
            result["attribute_matches"] += len(selected)
            differences = np.abs(predicted_frame[selected, 4:8] -
                                 true_frame[matches[selected], 3:7]).sum(axis=0)
            for key, value in zip(("rcs_abs_sum", "radial_abs_sum",
                                   "compensated_radial_abs_sum", "scan_age_abs_sum"),
                                  differences):
                result[key] += float(value)
    return result
