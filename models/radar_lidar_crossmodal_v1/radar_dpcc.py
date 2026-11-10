"""Radar adaptation of the density-preserving point-cloud autoencoder (D-PCC).

Reference: He et al., CVPR 2022, https://github.com/yunhe20/D-PCC .
This is a new PyTorch implementation of its three embedding encoder and
variable-cardinality point decoder, not a copy of the original CUDA code.
The only input and reconstruction target are the same filtered radar returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from .encoder import GRID


RATES = (0.5, 0.5, 0.5)
K = 16
MAX_CHILDREN = 8
WIDTH = 32
ATTRIBUTE_SCALES = (20.0, 10.0, 10.0, 1.0)


def _fps(xyz: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point sampling, computed once in the CPU cache."""
    if not len(xyz):
        return np.empty(0, dtype=np.int64)
    count = min(count, len(xyz))
    result = np.empty(count, dtype=np.int64)
    distances = np.full(len(xyz), np.inf, dtype=np.float64)
    index = int(np.argmax(np.sum((xyz - xyz.mean(0)) ** 2, axis=1)))
    for step in range(count):
        result[step] = index
        distances = np.minimum(distances, np.sum((xyz - xyz[index]) ** 2, axis=1))
        distances[result[:step + 1]] = -np.inf
        index = int(np.argmax(distances))
    return result


def build_hierarchy(points: torch.Tensor, rates=RATES, k=K) -> list[dict]:
    """D-PCC FPS hierarchy and each sampled point's collapsed neighborhood."""
    xyz = points[:, :3].numpy().astype(np.float64, copy=False)
    stages = []
    for rate in rates:
        if not len(xyz):
            stages.append({key: torch.empty(shape, dtype=dtype) for key, shape, dtype in (
                ("sample", (0,), torch.long), ("neighbors", (0, k), torch.long),
                ("mask", (0, k), torch.bool), ("counts", (0,), torch.float32),
                ("mean_distance", (0,), torch.float32))})
            continue
        sample = _fps(xyz, max(1, int(np.ceil(len(xyz) * rate))))
        anchors = xyz[sample]
        parent = cKDTree(anchors).query(xyz)[1]
        # Duplicated radar returns can have identical XYZ. Every FPS anchor
        # still owns itself, so no density embedding has an empty cluster.
        parent[sample] = np.arange(len(sample))
        counts = np.bincount(parent, minlength=len(sample))
        distance = np.linalg.norm(xyz - anchors[parent], axis=1)
        mean_distance = np.bincount(parent, weights=distance, minlength=len(sample)) / counts
        neighbors = np.zeros((len(sample), k), dtype=np.int64)
        mask = np.zeros((len(sample), k), dtype=bool)
        order = np.lexsort((distance, parent))
        sorted_parent = parent[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_parent)) + 1]
        ends = np.r_[starts[1:], len(order)]
        for anchor, (start, end) in enumerate(zip(starts, ends)):
            selected = order[start:min(end, start + k)]
            neighbors[anchor, :len(selected)] = selected
            mask[anchor, :len(selected)] = True
        stages.append({"sample": torch.from_numpy(sample),
                       "neighbors": torch.from_numpy(neighbors),
                       "mask": torch.from_numpy(mask),
                       "counts": torch.from_numpy(counts.astype(np.float32)),
                       "mean_distance": torch.from_numpy(mean_distance.astype(np.float32))})
        xyz = anchors
    return stages


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    xyz = points[:, :3] / points.new_tensor((51.2, 51.2, 2.0))
    attrs = torch.asinh(points[:, 3:] / points.new_tensor(ATTRIBUTE_SCALES))
    return torch.cat((xyz, attrs), dim=1)


class DownBlock(nn.Module):
    """Ancestor, local position, and density embeddings of collapsed points."""

    def __init__(self, width: int):
        super().__init__()
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.position = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.density = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.mix = nn.Sequential(nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, xyz, features, stage):
        sample, neighbors, mask = stage["sample"], stage["neighbors"], stage["mask"]
        center, center_feat = xyz[sample], features[sample]
        relative = xyz[neighbors] - center[:, None]
        local = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), dim=-1)
        position = self.position(local)
        attention = ((self.query(center_feat)[:, None] - self.key(features[neighbors])
                      + position) / features.shape[-1] ** 0.5).mean(-1)
        attention = attention.masked_fill(~mask, -1e4).softmax(-1)
        ancestor = (attention[..., None] * (self.value(features[neighbors]) + position)).sum(1)
        position = (attention[..., None] * position).sum(1)
        density = self.density(torch.log1p(stage["counts"])[:, None])
        return center, self.norm(center_feat + self.mix(torch.cat(
            (ancestor, position, density), dim=-1)))


class UpBlock(nn.Module):
    """D-PCC style learned count, child offsets, and inherited point features."""

    def __init__(self, width: int, max_children: int):
        super().__init__()
        self.max_children = max_children
        self.slot = nn.Embedding(max_children, width)
        self.count = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.child = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.offset = nn.Linear(width, 3)
        self.scale = nn.Linear(width, 1)
        self.refine = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        self.norm = nn.LayerNorm(width)
        nn.init.constant_(self.count[-1].bias, -2.1)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, xyz, features, *, threshold: float):
        continuous_count = 1 + (self.max_children - 1) * torch.sigmoid(self.count(features)).squeeze(-1)
        slots = torch.arange(self.max_children, device=xyz.device)
        # At 0.5 this is the rounded D-PCC cardinality. Other thresholds permit
        # the same validation count sweep as the existing radar AE.
        probability = torch.sigmoid(4 * (continuous_count[:, None] - slots - 0.5))
        selected = (probability >= threshold) | (slots[None] == 0)
        parent, slot = selected.nonzero(as_tuple=True)
        children = self.norm(features[parent] + self.child(features[parent] + self.slot(slot)))
        direction = F.normalize(self.offset(children), dim=-1)
        scale = F.softplus(self.scale(children))
        child_xyz = xyz[parent] + direction * scale + self.refine(children) * 0.05
        return child_xyz, children, continuous_count, parent


@dataclass
class Reconstruction:
    xyz: torch.Tensor
    attributes: torch.Tensor
    stage_xyz: list[torch.Tensor]
    stage_counts: list[torch.Tensor]
    stage_parents: list[torch.Tensor]
    latent_count: int


class RadarDPCC(nn.Module):
    """Radar-only, three-level D-PCC adaptation with no input skip path."""

    def __init__(self, width=WIDTH, max_children=MAX_CHILDREN):
        super().__init__()
        self.width, self.max_children = width, max_children
        self.embed = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.down = nn.ModuleList(DownBlock(width) for _ in RATES)
        self.up = nn.ModuleList(UpBlock(width, max_children) for _ in RATES)
        self.attributes = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 4))

    def forward(self, points: torch.Tensor, hierarchy: list[dict], *, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("Cardinality threshold must be between 0 and 1")
        if not len(points):
            zero = points.sum() * 0
            return Reconstruction(points[:, :3] + zero, points[:, 3:] + zero,
                                  [], [], [], 0)
        xyz, features = points[:, :3], self.embed(normalize_points(points))
        for block, stage in zip(self.down, hierarchy):
            xyz, features = block(xyz, features, stage)
        latent_count = len(xyz)
        stage_xyz, stage_counts, stage_parents = [], [], []
        for block in self.up:
            xyz, features, count, parent = block(xyz, features, threshold=threshold)
            stage_xyz.append(xyz)
            stage_counts.append(count)
            stage_parents.append(parent)
        # Fine model grid is the comparison domain used by the existing radar AE.
        low = points.new_tensor(GRID.minimum_xyz) + 1e-4
        high = points.new_tensor(GRID.maximum_xyz) - 1e-4
        xyz = torch.minimum(torch.maximum(xyz, low), high)
        attributes = torch.sinh(self.attributes(features).clamp(-8, 8)) * points.new_tensor(ATTRIBUTE_SCALES)
        return Reconstruction(xyz, attributes, stage_xyz, stage_counts,
                              stage_parents, latent_count)


def _nearest(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Exact KD-tree assignment; squared distances retain their torch gradient."""
    left_xyz = left.detach().cpu().numpy()
    right_xyz = right.detach().cpu().numpy()
    indices = cKDTree(right_xyz).query(left_xyz, workers=1)[1]
    return torch.as_tensor(indices, device=left.device, dtype=torch.long)


def _chamfer(predicted: torch.Tensor, target: torch.Tensor):
    if not len(predicted) or not len(target):
        return predicted.sum() * 0 + target.sum() * 0
    pred_idx = _nearest(predicted, target)
    target_idx = _nearest(target, predicted)
    pred_sq = (predicted - target[pred_idx]).square().sum(1)
    target_sq = (target - predicted[target_idx]).square().sum(1)
    return pred_sq.mean() + target_sq.mean(), pred_idx


def reconstruction_loss(result: Reconstruction, points: torch.Tensor,
                        hierarchy: list[dict]):
    if not len(points):
        zero = result.xyz.sum() * 0
        return zero, {"chamfer": zero.detach(), "features": zero.detach(),
                      "intermediate": zero.detach(), "density": zero.detach()}
    xyz_levels = [points[:, :3]]
    for stage in hierarchy:
        xyz_levels.append(xyz_levels[-1][stage["sample"]])
    chamfer, nearest = _chamfer(result.xyz, points[:, :3])
    # Reconstruct RCS, two Doppler channels and scan age at matched radar returns.
    feature = F.smooth_l1_loss(
        torch.asinh(result.attributes / points.new_tensor(ATTRIBUTE_SCALES)),
        torch.asinh(points[nearest, 3:] / points.new_tensor(ATTRIBUTE_SCALES)))
    intermediate = chamfer.new_zeros(())
    density = chamfer.new_zeros(())
    for i, (pred_xyz, count) in enumerate(zip(result.stage_xyz, result.stage_counts)):
        target = xyz_levels[-2-i]
        if i < len(result.stage_xyz) - 1:
            stage_cd, _ = _chamfer(pred_xyz, target)
            intermediate = intermediate + stage_cd
        # Each decoder parent is assigned the nearest encoder anchor at this scale.
        parent_xyz = xyz_levels[-1-i]
        predicted_parent_xyz = (xyz_levels[-1] if i == 0 else result.stage_xyz[i-1])
        nearest_parent = _nearest(predicted_parent_xyz, parent_xyz)
        target_count = hierarchy[-1-i]["counts"][nearest_parent]
        target_distance = hierarchy[-1-i]["mean_distance"][nearest_parent]
        predicted_distance = (pred_xyz - predicted_parent_xyz[result.stage_parents[i]]).norm(dim=1)
        predicted_sum = count.new_zeros(len(count))
        predicted_sum.index_add_(0, result.stage_parents[i], predicted_distance)
        predicted_mean = predicted_sum / torch.bincount(
            result.stage_parents[i], minlength=len(count)).clamp_min(1)
        density = (density + F.smooth_l1_loss(count, target_count.clamp(max=8))
                   + 0.1 * F.smooth_l1_loss(predicted_mean, target_distance))
    total = chamfer + 0.1 * intermediate + 0.1 * feature + 0.05 * density
    return total, {"chamfer": chamfer.detach(), "features": feature.detach(),
                   "intermediate": intermediate.detach(), "density": density.detach()}
