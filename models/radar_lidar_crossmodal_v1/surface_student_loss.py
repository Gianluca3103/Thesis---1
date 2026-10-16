"""Supervise radar-conditioned proposal geometry without a filled voxel domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.correspondence import local_neighbors_xyz
from models.radar_lidar_stage1.sparse import SparseSites, voxelize

from .encoder import GRID
from .student_loss import aligned_teacher_features
from .surface_student import SurfaceStudentConfig, SurfaceStudentOutput


def clean_surface_sites(clean_points: torch.Tensor,
                        clean_valid: torch.Tensor) -> SparseSites:
    coords, _, _, _ = voxelize(clean_points, clean_valid, GRID)
    features = clean_points.new_zeros((len(coords), 1))
    return SparseSites._from_validated(coords, features, GRID.shape_zyx, 1)


def surface_student_objective(output: SurfaceStudentOutput, teacher_sites: SparseSites | None,
                              clean: SparseSites, config: SurfaceStudentConfig):
    if config.distillation_weight:
        if teacher_sites is None:
            raise ValueError("Feature distillation requires teacher features")
        target_features = aligned_teacher_features(output, teacher_sites)
        distill = (F.smooth_l1_loss(output.conditioning.features, target_features)
                   if len(target_features) else output.conditioning.features.sum() * 0)
    else:
        distill = output.conditioning.features.sum() * 0
    p = output.surface
    if not len(p.anchor_xyz):
        zero = p.logits.sum() * 0
        pieces = {"distillation": distill, "geometry": zero, "confidence": zero,
                  "anchors": 0, "proposals": 0, "anchors_with_targets": 0,
                  "confidence_elements": 0, "geometry_elements": 0}
        return config.distillation_weight * distill, pieces
    neighbors = local_neighbors_xyz(p.anchor_xyz, p.anchor_batch, clean, GRID,
                                    config.target_radius_m, config.target_neighbors)
    valid = neighbors.valid
    has_target = valid.any(1)
    if len(clean.coords):
        target_xyz = clean.centers_xyz(GRID)[neighbors.indices.clamp_min(0)]
        delta = torch.linalg.vector_norm(p.xyz[:, :, None, :] - target_xyz[:, None, :, :],
                                         dim=-1).masked_fill(~valid[:, None, :], float("inf"))
        nearest = delta.min(-1).values
        if bool(has_target.any()):
            forward = F.smooth_l1_loss(nearest[has_target],
                                       torch.zeros_like(nearest[has_target]), beta=.2)
            reverse = delta.min(1).values[valid]
            reverse = F.smooth_l1_loss(reverse, torch.zeros_like(reverse), beta=.2)
            geometry = (forward + reverse) * .5
        else:
            geometry = p.xyz.sum() * 0
        quality = torch.exp(-nearest.detach() / config.confidence_sigma_m)
    else:
        geometry = p.xyz.sum() * 0
        quality = p.logits.new_zeros(p.logits.shape)
    weights = 1 + has_target[:, None].to(p.logits.dtype)
    per_site = F.binary_cross_entropy_with_logits(p.logits, quality, reduction="none")
    confidence = (per_site * weights).sum() / weights.expand_as(per_site).sum().clamp_min(1)
    total = (config.distillation_weight * distill + config.geometry_weight * geometry
             + config.confidence_weight * confidence)
    return total, {"distillation": distill, "geometry": geometry,
                   "confidence": confidence, "anchors": len(p.anchor_xyz),
                   "proposals": p.logits.numel(),
                   "anchors_with_targets": int(has_target.sum()),
                   "confidence_elements": p.logits.numel(),
                   "geometry_elements": int(valid.sum())}


@dataclass
class SurfaceMetricCounts:
    selected: int = 0
    correct: int = 0
    eligible_clean: int = 0
    covered_clean: int = 0
    all_clean: int = 0

    def add(self, other: "SurfaceMetricCounts") -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def report(self) -> dict:
        precision = self.correct / self.selected if self.selected else None
        recall = self.covered_clean / self.eligible_clean if self.eligible_clean else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall
              else 0.0 if precision is not None and recall is not None else None)
        return {"proposal_precision_20cm": precision,
                "radar_near_clean_recall_20cm": recall,
                "surface_f1_20cm": f1,
                "selected_proposals": self.selected,
                "correct_proposals": self.correct,
                "eligible_clean_sites": self.eligible_clean,
                "covered_clean_sites": self.covered_clean,
                "all_clean_sites": self.all_clean,
                "radar_near_clean_fraction": self.eligible_clean / self.all_clean
                if self.all_clean else None}


@torch.no_grad()
def surface_metric_counts(output: SurfaceStudentOutput, clean: SparseSites,
                          config: SurfaceStudentConfig) -> SurfaceMetricCounts:
    """Score raw proposals against LiDAR near radar-pattern anchors.

    All distances and counts are computed per batch. LiDAR is a reference only;
    it cannot create inference proposals.
    """
    p = output.surface
    result = SurfaceMetricCounts()
    xyz = p.xyz.detach().cpu().numpy().reshape(-1, 3)
    score = p.score.detach().cpu().numpy().reshape(-1)
    batch = np.repeat(p.anchor_batch.detach().cpu().numpy(), p.xyz.shape[1])
    anchor = p.anchor_xyz.detach().cpu().numpy()
    anchor_batch = p.anchor_batch.detach().cpu().numpy()
    clean_xyz = clean.centers_xyz(GRID).detach().cpu().numpy()
    clean_batch = clean.coords[:, 0].detach().cpu().numpy()
    for batch_id in np.unique(np.concatenate((batch, clean_batch))):
        selected = (batch == batch_id) & (score >= config.threshold)
        truth = clean_xyz[clean_batch == batch_id]
        sites = xyz[selected]
        result.selected += len(sites)
        result.all_clean += len(truth)
        if not len(truth):
            continue
        if len(sites):
            result.correct += int((cKDTree(truth).query(sites)[0]
                                   <= config.evaluation_tolerance_m).sum())
        anchors = anchor[anchor_batch == batch_id]
        if not len(anchors):
            continue
        eligible = cKDTree(anchors).query(truth)[0] <= config.target_radius_m
        result.eligible_clean += int(eligible.sum())
        if len(sites) and eligible.any():
            result.covered_clean += int((cKDTree(sites).query(truth[eligible])[0]
                                         <= config.evaluation_tolerance_m).sum())
    return result
