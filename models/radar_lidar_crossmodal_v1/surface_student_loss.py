"""Proposal-only supervision for radar-predicted fine LiDAR voxels.

Unproposed clean voxels never enter the loss. Unknown proposed voxels are
unlabeled; only observed occupancy and verified free space are supervised.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import SparseSites, encode_keys, voxelize
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import make_targets

from .encoder import GRID
from .pointer_voxels import PointerVoxelField, merge_surface_pointers
from .surface_student import SurfaceStudentConfig, SurfaceStudentOutput


def clean_surface_sites(clean_points: torch.Tensor,
                        clean_valid: torch.Tensor) -> SparseSites:
    """Fine clean-scan occupancy, retained for diagnostics and callers."""
    coords, _, _, _ = voxelize(clean_points, clean_valid, GRID)
    return SparseSites._from_validated(
        coords, clean_points.new_zeros((len(coords), 1)), GRID.shape_zyx, 1)


def _soft_occupied_at_pointers(xyz: torch.Tensor, batch: torch.Tensor,
                               clean_coords: torch.Tensor, *,
                               chunk_size: int = 65536) -> torch.Tensor:
    """Training-only trilinear occupancy sample at continuous pointer locations.

    The forward loss uses exact discrete candidate labels. This sample gives
    the unique-hit reward a local gradient with respect to pointer XYZ. It
    creates no candidate or output voxels and does not label unknown as free.
    """
    if not len(xyz) or not len(clean_coords):
        return xyz.new_zeros((len(xyz),))
    clean_keys = encode_keys(clean_coords, GRID.shape_zyx)
    origin = xyz.new_tensor(GRID.minimum_xyz)
    size = xyz.new_tensor(GRID.size_xyz)
    shape = batch.new_tensor(GRID.shape_zyx[::-1])
    samples = []
    for begin in range(0, len(xyz), chunk_size):
        point = xyz[begin:begin + chunk_size]
        point_batch = batch[begin:begin + chunk_size]
        centered = (point - origin) / size - 0.5
        base = torch.floor(centered).long()
        fraction = centered - base
        value = point.new_zeros(len(point))
        for corner in product((0, 1), repeat=3):
            offset = base.new_tensor(corner)
            cell = base + offset
            inside = ((cell >= 0) & (cell < shape)).all(-1)
            safe = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), shape - 1)
            rows = torch.stack((point_batch, safe[:, 2], safe[:, 1], safe[:, 0]), -1)
            key = encode_keys(rows, GRID.shape_zyx)
            where = torch.searchsorted(clean_keys, key)
            present = inside & (where < len(clean_keys)) & (
                clean_keys[where.clamp_max(len(clean_keys) - 1)] == key)
            weight = torch.where(offset.bool(), fraction, 1 - fraction).prod(-1)
            value = value + weight * present.to(weight.dtype)
        samples.append(value)
    return torch.cat(samples)


def surface_student_objective(output: SurfaceStudentOutput,
                              clean_points: torch.Tensor,
                              clean_valid: torch.Tensor,
                              config: SurfaceStudentConfig):
    """BCE + Dice + distinct-hit reward on known, pointed voxels only."""
    field = merge_surface_pointers(output.surface)
    domain = CandidateDomain(field.coords, field.confidence, GRID, {})
    targets = make_targets(domain, clean_points, clean_valid)
    known = targets.occupied | targets.known_free
    confidence = field.confidence
    zero = output.surface.logits.sum() * 0
    if bool(known.any()):
        predicted = confidence[known]
        target = targets.occupied[known].to(predicted.dtype)
        bce = F.binary_cross_entropy(predicted.clamp(1e-6, 1 - 1e-6), target)
        overlap = (predicted * target).sum()
        dice = 1 - (2 * overlap + 1e-6) / (predicted.sum() + target.sum() + 1e-6)
        hard_hits = (confidence * targets.occupied.to(confidence.dtype)).sum()

        hits = hard_hits
        if torch.is_grad_enabled() and field.pointer_xyz.requires_grad:
            # Hard voxel assignment has no XYZ derivative. Use the clean scan
            # for a local, training-only straight-through gradient. The
            # forward reward remains exact distinct occupied candidate hits.
            clean_coords, _, _, _ = voxelize(clean_points, clean_valid, GRID)
            soft_at_pointer = _soft_occupied_at_pointers(
                field.pointer_xyz, field.pointer_batch, clean_coords)
            pointer_weight = field.pointer_strength.detach()
            weighted_soft = confidence.new_zeros(len(confidence)).index_add(
                0, field.pointer_voxel_index, pointer_weight * soft_at_pointer)
            weight_sum = confidence.new_zeros(len(confidence)).index_add(
                0, field.pointer_voxel_index, pointer_weight)
            soft_by_voxel = weighted_soft / weight_sum.clamp_min(1e-8)
            soft_hits = (confidence.detach() * soft_by_voxel * known.to(confidence.dtype)).sum()
            hits = hard_hits + soft_hits - soft_hits.detach()
        hit_reward = torch.log1p(hits / (config.hit_scale * clean_points.shape[0]))
    else:
        bce = dice = hit_reward = hard_hits = zero
    total = (config.bce_weight * bce + config.dice_weight * dice
             - config.hit_weight * hit_reward)
    pieces = {"bce": bce, "dice": dice, "hit_reward": hit_reward,
              "distinct_hit_mass": hard_hits,
              "pointer_slots": output.surface.logits.numel(),
              "in_grid_pointers": len(field.pointer_xyz),
              "candidate_sites": len(field.coords),
              "known_sites": int(known.sum()),
              "occupied_sites": int(targets.occupied.sum()),
              "known_free_sites": int(targets.known_free.sum())}
    return total, pieces, field, targets


@dataclass
class SurfaceMetricCounts:
    candidate_sites: int = 0
    selected_sites: int = 0
    correct_unique: int = 0
    false_positive_known_free: int = 0
    unknown_selected: int = 0
    pointer_slots: int = 0
    in_grid_pointers: int = 0
    support_sources_on_selected: int = 0

    def add(self, other: "SurfaceMetricCounts") -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def report(self) -> dict:
        verified = self.correct_unique + self.false_positive_known_free
        precision = self.correct_unique / verified if verified else None
        return {
            "distinct_correct_voxels": self.correct_unique,
            "verified_proposal_precision": precision,
            "proposal_utility": self.correct_unique * precision if precision is not None else 0.0,
            "false_positive_known_free": self.false_positive_known_free,
            "unknown_selected": self.unknown_selected,
            "unknown_selected_fraction": self.unknown_selected / self.selected_sites
            if self.selected_sites else None,
            "candidate_sites": self.candidate_sites,
            "selected_sites": self.selected_sites,
            "pointer_slots": self.pointer_slots,
            "in_grid_pointers": self.in_grid_pointers,
            "mean_radar_sources_per_selected": self.support_sources_on_selected / self.selected_sites
            if self.selected_sites else None,
        }


@torch.no_grad()
def surface_metric_counts(field: PointerVoxelField, targets,
                          config: SurfaceStudentConfig, *,
                          pointer_slots: int = 0) -> SurfaceMetricCounts:
    selected = field.confidence >= config.threshold
    correct = selected & targets.occupied
    free = selected & targets.known_free
    unknown = selected & ~(targets.occupied | targets.known_free)
    return SurfaceMetricCounts(
        candidate_sites=len(field.coords), selected_sites=int(selected.sum()),
        correct_unique=int(correct.sum()), false_positive_known_free=int(free.sum()),
        unknown_selected=int(unknown.sum()), pointer_slots=pointer_slots,
        in_grid_pointers=len(field.pointer_xyz),
        support_sources_on_selected=int(field.source_count[selected].sum()))
