"""Masked student losses and precisely scoped radar-supported IoU."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import make_targets

from .encoder import GRID
from .student import StudentConfig, StudentOutput


def aligned_teacher_features(student: StudentOutput, teacher_sites):
    student_sites = student.conditioning
    if len(student_sites.coords) != len(teacher_sites.coords):
        raise ValueError("Teacher and student radar bottleneck coordinates differ")
    student_keys = encode_keys(student_sites.coords, student_sites.shape_zyx)
    teacher_keys = encode_keys(teacher_sites.coords, teacher_sites.shape_zyx)
    if not torch.equal(student_keys, teacher_keys):
        raise ValueError("Teacher and student radar bottleneck coordinates differ")
    return teacher_sites.features.detach()


def student_targets(output: StudentOutput, clean_points: torch.Tensor,
                    clean_valid: torch.Tensor):
    confidence = output.confidence_logits
    domain = CandidateDomain(confidence.coords, torch.sigmoid(confidence.features[:, 0]).detach(),
                             GRID, {})
    return make_targets(domain, clean_points, clean_valid)


def student_objective(output: StudentOutput, teacher_sites, targets, config: StudentConfig):
    target_features = aligned_teacher_features(output, teacher_sites)
    if len(target_features):
        distill = F.smooth_l1_loss(output.conditioning.features, target_features,
                                   beta=config.smooth_l1_beta)
    else:
        distill = output.conditioning.features.sum() * 0
    logits = output.confidence_logits.features[:, 0]
    occupied, free = targets.occupied, targets.known_free
    if occupied.shape != logits.shape or free.shape != logits.shape or (occupied & free).any():
        raise ValueError("Confidence labels must be disjoint and match candidate sites")
    known = occupied | free
    dice_intersection = logits.new_zeros(())
    dice_probability_sum = logits.new_zeros(())
    dice_target_sum = logits.new_zeros(())
    if known.any():
        bce = F.binary_cross_entropy_with_logits(logits[known], occupied[known].float(),
                                                  pos_weight=logits.new_tensor(config.positive_weight))
        probabilities = torch.sigmoid(logits[known])
        positives = occupied[known].float()
        dice_intersection = (probabilities * positives).sum()
        dice_probability_sum = probabilities.sum()
        dice_target_sum = positives.sum()
        dice = (1 - (2 * dice_intersection + 1e-6) /
                (dice_probability_sum + dice_target_sum + 1e-6)) if positives.any() else logits.sum() * 0
    else:
        bce = logits.sum() * 0
        dice = logits.sum() * 0
    spatial = bce + config.dice_weight * dice
    total = config.distillation_weight * distill + config.spatial_weight * spatial
    return total, {"distillation": distill, "spatial": spatial, "bce": bce,
                   "dice": dice, "feature_elements": target_features.numel(),
                   "known_sites": int(known.sum()), "occupied_sites": int(occupied.sum()),
                   "free_sites": int(free.sum()), "unknown_sites": int((~known).sum()),
                   "dice_intersection": dice_intersection.detach(),
                   "dice_probability_sum": dice_probability_sum.detach(),
                   "dice_target_sum": dice_target_sum.detach()}


def radar_supported_iou(output: StudentOutput, targets, threshold: float):
    """Conventional candidate-domain IoU and known-space masked variant."""
    predicted = torch.sigmoid(output.confidence_logits.features[:, 0]) >= threshold
    occupied, free = targets.occupied, targets.known_free
    known = occupied | free
    if predicted.shape != occupied.shape:
        raise ValueError("Prediction and candidate labels differ")

    def measure(mask):
        intersection = int((predicted & occupied & mask).sum())
        union = int(((predicted | occupied) & mask).sum())
        return {"intersection": intersection, "union": union,
                "iou": intersection / union if union else None}

    return {
        "conventional_clean_scan": measure(torch.ones_like(known)),
        "known_space_masked": measure(known),
        "candidate_sites": len(predicted),
        "known_sites": int(known.sum()),
        "occupied_sites": int(occupied.sum()),
        "free_sites": int(free.sum()),
        "unknown_sites": int((~known).sum()),
        "known_fraction": float(known.float().mean()) if len(known) else None,
        "clean_point_coverage": (targets.clean_points_in_candidates /
                                 max(targets.clean_points_in_grid, 1)),
        "clean_points_in_candidates": targets.clean_points_in_candidates,
        "clean_points_in_grid": targets.clean_points_in_grid,
    }
