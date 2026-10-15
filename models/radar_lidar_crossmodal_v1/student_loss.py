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


@torch.no_grad()
def teacher_correspondence_evidence(teacher_output, candidate_sites) -> torch.Tensor:
    """Project mean four-head attention to existing fine candidate parents.

    Attention supplies a coarse supervision weight, never a fine occupancy
    label or a candidate coordinate. Repeated neighbors of one radar query
    contribute together; evidence from different queries combines by max.
    """
    radar = teacher_output.radar_frozen
    lidar = teacher_output.lidar_frozen
    neighbors = teacher_output.neighbors.indices
    attention = teacher_output.attention
    coarse_shape = GRID.scale_shape(8)
    if (radar.stride != 8 or lidar.stride != 8 or candidate_sites.stride != 1
            or radar.shape_zyx != coarse_shape or lidar.shape_zyx != coarse_shape
            or candidate_sites.shape_zyx != GRID.shape_zyx):
        raise ValueError("Teacher and candidate coordinates must use the shared stride-8 grid")
    if (neighbors.ndim != 2 or neighbors.shape[0] != len(radar.coords)
            or attention.shape != (len(radar.coords), 4, neighbors.shape[1])):
        raise ValueError("Teacher attention and neighborhood shapes differ")
    if (radar.coords.device != lidar.coords.device
            or radar.coords.device != candidate_sites.coords.device
            or attention.device != radar.coords.device
            or neighbors.device != radar.coords.device):
        raise ValueError("Teacher and student coordinates must share a device")
    evidence = attention.new_zeros(len(candidate_sites.coords))
    valid_query, valid_slot = (neighbors >= 0).nonzero(as_tuple=True)
    if not len(valid_query) or not len(evidence):
        return evidence
    lidar_index = neighbors[valid_query, valid_slot]
    if bool((lidar_index >= len(lidar.coords)).any()):
        raise ValueError("Teacher neighborhood refers to a missing LiDAR site")
    if not torch.equal(radar.coords[valid_query, 0], lidar.coords[lidar_index, 0]):
        raise ValueError("Teacher neighborhood crosses sample boundaries")

    lidar_keys = encode_keys(lidar.coords[lidar_index], coarse_shape)
    query_parent = torch.stack((valid_query, lidar_keys), dim=1)
    unique_pairs, pair_inverse = torch.unique(query_parent, dim=0,
                                              sorted=True, return_inverse=True)
    mean_attention = attention.mean(dim=1)[valid_query, valid_slot]
    summed = mean_attention.new_zeros(len(unique_pairs))
    summed.index_add_(0, pair_inverse, mean_attention)
    parent_keys, parent_inverse = torch.unique(unique_pairs[:, 1], sorted=True,
                                               return_inverse=True)
    parent_scores = summed.new_zeros(len(parent_keys))
    parent_scores.scatter_reduce_(0, parent_inverse, summed, reduce="amax",
                                  include_self=True)

    parents = candidate_sites.coords.clone()
    parents[:, 1:] //= 8
    candidate_keys = encode_keys(parents, coarse_shape)
    position = torch.searchsorted(parent_keys, candidate_keys)
    matched = ((position < len(parent_keys))
               & (parent_keys[position.clamp(max=len(parent_keys) - 1)] == candidate_keys))
    evidence[matched] = parent_scores[position[matched]]
    return evidence


def student_objective(output: StudentOutput, teacher_sites, targets, config: StudentConfig,
                      teacher_evidence: torch.Tensor | None = None):
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
    if teacher_evidence is not None and teacher_evidence.shape != logits.shape:
        raise ValueError("Teacher evidence must match existing candidate sites")
    dice_intersection = logits.new_zeros(())
    dice_probability_sum = logits.new_zeros(())
    dice_target_sum = logits.new_zeros(())
    if known.any():
        labels = occupied[known].float()
        if config.lambda_teacher_evidence and teacher_evidence is not None:
            per_site = F.binary_cross_entropy_with_logits(
                logits[known], labels, pos_weight=logits.new_tensor(config.positive_weight),
                reduction="none")
            weights = 1 + config.lambda_teacher_evidence * teacher_evidence[known].detach() * labels
            bce_weight_sum = weights.sum()
            bce = (weights * per_site).sum() / bce_weight_sum
        else:
            # Preserve the original objective exactly when evidence is disabled.
            bce = F.binary_cross_entropy_with_logits(
                logits[known], labels, pos_weight=logits.new_tensor(config.positive_weight))
            bce_weight_sum = logits.new_tensor(len(labels))
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
        bce_weight_sum = logits.new_zeros(())
    spatial = bce + config.dice_weight * dice
    total = config.distillation_weight * distill + config.spatial_weight * spatial
    return total, {"distillation": distill, "spatial": spatial, "bce": bce,
                   "dice": dice, "feature_elements": target_features.numel(),
                   "bce_weight_sum": bce_weight_sum.detach(),
                   "known_sites": known.sum().detach(),
                   "occupied_sites": occupied.sum().detach(),
                   "free_sites": free.sum().detach(),
                   "unknown_sites": (~known).sum().detach(),
                   "dice_intersection": dice_intersection.detach(),
                   "dice_probability_sum": dice_probability_sum.detach(),
                   "dice_target_sum": dice_target_sum.detach()}


def radar_supported_iou(output: StudentOutput, targets, threshold: float,
                        teacher_evidence: torch.Tensor | None = None,
                        evidence_threshold: float = 0.2):
    """Conventional candidate-domain IoU and known-space masked variant."""
    predicted = torch.sigmoid(output.confidence_logits.features[:, 0]) >= threshold
    occupied, free = targets.occupied, targets.known_free
    known = occupied | free
    if predicted.shape != occupied.shape:
        raise ValueError("Prediction and candidate labels differ")
    if teacher_evidence is not None and teacher_evidence.shape != predicted.shape:
        raise ValueError("Teacher evidence must match existing candidate sites")

    def measure(mask):
        intersection = int((predicted & occupied & mask).sum())
        union = int(((predicted | occupied) & mask).sum())
        return {"intersection": intersection, "union": union,
                "iou": intersection / union if union else None}

    report = {
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
    if teacher_evidence is not None:
        report["teacher_correspondence_supported"] = measure(
            known & (teacher_evidence >= evidence_threshold))
    return report


@torch.no_grad()
def radar_supported_iou_sweep(output: StudentOutput, targets,
                              thresholds: tuple[float, ...], teacher_evidence: torch.Tensor,
                              evidence_threshold: float) -> torch.Tensor:
    """Return [threshold, known/conventional/teacher, intersection/union].

    Keep all reductions on the device so validation can transfer the complete
    epoch's counts once, instead of synchronizing for every threshold/frame.
    """
    logits = output.confidence_logits.features[:, 0]
    occupied, free = targets.occupied, targets.known_free
    if occupied.shape != logits.shape or free.shape != logits.shape or teacher_evidence.shape != logits.shape:
        raise ValueError("Prediction, labels, and teacher evidence must share candidate coordinates")
    known = occupied | free
    predicted = torch.sigmoid(logits)[:, None] >= logits.new_tensor(thresholds)[None, :]
    hit = predicted & occupied[:, None]
    union = predicted | occupied[:, None]
    teacher_known = known & (teacher_evidence >= evidence_threshold)
    masks = (known, torch.ones_like(known), teacher_known)
    return torch.stack([
        torch.stack(((hit & mask[:, None]).sum(0),
                     (union & mask[:, None]).sum(0)), dim=-1)
        for mask in masks
    ], dim=1)
