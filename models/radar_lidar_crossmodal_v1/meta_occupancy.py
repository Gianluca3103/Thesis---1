"""Radar-only occupancy with a LiDAR-conditioned, meta-trained teacher.

The teacher never supplies a decoder input. Its map-loss gradient travels
through a differentiable one-step feature-distillation update of the student.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from models.radar_lidar_stage1.sparse import encode_keys, voxelize
from models.radar_lidar_stage2.candidate_domain import voxel_centers_xyz

from .encoder import GRID
from .student_loss import student_targets


SCAN_TOLERANCE_M = 0.2


@dataclass(frozen=True)
class MetaOccupancyConfig:
    radius_m: float = 3.2
    max_candidates: int = 24
    teacher_sigma_m: float = 2.0
    distillation_weight: float = 1.0
    inner_lr: float = 0.05
    smooth_l1_beta: float = 1.0
    teacher_geometry_weight: float = 0.05
    teacher_fusion_weight: float = 0.05
    threshold: float = 0.5

    def __post_init__(self):
        if (self.radius_m <= 0 or self.max_candidates < 1 or self.teacher_sigma_m <= 0
                or self.distillation_weight < 0 or self.inner_lr <= 0
                or self.smooth_l1_beta <= 0 or not 0 < self.threshold < 1
                or min(self.teacher_geometry_weight, self.teacher_fusion_weight) < 0):
            raise ValueError("Invalid meta-occupancy configuration")


def _near_radar(points_xyz: np.ndarray, point_batches: np.ndarray,
                radar: torch.Tensor, radar_valid: torch.Tensor, radius_m: float,
                device: torch.device) -> torch.Tensor:
    """Exact Euclidean support, with no neighbors from another frame."""
    if radius_m <= 0:
        raise ValueError("Support radius must be positive")
    result = np.zeros(len(points_xyz), dtype=np.bool_)
    radar_xyz = radar[..., :3].detach().cpu().numpy()
    valid = radar_valid.detach().cpu().numpy()
    for batch in np.unique(point_batches):
        selected = np.flatnonzero(point_batches == batch)
        returns = radar_xyz[int(batch), valid[int(batch)]]
        if len(returns):
            distance, _ = cKDTree(returns).query(points_xyz[selected], k=1)
            result[selected] = distance <= radius_m + 1e-6
    return torch.from_numpy(result).to(device=device)


def candidate_support(coords: torch.Tensor, radar: torch.Tensor,
                      radar_valid: torch.Tensor, radius_m: float) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Candidate coordinates must be [N,4] in batch,z,y,x order")
    centers = voxel_centers_xyz(coords, GRID).detach().cpu().numpy()
    batches = coords[:, 0].detach().cpu().numpy()
    return _near_radar(centers, batches, radar, radar_valid, radius_m, coords.device)


def supported_clean_points(clean: torch.Tensor, clean_valid: torch.Tensor,
                           radar: torch.Tensor, radar_valid: torch.Tensor,
                           radius_m: float) -> int:
    """Denominator for coverage: in-grid clean returns inside radar support."""
    xyz = clean[..., :3]
    lower = xyz.new_tensor(GRID.minimum_xyz)
    upper = xyz.new_tensor(GRID.maximum_xyz)
    in_grid = ((xyz >= lower) & (xyz < upper)).all(-1)
    valid = clean_valid & in_grid
    batch_index, point_index = valid.nonzero(as_tuple=True)
    if not len(batch_index):
        return 0
    selected = xyz[batch_index, point_index].detach().cpu().numpy()
    mask = _near_radar(selected, batch_index.detach().cpu().numpy(),
                       radar, radar_valid, radius_m, clean.device)
    return int(mask.sum())


def supported_clean_voxels(clean: torch.Tensor, clean_valid: torch.Tensor,
                           radar: torch.Tensor, radar_valid: torch.Tensor,
                           radius_m: float) -> int:
    """All unique observed clean voxels in radar support, generated or not."""
    coordinates, _, _, _ = voxelize(clean, clean_valid, GRID)
    return int(candidate_support(coordinates, radar, radar_valid, radius_m).sum())


@torch.no_grad()
def scan_overlap_counts(output, clean: torch.Tensor, clean_valid: torch.Tensor,
                        radar: torch.Tensor, radar_valid: torch.Tensor,
                        support: torch.Tensor, threshold: float,
                        radius_m: float) -> dict[str, int]:
    """Compare thresholded radar map with clean-scan occupied voxels.

    Exact matches share a voxel key. Tolerant matches use a maximum-cardinality
    one-to-one matching of voxel centers, within the same frame. This is scan
    agreement: a missing clean return does not prove that space is empty.
    """
    if not 0 < threshold < 1:
        raise ValueError("Probability threshold must be in (0,1)")
    logits = output.confidence_logits.features[:, 0]
    coordinates = output.confidence_logits.coords
    if support.shape != logits.shape:
        raise ValueError("Radar support mask must match candidate logits")
    predicted = coordinates[support & (torch.sigmoid(logits) >= threshold)]
    observed, _, _, _ = voxelize(clean, clean_valid, GRID)
    observed = observed[candidate_support(observed, radar, radar_valid, radius_m)]
    return scan_overlap_counts_observed(predicted, observed)


def scan_overlap_counts_observed(predicted: torch.Tensor,
                                 observed: torch.Tensor) -> dict[str, int]:
    """Scan agreement from fixed, radar-supported clean voxel coordinates."""
    prediction_count, observation_count = len(predicted), len(observed)
    if not prediction_count or not observation_count:
        return {"predicted": prediction_count, "observed": observation_count,
                "tp_exact": 0, "tp_at_0_2m": 0}

    prediction_keys = encode_keys(predicted, GRID.shape_zyx).cpu().numpy()
    observation_keys = encode_keys(observed, GRID.shape_zyx).cpu().numpy()
    tp_exact = int(np.intersect1d(prediction_keys, observation_keys,
                                  assume_unique=True).size)
    prediction_xyz = voxel_centers_xyz(predicted, GRID).cpu().numpy()
    observation_xyz = voxel_centers_xyz(observed, GRID).cpu().numpy()
    prediction_batches = predicted[:, 0].cpu().numpy()
    observation_batches = observed[:, 0].cpu().numpy()
    tp_tolerant = 0
    for batch in np.intersect1d(np.unique(prediction_batches),
                                np.unique(observation_batches)):
        left = prediction_xyz[prediction_batches == batch]
        right = observation_xyz[observation_batches == batch]
        edges = cKDTree(left).sparse_distance_matrix(
            cKDTree(right), max_distance=SCAN_TOLERANCE_M + 1e-6,
            output_type="coo_matrix")
        if edges.nnz:
            graph = csr_matrix((np.ones(edges.nnz, dtype=np.int8),
                                (edges.row, edges.col)), shape=(len(left), len(right)))
            tp_tolerant += int((maximum_bipartite_matching(graph) >= 0).sum())
    return {"predicted": prediction_count, "observed": observation_count,
            "tp_exact": tp_exact, "tp_at_0_2m": tp_tolerant}


def scan_overlap_rates(counts: dict[str, int]) -> dict[str, float | None]:
    predicted, observed = counts["predicted"], counts["observed"]
    if (min(predicted, observed, counts["tp_exact"], counts["tp_at_0_2m"]) < 0
            or not counts["tp_exact"] <= counts["tp_at_0_2m"] <= min(predicted, observed)):
        raise ValueError("Invalid exact or tolerant scan-match counts")
    result = {}
    for label, tp in (("exact", counts["tp_exact"]),
                      ("at_0_2m", counts["tp_at_0_2m"])):
        union = predicted + observed - tp
        total = predicted + observed
        result[f"iou_{label}"] = tp / union if union else None
        result[f"f1_{label}"] = 2 * tp / total if total else None
    return result


@torch.no_grad()
def radar_only_probability_map(student: nn.Module, radar: torch.Tensor,
                               radar_valid: torch.Tensor, radius_m: float = 3.2):
    """Inference output excludes unsupervised candidates outside radar support."""
    output = student(radar, radar_valid).exported()
    if output.get("interface_version") == "crossmodal_fine_association_v1":
        trained_radius_m = student.config.target_radius_m
        if abs(radius_m - trained_radius_m) > 1e-6:
            raise ValueError("Fine-association pointers require their trained radius")
        # The pointer head already enforces its Euclidean radius from each
        # active radar voxel. A second raw-return radius filter would wrongly
        # discard boundary pointers because raw returns and voxel centers differ.
        output["radar_support_radius_m"] = radius_m
        return output
    support = candidate_support(output["confidence_coords"], radar, radar_valid,
                                radius_m)
    for key in ("confidence_coords", "confidence_logits", "confidence_probabilities"):
        output[key] = output[key][support]
    output["radar_support_radius_m"] = radius_m
    return output


def feature_distillation(student_sites, teacher_sites, *, beta: float,
                         detach_teacher: bool,
                         eligible: torch.Tensor | None = None) -> torch.Tensor:
    if (student_sites.shape_zyx != teacher_sites.shape_zyx
            or student_sites.stride != teacher_sites.stride
            or student_sites.features.shape != teacher_sites.features.shape
            or not torch.equal(encode_keys(student_sites.coords, student_sites.shape_zyx),
                               encode_keys(teacher_sites.coords, teacher_sites.shape_zyx))):
        raise ValueError("Teacher and student radar-coordinate features differ")
    target = teacher_sites.features.detach() if detach_teacher else teacher_sites.features
    if eligible is not None:
        if eligible.shape != (len(target),):
            raise ValueError("Teacher-eligible query mask differs from feature sites")
        target = target[eligible]
        student_features = student_sites.features[eligible]
    else:
        student_features = student_sites.features
    if not len(target):
        return student_sites.features.sum() * 0
    return F.smooth_l1_loss(student_features, target, beta=beta)


def occupancy_loss(output, targets, support: torch.Tensor, threshold: float):
    """Unweighted proper BCE on occupied or verified-free supported voxels."""
    logits = output.confidence_logits.features[:, 0]
    occupied, free = targets.occupied, targets.known_free
    if (logits.shape != support.shape or logits.shape != occupied.shape
            or logits.shape != free.shape or bool((occupied & free).any())):
        raise ValueError("Occupancy labels and radar support must match logits")
    known = support & (occupied | free)
    count = int(known.sum())
    if count:
        labels = occupied[known].to(logits.dtype)
        bce_sum = F.binary_cross_entropy_with_logits(logits[known], labels, reduction="sum")
        probabilities = torch.sigmoid(logits[known])
        brier_sum = ((probabilities - labels) ** 2).sum()
        predicted = probabilities >= threshold
        truth = occupied[known]
        intersection = int((predicted & truth).sum())
        union = int((predicted | truth).sum())
        loss = bce_sum / count
    else:
        loss = logits.sum() * 0
        bce_sum = loss.detach()
        brier_sum = loss.detach()
        intersection = union = 0
    return loss, {
        "bce_sum": float(bce_sum.detach()),
        "brier_sum": float(brier_sum.detach()),
        "known": count,
        "occupied": int((support & occupied).sum()),
        "free": int((support & free).sum()),
        "candidate_sites": len(logits),
        "supported_candidate_sites": int(support.sum()),
        "represented_clean_points": (
            targets.represented_clean_points_supported
            if hasattr(targets, "represented_clean_points_supported")
            else int(targets.clean_point_count[support].sum())),
        "intersection": intersection,
        "union": union,
    }


def relationship_loss(teacher_output, config: MetaOccupancyConfig):
    """Light LiDAR-geometry anchor; the quiz BCE remains the outer objective."""
    valid = teacher_output.neighbors.valid
    eligible = valid.any(-1)
    zero = teacher_output.fused.features.sum() * 0
    if not bool(eligible.any()):
        return zero, {"geometry": 0.0, "fusion": 0.0, "eligible": 0}
    distance = teacher_output.neighbors.distances_m
    logits = -0.5 * (distance / config.teacher_sigma_m).square()
    target = torch.softmax(logits.masked_fill(~valid, -1e4), dim=-1) * valid
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    attention = teacher_output.attention.mean(1)
    geometry = -(target * torch.log(attention.clamp_min(1e-8))).sum(-1)[eligible].mean()
    reference = teacher_output.lidar_frozen.features[
        teacher_output.neighbors.indices.clamp_min(0)]
    reference = (target[..., None] * reference).sum(1).detach()
    reference = F.layer_norm(reference, (reference.shape[-1],))
    fusion = F.smooth_l1_loss(teacher_output.fused.features[eligible], reference[eligible])
    total = config.teacher_geometry_weight * geometry + config.teacher_fusion_weight * fusion
    return total, {"geometry": float(geometry.detach()),
                   "fusion": float(fusion.detach()), "eligible": int(eligible.sum())}


class _EncodedStudent(nn.Module):
    """Decode radar sites for the virtual step without re-encoding the quiz."""

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student

    def forward(self, radar_sites):
        return self.student.forward_encoded(radar_sites)


def meta_teacher_step(teacher: nn.Module, student: nn.Module, paired_batch: dict,
                      quiz_batch: dict, teacher_optimizer: torch.optim.Optimizer,
                      config: MetaOccupancyConfig):
    """Update teacher from the radar-only map after one virtual distillation step.

    The quiz batch is a different *training* batch. Validation never updates
    weights. The virtual step moves student feature-head parameters. The real
    student step updates its encoder, and this outer step updates both teacher
    encoders through the quiz loss and relationship anchor.
    """
    if not student.config.decoder_from_conditioned:
        raise ValueError("Meta training requires the decoder to consume distilled features")
    teacher_output = teacher(paired_batch["radar"], paired_batch["radar_valid"],
                             paired_batch["clean_lidar"],
                             paired_batch["clean_lidar_valid"],
                             neighbors=paired_batch.get("neighbors"))
    if "radar_coords" in paired_batch:
        if (not torch.equal(teacher_output.radar_frozen.coords, paired_batch["radar_coords"])
                or not torch.equal(teacher_output.lidar_frozen.coords,
                                   paired_batch["lidar_coords"])):
            raise ValueError("Cached neighbor coordinates differ from live encoders")
    eligible = teacher_output.neighbors.valid.any(-1)
    if not bool(eligible.any()):
        return {"skipped": True, "reason": "no LiDAR neighbors"}
    with torch.no_grad():
        paired_radar = student.radar_encoder(paired_batch["radar"],
                                             paired_batch["radar_valid"])
    student_feature, _ = student.encode_conditioning_sites(paired_radar)
    distill = feature_distillation(student_feature, teacher_output.fused,
                                   beta=config.smooth_l1_beta, detach_teacher=False,
                                   eligible=eligible)
    selected = [(name, parameter) for name, parameter in student.named_parameters()
                if parameter.requires_grad and name.startswith(
                    ("shared.", "feature_head.", "feature_norm."))]
    if not selected:
        raise ValueError("Student has no trainable feature path")
    inner_grads = torch.autograd.grad(distill, [parameter for _, parameter in selected],
                                      create_graph=True, allow_unused=True)
    virtual = {name: parameter - config.inner_lr * config.distillation_weight * gradient
               for (name, parameter), gradient in zip(selected, inner_grads)
               if gradient is not None}
    with torch.no_grad():
        quiz_radar = student.radar_encoder(quiz_batch["radar"],
                                           quiz_batch["radar_valid"])
    quiz_output = functional_call(_EncodedStudent(student),
                                  {f"student.{name}": value
                                   for name, value in virtual.items()},
                                  (quiz_radar,))
    if "candidate_coords" in quiz_batch:
        if not torch.equal(quiz_output.confidence_logits.coords,
                           quiz_batch["candidate_coords"]):
            raise ValueError("Cached quiz candidate coordinates differ")
        quiz_targets = quiz_batch["targets"]
        support = quiz_batch["support"]
    else:
        quiz_targets = student_targets(quiz_output, quiz_batch["clean_lidar"],
                                       quiz_batch["clean_lidar_valid"])
        support = candidate_support(quiz_output.confidence_logits.coords,
                                    quiz_batch["radar"], quiz_batch["radar_valid"],
                                    config.radius_m)
    quiz_loss, quiz_stats = occupancy_loss(quiz_output, quiz_targets, support,
                                           config.threshold)
    if not quiz_stats["known"]:
        return {"skipped": True, "reason": "no observed voxels in radar support"}
    relation, relation_stats = relationship_loss(teacher_output, config)
    outer_loss = quiz_loss + relation
    teacher_optimizer.zero_grad(set_to_none=True)
    teacher_params = [parameter for parameter in teacher.parameters()
                      if parameter.requires_grad]
    gradients = torch.autograd.grad(outer_loss, teacher_params, allow_unused=True)
    for parameter, gradient in zip(teacher_params, gradients):
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(teacher_params, 5.0)
    teacher_optimizer.step()
    return {"quiz_bce": float(quiz_loss.detach()),
            "distillation": float(distill.detach()),
            "relationship": relation_stats, "quiz": quiz_stats}
