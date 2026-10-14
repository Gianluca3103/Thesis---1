"""Four independently switchable losses and counted teacher metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .teacher import TeacherConfig, TeacherOutput


@dataclass
class Counted:
    numerator: torch.Tensor
    denominator: int

    @property
    def mean(self):
        return self.numerator / max(self.denominator, 1)


def teacher_objective(output: TeacherOutput, radar_ids: torch.Tensor,
                      lidar_ids: torch.Tensor, config: TeacherConfig,
                      reliability: torch.Tensor | None = None,
                      class_by_id: dict[int, str] | None = None):
    """Object labels are point-dominant instance IDs; -1 means unverified."""
    neighbors = output.neighbors
    n, k = neighbors.indices.shape
    if radar_ids.shape != (n,) or lidar_ids.shape != (len(output.lidar_frozen.coords),):
        raise ValueError("Voxel instance labels do not match the frozen encoders")
    valid = neighbors.valid
    candidate_ids = (lidar_ids[neighbors.indices.clamp_min(0)] if len(lidar_ids)
                     else radar_ids.new_full((n, k), -1))
    same = valid & (radar_ids[:, None] >= 0) & (candidate_ids == radar_ids[:, None])
    verified_different = valid & (radar_ids[:, None] >= 0) & (candidate_ids >= 0) & ~same
    compatible = valid & ~verified_different
    if reliability is not None:
        if reliability.shape != valid.shape:
            raise ValueError("Reliability mask must match candidates")
        compatible &= reliability > 0
    mean_attention = output.attention.mean(1)
    zero = output.fused.features.sum() * 0

    object_use = same.any(-1)
    object_losses = -torch.log((mean_attention * same).sum(-1).clamp_min(1e-8))
    object_term = Counted(object_losses[object_use].sum() if object_use.any() else zero,
                          int(object_use.sum()))

    distance = neighbors.distances_m
    logits = -(distance / config.sigma_m).square() / 2
    logits = logits.masked_fill(~compatible, -1e4)
    target = torch.softmax(logits, -1) * compatible
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    geometry_use = compatible.any(-1)
    geometry_losses = -(target * torch.log(mean_attention.clamp_min(1e-8))).sum(-1)
    geometry_term = Counted(geometry_losses[geometry_use].sum() if geometry_use.any() else zero,
                            int(geometry_use.sum()))

    if len(lidar_ids):
        lidar_reference = output.lidar_frozen.features[neighbors.indices.clamp_min(0)]
        fusion_target = (target[..., None] * lidar_reference).sum(1).detach()
        # Fixed frozen-reference normalization matches the final LayerNorm output.
        fusion_target = F.layer_norm(fusion_target, (config.feature_dim,))
        fusion_error = F.smooth_l1_loss(output.fused.features, fusion_target,
                                        beta=config.fusion_beta, reduction="none").sum(-1)
        fusion_abs = (output.fused.features - fusion_target).abs().sum(-1)
    else:
        fusion_error = output.fused.features.new_zeros((n,))
        fusion_abs = fusion_error
    feature_count = int(geometry_use.sum()) * config.feature_dim
    fusion_term = Counted(fusion_error[geometry_use].sum() if geometry_use.any() else zero,
                          feature_count)
    mae_term = Counted(fusion_abs[geometry_use].sum().detach() if geometry_use.any() else zero.detach(),
                       feature_count)

    # Prototype contrastive alignment is independent of the Q/K attention logits.
    # Only verified different annotated instances enter the denominator.
    prototypes = {}
    for identity in torch.unique(lidar_ids[lidar_ids >= 0]).tolist():
        members = output.lidar_alignment[lidar_ids == identity]
        prototypes[int(identity)] = F.normalize(members.mean(0), dim=0)
    contrastive_losses, positive_cosines, negative_cosines = [], [], []
    for index in torch.nonzero(object_use, as_tuple=False).flatten().tolist():
        identity = int(radar_ids[index])
        if identity not in prototypes:
            continue
        batch = int(output.fused.coords[index, 0])
        negatives = [other for other in sorted(prototypes)
                     if other != identity and other // 1_000_000 == batch]
        if not negatives:
            continue
        query = output.radar_alignment[index]
        positive = torch.dot(query, prototypes[identity])
        negative = torch.stack([torch.dot(query, prototypes[other]) for other in negatives])
        contrastive_losses.append(torch.logsumexp(torch.cat((positive[None], negative)) /
                                                       config.temperature, 0) - positive / config.temperature)
        positive_cosines.append(positive.detach())
        negative_cosines.extend(value.detach() for value in negative)
    contrastive_term = Counted(torch.stack(contrastive_losses).sum() if contrastive_losses else zero,
                               len(contrastive_losses))

    terms = {"object": object_term, "geometry": geometry_term,
             "contrastive": contrastive_term, "fusion": fusion_term}
    total = (config.object_weight * object_term.mean +
             config.geometry_weight * geometry_term.mean +
             config.contrastive_weight * contrastive_term.mean +
             config.fusion_weight * fusion_term.mean)
    metrics = {
        "object_queries": int((radar_ids >= 0).sum()),
        "excluded_object_queries": int((radar_ids < 0).sum()),
        "object_candidate_coverage": Counted(object_use.sum().detach(), int((radar_ids >= 0).sum())),
        "geometry_eligible": geometry_term.denominator,
        "geometry_query_coverage": Counted(geometry_use.sum().detach(), n),
        "radar_active_sites": n,
        "lidar_active_sites": len(output.lidar_frozen.coords),
        "feature_variance": Counted(output.fused.features.var(0, unbiased=False).mean().detach() * n
                                    if n else zero.detach(), n),
        "fusion_mae": mae_term,
        "positive_cosine": Counted(torch.stack(positive_cosines).sum() if positive_cosines else zero.detach(),
                                    len(positive_cosines)),
        "negative_cosine": Counted(torch.stack(negative_cosines).sum() if negative_cosines else zero.detach(),
                                    len(negative_cosines)),
        "gate_mean": Counted(output.gate.sum().detach(), len(output.gate)),
        "fusion_magnitude": Counted((output.gate * output.lidar_contribution).norm(dim=-1).sum().detach(), n),
    }
    for top in (1, 3, 5):
        selected = mean_attention.topk(min(top, k), dim=-1).indices
        hit = same.gather(1, selected).any(-1)
        metrics[f"object_r{top}"] = Counted(hit[object_use].sum().detach(), object_term.denominator)
        if class_by_id:
            for cls in sorted(set(class_by_id.values())):
                cls_use = object_use & torch.as_tensor(
                    [class_by_id.get(int(identity)) == cls for identity in radar_ids],
                    device=radar_ids.device)
                metrics[f"object_r{top}/class={cls}"] = Counted(
                    hit[cls_use].sum().detach(), int(cls_use.sum()))
    return total, terms, metrics
