"""Observed within-cell point histograms for independent PFE pretraining."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.sparse import encode_keys

from .encoder import GRID


def observed_child_distribution(sites, points: torch.Tensor, valid: torch.Tensor):
    """512 child bins per occupied stride-8 site; no physical free-space claim."""
    if sites.stride != 8 or sites.shape_zyx != GRID.scale_shape(8):
        raise ValueError("Expected original stride-8 PFE sites")
    counts = points.new_zeros((len(sites.coords), 512))
    if not len(sites.coords):
        return counts
    minimum = points.new_tensor(GRID.minimum_xyz)
    step = points.new_tensor(GRID.size_xyz)
    fine_xyz = torch.floor((points[..., :3] - minimum) / step).long()
    shape_xyz = points.new_tensor(GRID.shape_zyx[::-1], dtype=torch.long)
    inside = valid & ((fine_xyz >= 0) & (fine_xyz < shape_xyz)).all(-1)
    xyz = fine_xyz[inside]
    batch = torch.arange(len(points), device=points.device)[:, None].expand_as(valid)[inside]
    coarse = xyz // 8
    parent = torch.stack((batch, coarse[:, 2], coarse[:, 1], coarse[:, 0]), -1)
    parent_keys = encode_keys(parent, GRID.scale_shape(8))
    site_keys = encode_keys(sites.coords, sites.shape_zyx)
    location = torch.searchsorted(site_keys, parent_keys)
    present = location < len(site_keys)
    present &= site_keys[location.clamp(max=len(site_keys) - 1)] == parent_keys
    child = xyz % 8
    child_index = (child[:, 2] * 8 + child[:, 1]) * 8 + child[:, 0]
    flat = location[present] * 512 + child_index[present]
    counts.view(-1).index_add_(0, flat, counts.new_ones(len(flat)))
    return counts


def observed_distribution_loss(logits: torch.Tensor, counts: torch.Tensor,
                               mask: torch.Tensor | None = None):
    """Cross entropy over observed return locations; no negative occupancy BCE."""
    if logits.shape != counts.shape or logits.ndim != 2 or logits.shape[1] != 512:
        raise ValueError("Expected [active bottleneck sites,512] local distributions")
    eligible = counts.sum(-1) > 0
    if mask is not None:
        if mask.shape != eligible.shape:
            raise ValueError("Mask and active-site counts differ")
        eligible &= mask
    if not eligible.any():
        return logits.sum() * 0, 0
    target = counts[eligible] / counts[eligible].sum(-1, keepdim=True)
    per_site = -(target * F.log_softmax(logits[eligible], -1)).sum(-1)
    return per_site.mean(), len(per_site)
