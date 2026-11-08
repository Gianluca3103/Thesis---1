"""Addition-only supervision for the conditioned clean-latent diagnostic.

The old Stage-II addition head is trained on unobserved radar-gated cells.
Clean occupancy and the synthetic injector footprint supply labels only.
Neither is an input to the addition head at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .addition import addition_context, healthy_region_negatives
from .clean_denoising import frame_mean
from .fault_region import footprint_mask
from .geometry_gate import offset_targets
from .sparse_residual import SparseResidualStage2, cosine_alpha_bars
from scripts.train_stage2_clean_denoiser import (
    _batch, _conditioned_inputs, sample_clean_latent)


def touched_cells(fine: torch.Tensor, regions: list[dict]) -> torch.Tensor:
    if fine.ndim != 2 or fine.shape[1] != 4:
        raise ValueError("Expected batched BZYX candidate cells")
    if len(fine) and (bool((fine[:, 0] < 0).any()) or
                      bool((fine[:, 0] >= len(regions)).any())):
        raise ValueError("Candidate batch IDs exceed injector regions")
    result = torch.zeros(len(fine), dtype=torch.bool, device=fine.device)
    for frame, region in enumerate(regions):
        mask = fine[:, 0] == frame
        result[mask] = footprint_mask(fine[mask], region)
    return result


def addition_masks(clean: torch.Tensor, clean_ids: torch.Tensor,
                   faulty: torch.Tensor, faulty_ids: torch.Tensor,
                   fine: torch.Tensor, regions: list[dict]) -> dict:
    """Reproduce old healthy negatives and label the larger untouched sector."""
    context = addition_context(faulty, fine, faulty_ids)
    occupied, _, _ = offset_targets(clean, fine, clean_ids)
    touched = touched_cells(fine, regions)
    candidates = ~context.observed
    positive = candidates & occupied
    negative = candidates & ~occupied
    healthy = healthy_region_negatives(
        clean, fine, occupied, context, clean_ids) & negative
    reliable = negative & ~touched
    return {"context": context, "occupied": occupied,
            "candidates": candidates, "positive": positive,
            "negative": negative, "healthy": healthy,
            "reliable": reliable, "touched": touched}


def addition_losses(logits: torch.Tensor, fine: torch.Tensor, masks: dict,
                    batch_size: int, *, positive_weight: float = 1.0,
                    negative_weight: float = 1.0,
                    healthy_weight: float = 1.0,
                    reliable_weight: float = 1.0) -> dict:
    """Balanced addition BCE, plus extra weight on reliable empty cells."""
    if logits.shape != (len(fine),) or batch_size < 1 or any(
            weight < 0 for weight in (positive_weight, negative_weight,
                                      healthy_weight, reliable_weight)):
        raise ValueError("Invalid addition scores or loss weights")
    frame_ids = fine[:, 0].long()

    def term(mask: torch.Tensor, positive: bool) -> torch.Tensor:
        if not bool(mask.any()):
            return logits.sum() * 0
        values = F.softplus(-logits[mask] if positive else logits[mask])
        return frame_mean(values, frame_ids[mask], batch_size)

    positive = term(masks["positive"], True)
    negative = term(masks["negative"], False)
    healthy = term(masks["healthy"], False)
    reliable = term(masks["reliable"], False)
    total = (positive_weight * positive + negative_weight * negative
             + healthy_weight * healthy + reliable_weight * reliable)
    return {"total": total, "positive": positive, "negative": negative,
            "healthy": healthy, "reliable": reliable,
            "positive_sites": int(masks["positive"].sum()),
            "negative_sites": int(masks["negative"].sum()),
            "healthy_sites": int(masks["healthy"].sum()),
            "reliable_sites": int(masks["reliable"].sum())}


def decoded_addition_batch(model: SparseResidualStage2, rows: list[dict],
                           alpha_bars: torch.Tensor,
                           generator: torch.Generator, device: str,
                           *, eval_timestep: int, train_head: bool,
                           labels: bool = True) -> dict:
    """Hold the denoiser and LPCR fixed while fitting the addition head."""
    with torch.no_grad():
        hierarchy, clean, clean_ids = _batch(rows, device)
        fine, coarse = hierarchy[0], hierarchy[-1]
        faults = [row["faulty_xyz"] for row in rows]
        faulty = torch.cat(faults).to(device, non_blocking=True)
        faulty_ids = torch.cat([
            torch.full((len(cloud),), index, dtype=torch.long)
            for index, cloud in enumerate(faults)]).to(device)
        if not len(coarse):
            return {"fine": fine, "clean": clean, "clean_ids": clean_ids,
                    "faulty": faulty, "faulty_ids": faulty_ids,
                    "logits": fine.new_empty((0,), dtype=torch.float32),
                    "masks": None}
        x0, _ = model.lidar_encoder(clean, coarse, clean_ids)
        conditions = _conditioned_inputs(model, rows, hierarchy, device)
        recovered = sample_clean_latent(
            model, x0, coarse, alpha_bars, generator,
            batch_size=len(rows), strategy="one_step",
            eval_timestep=eval_timestep, conditions=conditions)
        full_logits, _, hidden = model.lpcr(
            recovered.clamp(-1, 1), hierarchy,
            x0.new_ones((len(fine), 1)), return_hidden=True)
        if labels:
            masks = addition_masks(clean, clean_ids, faulty, faulty_ids, fine,
                                   [row["region"] for row in rows])
        else:
            context = addition_context(faulty, fine, faulty_ids)
            occupied, _, _ = offset_targets(clean, fine, clean_ids)
            masks = {"context": context, "occupied": occupied,
                     "candidates": ~context.observed}
    with torch.set_grad_enabled(train_head):
        logits = model.addition_head(
            hidden, full_logits, full_logits.new_ones((len(fine), 1)),
            masks["context"].observed,
            masks["context"].local_faulty_density)
    return {"fine": fine, "clean": clean, "clean_ids": clean_ids,
            "faulty": faulty, "faulty_ids": faulty_ids,
            "logits": logits, "masks": masks}
