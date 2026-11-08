"""Standard occupancy BCE on final generated LPCR logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def generated_occupancy_bce(logits: torch.Tensor, occupied: torch.Tensor,
                            allowed: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Mean, unweighted BCE on cells the reconstruction may add."""
    if (logits.ndim != 1 or occupied.shape != logits.shape or
            allowed.shape != logits.shape or occupied.dtype != torch.bool or
            allowed.dtype != torch.bool):
        raise ValueError("Expected aligned logits, clean occupancy, and allowed mask")
    selected = logits[allowed]
    targets = occupied[allowed]
    positive = int(targets.sum())
    negative = len(targets) - positive
    if not len(selected):
        return logits.sum() * 0, {"positive_count": 0, "negative_count": 0}
    loss = F.binary_cross_entropy_with_logits(selected, targets.to(selected.dtype))
    return loss, {"positive_count": positive, "negative_count": negative}
