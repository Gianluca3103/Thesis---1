"""False-positive-sensitive margin supervision on generated LPCR logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F


LOGIT_MARGIN = 0.5


def error_sums(logits: torch.Tensor, occupied: torch.Tensor,
               allowed: torch.Tensor | None = None) -> dict:
    """Margin violations, split by clean occupancy class."""
    if (logits.ndim != 1 or occupied.shape != logits.shape or
            occupied.dtype != torch.bool or
            (allowed is not None and
             (allowed.shape != logits.shape or allowed.dtype != torch.bool))):
        raise ValueError("Decoder logits and occupancy masks must align")
    if allowed is not None:
        logits, occupied = logits[allowed], occupied[allowed]
    positive = logits[occupied]
    negative = logits[~occupied]
    return {"positive_error_sum": F.relu(LOGIT_MARGIN - positive).sum(),
            "negative_error_sum": F.relu(LOGIT_MARGIN + negative).sum(),
            "positive_count": int(positive.numel()),
            "negative_count": int(negative.numel())}


def asymmetric_occupancy_loss(logits: torch.Tensor,
                               occupied: torch.Tensor,
                               allowed: torch.Tensor,
                               *, false_positive_weight: float) -> tuple[torch.Tensor, dict]:
    """Mean positive margin error plus weighted false-positive margin error.

    Classwise means prevent the much larger empty class from winning merely
    because it has more cells. The weight changes their relative importance.
    A high-confidence false positive receives a non-vanishing gradient.
    No BCE or offset loss is used.
    """
    if false_positive_weight <= 0:
        raise ValueError("False-positive weight must be positive")
    parts = error_sums(logits, occupied, allowed)
    positive = (parts["positive_error_sum"] /
                max(parts["positive_count"], 1))
    negative = (parts["negative_error_sum"] /
                max(parts["negative_count"], 1))
    return positive + false_positive_weight * negative, parts
