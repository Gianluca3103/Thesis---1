"""False-positive-sensitive occupancy supervision on generated LPCR outputs."""

from __future__ import annotations

import torch


def error_sums(logits: torch.Tensor, occupied: torch.Tensor,
               allowed: torch.Tensor | None = None) -> dict:
    """Squared probability errors, split by clean occupancy class."""
    if (logits.ndim != 1 or occupied.shape != logits.shape or
            occupied.dtype != torch.bool or
            (allowed is not None and
             (allowed.shape != logits.shape or allowed.dtype != torch.bool))):
        raise ValueError("Decoder logits and occupancy masks must align")
    if allowed is not None:
        logits, occupied = logits[allowed], occupied[allowed]
    probability = logits.sigmoid()
    positive = probability[occupied]
    negative = probability[~occupied]
    return {"positive_error_sum": (1 - positive).square().sum(),
            "negative_error_sum": negative.square().sum(),
            "positive_count": int(positive.numel()),
            "negative_count": int(negative.numel())}


def asymmetric_occupancy_loss(logits: torch.Tensor,
                               occupied: torch.Tensor,
                               allowed: torch.Tensor,
                               *, false_positive_weight: float) -> tuple[torch.Tensor, dict]:
    """Mean positive miss error plus weighted mean false-positive error.

    Classwise means prevent the much larger empty class from winning merely
    because it has more cells. The weight changes their relative importance.
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
