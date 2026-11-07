"""Micro occupancy threshold accounting for clean-LiDAR denoising."""

from __future__ import annotations

import torch


THRESHOLD_STEPS = 1000


def score_histograms(probabilities: torch.Tensor, target: torch.Tensor,
                     steps: int = THRESHOLD_STEPS
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin probabilities so all thresholds can be evaluated in one pass."""
    if (probabilities.ndim != 1 or probabilities.shape != target.shape or
            target.dtype != torch.bool or steps < 1):
        raise ValueError("Expected aligned 1D probabilities and Boolean labels")
    if bool(((probabilities < 0) | (probabilities > 1) |
             ~torch.isfinite(probabilities)).any()):
        raise ValueError("Occupancy probabilities must be finite in [0,1]")
    buckets = torch.floor(probabilities * steps).long().clamp(0, steps)
    positive = torch.bincount(buckets[target], minlength=steps + 1).cpu()
    negative = torch.bincount(buckets[~target], minlength=steps + 1).cpu()
    return positive, negative


def sweep_rows(positive: torch.Tensor, negative: torch.Tensor
               ) -> list[dict[str, float | int]]:
    if (positive.ndim != 1 or positive.shape != negative.shape or
            len(positive) < 2):
        raise ValueError("Expected equal, nonempty positive and negative histograms")
    steps = len(positive) - 1
    true_positive = positive.flip(0).cumsum(0).flip(0)
    predicted = true_positive + negative.flip(0).cumsum(0).flip(0)
    target = int(positive.sum())
    result = []
    for index in range(steps + 1):
        tp = int(true_positive[index])
        pred = int(predicted[index])
        union = pred + target - tp
        result.append({"threshold": index / steps, "tp": tp,
                       "predicted": pred, "target": target,
                       "precision": tp / pred if pred else 0.0,
                       "recall": tp / target if target else 0.0,
                       "f1": 2 * tp / (pred + target) if pred + target else 0.0,
                       "iou": tp / union if union else 0.0})
    return result


def selected_indices(rows: list[dict]) -> dict[str, int | None]:
    """Choose operating points using calibration rows only."""
    if not rows:
        raise ValueError("Threshold sweep is empty")

    def choose(metric: str, *, minimum_precision: float = 0.0,
               minimum_recall: float = 0.0) -> int | None:
        feasible = [index for index, row in enumerate(rows)
                    if row["precision"] >= minimum_precision and
                    row["recall"] >= minimum_recall]
        if not feasible:
            return None
        return max(feasible, key=lambda index: (
            rows[index][metric], rows[index]["precision"],
            rows[index]["recall"], rows[index]["threshold"]))

    return {"fixed_0p5": min(range(len(rows)),
                              key=lambda i: abs(rows[i]["threshold"] - 0.5)),
            "best_f1": choose("f1"),
            "best_iou": choose("iou"),
            "max_recall_at_precision_20pct": choose(
                "recall", minimum_precision=0.20),
            "max_precision_at_recall_90pct": choose(
                "precision", minimum_recall=0.90),
            "max_precision_at_recall_95pct": choose(
                "precision", minimum_recall=0.95)}
