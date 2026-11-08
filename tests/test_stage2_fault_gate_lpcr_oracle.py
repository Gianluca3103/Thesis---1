"""The LPCR oracle must compare the same full-scene baseline and gate."""

import torch

from models.radar_lidar_stage2.clean_thresholds import THRESHOLD_STEPS
from scripts.evaluate_stage2_fault_gate_lpcr_oracle import summarize


def test_oracle_summary_uses_calibration_threshold_and_perfect_gate_ceiling():
    stores = {}
    for name in ("calibration", "heldout", "all_validation"):
        positive = torch.zeros(THRESHOLD_STEPS + 1, dtype=torch.long)
        negative = torch.zeros_like(positive)
        positive[800] = 1
        negative[200] = 1
        stores[name] = {
            "positive": positive, "negative": negative,
            "baseline": {"tp": 1, "predicted": 2, "target": 2},
            "frames": 1,
        }

    report = summarize(stores)

    assert report["calibration_selected_threshold"] == 0.8
    heldout = report["groups"]["heldout"]
    assert heldout["faulty_baseline"]["iou"] == 1 / 3
    assert heldout["gate_perfect_classifier_ceiling"]["iou"] == 2 / 3
    assert heldout["lpcr_clean_latent_at_0p5"]["iou"] == 2 / 3
    assert heldout["lpcr_clean_latent_calibrated"]["addition_precision"] == 1
    assert heldout["lpcr_clean_latent_calibrated"]["addition_eligible_recall"] == 1
