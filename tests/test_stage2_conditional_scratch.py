"""Full-scene checkpoint selection for a clean-input-free validation pass."""

import pytest

from models.radar_lidar_stage2.clean_thresholds import THRESHOLD_STEPS
from scripts.train_stage2_conditional_from_scratch import (
    full_iou_at_half, reconstruction_validation_due,
    resume_batch_size_change)


def test_reconstruction_validation_is_periodic_and_includes_final_epoch():
    due = [epoch for epoch in range(1, 13)
           if reconstruction_validation_due(epoch, 12, 5)]
    assert due == [1, 5, 10, 12]


def test_resume_allows_only_explicit_batch_size_change():
    saved = {"batch_size": 8, "revision": "generated", "seed": 3103}
    larger = {**saved, "batch_size": 16}
    assert resume_batch_size_change(saved, saved, allow_change=False) is None
    with pytest.raises(ValueError):
        resume_batch_size_change(saved, larger, allow_change=False)
    assert resume_batch_size_change(saved, larger, allow_change=True) == 8
    with pytest.raises(ValueError):
        resume_batch_size_change(saved, {**larger, "seed": 99},
                                 allow_change=True)


def test_fixed_half_threshold_combines_all_validation_frames():
    empty = {"tp": 0, "predicted": 0, "target": 0}
    calibration = [dict(empty) for _ in range(THRESHOLD_STEPS + 2)]
    heldout = [dict(empty) for _ in range(THRESHOLD_STEPS + 2)]
    calibration[THRESHOLD_STEPS // 2] = {
        "tp": 3, "predicted": 4, "target": 5}
    heldout[THRESHOLD_STEPS // 2] = {
        "tp": 7, "predicted": 10, "target": 12}
    calibration[-1] = {"tp": 2, "predicted": 3, "target": 5}
    heldout[-1] = {"tp": 6, "predicted": 8, "target": 12}
    result = full_iou_at_half({"curves": {
        "calibration": calibration, "heldout": heldout}})
    assert result["tp"] == 10
    assert result["predicted"] == 14
    assert result["target"] == 17
    assert result["iou"] == 10 / 21
    assert result["faulty_baseline_iou"] == 8 / 20


def test_fixed_half_reports_separate_0p2m_occupancy_iou():
    empty = {"tp": 0, "predicted": 0, "target": 0}
    calibration = [dict(empty) for _ in range(THRESHOLD_STEPS + 2)]
    heldout = [dict(empty) for _ in range(THRESHOLD_STEPS + 2)]
    calibration[THRESHOLD_STEPS // 2] = {
        "tp": 2, "predicted": 4, "target": 5}
    heldout[THRESHOLD_STEPS // 2] = {
        "tp": 3, "predicted": 6, "target": 7}
    report = {"curves": {"calibration": calibration, "heldout": heldout},
              "fixed_half_tolerant": {
                  "calibration": {"predicted": 4, "eligible_target": 5,
                                  "exact_tp": 2, "tolerant_tp_0p2m": 3},
                  "heldout": {"predicted": 6, "eligible_target": 7,
                              "exact_tp": 3, "tolerant_tp_0p2m": 4}}}
    result = full_iou_at_half(report)
    assert result["iou"] == 5 / 17
    assert result["iou_0p2m"] == 7 / 15
