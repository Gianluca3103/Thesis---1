"""Full-scene checkpoint selection for a clean-input-free validation pass."""

from models.radar_lidar_stage2.clean_thresholds import THRESHOLD_STEPS
from scripts.train_stage2_conditional_from_scratch import full_iou_at_half


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
