"""Metrics must not gain true positives from duplicate or nearby voxels."""

import numpy as np

from models.radar_lidar_stage2.completion_eval import (
    chamfer_squared_l2, overlap_counts, rates, scoped_cloud, voxel_centers,
)
from scripts.sweep_stage1_stage2_completion import _summarize


def test_exact_and_tolerant_are_one_to_one():
    predicted = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    observed = np.array([[0, 0, 0]], dtype=np.int64)
    counts = overlap_counts(predicted, observed)
    assert counts == {"predicted": 2, "observed": 1,
                      "tp_exact": 1, "tp_at_0_2m": 1}
    assert rates(counts)["iou_exact"] == 0.5
    shifted = overlap_counts(predicted[1:], observed)
    assert shifted["tp_exact"] == 0
    assert shifted["tp_at_0_2m"] == 1


def test_scope_deduplicates_voxels_and_uses_voxel_center_support():
    near = voxel_centers(np.array([[0, 0, 0]], dtype=np.int64))[0]
    far = voxel_centers(np.array([[0, 0, 3]], dtype=np.int64))[0]
    points = np.stack((near, near + np.array([0.01, 0, 0]), far))
    scoped, voxels = scoped_cloud(points, radar_xyz=near[None], radius_m=0.2)
    assert len(scoped) == 2
    np.testing.assert_array_equal(voxels, [[0, 0, 0]])


def test_empty_prediction_has_undefined_chamfer():
    target = voxel_centers(np.array([[0, 0, 0]], dtype=np.int64))
    assert chamfer_squared_l2(np.empty((0, 3), dtype=np.float32), target) is None
    assert chamfer_squared_l2(target, target) == 0.0


def test_stage_thresholds_and_faulty_improvements_are_independent():
    def record(predicted, tp, chamfer):
        return {"predicted": predicted, "observed": 2,
                "tp_exact": tp, "tp_at_0_2m": tp,
                "chamfer_squared_l2": chamfer}

    baseline = record(1, 1, 0.4)
    frame = {"fault_type": "fog_sim",
             "stage1": {"faulty": baseline,
                        "rows": [record(2, 1, 0.3), record(2, 2, 0.2)]},
             "stage2": {"faulty": baseline,
                        "rows": [record(2, 2, 0.2), record(3, 1, 0.5)]}}
    result = _summarize([frame], (0.1, 0.2), "iou_exact")
    assert result["stage1"]["selected_threshold"] == 0.2
    assert result["stage2"]["selected_threshold"] == 0.1
    selected = result["stage2"]["scopes"]["overall"]["rows"][0]
    assert selected["iou_exact_gain"] == 0.5
    assert abs(selected["chamfer_squared_l2_reduction"] - 0.2) < 1e-8
    assert selected["chamfer_paired_frames"] == 1
