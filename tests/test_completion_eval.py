"""Metrics must not gain true positives from duplicate or nearby voxels."""

import numpy as np

from models.radar_lidar_stage2.completion_eval import (
    classify_addition_false_positives, chamfer_squared_l2, overlap_counts,
    rates, scoped_cloud, voxel_centers,
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


def test_precision_recall_and_precision_weighted_f_score():
    result = rates({"predicted": 10, "observed": 4,
                    "tp_exact": 2, "tp_at_0_2m": 3})
    assert result["precision_exact"] == 0.2
    assert result["recall_exact"] == 0.5
    assert result["precision_at_0_2m"] == 0.3
    assert result["recall_at_0_2m"] == 0.75
    assert result["f0_5_exact"] == 2.5 / 11


def test_false_positive_ray_proxy_partitions_additions():
    clean = np.array([[5.19, 0, 0], [0, 5, 0]], dtype=np.float32)
    faulty = clean[:1]
    additions = np.array([
        [5.15, 0, 0],  # Same clean occupied voxel.
        [5.21, 0, 0],  # Exact voxel mismatch, near the clean return.
        [3, 0, 0],     # Confirmed free before the clean return.
        [7, 0, 0],     # Behind a retained faulty return.
        [0, 3, 0],     # Clean-known free on a missing faulty ray.
        [0, 7, 0],     # Clean ray matched but no faulty return.
        [2, 2, 1],     # No measured return ray can be matched.
    ], dtype=np.float32)
    labels, no_retained = classify_addition_false_positives(additions, clean, faulty)
    assert labels.tolist() == [
        "true_positive", "near_clean_0_2m", "known_free_before_clean_return",
        "behind_retained_faulty_return", "known_free_before_clean_return",
        "no_retained_return_clean_ray", "no_retained_return_unmatched_ray",
    ]
    assert no_retained.tolist() == [False, False, False, False, True, True, True]


def test_ray_proxy_does_not_treat_other_beam_as_occluded():
    clean = np.array([[5, 0, 0]], dtype=np.float32)
    additions = np.array([[7, 0.2, 0]], dtype=np.float32)
    labels, no_retained = classify_addition_false_positives(additions, clean, clean)
    assert labels.tolist() == ["no_retained_return_unmatched_ray"]
    assert no_retained.tolist() == [True]


def test_return_beyond_roi_still_establishes_known_free_space():
    clean = np.array([[40, 0, 0]], dtype=np.float32)
    additions = np.array([[30, 0, 0]], dtype=np.float32)
    labels, no_retained = classify_addition_false_positives(
        additions, clean, np.empty((0, 3), dtype=np.float32))
    assert labels.tolist() == ["known_free_before_clean_return"]
    assert no_retained.tolist() == [True]


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
