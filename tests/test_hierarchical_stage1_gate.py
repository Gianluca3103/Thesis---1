"""Contracts that matter for a learned sparse Stage-I gate."""

import torch
from types import SimpleNamespace
from scipy.spatial import cKDTree

from models.radar_lidar_crossmodal_v1.hierarchical_gate import (
    HierarchicalGateConfig, HierarchicalStage1Gate, fine_ranking, missing_target,
    target_mask)
from models.radar_lidar_stage2.sparse_geometry import select_allowed
from scripts.train_hierarchical_stage1 import _validation_indices
from scripts.compare_hierarchical_stage1 import (
    _accumulate, _best_threshold, _new_totals, _radar_eligible,
    _stage2_centers, _threshold_counts, _threshold_rows)


def _radar(xyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((xyz, torch.zeros((len(xyz), 4))), dim=1)


def test_training_retains_all_positive_descendants_but_inference_has_no_target_input():
    torch.manual_seed(13)
    config = HierarchicalGateConfig(channels=16, coarse_parent_budget=1,
                                    middle_parent_budget=1)
    model = HierarchicalStage1Gate(config).eval()
    radar = _radar(torch.tensor([[1.0, -4.0, 0.0], [24.0, 8.0, 0.0]]))
    clean = torch.tensor([[1.0, -4.0, 0.0], [8.0, 0.0, 0.0],
                          [24.0, 8.0, 0.0]])
    faulty = clean[1:2]
    truth = missing_target(clean, faulty)
    with torch.no_grad():
        trained = model(radar, faulty, target_fine=truth, teacher_forcing=True)
        inferred = model(radar, faulty)
        ignored_target = model(radar, faulty, target_fine=truth,
                               teacher_forcing=False)
    assert int(target_mask(trained["fine_coords"], truth, 1).sum()) == len(truth)
    assert torch.equal(inferred["fine_coords"], ignored_target["fine_coords"])
    assert torch.equal(inferred["fine_logits"], ignored_target["fine_logits"])


def test_exported_coordinates_align_with_stage2_gate():
    torch.manual_seed(19)
    model = HierarchicalStage1Gate(HierarchicalGateConfig(
        channels=64, coarse_parent_budget=2, middle_parent_budget=2)).eval()
    radar = _radar(torch.tensor([[5.0, -3.0, 0.0]]))
    with torch.no_grad():
        output = model(radar, torch.empty((0, 3)))
        exported = model.exported(output, max_sites=12)
    allowed, scores = select_allowed(exported["confidence_coords"],
                                    exported["confidence_probabilities"], 1e-7)
    assert len(allowed) == len(scores) == 12
    assert set(map(tuple, allowed[:, 1:].tolist())) == set(map(
        tuple, (exported["confidence_coords"][:, 1:] - torch.tensor(
            [0, 176, 256])).tolist()))
    assert exported["conditioning_features"].shape[1] == 64


def test_final_ranking_respects_middle_parent_evidence():
    torch.manual_seed(23)
    model = HierarchicalStage1Gate(HierarchicalGateConfig(
        channels=16, coarse_parent_budget=2, middle_parent_budget=2)).eval()
    radar = _radar(torch.tensor([[5.0, -3.0, 0.0]]))
    with torch.no_grad():
        output = model(radar, torch.empty((0, 3)))
        original, probability = fine_ranking(output)
        altered = dict(output)
        altered["middle_logits"] = output["middle_logits"].clone()
        altered["middle_logits"][output["selected_middle_indices"][0]] += 10
        changed, _ = fine_ranking(altered)
    assert len(original) == len(probability) == len(output["fine_coords"])
    assert torch.all((probability > 0) & (probability < 1))
    assert torch.all(changed[:8] > original[:8])
    assert torch.equal(changed[8:], original[8:])


def test_block_validation_selection_stays_contiguous_within_each_sequence():
    frames = [SimpleNamespace(frame_id=f"{index:05d}")
              for index in (*range(10), *range(100, 110))]
    selected = _validation_indices(frames, 8, 17, "blocks")
    ids = [int(frames[index].frame_id) for index in selected]
    assert len(ids) == 8
    assert len(ids[:4]) == len(ids[4:]) == 4
    assert ids[:4] == list(range(ids[0], ids[0] + 4))
    assert ids[4:] == list(range(ids[4], ids[4] + 4))


def test_one_meter_radar_scope_filters_targets_and_candidates_before_budget():
    near = torch.tensor([[8, 80, 50], [8, 80, 55], [8, 80, 56]])
    tree = cKDTree(_stage2_centers(near[:1]).numpy())
    eligible = _radar_eligible(near, tree, 1.0)
    assert eligible.tolist() == [True, True, False]
    assert not _radar_eligible(near, None, 1.0).any()
    assert len(_radar_eligible(near[:0], tree, 1.0)) == 0

    # The highest raw score is outside the scope, so top-1 must come from
    # the in-scope pool and use only the in-scope target as denominator.
    scores = torch.tensor([0.2, 0.8, 0.99])
    totals = _new_totals((1,))
    _accumulate(totals, near[eligible], scores[eligible], near[1:2], (1,))
    assert totals["1"] == {"tp": 1, "selected": 1, "target": 1}
    assert totals["pool"] == {"tp": 1, "selected": 2, "target": 1}


def test_threshold_sweep_counts_exact_matches_and_selects_on_calibration():
    coords = torch.tensor([[8, 80, 50], [8, 80, 51], [8, 80, 52]])
    probabilities = torch.tensor([0.2, 0.5, 0.8])
    thresholds = (0.0, 0.5, 0.8, 1.0)
    counts = _threshold_counts(coords, probabilities, coords[[0, 2]],
                               torch.tensor(thresholds))
    assert counts.tolist() == [[2, 3, 2], [1, 2, 2], [1, 1, 2], [0, 0, 2]]
    rows = _threshold_rows(counts, thresholds)
    assert _best_threshold(rows, "f1")["threshold"] == 0.0
    assert _best_threshold(rows, "iou")["threshold"] == 0.0
    assert _best_threshold(rows, "recall")["threshold"] == 0.0
    assert _best_threshold(rows, "recall", 0.75)["threshold"] == 0.8
    assert _best_threshold(rows, "recall", 1.01) is None
    assert _threshold_counts(coords[:0], probabilities[:0], coords[:0],
                             torch.tensor(thresholds)).shape == (4, 3)
