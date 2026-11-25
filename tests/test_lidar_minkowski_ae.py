"""Contract checks for the clean VoD voxel autoencoder trial."""

import numpy as np
import torch

from models.lidar_minkowski_ae import Reconstruction, _targets_at
from models.lidar_minkowski_ae_data import (
    occupancy_counts, summarize_counts, voxelize_clean_lidar,
)
from scripts.train_lidar_minkowski_ae import _counts, _run
from scripts.audit_lidar_minkowski_ae import remove_coarse_blocks


def test_voxelization_matches_stage2_grid_and_fixed_forward_crop():
    points = np.asarray([
        [1.01, 0.01, 0.01, 1], [1.05, 0.02, 0.02, 2],
        [1.21, 0.01, 0.01, 3], [-1.0, 0.0, 0.0, 4],
        [1.0, 3.0, 0.0, 5], [33.0, 0.0, 0.0, 6],
    ], dtype=np.float32)
    cells = voxelize_clean_lidar(points, np.eye(4))
    np.testing.assert_array_equal(cells, [[5, 80, 8], [6, 80, 8]])


def test_exact_iou_uses_occupied_cells_not_empty_background():
    predicted = np.asarray([[5, 80, 8], [6, 80, 8]], np.int32)
    target = np.asarray([[5, 80, 8], [7, 80, 8]], np.int32)
    row = summarize_counts(occupancy_counts(predicted, target))
    assert row["tp"] == 1
    assert row["exact_iou"] == 1 / 3
    assert row["exact_f1"] == 1 / 2


def test_decoder_supervision_coarsens_targets_without_inserting_fine_sites():
    fine = torch.tensor([[0, 5, 80, 8], [0, 6, 80, 8]], dtype=torch.int32)
    candidates = torch.tensor([[0, 4, 80, 8], [0, 6, 80, 8],
                               [0, 8, 80, 8]], dtype=torch.int32)
    assert _targets_at(candidates, fine, 2).tolist() == [True, True, False]


def test_batched_iou_does_not_match_cells_across_different_frames():
    target = torch.tensor([[0, 5, 80, 8], [1, 6, 80, 8]], dtype=torch.int32)
    predicted = torch.tensor([[0, 6, 80, 8], [1, 5, 80, 8]], dtype=torch.int32)
    assert _counts(predicted, target) == {"tp": 0, "predicted": 2, "target": 2}


def test_block_ablation_removes_whole_latent_coordinate_blocks():
    cells = np.asarray([[0, 0, 0], [1, 1, 1], [2, 0, 0],
                        [3, 1, 1], [4, 0, 0]], dtype=np.int32)
    surviving, removed, blocks = remove_coarse_blocks(
        cells, fraction=1 / 3, seed=3103)
    assert blocks == 1
    assert len(surviving) + len(removed) == len(cells)
    assert not {tuple(x) for x in surviving // 2} & {tuple(x) for x in removed // 2}


def test_validation_decodes_before_targets_are_consulted():
    class ProbeModel:
        def train(self, training):
            assert not training

        def encode(self, fine):
            return fine

        def decode(self, latent, *, threshold):
            return Reconstruction(latent, torch.full((len(latent),), 9.0),
                                  None, len(latent), None)

        def __call__(self, *args, **kwargs):
            raise AssertionError("Validation must not call target-bearing forward")

    fine = torch.tensor([[0, 5, 80, 8]], dtype=torch.int32)
    result = _run(ProbeModel(), [{"frame_ids": ["demo"], "coordinates": fine}],
                  torch.device("cpu"), optimizer=None, threshold=0.5,
                  label="target-free-check")
    assert result["exact_iou"] == 1.0
    assert result["candidate_recall_ceiling"] == 1.0
