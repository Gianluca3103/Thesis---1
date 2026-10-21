"""Map-only radar ablation keeps cached LiDAR supervision and gradient flow."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.meta_occupancy import MetaOccupancyConfig
from models.radar_lidar_crossmodal_v1.student import StudentOutput
from models.radar_lidar_crossmodal_v1.train_radar_autoencoder import (
    _epoch, collate_radar_autoencoder,
)
from models.radar_lidar_stage1.sparse import SparseSites


class _OneVoxelRadarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1, 1))

    def forward(self, radar, radar_valid):
        coords = torch.tensor([[0, 0, 0, 0]], device=radar.device)
        sites = SparseSites(coords, self.logit, GRID.shape_zyx)
        return StudentOutput(sites, sites)


class RadarAutoencoderTests(unittest.TestCase):
    def test_collate_keeps_frame_labels_and_coordinates_separate(self):
        rows = []
        for frame_id, occupied in (("00000", True), ("00001", False)):
            rows.append({"frame_id": frame_id, "radar": torch.zeros((1, 7)),
                         "geometry": {
                             "candidate_coords": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
                             "observed_coords": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
                             if occupied else torch.empty((0, 4), dtype=torch.int32),
                             "occupied": torch.tensor([occupied]),
                             "known_free": torch.tensor([not occupied]),
                             "support": torch.tensor([True]),
                             "clean_supported_points": int(occupied),
                             "represented_clean_points_supported": int(occupied)}})
        batch = collate_radar_autoencoder(rows)
        self.assertEqual(batch["candidate_coords"].tolist(),
                         [[0, 1, 2, 3], [1, 1, 2, 3]])
        self.assertEqual(batch["occupied"].tolist(), [True, False])
        self.assertEqual(batch["known_free"].tolist(), [False, True])

    def test_cached_lidar_occupancy_trains_direct_radar_output(self):
        row = {"frame_id": "00000", "radar": torch.zeros((1, 7)),
               "geometry": {
                   "candidate_coords": torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
                   "observed_coords": torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
                   "occupied": torch.tensor([True]),
                   "known_free": torch.tensor([False]),
                   "support": torch.tensor([True]),
                   "clean_supported_points": 1,
                   "represented_clean_points_supported": 1}}
        batch = collate_radar_autoencoder([row])
        self.assertEqual(batch["candidate_coords"].tolist(), [[0, 0, 0, 0]])
        model = _OneVoxelRadarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = float(model.logit.detach())
        report = _epoch(model, [batch], optimizer, MetaOccupancyConfig(),
                        "cpu", name="CPU radar ablation")
        self.assertGreater(float(model.logit.detach()), before)
        self.assertEqual(report["known"], 1)
        self.assertEqual(report["iou_full_support"], 1.0)


if __name__ == "__main__":
    unittest.main()
