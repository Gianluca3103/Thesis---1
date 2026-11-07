"""Occupancy-only IoU on Stage-II eligible cells."""

import unittest

import torch

from models.radar_lidar_stage2.eligible_occupancy_metrics import (
    occupancy_counts, summarize_counts)


class EligibleOccupancyMetricsTest(unittest.TestCase):
    def test_exact_and_20cm_match_use_voxel_centers(self):
        target = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 13]])
        predicted = torch.tensor([[0, 8, 80, 11], [0, 8, 80, 14]])
        report = summarize_counts(occupancy_counts(predicted, target))
        self.assertEqual(report["exact_tp"], 0)
        self.assertEqual(report["tolerant_tp_0p2m"], 2)
        self.assertEqual(report["eligible_iou"], 0.0)
        self.assertEqual(report["eligible_iou_0p2m"], 1.0)

    def test_tolerance_is_one_to_one(self):
        target = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 12]])
        predicted = torch.tensor([[0, 8, 80, 11]])
        report = summarize_counts(occupancy_counts(predicted, target))
        self.assertEqual(report["tolerant_tp_0p2m"], 1)
        self.assertEqual(report["eligible_iou_0p2m"], 0.5)

    def test_ambiguous_neighbors_use_maximum_matching(self):
        target = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 12]])
        predicted = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
        report = summarize_counts(occupancy_counts(predicted, target))
        self.assertEqual(report["exact_tp"], 1)
        self.assertEqual(report["tolerant_tp_0p2m"], 2)

    def test_vertical_neighbor_is_beyond_20cm(self):
        target = torch.tensor([[0, 8, 80, 10]])
        predicted = torch.tensor([[0, 9, 80, 10]])
        report = summarize_counts(occupancy_counts(predicted, target))
        self.assertEqual(report["tolerant_tp_0p2m"], 0)

    def test_empty_union_does_not_create_perfect_iou(self):
        empty = torch.empty((0, 4), dtype=torch.long)
        report = summarize_counts(occupancy_counts(empty, empty))
        self.assertEqual(report["eligible_iou"], 0.0)
        self.assertEqual(report["eligible_iou_0p2m"], 0.0)


if __name__ == "__main__":
    unittest.main()
