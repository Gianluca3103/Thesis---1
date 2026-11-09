"""Occupancy-only IoU on Stage-II eligible cells."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage2 import eligible_occupancy_metrics as metrics
from models.radar_lidar_stage2.eligible_occupancy_metrics import (
    occupancy_counts, summarize_counts)
from models.radar_lidar_stage2.sparse_geometry import GRID


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

    def test_disconnected_ambiguous_groups_keep_maximum_matching(self):
        target = torch.tensor([[0, 8, 80, x] for x in (10, 12, 20, 22)])
        predicted = torch.tensor([[0, 8, 80, x] for x in (10, 11, 20, 21)])
        counts = occupancy_counts(predicted, target)
        self.assertEqual(counts["exact_tp"], 2)
        self.assertEqual(counts["tolerant_tp_0p2m"], 4)

    def test_exact_subset_skips_graph_matching(self):
        cells = torch.tensor([[0, 8, 80 + y, 10 + x]
                              for y in range(50) for x in range(50)])
        target = torch.cat((cells, torch.tensor([[0, 9, 80, 10]])))
        original_tree = metrics.cKDTree
        try:
            metrics.cKDTree = lambda *_: self.fail("KD-tree should be skipped")
            counts = occupancy_counts(cells, target)
        finally:
            metrics.cKDTree = original_tree
        self.assertEqual(counts["tolerant_tp_0p2m"], len(cells))

    def test_component_matching_equals_global_reference_on_small_random_sets(self):
        random = np.random.default_rng(3103)
        grid = [(8, y, x) for y in range(78, 83) for x in range(8, 17)]
        for _ in range(30):
            predicted = torch.tensor(
                [[0, *grid[index]] for index in random.choice(
                    len(grid), 14, replace=False)])
            target = torch.tensor(
                [[0, *grid[index]] for index in random.choice(
                    len(grid), 15, replace=False)])
            left = metrics._centers(predicted, GRID)
            right = metrics._centers(target, GRID)
            neighbors = cKDTree(right).query_ball_point(left, 0.2 + 1e-9)
            row = [i for i, indices in enumerate(neighbors) for _ in indices]
            column = [j for indices in neighbors for j in indices]
            graph = csr_matrix((np.ones(len(row), dtype=np.int8),
                                (row, column)), shape=(len(left), len(right)))
            expected = np.count_nonzero(
                maximum_bipartite_matching(graph, perm_type="row") >= 0)
            self.assertEqual(
                occupancy_counts(predicted, target)["tolerant_tp_0p2m"],
                expected)

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
