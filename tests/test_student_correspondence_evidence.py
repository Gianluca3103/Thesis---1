"""Training-only teacher evidence and diagnostic IoU on fixed radar candidates."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.student import RadarOnlyStudent, StudentConfig, StudentOutput
from models.radar_lidar_crossmodal_v1.student_loss import (
    radar_supported_iou, student_objective, teacher_correspondence_evidence,
)
from models.radar_lidar_crossmodal_v1.train_student import Summary


def sites(coordinates, features, stride):
    return SparseSites(torch.tensor(coordinates, dtype=torch.long).reshape(-1, 4), features,
                       GRID.scale_shape(stride), stride)


def labels(occupied, free, represented=1, in_grid=2):
    return SimpleNamespace(occupied=torch.tensor(occupied, dtype=torch.bool),
                           known_free=torch.tensor(free, dtype=torch.bool),
                           clean_points_in_candidates=represented,
                           clean_points_in_grid=in_grid)


def output(logits):
    conditioning = sites([[0, 0, 0, 0]], torch.zeros(1, 64), 8)
    coords = [[0, 0, 0, i] for i in range(len(logits))]
    confidence = sites(coords, torch.tensor(logits, dtype=torch.float32)[:, None], 1)
    return StudentOutput(conditioning, confidence)


class StudentCorrespondenceEvidenceTests(unittest.TestCase):
    def test_attention_maps_by_parent_and_max_query_without_crossing_batches(self):
        radar = sites([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                      torch.zeros(3, 64), 8)
        lidar = sites([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                      torch.zeros(3, 64), 8)
        indices = torch.tensor([[0, 1, 1], [0, 1, -1], [2, -1, -1]])
        attention = torch.tensor([
            [[.2, .3, .5], [.4, .2, .4], [.6, .1, .3], [0., .4, .6]],
            [[.9, .1, 0.]] * 4,
            [[1., 0., 0.]] * 4,
        ])
        teacher = SimpleNamespace(radar_frozen=radar, lidar_frozen=lidar,
                                  neighbors=Neighborhood(indices, torch.zeros_like(indices).float()),
                                  attention=attention)
        candidates = sites([[0, 0, 0, 0], [0, 0, 0, 8], [0, 0, 0, 16],
                            [1, 0, 0, 0]], torch.zeros(4, 1), 1)
        evidence = teacher_correspondence_evidence(teacher, candidates)
        torch.testing.assert_close(evidence, torch.tensor([.9, .7, 0., 1.]))

        teacher.neighbors = Neighborhood(torch.tensor([[2, -1, -1],
                                                       [-1, -1, -1],
                                                       [-1, -1, -1]]),
                                         torch.zeros_like(indices).float())
        with self.assertRaisesRegex(ValueError, "sample boundaries"):
            teacher_correspondence_evidence(teacher, candidates)

    def test_empty_neighborhood_has_zero_evidence(self):
        radar = sites([[0, 0, 0, 0]], torch.zeros(1, 64), 8)
        lidar = sites([], torch.empty(0, 64), 8)
        teacher = SimpleNamespace(radar_frozen=radar, lidar_frozen=lidar,
                                  neighbors=Neighborhood(torch.full((1, 2), -1),
                                                         torch.full((1, 2), float("inf"))),
                                  attention=torch.zeros(1, 4, 2))
        candidates = sites([[0, 0, 0, 0]], torch.zeros(1, 1), 1)
        torch.testing.assert_close(teacher_correspondence_evidence(teacher, candidates),
                                   torch.zeros(1))

    def test_zero_lambda_reproduces_old_bce_and_positive_weight(self):
        prediction = output([0., 2., -1.])
        truth = labels([True, True, False], [False, False, True])
        teacher_sites = prediction.conditioning.replace_features(torch.ones(1, 64))
        config = StudentConfig(lambda_teacher_evidence=0.)
        _, old = student_objective(prediction, teacher_sites, truth, config)
        _, zero = student_objective(prediction, teacher_sites, truth, config,
                                    torch.tensor([1., .2, 1.]))
        expected = F.binary_cross_entropy_with_logits(
            prediction.confidence_logits.features[:, 0], truth.occupied.float(),
            pos_weight=torch.tensor(config.positive_weight))
        torch.testing.assert_close(old["bce"], expected)
        torch.testing.assert_close(zero["bce"], expected)
        torch.testing.assert_close(zero["spatial"], old["spatial"])

    def test_evidence_weights_only_positive_occupancy(self):
        prediction = output([0., 2., -1.])
        truth = labels([True, True, False], [False, False, True])
        teacher_sites = prediction.conditioning.replace_features(torch.zeros(1, 64))
        config = StudentConfig(lambda_teacher_evidence=.5)
        evidence = torch.tensor([1., 0., 1.])
        _, pieces = student_objective(prediction, teacher_sites, truth, config, evidence)
        raw = F.binary_cross_entropy_with_logits(
            prediction.confidence_logits.features[:, 0], truth.occupied.float(),
            pos_weight=torch.tensor(config.positive_weight), reduction="none")
        expected = (1.5 * raw[0] + raw[1] + raw[2]) / 3.5
        torch.testing.assert_close(pieces["bce"], expected)
        torch.testing.assert_close(pieces["bce_weight_sum"], torch.tensor(3.5))

    def test_teacher_iou_is_diagnostic_and_primary_iou_is_unchanged(self):
        prediction = output([2., 2., -2., 2., 2.])
        truth = labels([True, False, True, False, False],
                       [False, True, False, True, False])
        old = radar_supported_iou(prediction, truth, .5)
        new = radar_supported_iou(prediction, truth, .5,
                                  torch.tensor([.3, .4, .1, 0., 1.]), .2)
        self.assertEqual(new["known_space_masked"], old["known_space_masked"])
        self.assertEqual(new["conventional_clean_scan"], old["conventional_clean_scan"])
        self.assertEqual(new["clean_point_coverage"], old["clean_point_coverage"])
        self.assertEqual(new["known_space_masked"]["iou"], .25)
        self.assertEqual(new["conventional_clean_scan"]["iou"], .2)
        self.assertEqual(new["teacher_correspondence_supported"]["iou"], .5)
        empty = radar_supported_iou(prediction, truth, .5, torch.zeros(5), .2)
        self.assertIsNone(empty["teacher_correspondence_supported"]["iou"])

    def test_teacher_iou_uses_global_intersection_and_union(self):
        config = StudentConfig(thresholds=(.5,))
        summary = Summary(config)
        for prediction, truth, evidence in (
            (output([2., 2.]), labels([True, False], [False, True]), torch.ones(2)),
            (output([2., 2., 2.]), labels([False] * 3, [True] * 3), torch.ones(3)),
        ):
            pieces = {"feature_elements": 0, "known_sites": len(evidence),
                      "distillation": torch.tensor(0.), "bce": torch.tensor(0.),
                      "bce_weight_sum": torch.tensor(float(len(evidence))),
                      "dice_intersection": torch.tensor(0.),
                      "dice_probability_sum": torch.tensor(0.),
                      "dice_target_sum": torch.tensor(0.),
                      "occupied_sites": int(truth.occupied.sum()),
                      "free_sites": int(truth.known_free.sum()), "unknown_sites": 0}
            summary.add(pieces, prediction, truth, 1, teacher_evidence=evidence,
                        evaluate=True)
        report = summary.finish(select_threshold=True)
        self.assertEqual(report["selected_threshold"], .5)
        self.assertEqual(report["radar_supported_iou_known"], .2)
        self.assertEqual(report["teacher_correspondence_supported_iou"], .2)
        self.assertEqual(report["threshold_sweep"]["0.5"]["teacher"],
                         {"intersection": 1, "union": 5, "iou": .2})

    def test_teacher_iou_does_not_select_prediction_threshold(self):
        config = StudentConfig(thresholds=(.3, .5))
        summary = Summary(config)
        probabilities = torch.tensor([.4, .7, .6, .4, .4, .4, .4])
        prediction = output(torch.logit(probabilities).tolist())
        truth = labels([True, True, False, False, False, False, False],
                       [False, False, True, True, True, True, True])
        evidence = torch.tensor([1., 0., 1., 0., 0., 0., 0.])
        pieces = {"feature_elements": 0, "known_sites": 7,
                  "distillation": torch.tensor(0.), "bce": torch.tensor(0.),
                  "bce_weight_sum": torch.tensor(7.),
                  "dice_intersection": torch.tensor(0.),
                  "dice_probability_sum": torch.tensor(0.),
                  "dice_target_sum": torch.tensor(0.),
                  "occupied_sites": 2, "free_sites": 5, "unknown_sites": 0}
        summary.add(pieces, prediction, truth, 1, teacher_evidence=evidence,
                    evaluate=True)
        report = summary.finish(select_threshold=True)
        self.assertEqual(report["selected_threshold"], .5)
        self.assertEqual(report["radar_supported_iou_known"], 1 / 3)
        self.assertEqual(report["teacher_correspondence_supported_iou"], 0.)
        self.assertEqual(report["threshold_sweep"]["0.3"]["teacher"]["iou"], .5)

    def test_student_forward_still_uses_radar_only(self):
        class FakeRadar(nn.Module):
            def forward(self, points, valid):
                self.seen = (points, valid)
                return sites([[0, 0, 0, 0]], torch.ones(1, 64), 8)

        class FakeDecoder(nn.Module):
            def forward(self, bottleneck):
                return sites([[0, 0, 0, 0]], bottleneck.features[:, :1], 1)

        student = RadarOnlyStudent.__new__(RadarOnlyStudent)
        nn.Module.__init__(student)
        student.radar_encoder = FakeRadar()
        student.shared = nn.Identity()
        student.feature_head = nn.Identity()
        student.feature_norm = nn.Identity()
        student.confidence_decoder = FakeDecoder()
        radar = torch.zeros(1, 1, 7)
        valid = torch.ones(1, 1, dtype=torch.bool)
        result = student(radar, valid)
        self.assertIs(student.radar_encoder.seen[0], radar)
        self.assertEqual(result.confidence_logits.features.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
