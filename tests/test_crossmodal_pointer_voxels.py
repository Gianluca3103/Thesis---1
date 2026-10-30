"""Pointer overlap, proposal-only labels, and decoder-location gradients."""

from types import SimpleNamespace

import torch
import pytest

from models.radar_lidar_crossmodal_v1.pointer_voxels import merge_surface_pointers
from models.radar_lidar_crossmodal_v1.surface_student import (
    SurfaceProposals, SurfaceStudentConfig, SurfaceStudentOutput,
    pointer_xyz_from_offsets,
)
from models.radar_lidar_crossmodal_v1.surface_student_loss import (
    SurfaceMetricCounts, surface_metric_counts, surface_student_objective,
)


def _proposals(xyz, logits, batches):
    xyz = torch.tensor(xyz, dtype=torch.float32)
    logits = torch.tensor(logits, dtype=torch.float32)
    batch = torch.tensor(batches, dtype=torch.long)
    return SurfaceProposals(xyz, logits, xyz[:, 0].detach(), batch,
                            torch.ones((len(batch), 32)))


def test_same_radar_source_counts_once_but_two_sources_raise_confidence():
    center = [.1, .1, .125]
    one = _proposals([[center, center]], [[0., 0.]], [0])
    one_field = merge_surface_pointers(one)
    assert len(one_field.coords) == 1
    assert torch.allclose(one_field.confidence, torch.tensor([.5]))
    assert one_field.source_count.tolist() == [1]

    two = _proposals([[center, center], [center, center]],
                     [[0., 0.], [0., 0.]], [0, 0])
    two_field = merge_surface_pointers(two)
    assert len(two_field.coords) == 1
    assert torch.allclose(two_field.confidence, torch.tensor([.75]))
    assert two_field.source_count.tolist() == [2]


def test_fine_association_has_eight_slots_within_euclidean_three_metres():
    config = SurfaceStudentConfig(proposals_per_anchor=8, anchor_stride=1,
                                  offset_limit_m=3.0, spherical_offsets=True,
                                  decoder_from_conditioned=True)
    assert config.proposals_per_anchor == 8
    centers = torch.tensor([[4.1, 0.1, 0.125], [8.1, 1.1, 0.125]])
    raw = torch.full((2, 8, 3), 100.0)
    xyz = pointer_xyz_from_offsets(centers, raw, config.offset_limit_m,
                                   config.spherical_offsets)
    assert xyz.shape == (2, 8, 3)
    assert torch.all(torch.linalg.vector_norm(xyz - centers[:, None], dim=-1)
                     <= 3.0 + 1e-6)


def test_fine_association_exports_all_candidates_and_distinct_radar_votes():
    center = [.1, .1, .125]
    proposals = _proposals([[center], [center]], [[0.], [0.]], [0, 0])
    conditioned = SimpleNamespace(features=torch.ones((1, 64)),
                                  coords=torch.zeros((1, 4), dtype=torch.long))
    exported = SurfaceStudentOutput(conditioned, proposals,
                                    full_confidence_field=True).exported()
    assert exported["interface_version"] == "crossmodal_fine_association_v1"
    assert len(exported["confidence_coords"]) == 1
    assert torch.allclose(exported["confidence_probabilities"],
                          torch.tensor([[.75]]))
    assert exported["confidence_radar_support_count"].tolist() == [2]


def test_pointer_voxels_never_merge_across_batch_boundaries():
    center = [.1, .1, .125]
    field = merge_surface_pointers(_proposals([[center], [center]], [[0.], [0.]],
                                              [0, 1]))
    assert len(field.coords) == 2
    assert field.coords[:, 0].tolist() == [0, 1]
    assert torch.allclose(field.confidence, torch.tensor([.5, .5]))


def test_twenty_four_slots_do_not_force_twenty_four_selected_voxels():
    locations = [[.1 + .2 * index, .1, .125] for index in range(24)]
    logits = [[5.] + [-30.] * 23]
    field = merge_surface_pointers(_proposals([locations], logits, [0]))
    assert len(field.coords) == 24
    assert int((field.confidence >= .25).sum()) == 1
    exported = SurfaceStudentOutput(
        SimpleNamespace(features=torch.empty((0, 64)), coords=torch.empty((0, 4), dtype=torch.long)),
        _proposals([locations], logits, [0])).exported()
    assert len(exported["confidence_coords"]) == 1


def test_unproposed_clean_occupancy_does_not_change_decoder_loss():
    p = _proposals([[[5.1, .1, .125]]], [[0.]], [0])
    output = SurfaceStudentOutput(None, p)
    config = SurfaceStudentConfig(proposals_per_anchor=1)
    valid_one = torch.tensor([[True]])
    single = torch.tensor([[[5.1, .1, .125]]])
    loss_one, pieces_one, _, _ = surface_student_objective(
        output, single, valid_one, config)
    extra = torch.tensor([[[5.1, .1, .125], [20.1, 20.1, .125]]])
    loss_extra, pieces_extra, _, _ = surface_student_objective(
        output, extra, torch.ones((1, 2), dtype=torch.bool), config)
    assert pieces_one["known_sites"] == pieces_extra["known_sites"] == 1
    assert torch.allclose(loss_one, loss_extra)


def test_verified_free_is_negative_and_unknown_is_ignored():
    p = _proposals([[[5.1, .1, .125], [.1, 10.1, .125]]], [[0., 0.]], [0])
    clean = torch.tensor([[[10.1, .1, .125]]])
    _, pieces, field, targets = surface_student_objective(
        SurfaceStudentOutput(None, p), clean,
        torch.ones((1, 1), dtype=torch.bool),
        SurfaceStudentConfig(proposals_per_anchor=2))
    assert pieces["occupied_sites"] == 0
    assert pieces["known_free_sites"] == 1
    assert pieces["known_sites"] == 1
    counts = surface_metric_counts(field, targets,
                                   SurfaceStudentConfig(proposals_per_anchor=2))
    assert counts.false_positive_known_free == 1
    assert counts.unknown_selected == 1


def test_training_only_soft_target_moves_pointer_toward_nearby_clean_voxel():
    xyz = torch.tensor([[[5.12, .1, .125]]], requires_grad=True)
    logits = torch.tensor([[0.]], requires_grad=True)
    p = SurfaceProposals(xyz, logits, xyz[:, 0].detach(), torch.tensor([0]),
                         torch.ones((1, 32)))
    clean = torch.tensor([[[5.31, .1, .125]]])
    loss, pieces, _, _ = surface_student_objective(
        SurfaceStudentOutput(None, p), clean,
        torch.ones((1, 1), dtype=torch.bool),
        SurfaceStudentConfig(proposals_per_anchor=1, hit_scale=1.))
    assert pieces["known_free_sites"] == 1
    loss.backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
    assert xyz.grad[0, 0, 0] < 0  # descent moves the pointer toward +x
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_unique_hits_times_verified_precision_selects_breadth_and_accuracy():
    small = SurfaceMetricCounts(correct_unique=80, false_positive_known_free=20)
    large = SurfaceMetricCounts(correct_unique=800, false_positive_known_free=200)
    assert small.report()["verified_proposal_precision"] == .8
    assert large.report()["verified_proposal_precision"] == .8
    assert large.report()["proposal_utility"] == 10 * small.report()["proposal_utility"]
    combined = SurfaceMetricCounts()
    combined.add(SurfaceMetricCounts(correct_unique=2, false_positive_known_free=1))
    combined.add(SurfaceMetricCounts(correct_unique=1, false_positive_known_free=1))
    assert combined.report()["verified_proposal_precision"] == 3 / 5
    assert combined.report()["proposal_utility"] == pytest.approx(9 / 5)
