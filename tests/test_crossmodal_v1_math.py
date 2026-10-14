"""CPU checks for the versioned teacher and raw radar-only output semantics."""

import torch

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.teacher import BottleneckTeacherCore, TeacherConfig
from models.radar_lidar_crossmodal_v1.teacher_loss import teacher_objective
from models.radar_lidar_crossmodal_v1.student import StudentConfig, StudentOutput
from models.radar_lidar_crossmodal_v1.student_loss import student_objective, radar_supported_iou


def sites(coords, features, stride):
    return SparseSites(torch.tensor(coords, dtype=torch.long), features,
                       GRID.scale_shape(stride), stride)


def test_teacher_attention_mask_coordinates_and_gradients():
    torch.manual_seed(3)
    config = TeacherConfig()
    radar_features = torch.randn(2, 64)
    lidar_features = torch.randn(2, 64)
    radar = sites([[0, 0, 1, 1], [1, 0, 1, 1]], radar_features.clone(), 8)
    lidar = sites([[0, 0, 1, 2], [1, 0, 1, 2]], lidar_features.clone(), 8)
    indices = torch.full((2, 24), -1, dtype=torch.long)
    indices[0, 0] = 0
    distances = torch.full((2, 24), float("inf"))
    distances[0, 0] = 0.8
    core = BottleneckTeacherCore(config)
    output = core(radar, lidar, Neighborhood(indices, distances))
    assert output.attention.shape == (2, 4, 24)
    torch.testing.assert_close(output.attention[0, :, 0], torch.ones(4))
    assert output.attention[0, :, 1:].sum() == 0
    assert output.attention[1].sum() == 0
    assert output.lidar_contribution[1].abs().sum() == 0
    assert torch.equal(output.fused.coords, radar.coords)
    torch.testing.assert_close(radar.features, radar_features)
    torch.testing.assert_close(lidar.features, lidar_features)
    total, terms, metrics = teacher_objective(output, torch.tensor([1, -1]),
                                               torch.tensor([1, 1]), config)
    assert torch.isfinite(total)
    assert terms["object"].denominator == 1
    assert metrics["object_r1"].mean == 1
    total.backward()
    assert core.gate.weight.grad is not None
    assert core.value.weight.grad is not None
    assert core.output.weight.grad is not None
    assert core.radar_adapter.second.weight.grad is not None


def test_teacher_empty_neighbors_finite_and_no_cross_batch_candidates():
    config = TeacherConfig()
    radar = sites([[0, 0, 1, 1]], torch.randn(1, 64), 8)
    lidar = sites([[1, 0, 1, 1]], torch.randn(1, 64), 8)
    from models.radar_lidar_stage1.correspondence import local_neighbors
    neighbors = local_neighbors(radar, lidar, GRID, 6, 24)
    assert not neighbors.valid.any()
    output = BottleneckTeacherCore(config)(radar, lidar, neighbors)
    total, terms, metrics = teacher_objective(output, torch.tensor([-1]),
                                               torch.tensor([-1]), config)
    assert torch.isfinite(total)
    assert terms["object"].denominator == 0
    assert metrics["geometry_eligible"] == 0
    assert output.attention.sum() == 0


class Targets:
    occupied = torch.tensor([True, False, False])
    known_free = torch.tensor([False, True, False])
    clean_points_in_candidates = 1
    clean_points_in_grid = 2


def test_student_losses_and_raw_iou_scope():
    feature = sites([[0, 0, 1, 1]], torch.randn(1, 64, requires_grad=True), 8)
    teacher = sites([[0, 0, 1, 1]], torch.ones(1, 64), 8)
    logits = sites([[0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]],
                   torch.tensor([[2.], [-2.], [2.]], requires_grad=True), 1)
    output = StudentOutput(feature, logits)
    total, pieces = student_objective(output, teacher, Targets(), StudentConfig())
    assert torch.isfinite(total)
    assert pieces["known_sites"] == 2
    total.backward()
    assert feature.features.grad is not None
    assert logits.features.grad is not None
    report = radar_supported_iou(output, Targets(), 0.5)
    assert report["conventional_clean_scan"]["iou"] == 0.5
    assert report["known_space_masked"]["iou"] == 1.0
    assert report["clean_point_coverage"] == 0.5


def test_student_feature_alignment_is_checked():
    feature = sites([[0, 0, 1, 1]], torch.zeros(1, 64), 8)
    teacher = sites([[0, 0, 1, 2]], torch.ones(1, 64), 8)
    logits = sites([[0, 0, 1, 1]], torch.zeros(1, 1), 1)
    try:
        student_objective(StudentOutput(feature, logits), teacher, Targets(), StudentConfig())
    except ValueError as error:
        assert "coordinates" in str(error)
    else:
        raise AssertionError("Mismatched teacher/student coordinates were accepted")


def test_oriented_box_membership_and_ambiguous_voxel_exclusion(tmp_path):
    from models.radar_lidar_crossmodal_v1.supervision import point_object_ids, voxel_object_ids
    label = tmp_path / "00001.txt"
    # KITTI: class, truncation, occlusion, alpha, 2D bbox, h,w,l, location, yaw.
    label.write_text("Car 0 0 0 0 0 0 0 2 0.8 4 0 1 0 0\n", encoding="utf-8")
    ids, names = point_object_ids(torch.tensor([[0., 0., 0.], [4., 0., 0.]]).numpy(),
                                  label, torch.eye(4).numpy())
    assert ids.tolist() == [0, -1]
    assert names[0] == "Car"
    sparse = sites([[0, 1, 32, 32]], torch.zeros(1, 64), 8)
    # Two near-center returns share the same original coarse cell, but one is
    # unannotated; a 0.8 dominance rule rejects the mixed identity.
    points = torch.tensor([[[0., 0., 0., 1.], [0.6, 0., 0., 1.]]])
    result, _ = voxel_object_ids(sparse, points, torch.tensor([[True, True]]),
                                 [label], [torch.eye(4).numpy()])
    assert result.item() == -1
