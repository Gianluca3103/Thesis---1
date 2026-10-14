"""CPU contract checks for the active previous Stage-I encoder interface."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models.radar_lidar_stage1.sparse import SparseSites, encode_keys
from models.radar_lidar_stage1.pfe_spconv import PFEPointVoxelEncoder
from models.radar_lidar_crossmodal_v1.encoder import GRID, masked_bottleneck_sites
from models.radar_lidar_crossmodal_v1 import previous_encoder
from models.radar_lidar_crossmodal_v1.pretrain_encoder import GeometryPretrainer
from models.radar_lidar_crossmodal_v1 import student as student_module
from models.radar_lidar_crossmodal_v1 import teacher as teacher_module


class FakePointEncoder(nn.Module):
    def __init__(self, grid, modality):
        super().__init__()
        self.modality = modality
        self.weight = nn.Parameter(torch.ones(32))

    def forward(self, points, valid, **kwargs):
        batch, point = valid.nonzero(as_tuple=True)
        xyz = ((points[batch, point, :3] - points.new_tensor(GRID.minimum_xyz))
               / points.new_tensor(GRID.size_xyz)).floor().long()
        coords = torch.stack((batch, xyz[:, 2], xyz[:, 1], xyz[:, 0]), -1)
        order = torch.argsort(encode_keys(coords, GRID.shape_zyx))
        sites = SparseSites._from_validated(coords[order], self.weight.expand(len(coords), -1),
                                            GRID.shape_zyx)
        return sites, {}


class FakeBackbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels == (16, 32, 64, 128)
        self.project = nn.Linear(32, 128)
        self.last_fine = None

    def forward(self, fine, **kwargs):
        self.last_fine = fine.features
        coarse = fine.coords.clone()
        coarse[:, 1:] //= 8
        coords, inverse = torch.unique(coarse, dim=0, sorted=True, return_inverse=True)
        pooled = fine.features.new_zeros((len(coords), 32))
        pooled.index_add_(0, inverse, fine.features)
        counts = torch.bincount(inverse, minlength=len(coords)).clamp_min(1)
        features = self.project(pooled / counts[:, None])
        return [SparseSites._from_validated(coords, features, GRID.scale_shape(8), 8)], []


class PreviousStage1EncoderTests(unittest.TestCase):
    def setUp(self):
        point = patch.object(previous_encoder, "PFEPointVoxelEncoder", FakePointEncoder)
        backbone = patch.object(previous_encoder, "PFESpconvBackbone", FakeBackbone)
        point.start()
        backbone.start()
        self.addCleanup(point.stop)
        self.addCleanup(backbone.stop)

    def sample(self, modality="radar"):
        columns = 7 if modality == "radar" else 4
        points = torch.zeros(2, 2, columns)
        origin = points.new_tensor(GRID.minimum_xyz)
        step = points.new_tensor(GRID.size_xyz)
        points[0, 0, :3] = origin + step * points.new_tensor([0.5, 0.5, 0.5])
        points[0, 1, :3] = origin + step * points.new_tensor([8.5, 0.5, 0.5])
        points[1, 0, :3] = points[0, 0, :3]
        valid = torch.tensor([[True, True], [True, False]])
        return points, valid

    def test_fine_and_coarse_masks_match_and_do_not_cross_batches(self):
        coarse = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        fine = coarse.clone()
        fine[:, 1:] *= 8
        for salt in range(10):
            expected = masked_bottleneck_sites(fine, 0.3, salt)
            torch.testing.assert_close(previous_encoder._selected_coarse_sites(coarse, 0.3, salt),
                                       expected)
        self.assertTrue(any(bool(masked_bottleneck_sites(fine, 0.3, salt).any())
                            for salt in range(10)))

    def test_trained_encoder_exports_strict_frozen_radar_and_lidar_weights(self):
        for modality in ("radar", "lidar"):
            with self.subTest(modality=modality):
                model = previous_encoder.PreviousStage1PFE(modality=modality)
                points, valid = self.sample(modality)
                output = model(points, valid)
                self.assertEqual(output.features.shape, (3, 64))
                self.assertEqual(output.stride, 8)
                self.assertEqual(output.shape_zyx, GRID.scale_shape(8))
                with tempfile.TemporaryDirectory() as folder:
                    saved = Path(folder) / "encoder.pth"
                    torch.save(model.state_dict(), saved)
                    frozen = previous_encoder.FrozenPreviousStage1PFE(saved, modality=modality)
                    loaded = frozen(points, valid)
                    torch.testing.assert_close(loaded.features, output.features)
                    self.assertFalse(any(parameter.requires_grad for parameter in frozen.parameters()))

    def test_phase_zero_trains_output_adapter_and_temporary_head(self):
        points, valid = self.sample()
        model = GeometryPretrainer("radar")
        loss, eligible, active = model(points, valid, 0.5, 7)
        self.assertGreaterEqual(active, eligible)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        if eligible:
            self.assertIsNotNone(model.encoder.output_adapter[0].weight.grad)
            self.assertIsNotNone(model.local_distribution.weight.grad)

    def test_active_radar_point_encoder_receives_doppler_and_scan_age(self):
        encoder = PFEPointVoxelEncoder(GRID, "radar").eval()
        captured = []
        handle = encoder.attribute_mlp.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        self.addCleanup(handle.remove)
        radar = torch.zeros(1, 2, 7)
        radar[:, :, :3] = radar.new_tensor([0.1, 0.1, 0.0])
        valid = torch.tensor([[True, True]])
        encoder(radar, valid)
        baseline = captured[-1]
        radar[0, 0, 4:7] = radar.new_tensor([6.0, 4.0, -3.0])
        encoder(radar, valid)
        changed = captured[-1]
        self.assertFalse(torch.equal(baseline[:, [1, 2, 5, 6, 3, 7]],
                                     changed[:, [1, 2, 5, 6, 3, 7]]))

    def test_teacher_and_student_construct_the_previous_encoder(self):
        class FakeFrozenEncoder(nn.Module):
            def __init__(self, checkpoint, *, modality):
                super().__init__()
                self.source_checkpoint = checkpoint
                self.modality = modality

        with patch.object(teacher_module, "FrozenPreviousStage1PFE", FakeFrozenEncoder), \
             patch.object(student_module, "FrozenPreviousStage1PFE", FakeFrozenEncoder), \
             patch.object(student_module, "RawSparseConfidenceDecoder", lambda config: nn.Identity()):
            teacher = teacher_module.CrossModalTeacher("radar.pth", "lidar.pth")
            student = student_module.RadarOnlyStudent("radar.pth")
        self.assertEqual(teacher.radar_encoder.modality, "radar")
        self.assertEqual(teacher.lidar_encoder.modality, "lidar")
        self.assertEqual(student.radar_encoder.source_checkpoint, "radar.pth")


if __name__ == "__main__":
    unittest.main()
