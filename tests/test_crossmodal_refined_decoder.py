"""Keep the learned decoder's fine radar path compatible with frozen weights."""

import torch
from torch import nn
from types import SimpleNamespace

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.previous_encoder import PreviousStage1Sparse
from models.radar_lidar_crossmodal_v1.student import GenerativeBlock, RadarOnlyStudent
from models.radar_lidar_stage1.sparse import SparseSites


def test_frozen_encoder_pyramid_preserves_original_bottleneck():
    encoder = PreviousStage1Sparse(modality="radar")
    encoder.frozen = True
    encoder.eval()
    points = torch.tensor([[[0., 0., 0., 1., 2., 3., 0.]]])
    valid = torch.ones((1, 1), dtype=torch.bool)
    ordinary = encoder(points, valid)
    coarse, pyramid = encoder.forward_pyramid(points, valid)
    assert [level.stride for level in pyramid] == [1, 2, 4]
    torch.testing.assert_close(ordinary.coords, coarse.coords)
    torch.testing.assert_close(ordinary.features, coarse.features)
    assert all(not level.features.requires_grad for level in pyramid)


def test_student_passes_only_radar_pyramid_to_decoder():
    class Adapter(nn.Module):
        def forward(self, value):
            return value

    class Sites:
        def __init__(self):
            self.features = torch.ones(1, 64)

        def replace_features(self, features):
            result = Sites()
            result.features = features
            return result

    class RadarSites(nn.Module):
        def forward_pyramid(self, points, valid):
            self.seen = (points, valid)
            return Sites(), ("fine", "middle", "coarse")

    class Decoder(nn.Module):
        def forward(self, sites, pyramid):
            self.pyramid = pyramid
            return sites

    student = RadarOnlyStudent.__new__(RadarOnlyStudent)
    nn.Module.__init__(student)
    student.radar_encoder = RadarSites()
    student.shared = Adapter()
    student.feature_head = Adapter()
    student.feature_norm = Adapter()
    student.confidence_decoder = Decoder()
    points = torch.zeros(1, 1, 7)
    valid = torch.ones((1, 1), dtype=torch.bool)
    result = student(points, valid)
    assert student.radar_encoder.seen[0] is points
    assert student.radar_encoder.seen[1] is valid
    assert student.confidence_decoder.pyramid == ("fine", "middle", "coarse")
    assert result.confidence_logits.features.shape == (1, 64)


def test_fine_radar_join_uses_exact_voxels_and_sample_indices():
    candidate = SimpleNamespace(
        C=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 2], [1, 0, 1, 1]]),
        F=torch.zeros(3, 1), tensor_stride=(1, 1, 1))
    radar = SparseSites(torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]]),
                        torch.tensor([[2.], [7.]]), GRID.shape_zyx, 1)
    block = SimpleNamespace(skip_projection=nn.Identity())
    joined = GenerativeBlock._joined_skip(block, candidate, radar)
    torch.testing.assert_close(joined[:, 0], torch.tensor([2., 0., 7.]))
