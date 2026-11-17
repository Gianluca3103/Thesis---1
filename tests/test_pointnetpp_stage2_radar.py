"""Stage-II radar crop, original-frame density rule, and seven-channel model."""

import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.pointnetpp_stage2_radar_data import (
    Stage2RadarScale, filter_and_crop,
)
from scripts.radar_encoder_ablation import RadarAutoencoder, reconstruction_loss
from scripts.train_pointnetpp_stage2_radar import FixedRadarInput, save_checkpoint


def test_filter_uses_boundary_support_before_stage2_crop():
    # Twelve distinct 0.2 m voxels form one cluster. Four supporting returns
    # are beyond x=32 m, but the eight inner returns still qualify.
    points = torch.zeros((13, 7), dtype=torch.float32)
    points[:12, 0] = torch.tensor([31.05, 31.25, 31.45, 31.65, 32.05, 32.25] * 2)
    points[:12, 1] = torch.tensor([0.05] * 6 + [0.25] * 6)
    points[:12, 2] = 0.1
    points[12, :3] = torch.tensor([10.0, 10.0, 0.1])
    filtered = filter_and_crop(points)
    assert len(filtered) == 8
    assert torch.all(filtered[:, 0] < 32)
    assert not torch.any(filtered[:, 0] == 10.0)


def test_empty_input_still_collates_to_seven_columns():
    class EmptyFrames:
        def __len__(self):
            return 1

        def load(self, index):
            return torch.empty((0, 7))

    scale = Stage2RadarScale([0, -16, -2, -10, -10, -10, -10],
                             [32, 16, 2, 10, 10, 10, 10])
    item = FixedRadarInput(EmptyFrames(), scale, 32, training=False)[0]
    assert item["input"].shape == (32, 7)
    assert item["usable"] is False


def test_large_model_contract_accepts_seven_radar_columns():
    model = RadarAutoencoder("pointnetpp", torch.tensor([0, -16, -2, -1, -1, -1, -1]),
                             torch.tensor([32, 16, 2, 1, 1, 1, 1]),
                             width=16, anchors=8, children=2, columns=7)
    radar = torch.rand((1, 16, 7)) * 1.8 - 0.9
    output = model(radar)
    assert output.shape == (1, 16, 7)
    terms = reconstruction_loss(output, radar, model.minimum, model.maximum)
    terms["total"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_dropout_is_training_only_and_does_not_change_old_weights():
    minimum = torch.tensor([0, -16, -2, -1, -1, -1, -1])
    maximum = torch.tensor([32, 16, 2, 1, 1, 1, 1])
    original = RadarAutoencoder("pointnetpp", minimum, maximum,
                                width=16, anchors=8, children=2, columns=7)
    regularized = RadarAutoencoder("pointnetpp", minimum, maximum,
                                   width=16, anchors=8, children=2,
                                   columns=7, dropout_p=0.1)
    assert not any(isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout1d))
                   for layer in original.modules())
    assert any(isinstance(layer, torch.nn.Dropout) and layer.p == 0.1
               for layer in regularized.encoder.modules())
    assert any(isinstance(layer, torch.nn.Dropout1d) and layer.p == 0.1
               for layer in regularized.decoder.modules())
    assert set(original.state_dict()) != set(regularized.state_dict())
    points = torch.rand((1, 16, 7)) * 1.8 - 0.9
    regularized.eval()
    assert torch.equal(regularized(points), regularized(points))
    regularized.train()
    assert not torch.equal(regularized(points), regularized(points))


def test_voxelnet_accepts_same_seven_column_radar_and_decoder():
    minimum = torch.tensor([0, -16, -2, -1, -1, -1, -1])
    maximum = torch.tensor([32, 16, 2, 1, 1, 1, 1])
    model = RadarAutoencoder("voxelnet", minimum, maximum,
                             width=16, anchors=8, children=2, columns=7)
    radar = torch.rand((1, 16, 7)) * 1.8 - 0.9
    output = model(radar)
    assert output.shape == (1, 16, 7)
    reconstruction_loss(output, radar, model.minimum,
                        model.maximum)["total"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_matched_capacity_parameter_counts():
    minimum = torch.tensor([0, -16, -2, -1, -1, -1, -1])
    maximum = torch.tensor([32, 16, 2, 1, 1, 1, 1])
    expected = (("pointnetpp", 256, 1_103_644),
                ("voxelnet", 108, 1_105_732),
                ("pointnetpp", 1024, 17_390_620),
                ("voxelnet", 430, 17_417_608))
    for variant, width, count in expected:
        model = RadarAutoencoder(variant, minimum, maximum, width=width,
                                 anchors=512, children=4, columns=7)
        assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_checkpoint_write_is_complete_before_publication():
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "last.ckpt"
        save_checkpoint(checkpoint, {"epoch": 1, "weight": torch.arange(8)})
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert loaded["epoch"] == 1
        assert torch.equal(loaded["weight"], torch.arange(8))
        assert not list(Path(directory).glob("*.tmp"))
