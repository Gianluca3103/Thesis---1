"""The 32-slot trial retains an available trained eight-slot association."""

from dataclasses import asdict, replace

import torch
import pytest

from models.radar_lidar_crossmodal_v1.fine_association import (
    FineAssociationStudent, VERSION)
from models.radar_lidar_crossmodal_v1.meta_occupancy import radar_only_probability_map
from models.radar_lidar_crossmodal_v1.surface_student import SurfaceStudentConfig
from models.radar_lidar_crossmodal_v1.surface_student import pointer_xyz_from_offsets


def test_eight_slot_checkpoint_initializes_first_eight_of_thirty_two_slots():
    eight_config = SurfaceStudentConfig(
        proposals_per_anchor=8, target_neighbors=8,
        anchor_stride=1, offset_limit_m=3.0, target_radius_m=3.0,
        context_radii_m=(0.8, 1.6, 2.4, 3.0),
        spherical_offsets=True, decoder_from_conditioned=True)
    old = FineAssociationStudent(eight_config)
    new = FineAssociationStudent(replace(
        eight_config, proposals_per_anchor=32, target_neighbors=32))
    with torch.no_grad():
        old.surface_head.slots.weight.fill_(0.123)
        old.surface_head.slots.bias.fill_(0.456)
    untouched_tail = new.surface_head.slots.weight.detach().view(32, 4, -1)[8:].clone()
    saved = {"version": VERSION, "source_student_sha256": "same-source",
             "config": asdict(eight_config), "model": old.state_dict()}

    new.initialize_from_eight_slot_checkpoint(saved, "same-source")

    for key, value in old.state_dict().items():
        if not key.startswith("surface_head.slots."):
            torch.testing.assert_close(new.state_dict()[key], value)
    torch.testing.assert_close(
        new.surface_head.slots.weight.view(32, 4, -1)[:8],
        old.surface_head.slots.weight.view(8, 4, -1))
    torch.testing.assert_close(
        new.surface_head.slots.bias.view(32, 4)[:8],
        old.surface_head.slots.bias.view(8, 4))
    torch.testing.assert_close(
        new.surface_head.slots.weight.view(32, 4, -1)[8:], untouched_tail)


def test_new_thirty_two_slot_trial_caps_pointers_at_one_point_six_metres():
    config = SurfaceStudentConfig(
        proposals_per_anchor=32, target_neighbors=32,
        anchor_stride=1, offset_limit_m=1.6, target_radius_m=1.6,
        context_radii_m=(0.8, 1.6, 2.4, 3.0),
        spherical_offsets=True, decoder_from_conditioned=True)
    model = FineAssociationStudent(config)
    center = torch.tensor([[4.1, .1, .125]])
    offsets = torch.full((1, 32, 3), 100.0)
    xyz = pointer_xyz_from_offsets(center, offsets, config.offset_limit_m,
                                   config.spherical_offsets)
    assert torch.all(torch.linalg.vector_norm(xyz - center[:, None], dim=-1)
                     <= 1.6 + 1e-6)
    assert model.config.context_radii_m[-1] == 3.0
    with pytest.raises(ValueError):
        FineAssociationStudent(replace(config, target_radius_m=3.0))


def test_sixty_four_slot_trial_keeps_the_one_point_six_metre_radius():
    config = SurfaceStudentConfig(
        proposals_per_anchor=64, target_neighbors=64,
        anchor_stride=1, offset_limit_m=1.6, target_radius_m=1.6,
        context_radii_m=(0.8, 1.6, 2.4, 3.0),
        spherical_offsets=True, decoder_from_conditioned=True)
    model = FineAssociationStudent(config)
    assert model.surface_head.slots.out_features == 64 * 4
    assert model.config.target_radius_m == 1.6


def test_fine_association_inference_requires_its_trained_radius():
    class FakeFineStudent:
        config = type("Config", (), {"target_radius_m": 1.6})()

        def __call__(self, radar, valid):
            return type("Output", (), {"exported": lambda self: {
                "interface_version": "crossmodal_fine_association_v1",
                "confidence_coords": torch.empty((0, 4), dtype=torch.long),
            }})()

    model = FakeFineStudent()
    radar = torch.zeros((1, 0, 7))
    valid = torch.zeros((1, 0), dtype=torch.bool)
    assert radar_only_probability_map(model, radar, valid, 1.6)[
        "radar_support_radius_m"] == 1.6
    with pytest.raises(ValueError):
        radar_only_probability_map(model, radar, valid, 3.0)
