"""The 32-slot trial retains an available trained eight-slot association."""

from dataclasses import asdict, replace

import torch

from models.radar_lidar_crossmodal_v1.fine_association import (
    FineAssociationStudent, VERSION)
from models.radar_lidar_crossmodal_v1.surface_student import SurfaceStudentConfig


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
