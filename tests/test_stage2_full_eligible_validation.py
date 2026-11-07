"""The per-epoch report must cover every frame and score voxel occupancy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.radar_lidar_stage2.eligible_validation import evaluate_all_eligible
from models.radar_lidar_stage2.sparse_geometry import GRID


def center(cell):
    xyz = torch.tensor([cell[3], cell[2], cell[1]], dtype=torch.float32)
    return (xyz + 0.5) * torch.tensor(GRID.voxel_xyz) + torch.tensor(GRID.minimum_xyz)


class Dataset:
    def __init__(self):
        self.frames = [SimpleNamespace(frame_id="a"), SimpleNamespace(frame_id="b")]
        self.target = torch.tensor([0, 8, 80, 10])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return {"frame_id": self.frames[index].frame_id,
                "radar": torch.tensor([[float(index)]], dtype=torch.float32),
                "clean_lidar": center(self.target)[None, :]}


class Model:
    def eval(self):
        return self

    def conditions(self, output, gate_threshold):
        assert gate_threshold == 0.25
        fine = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
        return SimpleNamespace(hierarchy=(fine,), frame_index=output["frame_index"])

    def sample(self, faulty, conditions, alpha_bars, **kwargs):
        assert kwargs["missing_only"]
        assert kwargs["steps"] == 4
        assert kwargs["occupancy_threshold"] == 0.5
        chosen = conditions.hierarchy[0][conditions.frame_index:
                                         conditions.frame_index + 1]
        return {"occupied_coordinates": chosen,
                "points_xyz": torch.full((1, 3), 999.0)}


class FullEligibleValidationTest(unittest.TestCase):
    def test_all_frames_and_voxel_centers_only(self):
        dataset = Dataset()
        cache = SimpleNamespace(load=lambda frame_id: None)
        regions = SimpleNamespace(get=lambda frame_id: {})
        with (patch("models.radar_lidar_stage2.eligible_validation.load_faulty",
                    return_value=torch.empty((0, 3))),
              patch("models.radar_lidar_stage2.eligible_validation.geometry_gate_output",
                    side_effect=lambda radar, *args: {
                        "frame_index": int(radar[0, 0])})):
            report = evaluate_all_eligible(
                Model(), dataset, faulty_root=None, support_cache=cache,
                region_cache=regions, fault_names={"a": "fog", "b": "loss"},
                alpha_bars=torch.ones(4), gate_threshold=0.25,
                occupancy_threshold=0.5, sampling_steps=4, device="cpu")
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["eligible_target"], 2)
        self.assertEqual(report["predicted"], 2)
        self.assertEqual(report["exact_tp"], 1)
        self.assertEqual(report["tolerant_tp_0p2m"], 2)
        self.assertAlmostEqual(report["eligible_iou"], 1 / 3)
        self.assertEqual(report["eligible_iou_0p2m"], 1)
        self.assertEqual(report["by_fault"]["fog"]["eligible_iou"], 1)
        self.assertEqual(report["by_fault"]["loss"]["eligible_iou"], 0)


if __name__ == "__main__":
    unittest.main()
