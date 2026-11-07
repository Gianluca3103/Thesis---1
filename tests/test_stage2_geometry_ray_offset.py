"""Contracts for the fixed Stage-II gate and the ray/offset-only objective."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import SHAPE, radar_support_mask
from models.radar_lidar_stage2.geometry_gate import (
    VERSION, geometry_gate_output, offset_targets,
    version_for_min_radar_voxels,
)
from models.radar_lidar_stage2.fault_region import (
    FaultRegionCache, footprint_mask, partial_azimuth_region)
from models.radar_lidar_stage2.mild_fault_data import partial_azimuth_loss
from models.radar_lidar_stage2.ray_label_cache import RayLabelCache
from models.radar_lidar_stage2.ray_loss import _first_ranges, addition_ray_losses
from models.radar_lidar_stage2.sparse_geometry import GRID, select_allowed
from models.radar_lidar_stage2.sparse_residual import (
    SparseResidualStage2, Stage2Conditions, _frame_mean,
)
from scripts.evaluate_stage2_geometry_baseline import _counts, _select
from scripts.train_stage2_sparse import _combine_conditions, _combine_points


class GeometryGateTest(unittest.TestCase):
    def test_twelve_voxel_gate_has_separate_checkpoint_version(self):
        self.assertEqual(version_for_min_radar_voxels(3), VERSION)
        self.assertNotEqual(version_for_min_radar_voxels(12), VERSION)
        with self.assertRaises(ValueError):
            version_for_min_radar_voxels(5)

    def test_twelve_voxel_gate_emits_only_supported_cells(self):
        radar = torch.zeros((12, 7), dtype=torch.float32)
        radar[:, 0] = torch.tensor([10.05 + 0.2 * i for i in range(12)])
        radar[:, 1:3] = 0.05
        support = radar_support_mask(radar, 1.6, 12)
        output = geometry_gate_output(
            radar, torch.empty((0, 3)), {"kind": "full_scan"},
            min_radar_voxels=12)
        fine, _ = select_allowed(output["confidence_coords"],
                                 output["confidence_probabilities"], 0.5)
        self.assertEqual(len(fine), int(support.sum()))
        chosen = fine[:, 1:].numpy()
        self.assertTrue(support[chosen[:, 0], chosen[:, 1], chosen[:, 2]].all())

    def test_only_supported_unobserved_cells_are_selected(self):
        mask = np.zeros(SHAPE, dtype=bool)
        mask[8, 80, 80:83] = True
        radar = torch.tensor([[16.1, 0.1, 0.1, 4, 2, 3, 1]],
                             dtype=torch.float32)
        # Occupy the middle supported voxel; all other faulty cells must not
        # be added and unsupported cells must never enter the gate.
        faulty = torch.tensor([[16.3, 0.1, 0.1], [16.9, 0.1, 0.1]],
                              dtype=torch.float32)
        output = geometry_gate_output(radar, faulty, {"kind": "full_scan"}, mask)
        fine, scores = select_allowed(output["confidence_coords"],
                                      output["confidence_probabilities"], 0.5)
        self.assertEqual(fine[:, 1:].tolist(), [[8, 80, 80], [8, 80, 82]])
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))
        self.assertEqual(output["conditioning_features"].shape[1], 64)

    def test_empty_radar_has_no_learned_fallback(self):
        output = geometry_gate_output(torch.empty((0, 7)), torch.empty((0, 3)),
                                      {"kind": "full_scan"},
                                      np.zeros(SHAPE, dtype=bool))
        self.assertEqual(len(output["confidence_coords"]), 0)
        self.assertEqual(len(output["conditioning_coords"]), 0)

    def test_partial_injector_sector_keeps_removed_and_excludes_untouched(self):
        angles = np.linspace(-1.1, 1.1, 101, dtype=np.float32)
        clean = np.stack((10 * np.cos(angles), 10 * np.sin(angles),
                          np.zeros_like(angles)), axis=1)
        seed = 4
        retained = partial_azimuth_loss(clean, 0.30, seed)
        region = partial_azimuth_region(clean, 0.30, seed)
        self.assertEqual(region["removed_points"], len(clean) - len(retained))
        order = np.argsort(np.arctan2(clean[:, 1], clean[:, 0]), kind="stable")
        start = int(np.random.default_rng(seed).integers(len(clean)))
        removed_ids = order[(start + np.arange(region["removed_points"])) % len(clean)]
        coords_xyz = np.floor((clean - np.asarray(GRID.minimum_xyz)) /
                              np.asarray(GRID.voxel_xyz)).astype(np.int64)
        fine = torch.from_numpy(np.column_stack((np.zeros(len(clean), dtype=np.int64),
                                                 coords_xyz[:, [2, 1, 0]])))
        inside = footprint_mask(fine, region).numpy()
        self.assertTrue(inside[removed_ids].all())
        self.assertGreater(int((~inside).sum()), 20)
        reliable = int(np.flatnonzero(~inside)[0])
        touched = int(removed_ids[0])
        mask = np.zeros(SHAPE, dtype=bool)
        for index in (reliable, touched):
            z, y, x = fine[index, 1:].tolist()
            mask[z, y, x] = True
        output = geometry_gate_output(
            torch.tensor([[10.0, 0.0, 0.0, 1, 1, 1, 1]]),
            torch.empty((0, 3)), region, mask)
        local, _ = select_allowed(output["confidence_coords"],
                                  output["confidence_probabilities"], 0.5)
        self.assertEqual(local[:, 1:].tolist(), [fine[touched, 1:].tolist()])

    def test_wrapped_partial_sector_covers_removed_voxels(self):
        angles = np.linspace(-np.pi, np.pi, 721, endpoint=False,
                             dtype=np.float32)
        clean = np.stack((12 * np.cos(angles), 12 * np.sin(angles),
                          np.zeros_like(angles)), axis=1)
        indices = np.floor((clean - np.asarray(GRID.minimum_xyz)) /
                           np.asarray(GRID.voxel_xyz)).astype(np.int64)
        fine = torch.from_numpy(np.column_stack((
            np.zeros(len(clean), dtype=np.int64), indices[:, [2, 1, 0]])))
        # The random start crosses the -pi/pi seam for some seeds.
        for seed in range(20):
            region = partial_azimuth_region(clean, 0.40, seed)
            count = region["removed_points"]
            order = np.argsort(np.arctan2(clean[:, 1], clean[:, 0]),
                               kind="stable")
            start = int(np.random.default_rng(seed).integers(len(clean)))
            removed = order[(start + np.arange(count)) % len(clean)]
            self.assertTrue(footprint_mask(fine, region)[removed].all().item())

    def test_fault_region_cache_rejects_incomplete_data(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            expected = {"version": "test", "frame_ids": ["1", "2"]}
            cache = FaultRegionCache(root, expected, create=True)
            with self.assertRaises(ValueError):
                cache.write({"1": {"kind": "full_scan"}})
            cache.write({"1": {"kind": "full_scan"},
                         "2": {"kind": "empty"}})
            self.assertEqual(FaultRegionCache(root, expected).get("2")["kind"],
                             "empty")

    def test_ray_label_cache_is_bound_to_ordered_coordinates(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            expected = {"frame_ids": ["1"], "grid": {"shape": (16, 160, 160)}}
            cache = RayLabelCache(Path(directory), expected, create=True)
            fine = torch.tensor([[0, 8, 80, 80], [0, 8, 80, 81]])
            labels = {"candidate_count": 2,
                      "hit": np.array([0], dtype=np.int32),
                      "front": np.array([1, 1], dtype=np.int32),
                      "extra": np.array([], dtype=np.int32)}
            cache.write("1", fine, labels)
            loaded = RayLabelCache(Path(directory), expected).load("1", fine)
            np.testing.assert_array_equal(loaded["front"], labels["front"])
            with self.assertRaises(ValueError):
                cache.load("1", fine.flip(0))

    def test_offset_target_is_centroid_relative_to_eligible_cell(self):
        fine = torch.tensor([[0, 8, 80, 80], [0, 8, 80, 81]])
        clean = torch.tensor([[16.02, 0.02, 0.02], [16.08, 0.08, 0.08]])
        occupied, offsets, coverage = offset_targets(clean, fine)
        self.assertEqual(occupied.tolist(), [True, False])
        self.assertTrue(torch.allclose(offsets[0], torch.tensor([-0.05, -0.05, -0.075]),
                                       atol=1e-5))
        self.assertEqual(coverage, 1.0)


class ObjectiveTest(unittest.TestCase):
    def test_eight_live_frames_keep_distinct_sparse_batches(self):
        rows = []
        for index in range(8):
            local = torch.tensor([[0, 8, 80, 80]], dtype=torch.long)
            rows.append(Stage2Conditions(
                (local,) * 5, torch.ones((1, 1)), torch.ones((1, 1)),
                torch.full((1, 32), float(index))))
        with self.assertRaises(ValueError):
            _combine_conditions(rows)
        combined = _combine_conditions(rows, allow_live_targets=True)
        self.assertIsNone(combined.cached_targets)
        for level in combined.hierarchy:
            self.assertEqual(level[:, 0].tolist(), list(range(8)))
        self.assertEqual(combined.radar_features[:, 0].tolist(),
                         [float(index) for index in range(8)])
        points, point_batches = _combine_points([
            torch.tensor([[16.1, 0.1, 0.125]]) for _ in range(8)])
        self.assertEqual(point_batches.tolist(), list(range(8)))
        occupied, offsets, coverage = offset_targets(
            points, combined.hierarchy[0], point_batches)
        self.assertTrue(occupied.all().item())
        self.assertEqual(offsets.shape, (8, 3))
        self.assertEqual(coverage, 1.0)
        weighted = _frame_mean(torch.tensor([1.0] + [10.0] * 8),
                               torch.tensor([0] + list(range(8))), 8)
        self.assertAlmostEqual(weighted.item(), 9.4375)

    def test_parallel_ray_queries_preserve_labels_and_gradients(self):
        rng = np.random.default_rng(7)
        query = rng.normal(size=(257, 3))
        returns = rng.normal(size=(41, 3))
        with patch.dict("os.environ", {"STAGE2_RAY_WORKERS": "1"}):
            serial = _first_ranges(query, returns, angular_chord=0.4,
                                   lateral_m=0.5)
        with patch.dict("os.environ", {"STAGE2_RAY_WORKERS": "4"}):
            parallel = _first_ranges(query, returns, angular_chord=0.4,
                                     lateral_m=0.5)
        np.testing.assert_allclose(serial, parallel, rtol=0, atol=0,
                                   equal_nan=True)

        fine = torch.tensor([[0, 8, 80, 20], [0, 8, 80, 25],
                             [0, 8, 80, 30]])
        clean = torch.tensor([[5.1, 0.1, 0.125]])
        outputs = []
        for workers in ("1", "4"):
            logits = torch.zeros(3, requires_grad=True)
            with patch.dict("os.environ", {"STAGE2_RAY_WORKERS": workers}):
                result = addition_ray_losses(
                    fine, logits, torch.zeros(3, dtype=torch.bool),
                    torch.tensor([False, True, False]), clean)
            (result["hit"] + result["extra"]).backward()
            outputs.append((result["hit"].item(), result["extra"].item(),
                            result["hit_rays"], result["extra_sites"],
                            logits.grad.clone()))
        self.assertEqual(outputs[0][:4], outputs[1][:4])
        self.assertTrue(torch.equal(outputs[0][4], outputs[1][4]))

    def test_calibration_threshold_is_applied_to_heldout_counts(self):
        calibration_pos = np.array([1, 2, 7, 10], dtype=np.int64)
        calibration_neg = np.array([20, 10, 1, 0], dtype=np.int64)
        choices = _select(calibration_pos, calibration_neg, 20)
        threshold_bin = choices["best_f1"]
        heldout = _counts(np.array([1, 1, 8, 10]),
                          np.array([10, 9, 2, 0]), 20, 25,
                          threshold_bin)
        self.assertEqual(heldout["eligible_target"], 20)
        self.assertEqual(heldout["full_missing_target"], 25)
        self.assertAlmostEqual(heldout["full_missing_recall"],
                               heldout["tp"] / 25)

    def test_ray_offset_mode_never_calls_addition_or_healthy_loss(self):
        class Fake:
            diffusion_target = "epsilon"
            grid = GRID

            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(0.2))

            def encode_lidar(self, xyz, conditions, batch_ids):
                count = len(conditions.hierarchy[-1])
                latent = xyz.new_zeros((count, 32))
                return latent, torch.zeros(count, dtype=torch.bool)

            def diffusion(self, noisy, faulty, radar, probability, present,
                          timestep, coords):
                return noisy * 0 + self.weight

            def lpcr(self, clean_hat, hierarchy, probabilities,
                     return_hidden=False):
                logits = clean_hat[:, 0]
                offsets = clean_hat[:, :3]
                hidden = clean_hat[:, :16]
                return logits, offsets, hidden

            def lpcr_loss(self, *args, **kwargs):
                raise AssertionError("Ray/offset mode called LPCR occupancy loss")

            def addition_head(self, hidden, logits, probability, observed, density):
                return logits

            def addition_loss(self, *args, **kwargs):
                raise AssertionError("Ray/offset mode called addition loss")

        fake = Fake()
        fine = torch.tensor([[0, 8, 80, 80]], dtype=torch.long)
        conditions = Stage2Conditions((fine, fine, fine, fine, fine),
                                      torch.ones((1, 1)), torch.ones((1, 1)),
                                      torch.zeros((1, 32)))
        ray = lambda fine, logits, observed, occupied, clean, **kwargs: {
            "hit": logits.square().mean(), "extra": logits.square().mean(),
            "hit_rays": 1, "extra_sites": 1}
        with patch("models.radar_lidar_stage2.sparse_residual.addition_ray_losses", ray):
            result = SparseResidualStage2.diffusion_loss(
                fake, torch.empty((0, 3)),
                torch.tensor([[16.12, 0.13, 0.12]]), conditions,
                torch.linspace(0.99, 0.01, 10), ray_offset_baseline=True,
                ray_hit_weight=1, ray_extra_weight=1, offset_weight=1)
        self.assertGreater(float(result["reconstruction"]["offset"].detach()), 0)
        expected = (result["noise"] + result["task_weight"] * (
            result["reconstruction"]["offset"] + result["rays"]["hit"]
            + result["rays"]["extra"]))
        self.assertTrue(torch.allclose(result["loss"], expected))
        result["loss"].backward()
        self.assertIsNotNone(fake.weight.grad)
        self.assertTrue(torch.isfinite(fake.weight.grad))

        fine8 = fine.repeat(8, 1)
        fine8[:, 0] = torch.arange(8)
        conditions8 = Stage2Conditions(
            (fine8,) * 5, torch.ones((8, 1)), torch.ones((8, 1)),
            torch.zeros((8, 32)))
        clean8 = torch.tensor([[16.12, 0.13, 0.12]]).repeat(8, 1)
        fake.weight.grad = None
        with patch("models.radar_lidar_stage2.sparse_residual.addition_ray_losses", ray):
            batched = SparseResidualStage2.diffusion_loss(
                fake, torch.empty((0, 3)), clean8, conditions8,
                torch.linspace(0.99, 0.01, 10), ray_offset_baseline=True,
                ray_hit_weight=1, ray_extra_weight=1, offset_weight=1,
                faulty_batch_ids=torch.empty(0, dtype=torch.long),
                clean_batch_ids=torch.arange(8), batch_size=8)
        self.assertTrue(torch.isfinite(batched["loss"]))
        batched["loss"].backward()
        self.assertTrue(torch.isfinite(fake.weight.grad))

        def fixed_step(high, size, *, device, generator):
            return torch.zeros(size, dtype=torch.long, device=device)

        def fixed_noise(shape, *, device, dtype, generator):
            return torch.zeros(shape, dtype=dtype, device=device)

        with (patch("models.radar_lidar_stage2.sparse_residual.addition_ray_losses", ray),
              patch("models.radar_lidar_stage2.sparse_residual.torch.randint", fixed_step),
              patch("models.radar_lidar_stage2.sparse_residual.torch.randn", fixed_noise)):
            single_fixed = SparseResidualStage2.diffusion_loss(
                fake, torch.empty((0, 3)),
                torch.tensor([[16.12, 0.13, 0.12]]), conditions,
                torch.linspace(0.99, 0.01, 10), ray_offset_baseline=True,
                ray_hit_weight=1, ray_extra_weight=1, offset_weight=1)
            batch_fixed = SparseResidualStage2.diffusion_loss(
                fake, torch.empty((0, 3)), clean8, conditions8,
                torch.linspace(0.99, 0.01, 10), ray_offset_baseline=True,
                ray_hit_weight=1, ray_extra_weight=1, offset_weight=1,
                faulty_batch_ids=torch.empty(0, dtype=torch.long),
                clean_batch_ids=torch.arange(8), batch_size=8)
        self.assertTrue(torch.allclose(single_fixed["loss"], batch_fixed["loss"],
                                       atol=1e-6))


if __name__ == "__main__":
    unittest.main()
