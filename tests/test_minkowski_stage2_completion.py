"""CPU checks for target-free Stage II sites and calibrated selection."""

import unittest

import torch

from models.radar_lidar_stage2.minkowski_completion import (
    gather_codes, gather_logits, radar_anchor_features, radar_parent_sites,
    voxelize_bxyz)
from scripts.train_stage2_minkowski_completion import _prepare, _sample, _select_threshold


class MinkowskiCompletionSitesTest(unittest.TestCase):
    def test_gate_alone_creates_parent_sites(self):
        fine, parents = radar_parent_sites(
            [torch.tensor([[1, 2, 3], [1, 2, 4]]),
             torch.tensor([[3, 8, 9]])], torch.device("cpu"))
        self.assertEqual(fine.tolist(), [[0, 3, 2, 1], [0, 4, 2, 1],
                                         [1, 9, 8, 3]])
        self.assertEqual(parents.tolist(), [[0, 2, 2, 0], [0, 4, 2, 0],
                                            [1, 8, 8, 2]])
        # The absent clean block still has a radar-derived query site.
        codes, found = gather_codes(
            torch.tensor([[0, 2, 2, 0]]), torch.ones((1, 64)), parents)
        self.assertEqual(found.tolist(), [True, False, False])
        self.assertEqual(float(codes[1:].abs().sum()), 0)

    def test_decoder_logits_cover_all_gate_cells_and_backpropagate(self):
        gate = torch.tensor([[0, 3, 2, 1], [0, 4, 2, 1]])
        logits = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        sites = torch.tensor([[0, 4, 2, 1], [0, 0, 0, 0], [0, 3, 2, 1]])
        aligned = gather_logits(sites, logits, gate)
        self.assertTrue(torch.allclose(aligned, torch.tensor([0.7, 0.2])))
        aligned.sum().backward()
        self.assertEqual(logits.grad.tolist(), [1.0, 0.0, 1.0])
        with self.assertRaises(RuntimeError):
            gather_logits(sites[:1], logits[:1], gate)

    def test_radar_features_interpolate_to_generated_sites(self):
        parents = torch.tensor([[0, 2, 2, 0], [1, 8, 8, 2]])
        features = radar_anchor_features(
            parents, [(torch.tensor([[0.6, -15.4, -1.625]]),
                       torch.ones((1, 768))),
                      (torch.empty((0, 3)), torch.empty((0, 768)))])
        self.assertEqual(features.shape, (2, 768))
        self.assertTrue(torch.allclose(features[0], torch.ones(768)))
        self.assertEqual(float(features[1].abs().sum()), 0.0)

    def test_outside_lidar_points_do_not_enter_stage2_grid(self):
        xyz = torch.tensor([[1.0, -15.0, -1.0], [100.0, 0.0, 0.0]])
        cells = voxelize_bxyz(xyz, torch.tensor([0, 0]))
        self.assertEqual(cells.tolist(), [[0, 5, 5, 4]])

    def test_full_scene_threshold_can_choose_no_additions(self):
        def bins(tp, base_pred, target, positives, negatives):
            all_hist = torch.zeros(1001, dtype=torch.long)
            true_hist = torch.zeros(1001, dtype=torch.long)
            all_hist[500] = positives + negatives
            true_hist[500] = positives
            return {"all": all_hist, "true": true_hist,
                    "baseline_tp": tp, "baseline_predicted": base_pred,
                    "target": target, "frames": 1}
        result = _select_threshold({
            "calibration": bins(8, 8, 10, 0, 10),
            "heldout": bins(7, 7, 10, 1, 10)})
        self.assertEqual(result["threshold"], 1.001)
        self.assertEqual(result["groups"]["heldout"]["added"], 0)

    def test_noise_start_uses_frame_seed_and_no_clean_input(self):
        class FixedEpsilon(torch.nn.Module):
            def forward(self, parents, noisy, faulty, radar, present, timestep):
                self.asserted = (parents.shape, faulty.shape, radar.shape,
                                 present.shape, timestep.shape)
                return torch.zeros_like(noisy)

        model = FixedEpsilon()
        parents = torch.tensor([[0, 2, 2, 0], [1, 8, 8, 2]])
        faulty = torch.zeros((2, 64))
        present = torch.zeros(2, dtype=torch.bool)
        radar = torch.zeros((2, 768))
        alpha = torch.linspace(0.999, 0.001, 1000)
        first = _sample(model, parents, faulty, present, radar, alpha,
                        ["00001", "00002"], steps=4, seed=3103,
                        gradient_steps=0)
        second = _sample(model, parents, faulty, present, radar, alpha,
                         ["00001", "00002"], steps=4, seed=3103,
                         gradient_steps=0)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(model.asserted, ((2, 4), (2, 64), (2, 768),
                                          (2,), (2,)))

    def test_inference_condition_does_not_read_clean_lidar(self):
        class FakeEncoder:
            def encode(self, occupied):
                latent = occupied.clone()
                latent[:, 1:] = (latent[:, 1:] // 2) * 2
                return type("Encoded", (), {
                    "C": latent, "F": torch.ones((len(latent), 64))})()

        row = {"fine_zyx": torch.tensor([[1, 2, 3]]),
               "faulty_xyz": torch.tensor([[0.7, -15.5, -1.6]]),
               "full_clean_xyz": object(),
               "learned_radar_xyz": torch.tensor([[0.7, -15.5, -1.6]]),
               "learned_radar_features": torch.ones((1, 768))}
        prepared = _prepare([row], FakeEncoder(), torch.device("cpu"),
                            training_target=False)
        self.assertIsNone(prepared[2])
        self.assertIsNone(prepared[7])
        self.assertEqual(prepared[1].tolist(), [[0, 2, 2, 0]])


if __name__ == "__main__":
    unittest.main()
