"""Geometry-only previews show each cached mild fault with its real label."""

import unittest

from scripts.predict_stage2_sparse import _sample_indices


class MildViewerSelectionTest(unittest.TestCase):
    def test_balanced_per_fault_selection_is_deterministic(self):
        frame_ids = [f"frame_{index}" for index in range(12)]
        faults = {frame_id: ("fog_sim_2" if index % 2 == 0
                             else "partial_loss_50")
                  for index, frame_id in enumerate(frame_ids)}
        first = _sample_indices(frame_ids, faults, 3, 1, 3103)
        self.assertEqual(first, _sample_indices(frame_ids, faults, 3, 1, 3103))
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)
        self.assertEqual(sum(faults[frame_ids[i]] == "fog_sim_2" for i in first), 3)
        self.assertEqual(sum(faults[frame_ids[i]] == "partial_loss_50"
                             for i in first), 3)

    def test_per_fault_rejects_insufficient_frames(self):
        with self.assertRaisesRegex(ValueError, "Only 1"):
            _sample_indices(["a"], {"a": "fog_sim_2"}, 2, 5, 3103)


if __name__ == "__main__":
    unittest.main()
