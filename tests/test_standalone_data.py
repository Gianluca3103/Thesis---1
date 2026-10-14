"""The extracted loader still discovers, calibrates, and batches VoD pairs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1


def _calibration(path: Path, x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Tr_velo_to_cam: 1 0 0 {x} 0 1 0 0 0 0 1 0\n")


class StandaloneVoDDataTests(unittest.TestCase):
    def test_paired_radar_is_aligned_and_collated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, frame_id in (("train", "00001"), ("val", "00002")):
                image_sets = root / "lidar" / "ImageSets"
                image_sets.mkdir(parents=True, exist_ok=True)
                (image_sets / f"{split}.txt").write_text(frame_id + "\n")
                lidar_root = root / "lidar" / "training"
                radar_root = root / "radar_20frames_verified_doppler_radial" / "training"
                (lidar_root / "velodyne").mkdir(parents=True, exist_ok=True)
                (radar_root / "velodyne").mkdir(parents=True, exist_ok=True)
                np.array([[2., 0., 0., 0.5]], dtype=np.float32).tofile(
                    lidar_root / "velodyne" / f"{frame_id}.bin")
                np.array([[1., 0., 0., 3., 4., 5., 0.]], dtype=np.float32).tofile(
                    radar_root / "velodyne" / f"{frame_id}.bin")
                _calibration(lidar_root / "calib" / f"{frame_id}.txt", 0.)
                # Derived radar variants inherit single-scan calibration.
                _calibration(root / "radar" / "training" / "calib" /
                             f"{frame_id}.txt", 1.)
            dataset = VoDStage1Dataset(root, "train")
            sample = dataset[0]
            self.assertEqual(sample["frame_id"], "00001")
            np.testing.assert_allclose(sample["radar"][0, :3], [2., 0., 0.])
            self.assertEqual(sample["radar"][0, 3:].tolist(), [3., 4., 5., 0.])
            batch = collate_stage1([sample])
            self.assertEqual(batch["radar"].shape, (1, 1, 7))
            self.assertTrue(batch["clean_lidar_valid"][0, 0])
            self.assertEqual(len(VoDStage1Dataset(root, "val")), 1)


if __name__ == "__main__":
    unittest.main()
