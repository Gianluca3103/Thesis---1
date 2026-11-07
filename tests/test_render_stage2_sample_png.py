"""Static sample rendering works from cached arrays without a browser."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.render_stage2_sample_png import main, render_sample


class StaticStage2SampleTest(unittest.TestCase):
    def test_png_and_exact_occupancy_counts(self):
        workspace = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(prefix="stage2_png_test_", dir=workspace) as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(workspace))
            archive = root / "00001_partial_loss_30_stage2_epoch_022.npz"
            clean = np.array([[0.1, 0.1, 0.125], [0.3, 0.1, 0.125]], dtype=np.float32)
            predicted = np.array([[0.1, 0.1, 0.125], [0.5, 0.1, 0.125]], dtype=np.float32)
            np.savez_compressed(archive, clean_xyz=clean, faulty_xyz=clean[1:],
                                xyz=predicted, allowed_xyz=np.concatenate((clean,
                                    predicted[1:])), fault_type="partial_loss_30")
            output = root / "sample.png"
            preview, report = render_sample(archive, output)
            data = output.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (1440, 776))
            self.assertEqual(preview.shape, (388, 360, 3))
            self.assertEqual(report["predicted"], 2)
            self.assertEqual(report["eligible_target"], 2)
            self.assertEqual(report["exact_tp"], 1)
            self.assertAlmostEqual(report["exact_iou"], 1 / 3)
            with patch.object(sys, "argv", ["render_stage2_sample_png",
                                            "--sample-root", str(root)]):
                main()
            self.assertTrue((root / "static_png" / "contact_sheet.png").is_file())
            self.assertTrue((root / "static_png" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
