"""Light 3D viewers retain paired values and full-result counts."""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.make_stage2_lite_viewer import compact_frame, main
from scripts.predict_stage2_sparse import _viewer_markup


DATA = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


class LightViewerTest(unittest.TestCase):
    def test_compacts_display_only_and_writes_index(self):
        workspace = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(prefix="stage2_lite_test_", dir=workspace) as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(workspace))
            source = root / "full"
            destination = root / "lite"
            source.mkdir()
            destination.mkdir()
            data = {"frame": "00064", "fault_type": "fog_sim_2",
                    "clean": [[index, 0, 0] for index in range(10)],
                    "faulty": [[index, 0, 0] for index in range(10)],
                    "radar": [[index, 0, 0] for index in range(3)],
                    "predicted": [[index, 0, 0] for index in range(10)],
                    "probabilities": [index / 10 for index in range(10)],
                    "allowed_xyz": [[index, 0, 0] for index in range(20)],
                    "allowed_probabilities": [index / 20 for index in range(20)],
                    "allowed_cells": 20, "predicted_cells": 10,
                    "totals": [10, 10, 3, 10]}
            frame = source / "00064_fog_sim_2.html"
            frame.write_text(_viewer_markup(json.dumps(data)), encoding="utf-8")
            report = compact_frame(frame, destination / frame.name,
                                   cloud_limit=4, gate_limit=5)
            lite = json.loads(DATA.search((destination / frame.name)
                                          .read_text(encoding="utf-8")).group(1))
            self.assertEqual(report["full_gate"], 20)
            self.assertEqual(lite["allowed_cells"], 20)
            self.assertEqual(lite["display_allowed_cells"], 5)
            self.assertEqual(len(lite["clean"]), 4)
            self.assertEqual(len(lite["predicted"]), 4)
            self.assertEqual([row[0] / 10 for row in lite["predicted"]],
                             lite["probabilities"])
            self.assertEqual(lite["totals"], [10, 10, 3, 10])
            with patch.object(sys, "argv", ["make_stage2_lite_viewer",
                                            "--input-root", str(source),
                                            "--output-root", str(destination),
                                            "--cloud-limit", "4",
                                            "--gate-limit", "5"]):
                main()
            self.assertIn(frame.name, (destination / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
