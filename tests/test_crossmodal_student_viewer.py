"""The raw viewer shows the configured LiDAR region without point caps."""

import json

import numpy as np

from scripts.visualize_crossmodal_student import _write_html


def test_viewer_keeps_every_supplied_point_and_marks_old_checkpoint(tmp_path):
    radar = np.array([[2., 0., 0.]], dtype=np.float32)
    lidar = np.array([[3., 0., 0.], [4., 0., 0.]], dtype=np.float32)
    candidates = np.array([[3., 1., 0.], [4., 1., 0.]], dtype=np.float32)
    path = tmp_path / "view.html"
    _write_html(path, "08236", 20, radar, lidar, candidates,
                np.array([0.3, 0.9], dtype=np.float32),
                checkpoint_uses_fov=False)
    html = path.read_text(encoding="utf-8")
    payload = html.split('<script id="data" type="application/json">', 1)[1].split(
        "</script>", 1)[0]
    data = json.loads(payload)
    assert [len(cloud["xyz"]) for cloud in data["clouds"]] == [1, 2, 2]
    assert data["radar_half_fov_deg"] == 70.
    assert data["radar_max_range_m"] == 80.
    assert data["checkpoint_uses_fov"] is False
