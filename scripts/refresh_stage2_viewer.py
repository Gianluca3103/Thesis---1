"""Update existing Stage-II HTML viewers without rerunning inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import torch

from models.radar_lidar_stage2.sparse_geometry import GRID, select_allowed
from scripts.predict_stage2_sparse import _allowed_view, _viewer_markup


DATA = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


def refresh(path: Path, stage2_cache_root: Path | None = None) -> None:
    match = DATA.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Stage-II viewer data is missing: {path}")
    data = json.loads(match.group(1))
    # Older full-estimate previews did not record these display-only fields.
    data.setdefault("full_estimate_cells", data["predicted_cells"])
    data.setdefault("overlap_observed_cells", 0)
    data.setdefault("missing_only", False)
    data.setdefault("zero_residual", False)
    data.setdefault("diffusion_target", "residual")
    if stage2_cache_root is not None:
        split = data["split"]
        frame_id = data["frame"]
        cache_split = stage2_cache_root / split
        manifest = json.loads((cache_split / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["split"] != split or
                manifest["grid"] != json.loads(json.dumps(asdict(GRID)))):
            raise ValueError(f"Stage-II cache split or grid differs: {cache_split}")
        if frame_id not in manifest["frame_ids"]:
            raise ValueError(f"Frame {frame_id} is absent from {cache_split}")
        with np.load(cache_split / f"{frame_id}.npz", allow_pickle=False) as cached:
            fine, scores = select_allowed(
                torch.from_numpy(cached["confidence_coords"].astype(np.int64)),
                torch.from_numpy(cached["confidence_probabilities"].astype(np.float32)),
                float(manifest["gate_threshold"]))
        allowed_xyz, allowed_probabilities = _allowed_view(fine, scores)
        if len(allowed_xyz) != data["allowed_cells"]:
            raise ValueError(f"Allowed-cell count differs for frame {frame_id}: "
                             f"viewer={data['allowed_cells']}, cache={len(allowed_xyz)}")
        data["allowed_xyz"] = np.round(allowed_xyz, 3).tolist()
        data["allowed_probabilities"] = np.round(
            allowed_probabilities.astype(np.float64), 4).tolist()
        data["gate_threshold"] = float(manifest["gate_threshold"])
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(_viewer_markup(payload), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory containing existing Stage-II frame HTML files")
    parser.add_argument("--stage2-cache-root", type=Path,
                        help="Add the exact Stage-I allowed cells and probabilities without inference")
    args = parser.parse_args()
    paths = sorted(path for path in args.output_root.glob("*.html")
                   if path.name != "index.html")
    if not paths:
        parser.error(f"No Stage-II frame HTML files found in {args.output_root}")
    for path in paths:
        refresh(path, args.stage2_cache_root)
    print(f"Updated {len(paths)} frame viewers under {args.output_root}"
          f"{' with Stage-I allowed cells' if args.stage2_cache_root else ''}")


if __name__ == "__main__":
    main()
