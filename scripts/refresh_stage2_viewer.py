"""Update existing Stage-II HTML viewers without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from scripts.predict_stage2_sparse import _viewer_markup


DATA = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


def refresh(path: Path) -> None:
    match = DATA.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Stage-II viewer data is missing: {path}")
    data = json.loads(match.group(1))
    # Older full-estimate previews did not record these display-only fields.
    data.setdefault("full_estimate_cells", data["predicted_cells"])
    data.setdefault("overlap_observed_cells", 0)
    data.setdefault("missing_only", False)
    data.setdefault("zero_residual", False)
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(_viewer_markup(payload), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory containing existing Stage-II frame HTML files")
    args = parser.parse_args()
    paths = sorted(path for path in args.output_root.glob("*.html")
                   if path.name != "index.html")
    if not paths:
        parser.error(f"No Stage-II frame HTML files found in {args.output_root}")
    for path in paths:
        refresh(path)
    print(f"Updated camera controls in {len(paths)} frame viewers under {args.output_root}")


if __name__ == "__main__":
    main()
