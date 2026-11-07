"""Make fast interactive Stage-II viewers from existing full-resolution HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

from scripts.predict_stage2_sparse import _viewer_markup


DATA = re.compile(r'<script id="data" type="application/json">(.*?)</script>', re.S)


def _even_indices(count: int, limit: int) -> range | list[int]:
    if count <= limit:
        return range(count)
    return [index * count // limit for index in range(limit)]


def compact_frame(source: Path, destination: Path, *, cloud_limit: int = 6000,
                  gate_limit: int = 4000) -> dict:
    match = DATA.search(source.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Stage-II HTML has no embedded sample data: {source}")
    data = json.loads(match.group(1))
    if (cloud_limit < 1 or gate_limit < 1
            or len(data["predicted"]) != len(data["probabilities"])
            or len(data["allowed_xyz"]) != len(data["allowed_probabilities"])):
        raise ValueError(f"Invalid Stage-II viewer arrays: {source}")
    counts = {key: len(data[key]) for key in
              ("clean", "faulty", "radar", "predicted")}
    for key in counts:
        indices = _even_indices(counts[key], cloud_limit)
        data[key] = [data[key][index] for index in indices]
        if key == "predicted":
            data["probabilities"] = [data["probabilities"][index]
                                     for index in indices]
    full_gate = len(data["allowed_xyz"])
    if full_gate != data["allowed_cells"]:
        raise ValueError(f"Full eligible gate count differs: {source}")
    indices = _even_indices(full_gate, gate_limit)
    data["allowed_xyz"] = [data["allowed_xyz"][index] for index in indices]
    data["allowed_probabilities"] = [data["allowed_probabilities"][index]
                                     for index in indices]
    data["display_allowed_cells"] = len(data["allowed_xyz"])
    data["display_counts"] = {key: len(data[key]) for key in counts}
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    destination.write_text(_viewer_markup(payload), encoding="utf-8")
    return {"frame": data["frame"], "fault": data["fault_type"],
            "source_bytes": source.stat().st_size,
            "lite_bytes": destination.stat().st_size,
            "full_gate": full_gate,
            "shown_gate": data["display_allowed_cells"],
            "full_clouds": counts, "shown_clouds": data["display_counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cloud-limit", type=int, default=6000)
    parser.add_argument("--gate-limit", type=int, default=4000)
    args = parser.parse_args()
    if args.cloud_limit < 1 or args.gate_limit < 1:
        parser.error("Display limits must be positive")
    sources = sorted(path for path in args.input_root.glob("*.html")
                     if path.name != "index.html")
    if not sources:
        parser.error(f"No frame HTML files under {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        destination = args.output_root / source.name
        row = compact_frame(source, destination,
                            cloud_limit=args.cloud_limit,
                            gate_limit=args.gate_limit)
        rows.append(row)
        print(f"{row['frame']} {row['fault']}: "
              f"{row['source_bytes'] / 1e6:.2f} MB -> "
              f"{row['lite_bytes'] / 1e6:.2f} MB", flush=True)
    links = "\n".join(
        f'<li><a target="viewer" href="{html.escape(path.name, quote=True)}">'
        f'{html.escape(row["frame"])} · {html.escape(row["fault"])}'
        f'</a></li>'
        for path, row in zip(sources, rows))
    first = html.escape(sources[0].name, quote=True)
    index = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
             "<title>Light Stage II 3D viewer</title>"
             "<style>body{margin:1rem;background:#101820;color:#edf4fa;"
             "font:16px system-ui}a{color:#88ceff}iframe{width:100%;"
             "height:85vh;border:1px solid #526474}</style>"
             "<h1>Light Stage II 3D samples</h1>"
             "<p>Display point counts are reduced for speed. Metrics and "
             "the original PLY/NPZ files use full-resolution predictions.</p>"
             f"<ul>{links}</ul><iframe name='viewer' src='{first}' "
             "title='Stage II sample'></iframe></html>")
    (args.output_root / "index.html").write_text(index, encoding="utf-8")
    (args.output_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
