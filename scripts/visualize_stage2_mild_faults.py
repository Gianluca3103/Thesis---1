"""Inspect cached Stage-II mild faults against their matching clean LiDAR scans.

Creates standalone interactive HTML views and full-resolution PLY files. No
model checkpoint, radar data, GPU, or VoD source files are needed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.radar_lidar_stage2.mild_fault_data import (
    MILD_FAULT_ENCODING, MILD_FAULT_VERSION,
    MILD_FOG2_VERSION, decode_mild_xyz, fault_types_for_version, frame_seed,
)


def _find_mild_root(search_root: Path, split: str) -> Path:
    matches = []
    for path in search_root.rglob("manifest.json"):
        if path.parent.name != split:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("version") in (MILD_FAULT_VERSION, MILD_FOG2_VERSION):
            matches.append(path.parent.parent)
    if len(matches) != 1:
        found = "\n".join(str(path) for path in matches) or "none"
        raise ValueError(f"Expected one mild-fault cache under {search_root}; found:\n{found}\n"
                         "Pass --mild-root explicitly if there are multiple caches.")
    return matches[0]


def _removed_mask(clean: np.ndarray, faulty: np.ndarray) -> np.ndarray:
    """Mark missing clean XYZ, accounting for duplicate point multiplicity."""
    clean = np.ascontiguousarray(clean, dtype=np.float32)
    faulty = np.ascontiguousarray(faulty, dtype=np.float32)
    if clean.ndim != 2 or faulty.ndim != 2 or clean.shape[1:] != (3,) or faulty.shape[1:] != (3,):
        raise ValueError("Expected clean and faulty XYZ [N,3]")
    if not np.isfinite(clean).all() or not np.isfinite(faulty).all():
        raise ValueError("Point cloud contains NaN or Inf")
    if not len(clean):
        if len(faulty):
            raise ValueError("Faulty scan has points absent from empty clean scan")
        return np.zeros(0, dtype=bool)
    row_type = np.dtype((np.void, 3 * np.dtype(np.float32).itemsize))
    clean_rows = clean.view(row_type).reshape(-1)
    faulty_rows = faulty.view(row_type).reshape(-1)
    unique_clean, clean_inverse, clean_counts = np.unique(
        clean_rows, return_inverse=True, return_counts=True)
    faulty_counts = np.zeros(len(unique_clean), dtype=np.int64)
    if len(faulty_rows):
        unique_faulty, counts = np.unique(faulty_rows, return_counts=True)
        positions = np.searchsorted(unique_clean, unique_faulty)
        safe = np.minimum(positions, len(unique_clean) - 1)
        if np.any((positions == len(unique_clean)) | (unique_clean[safe] != unique_faulty)):
            raise ValueError("Faulty scan contains XYZ absent from clean LiDAR")
        faulty_counts[positions] = counts
    if np.any(faulty_counts > clean_counts):
        raise ValueError("Faulty scan contains more copies of a point than clean LiDAR")
    order = np.argsort(clean_rows, kind="stable")
    sorted_rows = clean_rows[order]
    group_start = np.maximum.accumulate(np.where(
        np.r_[True, sorted_rows[1:] != sorted_rows[:-1]], np.arange(len(clean)), 0))
    occurrence = np.empty(len(clean), dtype=np.int64)
    occurrence[order] = np.arange(len(clean)) - group_start
    return occurrence >= faulty_counts[clean_inverse]


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    records = np.empty(len(xyz), dtype=[(axis, "<f4") for axis in "xyz"])
    for index, axis in enumerate("xyz"):
        records[axis] = xyz[:, index]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def _display_sample(xyz: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(xyz) <= limit:
        return xyz
    indices = np.random.default_rng(seed).choice(len(xyz), limit, replace=False)
    return xyz[np.sort(indices)]


def _write_html(path: Path, row: dict, clean: np.ndarray, faulty: np.ndarray,
                removed: np.ndarray, max_display_points: int, seed: int) -> None:
    clouds = [clean, faulty, removed]
    sampled = [_display_sample(points, max_display_points,
                               frame_seed(seed + index, row["split"], row["frame_id"]))
               for index, points in enumerate(clouds)]
    bounds = clean if len(clean) else np.zeros((1, 3), dtype=np.float32)
    data = {**row, "clouds": [np.round(points, 3).tolist() for points in sampled],
            "display_counts": [len(points) for points in sampled],
            "minimum": bounds.min(axis=0).tolist(),
            "maximum": bounds.max(axis=0).tolist()}
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mild LiDAR fault viewer</title>
<style>
body{margin:0;padding:14px;background:#101820;color:#eef5fa;font:14px system-ui,sans-serif}
a{color:#9dcafb}h1{font-size:20px;margin:0 0 8px}p{color:#b9c8d6;margin:6px 0 12px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:6px 10px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
section{min-width:0;background:#182633;border:1px solid #3b5164;border-radius:5px;padding:6px}
h2{font-size:14px;margin:0 0 5px}canvas{width:100%;height:42vh;min-height:280px;background:#0c1620;touch-action:none;cursor:grab}
@media(max-width:800px){.panels{display:block}section{margin-bottom:8px}}
</style>
<a href="index.html">← All examples</a><h1 id="title"></h1>
<p id="counts"></p><p>The clean scan is a reference. The faulty scan is the cached Stage II input. Removed points are clean XYZ returns missing from that input; this page shows no model prediction.</p>
<div class="controls"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset view</button>
<span>Drag to rotate · Shift-drag to pan · wheel to zoom · all panels share one camera</span></div>
<div id="panels" class="panels"></div><p id="downloads"></p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} frame ${d.frame_id} · ${d.fault_type}`;
document.getElementById('counts').textContent=`${d.clean_points.toLocaleString()} clean points · ${d.retained_points.toLocaleString()} retained · ${d.removed_points.toLocaleString()} removed (${(100*d.removed_points/Math.max(1,d.clean_points)).toFixed(1)}%)`;
const links=document.getElementById('downloads');for(const name of ['clean','faulty','removed']){const a=document.createElement('a');a.href=`${d.stem}_${name}.ply`;a.textContent=`Download full ${name} PLY`;a.style.marginRight='18px';links.append(a);}
const names=['Clean reference','Faulty input','Removed points','Overlay: retained + removed'];
const panes=document.getElementById('panels'),canvases=[],headings=[];
for(const name of names){const s=document.createElement('section'),h=document.createElement('h2'),c=document.createElement('canvas');h.textContent=name;s.append(h,c);panes.append(s);canvases.push(c);headings.push(h);}
const mid=d.minimum.map((x,i)=>(x+d.maximum[i])/2),extent=Math.hypot(...d.minimum.map((x,i)=>d.maximum[i]-x))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
function project(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
function plot(ctx,points,w,h,s,color){ctx.fillStyle=color;for(const p of points){const [x,y]=project(p,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}}
function draw(c,i){const ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const s=.95*Math.min(w,h)*view.zoom/extent;if(i===0)plot(ctx,d.clouds[0],w,h,s,'#aeb9c4');if(i===1||i===3)plot(ctx,d.clouds[1],w,h,s,'#65d798');if(i===2||i===3)plot(ctx,d.clouds[2],w,h,s,'#ff8b6c');headings[i].textContent=names[i]+(i<3?` · ${d.display_counts[i].toLocaleString()} shown`:'');}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;canvases.forEach(draw);});}
function zoom(k){view.zoom=Math.max(.1,Math.min(80,view.zoom*k));redraw();}
for(const c of canvases){let last=null;c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});c.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});for(const event of ['pointerup','pointercancel'])c.addEventListener(event,()=>{last=null;});c.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001));},{passive:false});}
document.getElementById('zin').onclick=()=>zoom(1.6);document.getElementById('zout').onclick=()=>zoom(1/1.6);document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", payload)
    path.write_text(markup, encoding="utf-8")


def render(mild_root: Path, output_root: Path, split: str, per_fault: int,
           frame_ids: list[str] | None, seed: int, max_display_points: int) -> list[dict]:
    manifest_path = mild_root / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("version")
    fault_types = fault_types_for_version(version)
    if manifest.get("split") != split:
        raise ValueError(f"Wrong {split} manifest: {manifest_path}")
    assignment = manifest["frame_faults"]
    if frame_ids:
        unknown = set(frame_ids) - set(assignment)
        if unknown:
            raise ValueError(f"Frame IDs absent from manifest: {sorted(unknown)}")
        selected = frame_ids
    else:
        rng = random.Random(seed)
        selected = []
        for fault_type in fault_types:
            candidates = [frame_id for frame_id in manifest["frame_ids"]
                          if assignment[frame_id] == fault_type
                          and (mild_root / split / f"{frame_id}.npz").is_file()]
            selected.extend(rng.sample(candidates, min(per_fault, len(candidates))))
    if not selected:
        raise ValueError(f"No cached mild-fault frames found in {mild_root / split}")
    output_root.mkdir(parents=True, exist_ok=True)
    clean_root = Path(manifest["data_cache_root"]) / split / "clean_lidar"
    rows = []
    for frame_id in selected:
        fault_type = assignment[frame_id]
        cache_path = mild_root / split / f"{frame_id}.npz"
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            expected = {"version": version, "encoding": MILD_FAULT_ENCODING,
                        "frame_id": frame_id, "split": split, "fault_type": fault_type,
                        "source_manifest_sha256": manifest["source_manifest_sha256"]}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Mild cache metadata differs: {cache_path}")
            faulty = decode_mild_xyz(archive["points_bytes"], int(metadata["retained_points"]))
        saved = torch.load(clean_root / f"{frame_id}.pt", map_location="cpu", weights_only=True)
        tensor = saved.get("points")
        if (saved.get("frame_id") != frame_id or not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2 or tensor.shape[1] != 4):
            raise ValueError(f"Invalid matching clean scan for {frame_id}")
        clean = np.ascontiguousarray(tensor[:, :3].numpy(), dtype=np.float32)
        if len(clean) != metadata["clean_points"]:
            raise ValueError(f"Clean point count differs for {frame_id}")
        removed = clean[_removed_mask(clean, faulty)]
        stem = f"{split}_{frame_id}_{fault_type}"
        row = {"split": split, "frame_id": frame_id, "fault_type": fault_type,
               "clean_points": len(clean), "retained_points": len(faulty),
               "removed_points": len(removed), "stem": stem, "html": f"{stem}.html"}
        if len(removed) != len(clean) - len(faulty):
            raise AssertionError(f"Missing point count mismatch for {frame_id}")
        for name, xyz in (("clean", clean), ("faulty", faulty), ("removed", removed)):
            _write_ply(output_root / f"{stem}_{name}.ply", xyz)
        _write_html(output_root / row["html"], row, clean, faulty, removed,
                    max_display_points, seed)
        rows.append(row)
        print(f"{frame_id}: {fault_type}, {len(clean):,} clean, {len(faulty):,} retained, "
              f"{len(removed):,} removed", flush=True)
    cards = "\n".join(
        f'<li><a href="{html.escape(row["html"], quote=True)}">'
        f'{html.escape(row["fault_type"])} · {html.escape(row["frame_id"])}'
        f'</a> — {row["removed_points"]:,}/{row["clean_points"]:,} points removed</li>'
        for row in rows)
    index = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
             "<title>Mild LiDAR faults</title><style>body{font:16px system-ui;max-width:780px;"
             "margin:35px auto;padding:0 18px;background:#101820;color:#eef5fa}"
             "a{color:#9dcafb}li{margin:12px 0}</style>"
             f"<h1>Stage II mild LiDAR faults · {html.escape(split)}</h1>"
             "<p>One cached fault per frame. Open an example to compare its clean scan, "
             "faulty input, and removed points in 3D.</p><ul>" + cards + "</ul></html>")
    (output_root / "index.html").write_text(index, encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Open: {output_root / 'index.html'}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mild-root", type=Path)
    source.add_argument("--search-root", type=Path,
                        help="Find the one mild-fault cache under this output directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--per-fault", type=int, default=2)
    parser.add_argument("--frame-id", action="append", dest="frame_ids")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--max-display-points", type=int, default=12000)
    args = parser.parse_args()
    if args.per_fault < 1 or args.max_display_points < 1:
        parser.error("--per-fault and --max-display-points must be positive")
    root = args.mild_root or _find_mild_root(args.search_root, args.split)
    print(f"Mild cache: {root}", flush=True)
    render(root, args.output_root, args.split, args.per_fault,
           args.frame_ids, args.seed, args.max_display_points)


if __name__ == "__main__":
    main()
