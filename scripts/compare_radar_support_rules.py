"""Compare seventeen geometry-only radar support rules on mild-fault validation.

Radar returns are deduplicated into radar voxels before neighbor counting.
These are full Stage-II ROI gates, before the synthetic injector footprint.
The HTML viewer and reported counts use every qualifying voxel.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.hierarchical_gate import missing_target, quantize
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import unique_radar_voxel_centers
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import MILD_FOG2_TYPES, MILD_FOG2_VERSION
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.compare_radar_geometry_radii import CENTERS, SHAPE, _empty_counts, _one, _summarize


RULES = tuple((count, radius) for radius, counts in ((0.8, (1, 2, 3, 4, 5)),
                                                  (1.6, (1, 2, 3, 6, 8, 10, 12)),
                                                  (3.2, (1, 2, 3, 4, 5)))
              for count in counts)
RULE_NAMES = tuple(f"{count}@{radius:g}m" for count, radius in RULES)
OBSERVED_BIT = 1 << len(RULES)
assert OBSERVED_BIT < 1 << 31
NEAREST_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
NEAREST_COLUMN = {count: column for column, count in enumerate(NEAREST_COUNTS)}
TAIL_COUNTS = NEAREST_COUNTS[5:]
CELL_COUNT = len(CENTERS)


def nearest_distances(radar) -> np.ndarray:
    """First five radar-voxel distances, plus 6/8/10/12 inside 1.6 m."""
    radar_centers = unique_radar_voxel_centers(radar)
    result = np.full((CELL_COUNT, len(NEAREST_COUNTS)), np.inf, dtype=np.float64)
    if not len(radar_centers):
        return result
    tree = cKDTree(radar_centers)
    for start in range(0, CELL_COUNT, 65536):
        stop = min(start + 65536, CELL_COUNT)
        distances, _ = tree.query(CENTERS[start:stop], k=5,
                                  distance_upper_bound=3.2 + 1e-9, workers=1)
        result[start:stop, :5] = distances
    # Only cells already within five radar voxels at 1.6 m can satisfy 6–12.
    eligible = np.flatnonzero(result[:, 4] <= 1.6 + 1e-9)
    for start in range(0, len(eligible), 65536):
        positions = eligible[start:start + 65536]
        tail, _ = tree.query(CENTERS[positions], k=TAIL_COUNTS,
                             distance_upper_bound=1.6 + 1e-9, workers=1)
        result[positions, 5:] = tail
    return result


def selected_mask(distances: np.ndarray, count: int, radius: float) -> np.ndarray:
    if (count not in NEAREST_COLUMN or radius <= 0 or
            (count > 5 and radius > 1.6)):
        raise ValueError("Invalid radar rule")
    return distances[:, NEAREST_COLUMN[count]] <= radius + 1e-9


def flat_indices(coords) -> np.ndarray:
    if not len(coords):
        return np.empty(0, dtype=np.int64)
    zyx = coords.detach().cpu().numpy()
    return np.ravel_multi_index(zyx.T, SHAPE)


def _sample(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.sort(rng.choice(len(points), limit, replace=False))]


def _xyz(coords) -> np.ndarray:
    if not len(coords):
        return np.empty((0, 3), dtype=np.float32)
    zyx = coords.detach().cpu().numpy()
    return (np.asarray(GRID.minimum_xyz) +
            (zyx[:, ::-1] + 0.5) * np.asarray(GRID.voxel_xyz)).astype(np.float32)


def _viewer_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complete radar geometry gates</title>
<style>
body{margin:0;background:#101820;color:#eaf3f8;font:14px system-ui,sans-serif}
main{display:grid;grid-template-columns:310px 1fr;min-height:100vh}
aside{padding:14px;background:#182633;overflow:auto}h1{font-size:19px;margin:0 0 8px}
p{color:#b9c8d6;line-height:1.4}a{color:#9dcafb}label{display:block;margin:8px 0}
button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:6px 10px;cursor:pointer;margin:3px}
#view{width:100%;height:100vh;background:#0c1620;touch-action:none;cursor:grab}
#slice{width:280px;height:280px;max-width:100%;image-rendering:pixelated;background:#0c1620}
.rule{border-top:1px solid #3b5164;padding:4px 0}.small{font-size:12px;color:#b9c8d6}
@media(max-width:800px){main{display:block}#view{height:65vh}aside{max-height:35vh}}
</style><main><aside><a href="index.html">← All frames</a><h1 id="title"></h1>
<p>Every blue square is a Stage II ROI voxel supported by the checked radar rules. All qualifying voxels are shown, with no display sampling. No model prediction or injector footprint is used.</p>
<button id="none">Clear rules</button><button id="all">All rules</button><button id="reset">Reset view</button>
<div id="rules"></div><label><input type="checkbox" id="exclude"> Exclude observed faulty LiDAR voxels (Stage II additions)</label>
<div id="layers"></div><p id="counts"></p>
<label for="z">Exact horizontal voxel slice: <span id="zlabel"></span></label>
<input id="z" type="range" min="0" max="15" value="8"><canvas id="slice" width="160" height="160"></canvas>
<p class="small">The slice draws exact 0.2 m × 0.2 m voxel squares at the selected height. Drag the 3D view to rotate, Shift-drag to pan, or use the wheel to zoom.</p>
</aside><canvas id="view"></canvas></main>
<script id="data" type="application/json">__DATA__</script>
<script>
(()=>{
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('view'),ctx=c.getContext('2d');
document.getElementById('title').textContent=`Frame ${d.frame_id} · ${d.fault}`;
const binary=atob(d.packed_candidates),raw=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)raw[i]=binary.charCodeAt(i);
const records=new DataView(raw.buffer),indices=new Uint32Array(d.candidate_count),membership=new Uint32Array(d.candidate_count);
for(let i=0;i<d.candidate_count;i++){indices[i]=records.getUint32(i*8,true);membership[i]=records.getUint32(i*8+4,true);}
const rules=document.getElementById('rules'),checks=[];
for(let i=0;i<d.rules.length;i++){let row=document.createElement('div');row.className='rule';let label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=i===0;box.onchange=redraw;checks.push(box);label.append(box,document.createTextNode(' '+d.rules[i].label));row.append(label);let sub=document.createElement('div');sub.className='small';sub.textContent=`${d.rules[i].raw_cells.toLocaleString()} supported · ${d.rules[i].candidates.toLocaleString()} additions · ${d.rules[i].missing_recall}% missing recall`;row.append(sub);rules.append(row);}
const layers=document.getElementById('layers'),layerNames=[['faulty','Faulty LiDAR','#65d798'],['missing','Missing clean LiDAR','#ff856a'],['radar','Radar voxel centers','#ffd166']];
const layerChecks={};for(const [key,name] of layerNames){let label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=key==='radar';box.onchange=redraw;layerChecks[key]=box;label.append(box,document.createTextNode(' '+name));layers.append(label);}
const exclude=document.getElementById('exclude'),zslider=document.getElementById('z'),zlabel=document.getElementById('zlabel');exclude.onchange=redraw;zslider.oninput=redraw;
const slice=document.getElementById('slice'),sctx=slice.getContext('2d'),[nz,ny,nx]=d.grid.shape;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},v={...initial},mid=[16,0,0],extent=46;
function project(p,w,h,s){let x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch),u=cy*x-sy*y,q=sy*x+cy*y;return[w/2+v.panX+u*s,h/2+v.panY-(cp*z-sp*q)*s];}
function plot(points,w,h,s,color,size){ctx.fillStyle=color;for(const p of points){let [x,y]=project(p,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-size/2,y-size/2,size,size);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(draw);}
function draw(){pending=false;const ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const s=.95*Math.min(w,h)*v.zoom/extent,box=Math.max(2,Math.min(14,.23*s)),cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch);let bits=0;for(let i=0;i<checks.length;i++)if(checks[i].checked)bits|=1<<i;const zi=Number(zslider.value),image=sctx.createImageData(nx,ny);let selected=0,sliceCount=0;ctx.fillStyle='#7cb8ff';for(let i=0;i<indices.length;i++){const m=membership[i];if(!(m&bits)||(exclude.checked&&(m&d.observed_bit)))continue;selected++;const flat=indices[i],x=flat%nx,y=Math.floor(flat/nx)%ny,z=Math.floor(flat/(nx*ny));if(z===zi){sliceCount++;const at=4*((ny-1-y)*nx+x);image.data[at]=124;image.data[at+1]=184;image.data[at+2]=255;image.data[at+3]=255;}const dx=d.grid.minimum[0]+(x+.5)*d.grid.size[0]-mid[0],dy=d.grid.minimum[1]+(y+.5)*d.grid.size[1]-mid[1],dz=d.grid.minimum[2]+(z+.5)*d.grid.size[2]-mid[2],u=cy*dx-sy*dy,q=sy*dx+cy*dy,px=w/2+v.panX+u*s,py=h/2+v.panY-(cp*dz-sp*q)*s;if(px>=0&&px<w&&py>=0&&py<h)ctx.fillRect(px-box/2,py-box/2,box,box);}sctx.putImageData(image,0,0);for(const [key,,color] of layerNames)if(layerChecks[key].checked)plot(d[key],w,h,s,color,key==='radar'?3:2);zlabel.textContent=`z=${(d.grid.minimum[2]+(zi+.5)*d.grid.size[2]).toFixed(3)} m · ${sliceCount.toLocaleString()} cells`;document.getElementById('counts').textContent=`${selected.toLocaleString()} complete ${exclude.checked?'addition':'support'} voxels shown · ${checks.filter(x=>x.checked).length} rules selected`;}
c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);c.last=[e.clientX,e.clientY]});c.addEventListener('pointermove',e=>{if(!c.last)return;const dx=e.clientX-c.last[0],dy=e.clientY-c.last[1];c.last=[e.clientX,e.clientY];if(e.shiftKey){v.panX+=dx;v.panY+=dy}else{v.yaw+=dx*.008;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*.008))}redraw()});for(const name of ['pointerup','pointercancel'])c.addEventListener(name,()=>{c.last=null});c.addEventListener('wheel',e=>{e.preventDefault();v.zoom=Math.max(.1,Math.min(80,v.zoom*Math.exp(-e.deltaY*.001)));redraw()},{passive:false});
document.getElementById('none').onclick=()=>{checks.forEach(x=>x.checked=false);redraw()};document.getElementById('all').onclick=()=>{checks.forEach(x=>x.checked=true);redraw()};document.getElementById('reset').onclick=()=>{Object.assign(v,initial);redraw()};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", data)


def write_viewer_frame(path: Path, *, frame_id: str, fault: str, distances: np.ndarray,
                       observed_index: np.ndarray, clean, faulty, missing, radar,
                       display_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # The largest requested rule contains every other rule.
    union = selected_mask(distances, 1, 3.2)
    candidate_index = np.flatnonzero(union)
    membership = np.zeros(len(candidate_index), dtype=np.uint32)
    rules = []
    missing_index = flat_indices(missing)
    for i, (count, radius) in enumerate(RULES):
        mask = selected_mask(distances, count, radius)
        membership |= (mask[candidate_index].astype(np.uint32) << i)
        candidates = int(mask.sum()) - int(mask[observed_index].sum())
        eligible = int(mask[missing_index].sum())
        rules.append({"label": f"≥{count} radar voxel{'s' if count != 1 else ''} within {radius:g} m",
                      "raw_cells": int(mask.sum()), "candidates": candidates,
                      "missing_recall": round(100 * eligible / max(len(missing), 1), 2)})
    membership |= (np.isin(candidate_index, observed_index,
                            assume_unique=True).astype(np.uint32) << len(RULES))
    packed = np.empty(len(candidate_index),
                      dtype=np.dtype([("index", "<u4"), ("membership", "<u4")]))
    packed["index"] = candidate_index
    packed["membership"] = membership
    radar_xyz = unique_radar_voxel_centers(radar)
    payload = {"frame_id": frame_id, "fault": fault, "rules": rules,
               "candidate_count": len(candidate_index),
               "observed_bit": OBSERVED_BIT,
               "packed_candidates": base64.b64encode(packed.tobytes()).decode("ascii"),
               "grid": {"minimum": GRID.minimum_xyz, "size": GRID.voxel_xyz,
                        "shape": GRID.shape_zyx},
               "faulty": np.round(_sample(_xyz(faulty), display_points, rng), 2).tolist(),
               "missing": np.round(_sample(_xyz(missing), display_points, rng), 2).tolist(),
               "radar": np.round(_sample(radar_xyz, display_points, rng), 2).tolist()}
    path.write_text(_viewer_html(payload), encoding="utf-8")


def _viewer_index(rows: list[dict]) -> str:
    links = "".join(f'<li><a href="{html.escape(row["file"], quote=True)}">'
                    f'{html.escape(row["frame_id"])} · {html.escape(row["fault"])}</a></li>'
                    for row in rows)
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Radar support rule viewer</title><style>body{background:#101820;"
            "color:#eaf3f8;font:16px system-ui;padding:24px}a{color:#9dcafb}"
            f"li{{margin:10px}}</style><h1>{len(RULES)} radar support rules</h1>"
            "<p>Open a frame to toggle the gates in 3D. These are geometry-only "
            "candidate sets, not reconstructed points.</p><ul>" + links + "</ul></html>")


def evaluate(dataset, indices: list[int], faulty_root: Path, frame_faults: dict[str, str],
             *, viewer_root: Path | None = None, per_fault: int = 2,
             display_points: int = 5000,
             seed: int = 3103) -> tuple[dict, list[dict]]:
    groups = {name: {rule: _empty_counts() for rule in RULE_NAMES}
              for name in ("all", *MILD_FOG2_TYPES)}
    rng = random.Random(seed)
    viewer_ids = set()
    if viewer_root is not None:
        for fault in MILD_FOG2_TYPES:
            candidates = [dataset.frames[i].frame_id for i in indices
                          if frame_faults[dataset.frames[i].frame_id] == fault]
            viewer_ids.update(rng.sample(candidates, min(per_fault, len(candidates))))
        viewer_root.mkdir(parents=True, exist_ok=True)
    viewer_rows = []
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        collate_fn=_one, num_workers=0)
    for row in tqdm(loader, desc=f"{len(RULES)} radar geometry rules", unit="frame"):
        frame_id = row["frame_id"]
        fault = frame_faults[frame_id]
        faulty = load_faulty(faulty_root, "val", frame_id)
        clean = quantize(row["clean_lidar"][:, :3], 1)
        missing = missing_target(row["clean_lidar"][:, :3], faulty)
        observed = quantize(faulty, 1)
        clean_index, missing_index, observed_index = map(flat_indices,
                                                        (clean, missing, observed))
        distances = nearest_distances(row["radar"])
        for (count, radius), name in zip(RULES, RULE_NAMES):
            mask = selected_mask(distances, count, radius)
            supported = int(mask.sum())
            supported_clean = int(mask[clean_index].sum())
            supported_missing = int(mask[missing_index].sum())
            observed_inside = int(mask[observed_index].sum())
            for group in ("all", fault):
                record = groups[group][name]
                record["frames"] += 1
                record["radar_supported_cells"] += supported
                record["clean_lidar_voxels"] += len(clean)
                record["supported_clean_lidar_voxels"] += supported_clean
                record["missing_lidar_voxels"] += len(missing)
                record["supported_missing_lidar_voxels"] += supported_missing
                record["addition_candidates"] += supported - observed_inside
        if frame_id in viewer_ids:
            file = f"{frame_id}_{fault}.html"
            write_viewer_frame(viewer_root / file, frame_id=frame_id, fault=fault,
                               distances=distances, observed_index=observed_index,
                               clean=clean, faulty=observed, missing=missing,
                               radar=row["radar"], display_points=display_points,
                               seed=seed + sum(frame_id.encode("utf-8")))
            viewer_rows.append({"frame_id": frame_id, "fault": fault, "file": file})
    if viewer_root is not None:
        (viewer_root / "index.html").write_text(_viewer_index(viewer_rows),
                                                 encoding="utf-8")
    return ({group: {name: _summarize(counts, has_faulty=True)
                    for name, counts in by_rule.items()}
             for group, by_rule in groups.items()}, viewer_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer-root", type=Path)
    parser.add_argument("--per-fault", type=int, default=2)
    parser.add_argument("--display-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--limit", type=int, help="Smoke test only; default is all validation")
    args = parser.parse_args()
    if any(value < 1 for value in (args.per_fault, args.display_points)) or (
            args.limit is not None and args.limit < 1):
        parser.error("Counts and limit must be positive")
    dataset = paired_dataset(args.vod_root, "val", args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    manifest = json.loads((args.faulty_root / "val" / "manifest.json").read_text(
        encoding="utf-8"))
    if (manifest.get("version") != MILD_FOG2_VERSION or
            manifest.get("radar_variant") != args.radar_variant or
            manifest.get("frame_ids") != [frame.frame_id for frame in dataset.frames]):
        parser.error("Mild-fog2 cache differs from validation dataset")
    indices = list(range(min(len(dataset), args.limit or len(dataset))))
    results, viewers = evaluate(dataset, indices, args.faulty_root,
                                manifest["frame_faults"], viewer_root=args.viewer_root,
                                per_fault=args.per_fault,
                                display_points=args.display_points, seed=args.seed)
    report = {"rules": [{"name": name, "minimum_radar_voxels": count,
                          "radius_m": radius} for name, (count, radius)
                         in zip(RULE_NAMES, RULES)],
              "frame_count": len(indices),
              "frame_ids": [dataset.frames[i].frame_id for i in indices],
              "scope": "Stage-II ROI, raw radar support; additions exclude observed faulty LiDAR voxels; no injector footprint",
              "radar_deduplication": "global radar voxel grid",
              "voxel_size_xyz_m": GRID.voxel_xyz,
              "results": results, "viewer_frames": viewers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(indices)} of {len(dataset)} validation frames.", flush=True)
    for group, by_rule in results.items():
        print(f"\n{group}", flush=True)
        print("Rule          Clean P  Full clean R  Addition P  Full missing R  Additions/frame")
        for name in RULE_NAMES:
            row = by_rule[name]
            clean = row["clean_occupancy"]
            added = row["missing_additions"]
            print(f"{name:12s} {clean['precision']:7.2%} {clean['full_recall']:12.2%} "
                  f"{added['precision']:11.2%} {added['full_recall']:15.2%} "
                  f"{added['mean_addition_candidates_per_frame']:15,.0f}", flush=True)
    print(f"Report: {args.output}", flush=True)
    if args.viewer_root:
        print(f"Viewer: {args.viewer_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
