"""Show the exact radar returns surviving the autoencoder's input filter.

This viewer needs no trained checkpoint. It reads the same fixed radar frames
used for encoder pretraining and applies the model's input-validity filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_voxel_gate import filter_radar_valid


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in xyz:
            stream.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _write_html(path: Path, frame_id: str, kept: np.ndarray,
                rejected: np.ndarray, outside_count: int) -> None:
    shown = np.concatenate((kept, rejected), axis=0)
    if len(shown):
        minimum = shown.min(axis=0).tolist()
        maximum = shown.max(axis=0).tolist()
    else:
        minimum, maximum = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    payload = json.dumps({
        "frame": frame_id,
        "kept": np.round(kept, 3).tolist(),
        "rejected": np.round(rejected, 3).tolist(),
        "outside": outside_count,
        "minimum": minimum, "maximum": maximum,
    }, separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Filtered radar encoder input</title>
<style>
body{margin:0;padding:14px;background:#0b1721;color:#eaf1f7;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 5px}p{margin:5px 0 11px;color:#b8c9d7}
label{margin-right:18px}button{background:#213746;color:#eaf1f7;border:1px solid #617889;border-radius:4px;padding:5px 9px;cursor:pointer;margin-right:7px}
canvas{display:block;width:100%;height:78vh;min-height:420px;background:#07121b;touch-action:none;cursor:grab;border:1px solid #3a5264;border-radius:5px}
</style>
<h1 id="title"></h1>
<p id="summary"></p>
<p>Every cyan point is an original radar return passed to the encoder. Red points were inside the encoder grid but removed by the ≥12 distinct radar voxels within 1.6 m rule. The filter is evaluated once against the original frame.</p>
<div><label><input id="show-kept" type="checkbox" checked> Retained radar input</label>
<label><input id="show-rejected" type="checkbox"> Rejected in-grid returns</label>
<button id="reset">Reset view</button><button id="zin">Zoom in</button><button id="zout">Zoom out</button></div>
<p>Drag to rotate; Shift-drag to pan; wheel to zoom. X is forward, Y lateral, Z vertical.</p>
<canvas id="cloud"></canvas>
<script id="data" type="application/json">__DATA__</script>
<script>
(()=>{
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('cloud');
document.getElementById('title').textContent=`Frame ${d.frame} · radar encoder input`;
document.getElementById('summary').textContent=`${d.kept.length.toLocaleString()} retained returns · ${d.rejected.length.toLocaleString()} removed inside grid · ${d.outside.toLocaleString()} outside encoder grid`;
const mid=d.minimum.map((v,i)=>(v+d.maximum[i])/2),span=Math.max(1,...d.minimum.map((v,i)=>d.maximum[i]-v));
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},v={...initial};
function projection(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch);
  const u=cy*x-sy*y,q=sy*x+cy*y;return [w/2+v.panX+u*s,h/2+v.panY-(cp*z-sp*q)*s];}
function draw(){const ratio=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;
  c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,w,h);const scale=.86*Math.min(w,h)*v.zoom/span;
  function cloud(points,color,size){ctx.fillStyle=color;for(const p of points){const [x,y]=projection(p,w,h,scale);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-size/2,y-size/2,size,size);}}
  if(document.getElementById('show-rejected').checked)cloud(d.rejected,'#ee6a62',2);
  if(document.getElementById('show-kept').checked)cloud(d.kept,'#55d6e6',2.5);
  const o=projection([0,0,0],w,h,scale);for(const [name,color,end] of [['X','#ee7777',[5,0,0]],['Y','#7bd58a',[0,5,0]],['Z','#a8b9ff',[0,0,2]]]){
    const t=projection(end,w,h,scale);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(...o);ctx.lineTo(...t);ctx.stroke();ctx.fillStyle=color;ctx.fillText(name,t[0],t[1]);}}
let pending=false;function redraw(){if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;draw();});}}
function zoom(f,x,y){const next=Math.max(.1,Math.min(80,v.zoom*f)),k=next/v.zoom;v.panX=x-c.clientWidth/2-(x-c.clientWidth/2-v.panX)*k;v.panY=y-c.clientHeight/2-(y-c.clientHeight/2-v.panY)*k;v.zoom=next;redraw();}
let last=null;c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});
c.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(e.shiftKey){v.panX+=dx;v.panY+=dy;}else{v.yaw+=dx*.008;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*.008));}redraw();});
for(const event of ['pointerup','pointercancel'])c.addEventListener(event,()=>{last=null;});
c.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY);},{passive:false});
document.getElementById('show-kept').addEventListener('change',redraw);document.getElementById('show-rejected').addEventListener('change',redraw);
document.getElementById('reset').onclick=()=>{Object.assign(v,initial);redraw();};
document.getElementById('zin').onclick=()=>zoom(1.6,c.clientWidth/2,c.clientHeight/2);
document.getElementById('zout').onclick=()=>zoom(1/1.6,c.clientWidth/2,c.clientHeight/2);
window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", payload)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path,
                        help="Compatible fixed-frame cache; otherwise read VoD radar directly")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-id", nargs="+")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=3103)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=False, data_cache_root=args.data_cache_root)
    id_to_index = {frame.frame_id: i for i, frame in enumerate(dataset.frames)}
    if args.frame_id:
        unknown = set(args.frame_id) - id_to_index.keys()
        if unknown:
            parser.error(f"Unknown frame IDs: {sorted(unknown)}")
        indices = [id_to_index[frame_id] for frame_id in args.frame_id]
    else:
        rng = np.random.default_rng(args.random_seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.limit, len(dataset)),
                                    replace=False).tolist())
    args.output_root.mkdir(parents=True, exist_ok=True)
    minimum = np.asarray(GRID.minimum_xyz)
    maximum = np.asarray(GRID.maximum_xyz)
    index_rows = []
    for index in indices:
        row = dataset[index]
        radar = row["radar"].float()
        valid = torch.ones((1, len(radar)), dtype=torch.bool)
        keep = filter_radar_valid(radar[None], valid)[0].numpy()
        raw = radar.numpy()
        inside = ((raw[:, :3] >= minimum) & (raw[:, :3] < maximum)).all(axis=1)
        kept = raw[keep]
        rejected = raw[inside & ~keep]
        frame_id = row["frame_id"]
        np.savez_compressed(args.output_root / f"{frame_id}.npz",
                            raw_radar=raw, retained_mask=keep,
                            grid_minimum_xyz=minimum, grid_maximum_xyz=maximum)
        _write_ply(args.output_root / f"{frame_id}_encoder_input.ply", kept[:, :3])
        _write_html(args.output_root / f"{frame_id}.html", frame_id,
                    kept[:, :3], rejected[:, :3], int((~inside).sum()))
        index_rows.append((frame_id, len(raw), len(kept), f"{frame_id}.html"))
        print(f"{frame_id}: {len(raw):,} raw returns, {len(kept):,} encoder inputs; "
              f"{len(rejected):,} removed in grid, {(~inside).sum():,} outside grid",
              flush=True)
    links = "\n".join(
        f'<li><a href="{name}">{frame_id}</a>: {kept:,}/{raw:,} retained · '
        f'<a href="{frame_id}_encoder_input.ply">PLY</a> · '
        f'<a href="{frame_id}.npz">NPZ</a></li>'
        for frame_id, raw, kept, name in index_rows)
    (args.output_root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Radar encoder inputs</title>'
        '<style>body{background:#0b1721;color:#eaf1f7;font:16px system-ui;'
        'padding:24px}a{color:#76d5ed}li{margin:10px 0}</style>'
        '<h1>Radar encoder inputs after ≥12-voxel / 1.6 m filtering</h1>'
        f'<ul>{links}</ul>', encoding="utf-8")
    print(f"Viewer: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
