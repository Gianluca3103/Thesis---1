"""View raw radar-only confidence beside radar and forward-sector clean LiDAR.

Every radar point and confidence candidate is shown. Clean LiDAR uses the
same fixed calibration-aware radar sector as cross-modal training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.radar_fov import (
    RADAR_FOV_SPEC, RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M,
)
from models.radar_lidar_crossmodal_v1.meta_occupancy import radar_only_probability_map
from models.radar_lidar_crossmodal_v1.train_meta_occupancy import (
    VERSION as META_VERSION, load_meta_student,
)
from models.radar_lidar_crossmodal_v1.train_student import load_student


def _write_ply(path: Path, xyz: np.ndarray, confidence: np.ndarray | None = None) -> None:
    fields = [(axis, "<f4") for axis in "xyz"]
    if confidence is not None:
        fields.append(("confidence", "<f4"))
    records = np.empty(len(xyz), dtype=fields)
    for column, axis in enumerate("xyz"):
        records[axis] = xyz[:, column]
    if confidence is not None:
        records["confidence"] = confidence
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              + ("property float confidence\n" if confidence is not None else "")
              + "end_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def _write_html(path: Path, frame_id: str, epoch: int, radar: np.ndarray,
                lidar: np.ndarray, centers: np.ndarray, confidence: np.ndarray,
                *, checkpoint_uses_fov: bool,
                support_radius_m: float | None = None) -> None:
    clouds = [
        {"name": "Radar", "xyz": np.round(radar, 3).tolist()},
        {"name": "Clean LiDAR in forward radar sector", "xyz": np.round(lidar, 3).tolist()},
        {"name": "Raw student confidence", "xyz": np.round(centers, 3).tolist(),
         "score": np.round(confidence, 4).tolist()},
    ]
    all_xyz = np.concatenate((radar, lidar, centers), axis=0)
    minimum = all_xyz.min(axis=0).tolist()
    maximum = all_xyz.max(axis=0).tolist()
    packed = json.dumps({"frame": frame_id, "epoch": epoch, "clouds": clouds,
                         "radar_half_fov_deg": RADAR_HALF_FOV_DEG,
                         "radar_max_range_m": RADAR_MAX_RANGE_M,
                         "checkpoint_uses_fov": checkpoint_uses_fov,
                         "support_radius_m": support_radius_m,
                         "minimum": minimum, "maximum": maximum},
                        separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raw student confidence</title>
<style>
body{margin:0;padding:12px;background:#101820;color:#edf4fa;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}p{color:#b9c8d6;margin:4px 0 10px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:12px 0}
input[type=range]{width:240px}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
section{min-width:0;background:#182633;border:1px solid #3b5164;border-radius:5px;padding:6px}
h2{font-size:14px;margin:0 0 5px}canvas{width:100%;height:70vh;min-height:420px;background:#0c1620;touch-action:none;cursor:grab}
.legend{margin-top:8px;color:#b9c8d6}.ramp{display:inline-block;vertical-align:middle;width:135px;height:12px;background:linear-gradient(90deg,#2862be,#51cad1,#ffe173,#f56868)}
@media(max-width:900px){.panels{display:block}section{margin-bottom:8px}canvas{height:55vh;min-height:300px}}
</style>
<h1 id="title"></h1>
<p id="map-note"></p>
<p id="range-note"></p>
<div class="controls"><label>Minimum confidence <input id="threshold" type="range" min="0" max="1" step="0.01" value="0"><output id="value">0.00</output></label>
<button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset view</button></div>
<div id="panels" class="panels"></div>
<div class="legend">Confidence 0 <span class="ramp"></span> 1 · model score, not measured surface accuracy</div>
<p>Drag to rotate; Shift-drag to pan; wheel to zoom. All panels share the same camera.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const data=JSON.parse(document.getElementById('data').textContent), panes=document.getElementById('panels');
document.getElementById('title').textContent=`Frame ${data.frame} · student epoch ${data.epoch}`;
document.getElementById('map-note').textContent=`Raw radar-only confidence at fine voxel centers${data.support_radius_m===null?'':` within ${data.support_radius_m.toFixed(1)} m of a radar point`}. No interpolation or point limit. Clean LiDAR is shown only for comparison; it is not an inference input.`;
document.getElementById('range-note').textContent=`Clean LiDAR reference: ${data.clouds[1].xyz.length.toLocaleString()} points in the calibrated forward sector (±${data.radar_half_fov_deg.toFixed(0)}° azimuth, up to ${data.radar_max_range_m.toFixed(0)} m). This is a fixed crop, not measured free space or per-frame radar visibility.${data.checkpoint_uses_fov?'':' This checkpoint predates the training crop; the green reference is cropped for display only.'}`;
const canvases=[], headings=[];
for(const cloud of data.clouds){const section=document.createElement('section'),h=document.createElement('h2'),canvas=document.createElement('canvas');
  h.textContent=`${cloud.name}: ${cloud.xyz.length.toLocaleString()} points`;section.append(h,canvas);panes.append(section);canvases.push(canvas);headings.push(h);}
const mid=data.minimum.map((a,i)=>(a+data.maximum[i])/2);
const extent=Math.hypot(...data.minimum.map((a,i)=>data.maximum[i]-a))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
const slider=document.getElementById('threshold');
function color(v){const stops=[[40,98,190],[81,202,209],[255,225,115],[245,104,104]],a=Math.max(0,Math.min(.9999,v))*3,i=Math.floor(a),t=a-i;
  return `rgb(${stops[i].map((n,k)=>Math.round(n*(1-t)+stops[i+1][k]*t)).join(',')})`;}
function project(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
  const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
function draw(canvas,index){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
  if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const cloud=data.clouds[index],scale=.88*Math.min(w,h)*view.zoom/extent,threshold=Number(slider.value);
  let selected=0;for(let i=0;i<cloud.xyz.length;i++){if(index===2&&cloud.score[i]<threshold)continue;selected++;
    const [x,y]=project(cloud.xyz[i],w,h,scale);if(x<0||x>=w||y<0||y>=h)continue;
    ctx.fillStyle=index===0?'#ffab40':index===1?'#45b975':color(cloud.score[i]);
    const size=index===2?2:1.4;ctx.fillRect(x-size/2,y-size/2,size,size);}
  headings[index].textContent=`${cloud.name}: ${cloud.xyz.length.toLocaleString()} total${index===2?`, ${selected.toLocaleString()} above score ${threshold.toFixed(2)}`:''}`;
  const origin=project([0,0,0],w,h,scale);for(const [name,tint,end] of [['X','#eb7777',[5,0,0]],['Y','#78d18b',[0,5,0]],['Z','#9eabf0',[0,0,2]]]){
    const tip=project(end,w,h,scale);ctx.strokeStyle=tint;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...tip);ctx.stroke();ctx.fillStyle=tint;ctx.fillText(name,tip[0],tip[1]);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;canvases.forEach(draw);});}
function zoom(mult,x,y,w,h){const next=Math.max(.1,Math.min(80,view.zoom*mult)),k=next/view.zoom;
  view.panX=x-w/2-(x-w/2-view.panX)*k;view.panY=y-h/2-(y-h/2-view.panY)*k;view.zoom=next;}
for(const canvas of canvases){let last=null;canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});
  canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];
    if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});
  for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>{last=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);redraw();},{passive:false});}
slider.addEventListener('input',()=>{document.getElementById('value').textContent=Number(slider.value).toFixed(2);redraw();});
document.getElementById('zin').onclick=()=>{zoom(1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};
document.getElementById('zout').onclick=()=>{zoom(1/1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};
document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", packed)
    path.write_text(html, encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-id", nargs="+")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of split frames to render when --frame-id is omitted")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    version = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("version")
    meta = version in (META_VERSION, META_VERSION + "_student")
    loader = load_meta_student if meta else load_student
    student, saved = loader(args.checkpoint, args.device)
    dataset = VoDStage1Dataset(args.vod_root, args.split,
                               radar_variant=saved["radar_variant"],
                               frame_ids=args.frame_id, include_clean=True,
                               lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
                               lidar_radar_max_range_m=RADAR_MAX_RANGE_M)
    if args.frame_id is None:
        dataset.frames = dataset.frames[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    found = {frame.frame_id for frame in dataset.frames}
    if args.frame_id and (missing := set(args.frame_id) - found):
        raise ValueError(f"Frames absent from {args.split}: {sorted(missing)}")
    for row in dataset:
        batch = collate_stage1([row])
        radar_input = batch["radar"].to(args.device)
        radar_valid = batch["radar_valid"].to(args.device)
        if meta:
            output = radar_only_probability_map(
                student, radar_input, radar_valid, saved["radius_m"])
            coords = output["confidence_coords"].detach().cpu().numpy()
            confidence = output["confidence_probabilities"].detach().cpu().numpy().reshape(-1)
            grid = output["grid_metadata"]
        else:
            output = student(radar_input, radar_valid)
            sites = output.confidence_logits
            coords = sites.coords.detach().cpu().numpy()
            confidence = sites.features.sigmoid().detach().cpu().numpy().reshape(-1)
            grid = output.exported()["grid_metadata"]
        origin = np.asarray(grid["origin_xyz_m"], dtype=np.float32)
        size = np.asarray(grid["voxel_xyz_m"], dtype=np.float32)
        centers = origin + (coords[:, [3, 2, 1]].astype(np.float32) + 0.5) * size
        radar = row["radar"][:, :3].numpy()
        lidar = row["clean_lidar"][:, :3].numpy()
        stem = args.output_root / f"{row['frame_id']}_student_epoch_{saved['epoch']:03d}"
        _write_ply(stem.with_name(stem.name + "_radar.ply"), radar)
        _write_ply(stem.with_name(stem.name + "_clean_lidar.ply"), lidar)
        _write_ply(stem.with_name(stem.name + "_confidence.ply"), centers, confidence)
        html = stem.with_suffix(".html")
        _write_html(html, row["frame_id"], saved["epoch"], radar, lidar,
                    centers, confidence,
                    checkpoint_uses_fov=saved.get("lidar_fov") == RADAR_FOV_SPEC,
                    support_radius_m=saved["radius_m"] if meta else None)
        support_note = f", radar support {saved['radius_m']:.1f} m" if meta else ""
        print(f"{row['frame_id']}: {len(centers)} raw confidence cells, "
              f"{len(radar)} radar points, {len(lidar)} clean LiDAR points "
              f"within forward ±{RADAR_HALF_FOV_DEG:.0f}° and {RADAR_MAX_RANGE_M:.0f} m"
              f"{support_note} | {html}",
              flush=True)


if __name__ == "__main__":
    main()
