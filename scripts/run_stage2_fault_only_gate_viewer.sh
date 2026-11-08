#!/usr/bin/env bash
# Full-voxel 3D view of the radar gate and the injected-fault-only active gate.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
FAULT_REGIONS=${FAULT_REGIONS:-$DATA/stage2_mild_fog2_injector_regions_20261106}
OUT=${OUT:-$DATA/stage2_conditional_cleanstart_faultonly_12vox_1p6m_20261108/active_region_viewer}
SPLIT=${SPLIT:-val}
PER_FAULT=${PER_FAULT:-2}
PORT=${PORT:-8768}

for required in "$PY" "$RAW/$SPLIT/manifest.json" \
  "$SUPPORT/$SPLIT/manifest.json" "$MILD/$SPLIT/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" -u -m scripts.cache_stage2_fault_regions \
  --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
  --output-root "$FAULT_REGIONS" --split "$SPLIT"

"$PY" -u -m scripts.visualize_stage2_fault_only_gate \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  --fault-region-root "$FAULT_REGIONS" --output-root "$OUT" \
  --split "$SPLIT" --per-fault "$PER_FAULT"

url="http://127.0.0.1:$PORT/index.html"
printf 'Open in Firefox: %s\n' "$url"
"$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory "$OUT" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$server_pid" 2>/dev/null || {
  echo 'Viewer server failed to start. Try another PORT.' >&2
  exit 1
}
if command -v firefox >/dev/null 2>&1; then
  firefox --new-tab "$url" >/dev/null 2>&1 &
else
  xdg-open "$url" >/dev/null 2>&1 || true
fi
printf 'Viewer server is running. Press Ctrl+C to stop it.\n'
wait "$server_pid"
