#!/usr/bin/env bash
# Rebuild only the previously selected viewer frames with every eligible voxel.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
OUT=${OUT:-$DATA/radar_geometry_14rules_mild_fog2_20261107}
PORT=${PORT:-8766}

for required in "$PY" "$RAW/val/manifest.json" "$MILD/val/manifest.json" \
  "$OUT/all_validation.json" "$REPO/scripts/render_full_radar_support_viewer.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" -u -m scripts.render_full_radar_support_viewer \
  --report "$OUT/all_validation.json" \
  --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
  --output-root "$OUT/viewer_full"

printf 'Viewer URL: http://127.0.0.1:%s/index.html\n' "$PORT"
"$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory "$OUT/viewer_full" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$server_pid" 2>/dev/null || { echo 'Viewer server failed to start' >&2; exit 1; }
xdg-open "http://127.0.0.1:$PORT/index.html" || true
printf 'Viewer server is running. Press Ctrl+C to stop it.\n'
wait "$server_pid"
