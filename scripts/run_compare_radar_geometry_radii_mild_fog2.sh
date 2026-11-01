#!/usr/bin/env bash
# Geometry-only 3-distinct-radar-voxel baselines at 1.6 m and 3.2 m.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
MATCH=${MATCH:-$DATA/hierarchical_stage1_mild_fog2_20261101/threshold_sweep_val20/full_roi_new.json}
OUT=${OUT:-$DATA/radar_geometry_3vox_1p6m_3p2m_20261102}

for required in "$PY" "$RAW/val/manifest.json" "$MILD/val/manifest.json" \
  "$REPO/scripts/compare_radar_geometry_radii.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Evaluating every mild-fault validation frame.\n'
"$PY" -u -m scripts.compare_radar_geometry_radii \
  --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
  --output "$OUT/all_validation.json" \
  2>&1 | tee "$OUT/all_validation.log"

if [[ -s "$MATCH" ]]; then
  printf 'Evaluating the previous Stage-I sweep held-out frame IDs.\n'
  "$PY" -u -m scripts.compare_radar_geometry_radii \
    --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
    --match-report "$MATCH" --output "$OUT/matched_heldout.json" \
    2>&1 | tee "$OUT/matched_heldout.log"
else
  printf 'Matched comparison skipped; report not found: %s\n' "$MATCH"
fi

printf 'Results: %s\n' "$OUT"
