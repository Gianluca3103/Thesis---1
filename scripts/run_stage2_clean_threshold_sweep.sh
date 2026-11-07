#!/usr/bin/env bash
# Full 1,296-frame one-step clean-LiDAR occupancy sweep for best and last.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
OUT=${OUT:-$DATA/stage2_clean_denoiser_12vox_1p6m_batch16_20261107}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" "$OUT/best.ckpt" "$OUT/last.ckpt" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for name in best last; do
  printf 'Sweeping %s checkpoint over every clean validation frame.\n' "$name"
  "$PY" -u -m scripts.evaluate_stage2_clean_denoiser_thresholds \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --support-cache-root "$SUPPORT" --lpcr-checkpoint "$LPCR" \
    --checkpoint "$OUT/$name.ckpt" \
    --output-json "$OUT/threshold_sweep_clean_$name.json" \
    --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/threshold_sweep_clean_$name.log"
done
