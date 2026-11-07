#!/usr/bin/env bash
# Render compact 3D HTML, PNG, NPZ, and PLY samples for val and test.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
OUT=${OUT:-$DATA/stage2_clean_denoiser_12vox_1p6m_batch16_20261107}
NAME=${NAME:-last}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" "$OUT/$NAME.ckpt" \
  "$OUT/threshold_sweep_clean_$NAME.json" \
  "$RAW/val/manifest.json" "$RAW/test/manifest.json" \
  "$SUPPORT/val/manifest.json" "$SUPPORT/test/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$DATA/stage2_clean_tmp" "$OUT/plot_config"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp" MPLCONFIGDIR="$OUT/plot_config"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for split in val test; do
  root="$OUT/clean_samples_${NAME}_${split}"
  "$PY" -u -m scripts.visualize_stage2_clean_denoiser \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --support-cache-root "$SUPPORT" --lpcr-checkpoint "$LPCR" \
    --checkpoint "$OUT/$NAME.ckpt" \
    --sweep-json "$OUT/threshold_sweep_clean_$NAME.json" \
    --split "$split" --candidate-count 24 --count 6 \
    --output-root "$root" --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/clean_samples_${NAME}_${split}.log"
  printf '3D index: %s\n' "$root/index.html"
  printf 'PNG examples: %s\n' "$root"
done
