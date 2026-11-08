#!/usr/bin/env bash
# Render balanced mild-fault test examples for the conditioned one-step model.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
OUT=${OUT:-$DATA/stage2_clean_one_step_faulty_radar_12vox_1p6m_batch16_20261107}
NAME=${NAME:-best_iou}
SPLIT=${SPLIT:-test}
PER_FAULT=${PER_FAULT:-2}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
SOURCE=$OUT/epoch125_snapshot/$NAME.ckpt
if [[ ! -s "$SOURCE" ]]; then
  SOURCE=$OUT/$NAME.ckpt
fi
for required in "$PY" "$LPCR" "$SOURCE" \
  "$RAW/$SPLIT/manifest.json" "$SUPPORT/$SPLIT/manifest.json" \
  "$MILD/$SPLIT/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

cd "$REPO"
mkdir -p "$OUT/plot_config" "$DATA/stage2_clean_tmp"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp" MPLCONFIGDIR="$OUT/plot_config"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SWEEP=$OUT/visual_samples_sweep_${NAME}_${SPLIT}.json
ROOT=$OUT/conditioned_completion_samples_${NAME}_${SPLIT}

"$PY" -u -m scripts.evaluate_stage2_clean_denoiser_thresholds \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  --lpcr-checkpoint "$LPCR" --checkpoint "$SOURCE" \
  --sampling-strategy one_step --output-json "$SWEEP" \
  --num-workers 2 --device cuda

"$PY" -u -m scripts.visualize_stage2_conditioned_completion \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  --lpcr-checkpoint "$LPCR" --checkpoint "$SOURCE" \
  --sweep-json "$SWEEP" --split "$SPLIT" \
  --per-fault "$PER_FAULT" --output-root "$ROOT" \
  --num-workers 2 --device cuda

printf 'Open in Firefox: %s\n' "$ROOT/index.html"
printf 'PNG files: %s\n' "$ROOT"
