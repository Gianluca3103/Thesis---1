#!/usr/bin/env bash
# Pure clean-LiDAR epsilon denoising diagnostic on the 12-voxel/1.6 m gate.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
OUT=${OUT:-$DATA/stage2_clean_denoiser_12vox_1p6m_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
CONDITION_FAULTY_RADAR=${CONDITION_FAULTY_RADAR:-0}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi

for required in "$PY" "$LPCR" "$RAW/train/manifest.json" \
  "$RAW/val/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
if [[ "$CONDITION_FAULTY_RADAR" == 1 ]]; then
  for required in "$MILD/train/manifest.json" "$MILD/val/manifest.json"; do
    [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
  done
fi
mkdir -p "$OUT" "$SUPPORT" "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf '1/3: precompute or reuse the 12-radar-voxel/1.6 m masks.\n'
"$PY" -u -m scripts.cache_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$SUPPORT" --split both --radius-m 1.6 \
  --min-distinct-radar-voxels 12 --num-workers 2 \
  2>&1 | tee "$OUT/cache_support.log"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --lpcr-checkpoint "$LPCR"
  --batch-size "${BATCH_SIZE:-8}" --num-workers "${WORKERS:-2}"
  --device cuda)
if [[ "$CONDITION_FAULTY_RADAR" == 1 ]]; then
  common+=(--faulty-root "$MILD" --condition-faulty-radar)
fi
printf '2/3: run a 32-train/8-validation frame GPU smoke test.\n'
if [[ ! -s "$OUT/smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_stage2_clean_denoiser \
    "${common[@]}" --output-root "$OUT/smoke" \
    --train-limit 32 --val-limit 8 --epochs 1 \
    2>&1 | tee "$OUT/smoke.log"
fi

printf '3/3: train on all clean frames and validate on every clean frame.\n'
resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
fi
"$PY" -u -m scripts.train_stage2_clean_denoiser \
  "${common[@]}" --output-root "$OUT" --epochs "${EPOCHS:-25}" \
  "${resume[@]}" 2>&1 | tee -a "$OUT/train.log"
printf 'Best checkpoint: %s\n' "$OUT/best.ckpt"
printf 'Epoch metrics: %s\n' "$OUT/history.jsonl"
