#!/usr/bin/env bash
# Fixed radar support intersected with the synthetic injector footprint.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
SUPPORT=${SUPPORT:-$DATA/radar_pruner3vox_1p6m_mild_fog2_20261103/support_masks}
REGIONS=${REGIONS:-$DATA/stage2_mild_fog2_injector_regions_20261106}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
OUT=${OUT:-$DATA/stage2_geometry1p6m_injector_region_epsilon_ray_offset_mild_batch8_20261107}
WORKERS=${WORKERS:-2}
RAY_WORKERS=${RAY_WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
RAY_LABELS=${RAY_LABELS:-}
EPOCHS=${EPOCHS:-25}
VAL_LIMIT=${VAL_LIMIT:-64}
TRAIN_LIMIT=${TRAIN_LIMIT:-0}
EVAL=${EVAL:-1}
EVAL_LIMIT=${EVAL_LIMIT:-512}
SAMPLING_STEPS=${SAMPLING_STEPS:-64}

for required in "$PY" "$LPCR" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$MILD/train/manifest.json" "$MILD/val/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json" \
  "$REGIONS/train/manifest.json" "$REGIONS/val/manifest.json" \
  "$REGIONS/train/regions.json" "$REGIONS/val/regions.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export STAGE2_RAY_WORKERS="$RAY_WORKERS"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'GPU batch size=%s; ray query workers=%s; data loader workers=%s\n' \
  "$BATCH_SIZE" "$RAY_WORKERS" "$WORKERS"

resume=()
log_mode=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
  log_mode=(-a)
fi
ray_args=()
if [[ -n "$RAY_LABELS" ]]; then
  for required in "$RAY_LABELS/train/manifest.json" "$RAY_LABELS/val/manifest.json"; do
    [[ -s "$required" ]] || { printf 'Missing ray label cache: %s\n' "$required" >&2; exit 1; }
  done
  ray_args=(--ray-label-cache-root "$RAY_LABELS")
  printf 'Using cached ray labels: %s\n' "$RAY_LABELS"
fi
limit_args=()
if (( TRAIN_LIMIT > 0 )); then
  limit_args=(--train-limit "$TRAIN_LIMIT")
fi

"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --diffusion-target epsilon --ray-offset-baseline \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --fault-region-root "$REGIONS" "${ray_args[@]}" \
  --lpcr-checkpoint "$LPCR" --output-root "$OUT" \
  --radar-variant radar_20frames_verified_doppler_radial \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
  --lr 0.0002 --val-limit "$VAL_LIMIT" "${limit_args[@]}" \
  --offset-weight 1 --ray-hit-weight 1 --ray-extra-weight 1 \
  --residual-weight 0 --reconstruction-weight 0 \
  --addition-weight 0 --healthy-weight 0 \
  --device cuda "${resume[@]}" \
  2>&1 | tee "${log_mode[@]}" "$OUT/train.log"

printf 'Best checkpoint: %s\n' "$OUT/best.ckpt"
if [[ "$EVAL" == 1 ]]; then
  "$PY" -u -m scripts.evaluate_stage2_geometry_baseline \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
    --fault-region-root "$REGIONS" \
    --checkpoint "$OUT/best.ckpt" \
    --output "$OUT/validation_512_calibrated.json" \
    --limit "$EVAL_LIMIT" --sampling-steps "$SAMPLING_STEPS" \
    --device cuda 2>&1 | tee "$OUT/evaluate.log"
fi
