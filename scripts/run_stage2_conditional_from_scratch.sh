#!/usr/bin/env bash
# Conditional Stage II training with frozen pretrained LPCR and LiDAR encoder.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
FAULT_REGIONS=${FAULT_REGIONS:-}
if [[ ${BCE:-0} != 0 ]]; then
  printf 'The one-step BCE trial was removed; use DECODER_LOSS=1\n' >&2
  exit 1
fi
DECODER_LOSS=${DECODER_LOSS:-0}
INIT=${INIT:-}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-64}
GRADIENT_STEPS=${GRADIENT_STEPS:-4}
OCCUPANCY_LOSS_WEIGHT=${OCCUPANCY_LOSS_WEIGHT:-0.05}
FALSE_POSITIVE_WEIGHT=${FALSE_POSITIVE_WEIGHT:-2.0}
if [[ "$DECODER_LOSS" != 0 && "$DECODER_LOSS" != 1 ]]; then
  printf 'DECODER_LOSS must be 0 or 1\n' >&2; exit 1
fi
if [[ "$DECODER_LOSS" == 1 && -z "$FAULT_REGIONS" ]]; then
  printf 'DECODER_LOSS=1 requires FAULT_REGIONS\n' >&2; exit 1
fi
if [[ -n "$INIT" && "$DECODER_LOSS" != 1 ]]; then
  printf 'INIT requires DECODER_LOSS=1\n' >&2; exit 1
fi
if [[ -n "$FAULT_REGIONS" ]]; then
  if [[ "$DECODER_LOSS" == 1 ]]; then
    OUT=${OUT:-$DATA/stage2_conditional_cleanstart_faultonly_decoder_loss_12vox_1p6m_20261108}
  else
    OUT=${OUT:-$DATA/stage2_conditional_cleanstart_faultonly_12vox_1p6m_20261108}
  fi
else
  OUT=${OUT:-$DATA/stage2_conditional_scratch_12vox_1p6m_20261108}
fi
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ "$DECODER_LOSS" == 1 ]]; then
  BATCH_SIZE=${BATCH_SIZE:-8}
  EPOCHS=${EPOCHS:-25}
  RECON_VAL_EVERY=${RECON_VAL_EVERY:-5}
else
  BATCH_SIZE=${BATCH_SIZE:-16}
  EPOCHS=${EPOCHS:-125}
  RECON_VAL_EVERY=1
fi
VAL_STEPS=${VAL_STEPS:-64}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" "$RAW/test/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json" \
  "$SUPPORT/test/manifest.json" "$MILD/train/manifest.json" \
  "$MILD/val/manifest.json" "$MILD/test/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
if [[ -n "$INIT" && ! -s "$INIT" ]]; then
  printf 'Missing initialization checkpoint: %s\n' "$INIT" >&2; exit 1
fi

mkdir -p "$OUT" "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

region_arg=()
decoder_arg=()
smoke_decoder_arg=()
if [[ "$DECODER_LOSS" == 1 ]]; then
  decoder_arg=(--generated-occupancy-loss
    --rollout-steps "$ROLLOUT_STEPS"
    --gradient-steps "$GRADIENT_STEPS"
    --occupancy-loss-weight "$OCCUPANCY_LOSS_WEIGHT"
    --false-positive-weight "$FALSE_POSITIVE_WEIGHT"
    --reconstruction-val-every "$RECON_VAL_EVERY")
  smoke_decoder_arg=(--rollout-steps 4 --gradient-steps 2)
  if [[ -n "$INIT" ]]; then
    decoder_arg+=(--initialize-from "$INIT")
  fi
fi
if [[ -n "$FAULT_REGIONS" ]]; then
  printf 'Caching synthetic injector footprints for train, validation, and test.\n'
  "$PY" -u -m scripts.cache_stage2_fault_regions \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --faulty-root "$MILD" --output-root "$FAULT_REGIONS" --split all \
    2>&1 | tee "$OUT/cache_fault_regions.log"
  region_arg=(--fault-region-root "$FAULT_REGIONS")
fi

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --faulty-root "$MILD"
  "${region_arg[@]}" "${decoder_arg[@]}"
  --lpcr-checkpoint "$LPCR" --batch-size "$BATCH_SIZE"
  --num-workers 2 --device cuda)

printf 'Smoke test: 32 train / 8 validation frames; 4 reverse steps.\n'
if [[ ! -s "$OUT/smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_stage2_conditional_from_scratch \
    "${common[@]}" --output-root "$OUT/smoke" \
    --train-limit 32 --val-limit 8 --validation-steps 4 --epochs 1 \
    "${smoke_decoder_arg[@]}" \
    2>&1 | tee "$OUT/smoke.log"
fi

resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
fi
printf 'Training %s epochs; noise loss each epoch; fixed-0.5 reconstruction validation every %s epochs with %s steps.\n' \
  "$EPOCHS" "$RECON_VAL_EVERY" "$VAL_STEPS"
"$PY" -u -m scripts.train_stage2_conditional_from_scratch \
  "${common[@]}" --output-root "$OUT" --epochs "$EPOCHS" \
  --validation-steps "$VAL_STEPS" "${resume[@]}" \
  2>&1 | tee -a "$OUT/train.log"

printf 'Sweeping thresholds once on the best validation-IoU checkpoint, then scoring official validation and test.\n'
"$PY" -u -m scripts.evaluate_stage2_conditional_reconstruction \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  "${region_arg[@]}" \
  --lpcr-checkpoint "$LPCR" --checkpoint "$OUT/best_iou.ckpt" \
  --steps "$VAL_STEPS" --num-workers 2 --device cuda \
  --output-json "$OUT/best_full_noise_${VAL_STEPS}steps_val_test.json" \
  2>&1 | tee "$OUT/best_full_noise_${VAL_STEPS}steps_val_test.log"

printf 'History: %s\n' "$OUT/history.jsonl"
printf 'Report: %s\n' "$OUT/best_full_noise_${VAL_STEPS}steps_val_test.json"
