#!/usr/bin/env bash
# Filter radar once, then train a radar -> same filtered radar point-cloud AE.
set -euo pipefail

BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
ROOT=${ROOT:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109}
PRIOR=${PRIOR:-$ROOT/radar_encoder_decoder_fullgrid12_1p6m_prefilter_100ep_20261109}
RAW=${RAW:-$PRIOR/full_clean_fixed_frames}
ENCODER=${ENCODER:-$PRIOR/radar_pretrain/best_encoder.pth}
OUT=${OUT:-$ROOT/radar_self_autoencoder_12vox_1p6m_20261110}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
TEMP_ROOT=${TEMP_ROOT:-$BASE/rtmp}
CACHE_WORKERS=${CACHE_WORKERS:-4}
WORKERS=${WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-100}
PHASE=${1:-all}

for required in "$REPO/models/radar_lidar_crossmodal_v1/radar_self_autoencoder.py" \
  "$REPO/scripts/cache_radar_self_autoencoder.py" "$PY" "$ENCODER" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$BASE/stage2-openblas-runtime/lib/libopenblas.so.0"; do
  [[ -s "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
mkdir -p "$OUT" "$TEMP_ROOT"
export TMPDIR="$TEMP_ROOT" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export CROSSMODAL_GPU_NEIGHBORS=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"

cache() {
  for split in train val; do
    "$PY" -u -m scripts.cache_radar_self_autoencoder \
      --raw-root "$RAW" --cache-root "$OUT/filtered_radar" \
      --split "$split" --num-workers "$CACHE_WORKERS"
  done
}

train() {
  local resume=()
  if [[ -s "$OUT/model/last.ckpt" ]]; then
    resume=(--resume "$OUT/model/last.ckpt")
    echo "Resuming radar-to-radar AE from $OUT/model/last.ckpt"
  else
    echo "Starting radar-to-radar AE with filtered-radar pretrained encoder"
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_self_autoencoder \
    --raw-root "$RAW" --cache-root "$OUT/filtered_radar" \
    --encoder-checkpoint "$ENCODER" --output-root "$OUT/model" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --num-workers "$WORKERS" --device cuda "${resume[@]}" \
    2>&1 | tee -a "$OUT/train.log"
}

evaluate() {
  test -s "$OUT/model/best.ckpt" || {
    echo "Missing radar self-autoencoder best checkpoint" >&2; exit 1;
  }
  "$PY" -u -m scripts.evaluate_radar_self_autoencoder \
    --raw-root "$RAW" --cache-root "$OUT/filtered_radar" \
    --checkpoint "$OUT/model/best.ckpt" --split val \
    --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
    --output-json "$OUT/validation_threshold_sweep.json" \
    2>&1 | tee "$OUT/validation_threshold_sweep.log"
}

smoke() {
  local resume=()
  if [[ -s "$OUT/smoke/last.ckpt" ]]; then
    echo "Smoke checkpoint already exists: $OUT/smoke/last.ckpt"
    return
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_self_autoencoder \
    --raw-root "$RAW" --cache-root "$OUT/filtered_radar" \
    --encoder-checkpoint "$ENCODER" --output-root "$OUT/smoke" \
    --epochs 1 --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
    --train-limit 32 --val-limit 16 --device cuda \
    2>&1 | tee "$OUT/smoke.log"
}

case "$PHASE" in
  cache) cache ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  all) cache; smoke; train; evaluate ;;
  *) echo "Usage: $0 [cache|smoke|train|evaluate|all]" >&2; exit 2 ;;
esac
