#!/usr/bin/env bash
# Independent radar-to-radar D-PCC adaptation; same filtered returns and metrics.
set -euo pipefail

BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
ROOT=${ROOT:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109}
PRIOR=${PRIOR:-$ROOT/radar_encoder_decoder_fullgrid12_1p6m_prefilter_100ep_20261109}
SELF=${SELF:-$ROOT/radar_self_autoencoder_12vox_1p6m_20261110}
RAW=${RAW:-$PRIOR/full_clean_fixed_frames}
FILTERED=${FILTERED:-$SELF/filtered_radar}
OUT=${OUT:-$ROOT/radar_dpcc_12vox_1p6m_20261110}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
TEMP_ROOT=${TEMP_ROOT:-$BASE/rtmp}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-4}
CACHE_WORKERS=${CACHE_WORKERS:-4}
PHASE=${1:-all}
if [[ ! -s "$FILTERED/train/manifest.json" || ! -s "$FILTERED/val/manifest.json" ]]; then
  FILTERED="$OUT/filtered_radar"
fi

for required in "$REPO/models/radar_lidar_crossmodal_v1/radar_dpcc.py" \
  "$REPO/scripts/cache_radar_dpcc.py" "$REPO/scripts/evaluate_radar_dpcc.py" \
  "$PY" "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$BASE/stage2-openblas-runtime/lib/libopenblas.so.0"; do
  [[ -s "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
mkdir -p "$OUT" "$TEMP_ROOT"
export TMPDIR="$TEMP_ROOT" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"

filtered() {
  for split in train val; do
    "$PY" -u -m scripts.cache_radar_self_autoencoder \
      --raw-root "$RAW" --cache-root "$FILTERED" \
      --split "$split" --num-workers "$CACHE_WORKERS"
  done
}

geometry() {
  for split in train val; do
    "$PY" -u -m scripts.cache_radar_dpcc \
      --raw-root "$RAW" --filtered-root "$FILTERED" \
      --geometry-root "$OUT/fps_geometry" --split "$split" \
      --workers "$CACHE_WORKERS"
  done
}

cache_test() {
  test -s "$RAW/test/manifest.json" || {
    echo "No complete raw official-test manifest at $RAW/test/manifest.json" >&2
    exit 1
  }
  "$PY" -u -m scripts.cache_radar_self_autoencoder \
    --raw-root "$RAW" --cache-root "$FILTERED" \
    --split test --num-workers "$CACHE_WORKERS"
  "$PY" -u -m scripts.cache_radar_dpcc \
    --raw-root "$RAW" --filtered-root "$FILTERED" \
    --geometry-root "$OUT/fps_geometry" --split test \
    --workers "$CACHE_WORKERS"
}

smoke() {
  if [[ -s "$OUT/smoke/last.ckpt" ]]; then
    echo "Smoke checkpoint already exists"
    return
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_dpcc \
    --raw-root "$RAW" --filtered-root "$FILTERED" \
    --geometry-root "$OUT/fps_geometry" --output-root "$OUT/smoke" \
    --epochs 1 --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
    --train-limit 16 --val-limit 8 --device cuda \
    2>&1 | tee "$OUT/smoke.log"
}

train() {
  local resume=()
  if [[ -s "$OUT/model/last.ckpt" ]]; then
    resume=(--resume "$OUT/model/last.ckpt")
    echo "Resuming radar D-PCC from $OUT/model/last.ckpt"
  else
    echo "Starting a fresh radar D-PCC adaptation in $OUT/model"
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_dpcc \
    --raw-root "$RAW" --filtered-root "$FILTERED" \
    --geometry-root "$OUT/fps_geometry" --output-root "$OUT/model" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
    --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/train.log"
}

evaluate() {
  for name in best last; do
    test -s "$OUT/model/$name.ckpt"
    "$PY" -u -m scripts.evaluate_radar_dpcc \
      --raw-root "$RAW" --filtered-root "$FILTERED" \
      --geometry-root "$OUT/fps_geometry" \
      --checkpoint "$OUT/model/$name.ckpt" --split val \
      --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
      --output-json "$OUT/${name}_validation_threshold_sweep.json" \
      2>&1 | tee "$OUT/${name}_validation_threshold_sweep.log"
    if [[ -s "$OUT/fps_geometry/test/manifest.json" ]]; then
      "$PY" -u -m scripts.evaluate_radar_dpcc \
        --raw-root "$RAW" --filtered-root "$FILTERED" \
        --geometry-root "$OUT/fps_geometry" \
        --checkpoint "$OUT/model/$name.ckpt" --split test \
        --selected-validation "$OUT/${name}_validation_threshold_sweep.json" \
        --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
        --output-json "$OUT/${name}_test_selected.json" \
        2>&1 | tee "$OUT/${name}_test_selected.log"
    fi
    if [[ -s "$SELF/validation_threshold_sweep.json" ]]; then
      "$PY" -u -m scripts.compare_radar_dpcc \
        --old "$SELF/validation_threshold_sweep.json" \
        --dpcc "$OUT/${name}_validation_threshold_sweep.json" \
        2>&1 | tee "$OUT/${name}_comparison.log"
    fi
  done
}

case "$PHASE" in
  cache) filtered; geometry ;;
  cache-test) cache_test ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  all) filtered; geometry; smoke; train; evaluate ;;
  *) echo "Usage: $0 [cache|cache-test|smoke|train|evaluate|all]" >&2; exit 2 ;;
esac
