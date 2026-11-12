#!/usr/bin/env bash
# Paper-guided 4D-RaDiff VAE on native five-sweep View-of-Delft radar.
set -euo pipefail

PHASE=${1:-all}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
ROOT=${ROOT:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109}
OUT=${OUT:-$ROOT/radiff_vae_vod5_20261112}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
CACHE=$OUT/vod5_radar
UPSTREAM=$OUT/upstream/D-PCC
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-32}
WORKERS=${WORKERS:-2}

mkdir -p "$OUT" "$BASE/rtmp"
export TMPDIR="$BASE/rtmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -x "$PY" && -s "$REPO/scripts/train_radiff_vae.py" ]] || {
  echo "Missing Python or source files in $REPO" >&2; exit 1;
}
cd "$REPO"

cache() {
  for split in train val test; do
    "$PY" -u -m scripts.cache_radiff_vod5 --vod-root "$VOD" \
      --cache-root "$CACHE" --split "$split"
  done
}

setup() {
  REPO="$REPO" OUT="$OUT" PY="$PY" BASE="$BASE" \
    bash "$REPO/scripts/run_radar_dpcc_official_professor.sh" setup
}

ready() {
  for required in "$CACHE/train/manifest.json" "$CACHE/val/manifest.json" \
    "$OUT/upstream_radar_patch.json"; do
    [[ -s "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
  done
  compgen -G "$UPSTREAM/models/pointops/pointops_cuda*.so" >/dev/null || {
    echo 'D-PCC pointops CUDA extension missing; run setup' >&2; exit 1;
  }
  compgen -G "$UPSTREAM/models/Chamfer3D/chamfer_3D*.so" >/dev/null || {
    echo 'D-PCC Chamfer CUDA extension missing; run setup' >&2; exit 1;
  }
  export PYTHONPATH="$UPSTREAM:$REPO:$UPSTREAM/models/pointops:$UPSTREAM/models/Chamfer3D${PYTHONPATH:+:$PYTHONPATH}"
}

smoke() {
  ready
  "$PY" -u -m scripts.train_radiff_vae --cache-root "$CACHE" \
    --output-root "$OUT/smoke" --epochs 1 --batch-size "$BATCH_SIZE" \
    --train-limit 32 --val-limit 16 --num-workers 0 \
    2>&1 | tee "$OUT/smoke.log"
}

train() {
  ready
  "$PY" -u -m scripts.train_radiff_vae --cache-root "$CACHE" \
    --output-root "$OUT/model" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
    2>&1 | tee -a "$OUT/train.log"
}

evaluate() {
  ready
  for split in val test; do
    [[ -s "$CACHE/$split/manifest.json" ]] || continue
    "$PY" -u -m scripts.train_radiff_vae --phase evaluate \
      --cache-root "$CACHE" --output-root "$OUT/model" \
      --checkpoint "$OUT/model/best_voxel_iou.ckpt" --split "$split" \
      --report-json "$OUT/best_voxel_iou_${split}.json" \
      --num-workers "$WORKERS" 2>&1 | tee "$OUT/best_voxel_iou_${split}.log"
  done
}

case "$PHASE" in
  cache) cache ;;
  setup) setup ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  all) cache; setup; smoke; train; evaluate ;;
  *) echo "Usage: $0 [cache|setup|smoke|train|evaluate|all]" >&2; exit 2 ;;
esac
