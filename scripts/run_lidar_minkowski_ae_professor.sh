#!/usr/bin/env bash
# Clean LiDAR -> sparse latent -> clean voxel occupancy on the Stage-II grid.
set -euo pipefail

PHASE=${1:-all}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
REPO=${REPO:-$BASE/Thesis-1-minkowski-lidar-ae}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-$BASE/sensor_fusion_outputs/minkowski_lidar_ae_stage2grid_20261125}
CACHE=${CACHE:-$OUT/clean_voxels}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-1}
WIDTH=${WIDTH:-32}
LEVELS=${LEVELS:-1}
WORKERS=${WORKERS:-2}
THRESHOLD=${THRESHOLD:-0.5}

[[ -x "$PY" && -d "$VOD" && -s "$REPO/scripts/train_lidar_minkowski_ae.py" ]] || {
  echo 'Missing Python, VoD data, or the Minkowski LiDAR trial code' >&2; exit 1;
}
mkdir -p "$OUT" "$BASE/rtmp"
export TMPDIR="$BASE/rtmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TORCH_LIB=$("$PY" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PY" -c 'import torch, MinkowskiEngine as ME; print("MinkowskiEngine ready; CUDA:", torch.cuda.get_device_name())'

cache() {
  "$PY" -u -m scripts.train_lidar_minkowski_ae \
    --phase cache --vod-root "$VOD" --cache-root "$CACHE" \
    --output-root "$OUT/model"
}

smoke() {
  if [[ -s "$OUT/smoke/last.ckpt" ]]; then
    echo 'Smoke checkpoint already exists; skipping smoke'
    return
  fi
  "$PY" -u -m scripts.train_lidar_minkowski_ae \
    --phase train --vod-root "$VOD" --cache-root "$CACHE" \
    --output-root "$OUT/smoke" --epochs 1 --batch-size 1 \
    --width "$WIDTH" --levels "$LEVELS" --num-workers "$WORKERS" \
    --train-limit 8 --val-limit 4 --threshold "$THRESHOLD"
}

train() {
  resume=()
  [[ ! -s "$OUT/model/last.ckpt" ]] || resume=(--resume "$OUT/model/last.ckpt")
  "$PY" -u -m scripts.train_lidar_minkowski_ae \
    --phase train --vod-root "$VOD" --cache-root "$CACHE" \
    --output-root "$OUT/model" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --width "$WIDTH" --levels "$LEVELS" \
    --num-workers "$WORKERS" --threshold "$THRESHOLD" "${resume[@]}"
}

evaluate() {
  [[ -s "$OUT/model/best.ckpt" ]] || {
    echo 'Train the model first; best.ckpt is missing' >&2; exit 1;
  }
  for split in val test; do
    "$PY" -u -m scripts.train_lidar_minkowski_ae \
      --phase evaluate --vod-root "$VOD" --cache-root "$CACHE" \
      --output-root "$OUT/model" --checkpoint "$OUT/model/best.ckpt" \
      --split "$split" --batch-size 1 --num-workers "$WORKERS" \
      --threshold "$THRESHOLD"
  done
}

case "$PHASE" in
  cache) cache ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  all) cache; smoke; train; evaluate ;;
  *) echo "Usage: bash $0 [cache|smoke|train|evaluate|all]" >&2; exit 2 ;;
esac
