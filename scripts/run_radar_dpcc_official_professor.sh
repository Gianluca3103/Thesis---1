#!/usr/bin/env bash
# Pinned upstream D-PCC on the existing full-frame >=12/1.6 m filtered radar.
set -euo pipefail

BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
ROOT=${ROOT:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109}
FILTERED=${FILTERED:-$ROOT/radar_self_autoencoder_12vox_1p6m_20261110/filtered_radar}
OUT=${OUT:-$ROOT/radar_dpcc_official_12vox_1p6m_20261111}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
REV=d738f59284e505166d26127dbe21cc71e2fa62c9
UPSTREAM=$OUT/upstream/D-PCC
EPOCHS=${EPOCHS:-200}
PHASE=${1:-all}

mkdir -p "$OUT" "$BASE/rtmp"
export TMPDIR="$BASE/rtmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for required in "$PY" "$REPO/scripts/train_radar_dpcc_official.py" \
  "$REPO/scripts/official_dpcc_radar_data.py" \
  "$FILTERED/train/manifest.json" "$FILTERED/val/manifest.json"; do
  [[ -s "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done

setup() {
  if [[ ! -d "$UPSTREAM/.git" ]]; then
    mkdir -p "$(dirname "$UPSTREAM")"
    git clone --quiet https://github.com/yunhe20/D-PCC.git "$UPSTREAM"
  fi
  git -C "$UPSTREAM" checkout --quiet --detach "$REV"
  [[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$REV" ]] || exit 1
  "$PY" "$REPO/scripts/patch_official_dpcc_radar.py" "$UPSTREAM" \
    --manifest "$OUT/upstream_radar_patch.json"
  git -C "$UPSTREAM" diff > "$OUT/upstream_radar_patch.diff"

  if ! "$PY" -c 'import compressai, einops' >/dev/null 2>&1; then
    "$PY" -m pip install --only-binary=:all: 'compressai==1.2.8' 'einops==0.3.2'
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    for root in /usr/local/cuda /usr/local/cuda-12.1 /usr/local/cuda-12.2; do
      if [[ -x "$root/bin/nvcc" ]]; then
        export CUDA_HOME="$root" PATH="$root/bin:$PATH"
        break
      fi
    done
  fi
  command -v nvcc >/dev/null 2>&1 || {
    echo 'CUDA toolkit nvcc is required to compile original D-PCC CUDA operators.' >&2
    exit 1
  }
  echo "PyTorch: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
  echo "nvcc: $(nvcc --version | tail -n 1)"
  if ! compgen -G "$UPSTREAM/models/pointops/pointops_cuda*.so" >/dev/null; then
    (cd "$UPSTREAM/models/pointops" && "$PY" setup.py build_ext --inplace)
  fi
  if ! compgen -G "$UPSTREAM/models/Chamfer3D/chamfer_3D*.so" >/dev/null; then
    (cd "$UPSTREAM/models/Chamfer3D" && "$PY" setup.py build_ext --inplace)
  fi
  export PYTHONPATH="$UPSTREAM:$REPO:$UPSTREAM/models/pointops:$UPSTREAM/models/Chamfer3D${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -c 'import torch, compressai, einops, pointops_cuda, chamfer_3D; from models.autoencoder import AutoEncoder; print("Official D-PCC CUDA modules ready")'
}

ready() {
  [[ -s "$OUT/upstream_radar_patch.json" ]] || {
    echo 'Run setup first' >&2; exit 1;
  }
  export PYTHONPATH="$UPSTREAM:$REPO:$UPSTREAM/models/pointops:$UPSTREAM/models/Chamfer3D${PYTHONPATH:+:$PYTHONPATH}"
}

smoke() {
  ready
  "$PY" -u -m scripts.train_radar_dpcc_official \
    --filtered-root "$FILTERED" --output-root "$OUT/smoke" \
    --epochs 1 --train-limit 1 --val-limit 1 \
    2>&1 | tee "$OUT/smoke.log"
}

train() {
  ready
  "$PY" -u -m scripts.train_radar_dpcc_official \
    --filtered-root "$FILTERED" --output-root "$OUT/model" \
    --epochs "$EPOCHS" 2>&1 | tee -a "$OUT/train.log"
}

evaluate() {
  ready
  for name in best_voxel_iou best_chamfer last; do
    [[ -s "$OUT/model/$name.ckpt" ]] || continue
    for split in val test; do
      [[ -s "$FILTERED/$split/manifest.json" ]] || continue
      "$PY" -u -m scripts.train_radar_dpcc_official \
        --phase evaluate --filtered-root "$FILTERED" --output-root "$OUT/model" \
        --checkpoint "$OUT/model/$name.ckpt" --split "$split" \
        --report-json "$OUT/${name}_${split}.json" \
        2>&1 | tee "$OUT/${name}_${split}.log"
    done
  done
}

case "$PHASE" in
  setup) setup ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  all) setup; smoke; train; evaluate ;;
  *) echo "Usage: $0 [setup|smoke|train|evaluate|all]" >&2; exit 2 ;;
esac
