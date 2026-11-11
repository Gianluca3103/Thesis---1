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

select_cuda_compiler() {
  local torch_cuda root output version chosen="" fallback="" seen=""
  local -a roots=("$BASE/cuda-12.1" /usr/local/cuda-12.1 /opt/cuda-12.1 \
    /usr/local/cuda-12.2 /opt/cuda-12.2 \
    "${CUDA_HOME:-}" /usr/local/cuda-12.* /opt/cuda-12.* /usr/local/cuda)
  local -a path_nvcc=()
  readarray -t path_nvcc < <(type -a -p nvcc 2>/dev/null || true)
  for output in "${path_nvcc[@]}"; do
    roots+=("$(dirname "$(dirname "$output")")")
  done
  torch_cuda=$("$PY" -c 'import torch; print(torch.version.cuda or "")')
  [[ -n "$torch_cuda" ]] || { echo 'PyTorch has no CUDA build' >&2; exit 1; }
  for root in "${roots[@]}"; do
    [[ -n "$root" && -x "$root/bin/nvcc" ]] || continue
    output=$("$root/bin/nvcc" --version)
    [[ "$output" =~ release[[:space:]]+([0-9]+\.[0-9]+) ]] || continue
    version=${BASH_REMATCH[1]}
    seen+=" $root=$version"
    if [[ "$version" == "$torch_cuda" ]]; then
      chosen="$root"
      break
    fi
    if [[ "${version%%.*}" == "${torch_cuda%%.*}" && -z "$fallback" ]]; then
      fallback="$root"
    fi
  done
  [[ -n "$chosen" ]] || chosen="$fallback"
  if [[ -z "$chosen" ]]; then
    echo "PyTorch needs CUDA $torch_cuda nvcc; found:${seen:- none}." >&2
    echo 'Install CUDA Toolkit 12.1 (or another 12.x toolkit) alongside 11.8, then rerun setup.' >&2
    exit 1
  fi
  export CUDA_HOME="$chosen" PATH="$chosen/bin:$PATH"
  echo "Selected CUDA_HOME=$CUDA_HOME for PyTorch CUDA $torch_cuda"
  "$CUDA_HOME/bin/nvcc" --version | tail -n 1
}

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
    # Ubuntu 20.04 has glibc 2.31; the published 1.2.8 Linux wheel requires
    # glibc 2.34. Build the pinned source distribution against this venv.
    "$PY" -m pip install --no-cache-dir --no-binary=compressai \
      --index-url "${DPCC_PYPI_INDEX:-https://pypi.org/simple}" \
      'compressai==1.2.8' 'einops==0.3.2'
  fi
  select_cuda_compiler
  echo "PyTorch: $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
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
    --epochs 1 --first-usable-smoke \
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
