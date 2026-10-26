#!/usr/bin/env bash
# Install an isolated SVEFusion detector environment on the professor's Linux machine.
set -euo pipefail

BASE=${SVE_BASE:-/mnt/3D10B36523559581/Gianluca}
SVE=${1:-$BASE/SVEFusion-9102873}
ENV_PREFIX=${2:-$BASE/.env_svefusion_py38_cu118}
MINIFORGE_PREFIX=${SVE_MINIFORGE_PREFIX:-$BASE/.miniforge_sve}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONSTRAINTS=$REPO/scripts/svefusion_detector_constraints.txt

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'This setup is for the professor machine (Linux x86_64).' >&2
  exit 2
fi
for required in "$SVE/setup.py" "$SVE/requirements.txt" "$SVE/mamba/setup.py" "$CONSTRAINTS"; do
  if [[ ! -s $required ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
for command in curl sha256sum g++; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing system build tool: $command" >&2
    exit 1
  fi
done

if [[ -n ${SVE_CONDA_EXE:-} ]]; then
  CONDA=$SVE_CONDA_EXE
elif command -v conda >/dev/null 2>&1; then
  CONDA=$(command -v conda)
elif [[ -x $MINIFORGE_PREFIX/bin/conda ]]; then
  CONDA=$MINIFORGE_PREFIX/bin/conda
else
  if [[ -e $MINIFORGE_PREFIX ]]; then
    echo "Incomplete Miniforge installation at $MINIFORGE_PREFIX; choose another SVE_MINIFORGE_PREFIX." >&2
    exit 1
  fi
  echo 'Installing Miniforge for the new detector environment...'
  INSTALLER=Miniforge3-Linux-x86_64.sh
  DOWNLOAD_DIR=$(mktemp -d)
  trap 'rm -f "$DOWNLOAD_DIR/$INSTALLER" "$DOWNLOAD_DIR/$INSTALLER.sha256"; rmdir "$DOWNLOAD_DIR"' EXIT
  URL=https://github.com/conda-forge/miniforge/releases/latest/download
  curl -fL --retry 3 "$URL/$INSTALLER" -o "$DOWNLOAD_DIR/$INSTALLER"
  curl -fL --retry 3 "$URL/$INSTALLER.sha256" -o "$DOWNLOAD_DIR/$INSTALLER.sha256"
  (cd "$DOWNLOAD_DIR" && sha256sum -c "$INSTALLER.sha256")
  bash "$DOWNLOAD_DIR/$INSTALLER" -b -p "$MINIFORGE_PREFIX"
  CONDA=$MINIFORGE_PREFIX/bin/conda
fi

if [[ ! -x $ENV_PREFIX/bin/python ]]; then
  if [[ -e $ENV_PREFIX ]]; then
    echo "Incomplete detector environment at $ENV_PREFIX; choose another prefix." >&2
    exit 1
  fi
  echo "Creating Python 3.8 environment at $ENV_PREFIX..."
  "$CONDA" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    python=3.8 pip 'setuptools<70' wheel 'numpy=1.23.5'
fi
PY=$ENV_PREFIX/bin/python
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'

export MAX_JOBS=${MAX_JOBS:-2}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PATH="$ENV_PREFIX/bin:$PATH"

echo 'Installing PyTorch 2.1.0 with CUDA 11.8...'
"$PY" -m pip install --upgrade 'pip<25' 'setuptools<70' wheel
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
"$PY" -c 'import torch; assert torch.version.cuda == "11.8", torch.version.cuda'

# Native OpenPCDet and Mamba operators must be compiled by nvcc 11.8.
if [[ -n ${SVE_CUDA_HOME:-} ]]; then
  CUDA_HOME=$SVE_CUDA_HOME
elif [[ -x /usr/local/cuda-11.8/bin/nvcc ]]; then
  CUDA_HOME=/usr/local/cuda-11.8
elif [[ -x $ENV_PREFIX/bin/nvcc ]]; then
  CUDA_HOME=$ENV_PREFIX
else
  echo 'Installing CUDA 11.8 build toolkit inside the detector environment...'
  "$CONDA" install -y -p "$ENV_PREFIX" --override-channels \
    -c nvidia/label/cuda-11.8.0 -c conda-forge cuda-toolkit=11.8
  CUDA_HOME=$ENV_PREFIX
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
if [[ ! -x $CUDA_HOME/bin/nvcc ]] || ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 11.8'; then
  echo "A CUDA 11.8 nvcc compiler is required at $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi
echo "Using CUDA compiler: $CUDA_HOME/bin/nvcc"

echo 'Installing SVEFusion Python dependencies...'
"$PY" -m pip install -c "$CONSTRAINTS" spconv-cu118
"$PY" -m pip install -c "$CONSTRAINTS" -r "$SVE/requirements.txt"
"$PY" -m pip install -c "$CONSTRAINTS" packaging ninja einops transformers
"$PY" -m pip install --no-build-isolation -c "$CONSTRAINTS" 'causal-conv1d>=1.1,<1.2'

echo 'Building OpenPCDet CUDA operators...'
"$PY" -m pip install --no-build-isolation -c "$CONSTRAINTS" -e "$SVE"
echo 'Installing the Mamba version included in this SVEFusion checkout...'
"$PY" -m pip install --no-build-isolation -c "$CONSTRAINTS" -e "$SVE/mamba"

"$PY" -m pip check
"$PY" "$REPO/scripts/check_svefusion_detector_env.py" --sve-root "$SVE"
echo "Detector environment ready: $PY"
echo "Train and compare: bash $REPO/scripts/train_new_svefusion_and_compare.sh $SVE $PY"
