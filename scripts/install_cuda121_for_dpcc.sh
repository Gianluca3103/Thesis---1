#!/usr/bin/env bash
# Install CUDA 12.1.1 toolkit only into the professor data drive, without sudo.
set -euo pipefail

BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
INSTALLER_NAME=cuda_12.1.1_530.30.02_linux.run
INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/$INSTALLER_NAME"
EXPECTED_MD5=2f0a4127bf797bf4eab0be2a547cb8d0
DOWNLOAD_DIR="$BASE/cuda-installers"
TOOLKIT="$BASE/cuda-12.1"
TEMP_DIR="$BASE/rtmp"
INSTALLER="$DOWNLOAD_DIR/$INSTALLER_NAME"

mkdir -p "$DOWNLOAD_DIR" "$TEMP_DIR"
if [[ -x "$TOOLKIT/bin/nvcc" ]] && \
   "$TOOLKIT/bin/nvcc" --version | grep -q 'release 12\.1'; then
  echo "CUDA 12.1 toolkit already installed: $TOOLKIT"
  exit 0
fi

# The runfile plus extraction and toolkit need space on this same data drive.
available_kib=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
if (( available_kib < 15 * 1024 * 1024 )); then
  echo "Need at least 15 GiB free under $BASE for the CUDA toolkit installer" >&2
  df -h "$BASE" >&2
  exit 1
fi

if [[ -f "$INSTALLER" ]] && \
   printf '%s  %s\n' "$EXPECTED_MD5" "$INSTALLER" | md5sum --check --status; then
  echo "Verified existing NVIDIA installer: $INSTALLER"
else
  echo "Downloading NVIDIA CUDA 12.1.1 toolkit installer to $INSTALLER"
  if ! curl --fail --location --retry 3 --continue-at - \
      "$INSTALLER_URL" --output "$INSTALLER"; then
    # A corrupt full-length old file can make a resumed request return 416.
    curl --fail --location --retry 3 "$INSTALLER_URL" --output "$INSTALLER"
  fi
  printf '%s  %s\n' "$EXPECTED_MD5" "$INSTALLER" | md5sum --check
fi

echo "Installing toolkit only under $TOOLKIT; existing NVIDIA driver is untouched"
sh "$INSTALLER" --silent --toolkit --toolkitpath="$TOOLKIT" \
  --defaultroot="$TOOLKIT" --tmpdir="$TEMP_DIR"
"$TOOLKIT/bin/nvcc" --version | grep 'release 12\.1'
echo "CUDA_HOME=$TOOLKIT"
