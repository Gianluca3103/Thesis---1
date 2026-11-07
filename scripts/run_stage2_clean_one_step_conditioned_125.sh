#!/usr/bin/env bash
# One-step clean-latent diagnostic with faulty LiDAR and radar conditioning.
set -euo pipefail

DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
OUT=${OUT:-$DATA/stage2_clean_one_step_faulty_radar_12vox_1p6m_batch16_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
BATCH_SIZE=16
EPOCHS=125
CONDITION_FAULTY_RADAR=1
export OUT MILD BATCH_SIZE EPOCHS CONDITION_FAULTY_RADAR
exec bash "$(dirname "$0")/run_stage2_clean_12vox_denoiser.sh"
