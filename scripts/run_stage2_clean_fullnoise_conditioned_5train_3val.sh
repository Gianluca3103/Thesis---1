#!/usr/bin/env bash
# Full-noise five/three-step trial conditioned on faulty LiDAR and radar.
set -euo pipefail

DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
OUT=${OUT:-$DATA/stage2_clean_fullnoise_faulty_radar_12vox_1p6m_5train_3val_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
CONDITION_FAULTY_RADAR=1
export OUT MILD CONDITION_FAULTY_RADAR
exec bash "$(dirname "$0")/run_stage2_clean_fullnoise_5train_3val.sh"
