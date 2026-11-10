# Radar-to-radar D-PCC adaptation

This is an independent implementation of the density-preserving autoencoder
from [D-PCC](https://github.com/yunhe20/D-PCC), adapted to variable-size 20-sweep
View-of-Delft radar. It uses farthest-point sampling; ancestor, local-position,
and density embeddings; and three learned variable-cardinality upsampling
blocks. The original D-PCC CUDA code is not copied or run. All seven filtered
radar channels are input; XYZ, RCS, two Doppler values, and scan age are
reconstructed. There is no LiDAR supervision or fine-resolution skip path.
The run uses three 2x point-downsampling blocks and 32 latent feature channels,
plus the latent XYZ anchors. The reported `latent_scalar_ratio` compares the
number of latent feature/coordinate scalars with the input's seven scalars per
point. Entropy coding and coordinate quantization from the compression paper
are omitted because this run tests reconstruction quality, not bit rate.

It reuses the *exact* filtered-radar cache from the previous radar-to-radar
run when present. The FPS hierarchy is cached separately. No existing cache
or checkpoint is deleted. `OUT` is a new experiment directory.

Run on the professor machine after checking out the pinned revision:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109"
ROOT="$BASE/sensor_fusion_outputs/sensor_fusion_outputs"
OUT="$ROOT/radar_dpcc_12vox_1p6m_20261110"
mkdir -p "$OUT"
cd "$REPO"
git fetch origin main
git checkout --detach REVISION_TO_FILL
REPO="$REPO" OUT="$OUT" EPOCHS=200 BATCH_SIZE=8 WORKERS=4 \
  bash scripts/run_radar_dpcc_professor.sh all 2>&1 | tee -a "$OUT/console.log"
```

The first phase builds only the new FPS geometry. The launcher runs a small
smoke test before the full training run. It resumes from `model/last.ckpt` on
subsequent invocations. It reports full validation radar-voxel IoU at a
cardinality threshold of 0.5 and one-to-one radar-point F1 within 0.2 m
after every epoch. After training, it sweeps thresholds on all validation
frames. Official test can be prepared afterward with
`bash scripts/run_radar_dpcc_professor.sh cache-test` and scored with
`bash scripts/run_radar_dpcc_professor.sh evaluate`; it uses validation-selected
thresholds. Outputs include Chamfer distance and
matched errors for RCS, both Doppler channels, and scan age.

The cardinality threshold acts on each D-PCC-style upsampling block's
predicted child count. This adapts the original continuous point-count
prediction to the threshold comparison used by the existing radar run.

For a fair comparison, use the existing radar self-autoencoder's held-out
validation results on the same filtered manifest. Their training losses
have different scales; compare the point and voxel metrics instead.
