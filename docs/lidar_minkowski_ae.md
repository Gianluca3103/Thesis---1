# Clean VoD LiDAR voxel autoencoder trial

This is an adaptation of the [MinkowskiEngine sparsity-pattern reconstruction
example](https://nvidia.github.io/MinkowskiEngine/demo/sparse_tensor_reconstruction.html),
not a claim to reproduce its ModelNet40 class-label experiment. The original
example has a generative sparse decoder but no LiDAR encoder. This trial adds a
sparse LiDAR encoder and trains it end to end on the official View-of-Delft
train split. The official validation split selects the best checkpoint; the
official test split is evaluated afterward.

The only input to the model is the **clean LiDAR occupancy grid**. Radar is not
used as a feature or a target; its per-frame calibration defines the same
fixed 70-degree, 80 m forward crop used in Stage II. The grid is exactly the
Stage-II `(x,y,z) = (0..32, -16..16, -2..2) m` grid with
`(0.2, 0.2, 0.25) m` voxels. No point-count cap or radar-density gate is
applied. The model learns one or two stride-2 sparse downsampling layers and
mirrored generative upsampling layers with voxel BCE. There are no skip paths
from full-resolution input voxels to the decoder. The decoder receives a new
sparse tensor made only from latent coordinates and features.

Default run: one stride-2 level, 32 base channels, batch size 1, 100 epochs.
Every epoch prints train and validation BCE, exact occupied-voxel IoU and F1,
and the decoder's maximum possible recall from its generated candidates.
The checkpoint is selected by validation exact IoU at threshold 0.5.

On the professor machine, use a checkout of this revision and run:

```bash
REPO=/path/to/checkout \
OUT=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/minkowski_lidar_ae_stage2grid_20261125 \
bash /path/to/checkout/scripts/run_lidar_minkowski_ae_professor.sh all
```

Use `smoke`, `cache`, `train`, or `evaluate` in place of `all` for individual
phases. The full run resumes from `model/last.ckpt` when present. A separate
output root is needed to change model width, levels, or the data split.

This trial measures clean-to-clean bottleneck fidelity. A high IoU here does
not establish that the latent can recover cells absent from faulty LiDAR, or
that diffusion can predict the corresponding latent features from radar.
