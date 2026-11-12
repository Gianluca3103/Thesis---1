# 4D-RaDiff radar VAE trial on VoD

This run trains **only the radar variational autoencoder**, so its latent
points can later be assessed for radar conditioning. It does not train the
paper's foreground or background diffusion model.

Reference: [4D-RaDiff, arXiv:2512.14235v2, Appendix B and C](https://arxiv.org/html/2512.14235v2).

## Published details reproduced

- Native five-sweep `radar_5frames` View-of-Delft returns; no 12-voxel density
  filter, no LiDAR, and no simulated fault.
- Paper ROI in radar coordinates: x `[0, 51.2)`, y `[-25.6, 25.6)`, z `[-3, 2)` m.
- Five input fields: XYZ, compensated Doppler, RCS. Global train-only
  normalization into `[-1, 1]`.
- Three D-PCC-derived density, position, ancestor, and Point Transformer
  downsampling blocks, followed by Gaussian per-point latent features of
  dimension four. A PointNet-style latent-coordinate branch and an MLP
  feature branch feed three symmetric upsampling blocks.
- Final and intermediate Chamfer, radar feature MSE, density, cardinality,
  and KL losses with the paper's published weights: 1, 0.1, 0.05, 1e-4,
  5e-7, and 1e-5 respectively; density mean-distance weight 50.
- Adam at 1e-3, StepLR every 45 epochs with gamma 0.5, 300 epochs, batch 32
  (the paper's background VAE batch size).

## Necessary choices and comparability

The authors have not published their VAE implementation or the exact input
point count and branch widths. This run therefore pins upstream D-PCC at
`d738f59284e505166d26127dbe21cc71e2fa62c9`, adapts its final radar
attribute width to two, and uses 1,024 fixed input points, feature width 8,
hidden width 64, neighbor count 16, downsampling factor 3, and maximum
upsampling factor 8. Training randomly downsamples or repeats points; the
validation input is deterministic. Each choice is saved in `run_identity.json`
and the checkpoint.

The paper trains separate VAE components for annotated foreground and
background. Here **one VAE learns the full cropped radar frame** because the
requested use is a scene-level radar conditioning latent. This is a
paper-guided architecture and objective reproduction, not the authors' exact
trained model. Its scores should not be compared directly with the earlier
20-sweep, density-filtered radar autoencoders because the input distributions
and target sets differ.

## Professor-machine run

After fetching the published revision into the existing checkout, use:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109"
OUT="$BASE/sensor_fusion_outputs/sensor_fusion_outputs/radiff_vae_vod5_20261112"
REPO="$REPO" OUT="$OUT" bash "$REPO/scripts/run_radiff_vae_professor.sh" all
```

`all` runs `cache`, `setup`, a 32-frame train/16-frame validation GPU smoke
test, the 300-epoch training run, and best-checkpoint official validation and
test evaluation. The `setup` phase fetches and compiles the pinned D-PCC CUDA
operators using the already established professor-machine CUDA toolchain.
It is resumable: rerun `train` with the same `OUT` to continue from `last.ckpt`.
Use `cache`, `setup`, `smoke`, `train`, or `evaluate` individually to isolate a
failure. `BATCH_SIZE=8` is available if the 32-sample smoke test exhausts VRAM;
use a fresh `OUT` for a changed batch size.

Each epoch logs train/validation loss, exact 0.2 m-grid radar voxel IoU in
the paper ROI, one-to-one point F1 at 0.2 m, and squared Chamfer distance.
Points decoded outside the ROI count as false-positive voxels. The full JSON
record also contains the five loss terms, point precision/recall, radar
attribute MAE for 0.2 m matched points and the paper's nearest-neighbor
attribute error over all predicted points, plus frame counts. Validation selects the
checkpoint; the official test split is evaluated only after selection.

Outputs are under `$OUT/model`, with `best_voxel_iou_val.json` and
`best_voxel_iou_test.json` as final reports.
