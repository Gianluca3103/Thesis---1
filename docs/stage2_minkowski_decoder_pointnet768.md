# Stage II trial: radar-seeded Minkowski latent completion

This is a new Stage II experiment, separate from the LPCR-based checkpoint.
The old latent has 32 channels and stride `(16,4,4)` in ZYX order. The audited
Minkowski LiDAR autoencoder has 64 channels and stride `(2,2,2)` in XYZ order.
Its decoder weights cannot be loaded into the old Stage II architecture.

## Model and data

- Reuse the saved width-32, one-level Minkowski LiDAR autoencoder. Freeze its
  encoder and decoder. The clean-to-clean 100% IoU is a copying diagnostic,
  **not** evidence it can fill missing blocks; the 30%-masked audit recovered
  none. This experiment measures whether radar-conditioned latent generation
  changes that result.
- Retrain the established PointNet++ radar autoencoder at **width 768** instead
  of 1024, with 2048 fixed radar input points and 512 anchors. It uses the
  filtered 20-sweep radar, full Stage II spatial crop, and all official train
  and validation frames. Cache its frozen anchor features for all train,
  validation, and test frames.
- Use the ≥12 distinct radar voxels within 1.6 m gate, intersected with the
  injected partial-loss sector. The latter is a synthetic oracle footprint;
  a real deployment would require a fault-localization model.
- Create every stride-2 coarse decoder coordinate from this gate, including
  blocks in which faulty LiDAR has **no** return. No clean voxel adds a decoder
  coordinate. All fine cells within the gate are scored; there is no cap on
  candidate voxel count.
- Encode full clean LiDAR only for the training target. Encode surviving faulty
  LiDAR for conditioning. Interpolate PointNet++ anchor features to each
  radar-selected coarse site. A new sparse U-Net predicts the noise in the
  64-channel Minkowski latent. Inference starts from Gaussian noise and uses
  only faulty LiDAR, radar features, and the known synthetic fault mask.
- Train with noise MSE plus ordinary BCE (weight 1) on the frozen decoder's
  generated occupancy logits at unobserved gated voxels. The decoder is
  frozen, but BCE gradients pass through it to the denoiser. Training uses a
  64-step rollout with gradients on its final four steps. Validation uses
  64 steps. An independent noise MSE validation runs every epoch; full
  reconstruction validation runs at epoch 1, every five epochs, and the end.

The decoder has only seen latent sites derived from occupied clean LiDAR in
pretraining. It may not have learned a useful empty code for radar-generated
sites. That is a substantive risk this trial will test, not an assumed result.

## Professor-machine run

The launcher reuses the earlier partial-only caches if present. Its `prepare`
phase builds missing train/validation/test radar stacks, fixed frames, radar
support masks, partial-loss faults, filtered radar inputs, and injector-region
cache. It does not delete old outputs. The saved Minkowski checkpoint is a
required input; its default location is
`$BASE/sensor_fusion_outputs/minkowski_lidar_ae_stage2grid_20261125/model/best.ckpt`.
Set `LIDAR_CKPT` explicitly if yours is elsewhere.

```bash
BASE=/mnt/3D10B36523559581/Gianluca
DATA="$BASE/sensor_fusion_outputs/sensor_fusion_outputs"
REPO="$BASE/sensor_fusion_outputs/Thesis-1-stage2-minkowski-pointnet768"
OUT="$DATA/stage2_minkowski_pointnet768_partial_only_20261125"
mkdir -p "$OUT"
REPO="$REPO" BASE="$BASE" DATA="$DATA" OUT="$OUT" \
  RADAR_EPOCHS=300 STAGE2_EPOCHS=75 RADAR_BATCH=4 STAGE2_BATCH=4 WORKERS=4 \
  bash "$REPO/scripts/run_stage2_minkowski_pointnetpp_professor.sh" all \
  2>&1 | tee -a "$OUT/pipeline.log"
```

Run phases individually with `prepare`, `radar`, `features`, `smoke`,
`stage2`, or `evaluate`. Radar and Stage II automatically resume their saved
`last.ckpt` files. The runner checks the compiled PointNet++ CUDA operator
before preparing caches. `smoke` runs 16 training and eight validation frames
in a separate directory; the actual train phase has no frame limit.

The best Stage II checkpoint is chosen by full-scene exact IoU at threshold
0.5 on full validation. After training, a threshold is selected on the first
260 validation frames, then the remaining validation frames and all official
test frames are scored at that fixed threshold. The report includes
addition-only IoU and IoU@0.2 m, full-scene before/after IoU and F1, and
the corresponding 0.2 m metrics. A no-addition threshold is permitted if
the model cannot improve the faulty baseline.
