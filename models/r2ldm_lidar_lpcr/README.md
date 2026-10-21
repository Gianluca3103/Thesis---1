# Independent R2LDM LiDAR encoder and decoder

This is a separately trained LiDAR-only reconstruction baseline adapted from
[R2LDM](https://github.com/NathanDrake67/R2LDM). It does not use radar,
cross-attention, distillation, or the occupancy model's caches and checkpoints.
See [NOTICE.md](NOTICE.md) for provenance and reproducibility limits.

The input is raw VoD LiDAR XYZ from the official `lidar/ImageSets/train.txt`
and `val.txt` splits. The data loader retains up to eight points per voxel and
40,000 voxels per frame, then supplies their mean XYZ to the R2LDM sparse
encoder. The decoder predicts two occupancy logits and an XYZ offset for each
voxel in R2LDM's `40 × 320 × 320` grid. Training uses fine-voxel occupancy
cross-entropy plus `0.04 ×` occupied-voxel offset L1, matching the released
Stage-0 objective. The empty class means no LiDAR return in that cell; it is
not a measured-free-space label. Its IoU is therefore a *LiDAR reconstruction*
metric and is not directly comparable to the radar model's Known/Full IoU.

The released R2LDM grid and spatial dimensions are fixed here because its
ConvNeXt block and volume reshape depend on them. The default offset mode is
R2LDM's `legacy_sigmoid`, which cannot represent negative center-relative
offsets. Pass `--offset-activation centered_tanh` for a separate corrected
experiment; use a different output root for that run. The implementation keeps
the encoder, active 2D S2D layers, three 3D upsampling stages, and fine
reconstruction heads. It omits R2LDM's unused RPN and coarse heads from this
Stage-0 run.

The standalone run now uses the **released repository's LPCR defaults**:

| Setting | Value |
| --- | --- |
| LiDAR grid | XYZ `[0,32] × [-16,16] × [-2,4]` m; voxel `0.1 × 0.1 × 0.15` m |
| Voxelizer | At most 8 points per voxel and 40,000 voxels per frame |
| Encoder output | `tanh` bounded `8 × 80 × 80` latent |
| Training | 30 epochs, batch size 1, seed 3407, AdamW `1e-3`, betas `(0.9, 0.99)`, weight decay 0, epsilon `1e-8` |
| Schedule | StepLR every 5 epochs at factor 0.7; gradient norm clip 1.0 |
| LPCR loss | Fine occupancy plus `0.04 ×` point offset L1 |
| EMA | `ema-pytorch` decay 0.995, update every 10 steps after 2 steps; online weights drive training and validation |
| Precision | Full precision, as in the released Stage-0 training loop (`--amp` is an experimental option) |

The [paper](https://arxiv.org/pdf/2503.17097) reports a *different* VoD
region, `[0,16] × [-8,8] × [-0.5,3.5]` m, and 0.0625 m voxels. Those
dimensions cannot be substituted into the released decoder's fixed latent and
output shapes. The paper reports batch size 1, AdamW, and initial LPCR learning
rate `1e-3`; its epoch count and other optimizer settings come from the code.
This run follows the executable source grid and schedule, not the paper grid.

The data input and exact loss are still an adaptation: this run uses the
official raw VoD LiDAR train/val splits because R2LDM did not release its
paper preprocessing. Its two-class occupancy cross-entropy is equivalent to
binary cross-entropy on the occupied softmax probability away from numerical
clamping. We supervise point offsets in occupied voxels; the released
`legacy_sigmoid` branch masks individual nonzero offset components across the
dense tensor, including empty voxels. This correction prevents the empty-cell
offsets from dominating training. These differences mean published scores
cannot be claimed as an exact reproduction.

## Run on the professor machine

Use the CUDA Python environment with PyTorch, NumPy, tqdm, `ema-pytorch`, and
`spconv.pytorch`. The default is 30 epochs, batch size 1, AdamW at `1e-3`,
and a StepLR decay every five epochs, following R2LDM's released schedule.
The model uses a new output directory and does not need the existing fixed
frame or geometry caches.

Build its separate voxel-mean cache before training. The cache uses compressed
NumPy files and a manifest tied to the VoD split, grid, and voxelization code.

```bash
cd /mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=8
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
OUT=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/r2ldm_lidar_lpcr

"$PY" -u -m scripts.cache_r2ldm_lidar_lpcr \
  --vod-root "$VOD" --cache-root "$OUT/voxel_cache" \
  --split both --num-workers 8

"$PY" -u -m scripts.train_r2ldm_lidar_lpcr \
  --vod-root "$VOD" --output-root "$OUT" --cache-root "$OUT/voxel_cache" \
  --epochs 30 --batch-size 1 --num-workers 1 --device cuda
```

Before committing to a full run, use a separate output root with
`--epochs 1 --train-limit 2 --val-limit 2 --num-workers 0` to check the
environment and tensor shapes. The full run writes `last.ckpt`, `best.ckpt`,
and `metrics.jsonl`; resume with `--resume "$OUT/last.ckpt"` and the same
data and loss settings. Validation and best-checkpoint selection use online
weights, matching the released trainer. Prediction uses EMA weights by default,
matching released inference; pass `--weights online` to compare.

After training, reconstruct five validation frames as little-endian float32
`N × 3` XYZ `.bin` files:

```bash
"$PY" -u -m scripts.predict_r2ldm_lidar_lpcr \
  --vod-root "$VOD" --checkpoint "$OUT/best.ckpt" \
  --cache-root "$OUT/voxel_cache" \
  --output-root "$OUT/val_predictions" --split val --limit 5 \
  --threshold 0.5 --device cuda
```

R2LDM did not release its complete paper preprocessing or checkpoints. This
run uses raw VoD LiDAR within the LPCR grid and should be described as an
independent R2LDM-style baseline, not an exact published-score reproduction.
