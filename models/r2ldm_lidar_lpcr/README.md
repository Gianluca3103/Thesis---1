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
ConvNeXt block and volume reshape depend on them. The default signed-offset
mode is R2LDM's `centered_tanh` correction, appropriate for fresh training.
Pass `--offset-activation legacy_sigmoid` only to reproduce the historical
nonnegative-offset branch. The implementation keeps the encoder, active 2D
S2D layers, three 3D upsampling stages, and fine reconstruction heads. It
omits R2LDM's unused RPN and coarse heads from this Stage-0 run.

## Run on the professor machine

Use the CUDA Python environment with PyTorch, NumPy, tqdm, and
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
  --epochs 30 --batch-size 1 --num-workers 8 --device cuda
```

Before committing to a full run, use a separate output root with
`--epochs 1 --train-limit 2 --val-limit 2 --num-workers 0` to check the
environment and tensor shapes. The full run writes `last.ckpt`, `best.ckpt`,
and `metrics.jsonl`; resume with `--resume "$OUT/last.ckpt"` and the same
data and loss settings.

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
