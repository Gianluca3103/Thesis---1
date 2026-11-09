# Local radar encoder–decoder reconstruction test

Run from the laptop's WSL terminal:

```bash
bash /mnt/c/Users/gianl/Desktop/Thesis/radar-lidar-crossmodal/scripts/run_radar_autoencoder_gate12_local.sh all
```

The script runs in the foreground and resumes saved pretraining and model checkpoints if interrupted. Outputs default to `/home/gianl/radar_encoder_decoder_local_20261109/full`; set `OUT=/another/WSL/path` for a separate trial. You can also run `pretrain`, `cache`, `train`, and `evaluate` as separate phases. The fixed, aligned train/validation/test frame cache from the earlier local Stage II setup must already exist at `/home/gianl/stage2_conditional_local_from_scratch_20261108/fixed_frames`.

This is the direct **radar encoder → generative sparse occupancy decoder** ablation. It loads the 20-frame radar and uses clean LiDAR only as a training label and evaluation target. No faulty LiDAR, clean LiDAR encoder, Stage I learned gate, LPCR decoder, or diffusion model enters the forward pass.

The candidate domain is the Stage II grid (`x=0..32 m`, `y=-16..16 m`, `z=-2..2 m`, voxel `0.2 × 0.2 × 0.25 m`) restricted to cells with **at least 12 distinct radar voxels within 1.6 m**. The decoder generates more candidate cells internally, but only cells inside this gate can contribute to training metrics or be selected for evaluation. The gate is precomputed from radar alone, and the geometry cache reports the fraction of eligible clean voxels that the decoder can represent.

The model is trained with its existing occupancy BCE over supported cells labeled either occupied by clean LiDAR or verified free by the LiDAR ray target builder. Unobserved cells outside those labels do not enter the BCE. Exact scan IoU and F1 use *all* gate-eligible clean LiDAR voxels as targets, so predictions in unlabeled cells still count as false positives in the final reconstruction metric. IoU/F1 at 0.2 m use one-to-one voxel-center matches within 0.2 m. The validation sweep tries thresholds on the best validation-BCE and best fixed-0.5 IoU checkpoints, chooses the better exact-IoU configuration on validation, then scores the official test once at that selection.

Reports are written to:

- `full/radar_autoencoder/metrics.jsonl` — train/validation loss and gate metrics by epoch.
- `full/evaluation/val_best_bce/threshold_sweep.json` — validation sweep for the BCE-selected checkpoint.
- `full/evaluation/val_best_iou/threshold_sweep.json` — validation sweep for the fixed-0.5-IoU checkpoint.
- `full/evaluation/selected_student.ckpt` — link to the validation-selected model.
- `full/evaluation/test/threshold_sweep.json` — official-test exact and 0.2 m IoU/F1 at the validation-selected threshold.

To inspect geometry coverage separately after the cache is built:

```bash
cd /mnt/c/Users/gianl/Desktop/Thesis/radar-lidar-crossmodal
PYTHONPATH=/usr/lib/python3/dist-packages:$PWD \
  /home/gianl/venvs/stage2-local/bin/python \
  -m scripts.report_radar_gate_decoder_coverage \
  --meta-cache-root /home/gianl/radar_encoder_decoder_local_20261109/full/geometry_gate12 \
  --split val
```

The small smoke run lives under `/home/gianl/radar_encoder_decoder_local_20261109/smoke` and uses only 16 train and 8 validation frames. Its metrics verify that the pipeline works; they are not an estimate of full-dataset reconstruction quality.

To render three validation examples from the completed full run using the same exact gate as the metrics:

```bash
cd /mnt/c/Users/gianl/Desktop/Thesis/radar-lidar-crossmodal
PYTHONPATH=/usr/lib/python3/dist-packages:$PWD \
  /home/gianl/venvs/stage2-local/bin/python \
  -m scripts.visualize_crossmodal_student \
  --vod-root '/mnt/c/Users/gianl/Desktop/Thesis/View-Of-Delft dataset/view_of_delft_PUBLIC' \
  --data-cache-root /home/gianl/stage2_conditional_local_from_scratch_20261108/fixed_frames \
  --gate-support-root /home/gianl/radar_encoder_decoder_local_20261109/full/support12vox_1p6m \
  --checkpoint /home/gianl/radar_encoder_decoder_local_20261109/full/evaluation/selected_student.ckpt \
  --split val --limit 3 --random-seed 3103 \
  --output-root /home/gianl/radar_encoder_decoder_local_20261109/full/views_val
```

The viewer writes a standalone `index.html` and PLY clouds. Its confidence slider lets you inspect the predicted voxels at different thresholds. For a Windows browser, run `cd /home/gianl/radar_encoder_decoder_local_20261109/full/views_val && python3 -m http.server 8765 --bind 127.0.0.1` in a second WSL terminal, then open `http://localhost:8765/index.html`.
