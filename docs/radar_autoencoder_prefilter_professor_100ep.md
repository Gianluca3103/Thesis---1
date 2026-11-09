# Radar input-prefilter trial

Run `bash scripts/run_radar_autoencoder_prefilter_professor_100ep.sh pretrain` from a pinned professor-machine checkout to build or reuse compatible fixed-frame caches and train only the filtered radar encoder for 25 epochs. Afterward, run the same script with `cache`, `train`, and `evaluate` phases in that order, or omit the phase to run all steps. This is a separate 100-epoch radar occupancy encoder–decoder experiment. Its output root is `/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/radar_encoder_decoder_fullgrid12_1p6m_prefilter_100ep_20261109`.

For each frame, voxelize raw radar within the existing encoder grid (`x,y` ±51.2 m and `z` ±2 m). A radar voxel survives if its center has at least 12 **distinct radar voxel centers**, including itself, within 1.6 m. Every original return in a surviving voxel is kept; returns from other voxels are removed before radar encoder pretraining and before the occupancy encoder–decoder. The filter is computed once against the original frame, so it does not recursively erode a cluster.

The reconstruction target and metric gate stay defined by the **original unfiltered radar frame**. This isolates the effect of filtering the radar input: eligible clean LiDAR targets, output eligibility, and the fixed 0.5 IoU reported each epoch have the same definition as the unfiltered-input run. Decoder candidates and representable-target coverage can change because their radar input is now filtered. There is no faulty-LiDAR region exclusion. Clean LiDAR supplies training labels and evaluation targets, never an inference input.

The script builds separate geometry caches and checkpoints, validates exact IoU and IoU@0.2 m over all 1,296 validation frames every occupancy epoch, then sweeps thresholds once and applies the validation-selected threshold to the official test. The final geometry coverage report can be produced with `scripts.report_radar_gate_decoder_coverage` against the trial's `geometry_gate12` directory.

The professor launcher sets `RADAR_AE_TMPDIR` to a short `/tmp` path for Python multiprocessing sockets. Training output remains under `OUT`. Override `RADAR_AE_TMPDIR` with another short, writable directory if needed.

Radar pretraining on the professor environment disables optional matplotlib plots because the installed matplotlib/Pillow combination cannot import `_imaging`. Metrics and checkpoints still save every epoch. If a prior run wrote a metrics row but crashed before its first checkpoint, the launcher archives that incomplete metrics file and starts epoch 1 again.
