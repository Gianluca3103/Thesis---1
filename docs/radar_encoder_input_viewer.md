# Filtered radar encoder input viewer

`scripts.visualize_radar_encoder_input` visualizes the original radar returns
that pass the exact input filter used by the radar encoder and occupancy
autoencoder: at least 12 distinct radar voxels (including the queried voxel)
within 1.6 m, restricted to the existing ±51.2 m XY and ±2 m Z encoder grid.
The filter evaluates the original frame once. It does not use a checkpoint or
clean LiDAR.

For each selected frame, the viewer writes an interactive HTML page with
retained and rejected returns as separately controlled layers, an unsampled
PLY of retained XYZ, and an NPZ containing all seven raw radar columns plus
the exact retained mask. The HTML shows every retained return and every
rejected return inside the encoder grid.

On the professor machine, render five validation frames using the pretraining
fixed-frame cache:

```bash
SRC=/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-radar-autoencoder-prefilter-20261109
DATA=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
VIEW="$DATA/radar_encoder_decoder_fullgrid12_1p6m_prefilter_100ep_20261109/radar_input_viewer"
cd "$SRC"
PYTHONPATH="$SRC" "$PY" -u -m scripts.visualize_radar_encoder_input \
  --vod-root "$VOD" \
  --data-cache-root "$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames" \
  --output-root "$VIEW" --split val --limit 5
```

To view it in Firefox, run `"$PY" -m http.server 8768 --bind 127.0.0.1
--directory "$VIEW"` in a terminal and open `http://127.0.0.1:8768/`.
Directly opening an HTML file in an editor may display its source instead.
