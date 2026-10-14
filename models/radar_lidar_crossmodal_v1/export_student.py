"""Export raw radar-only student tensors without interpolation or point caps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1

from .train_student import load_student


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, saved = load_student(args.checkpoint, args.device)
    dataset = VoDStage1Dataset(args.vod_root, args.split,
                               radar_variant=saved["radar_variant"],
                               frame_ids=args.frame_id, include_clean=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        row = dataset[index]
        batch = collate_stage1([row])
        output = model(batch["radar"].to(args.device), batch["radar_valid"].to(args.device))
        result = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                  for key, value in output.exported().items()}
        result["frame_id"] = row["frame_id"]
        path = args.output_root / f"{row['frame_id']}_radar_only_raw.pt"
        torch.save(result, path)
        print(f"{row['frame_id']}: {len(result['confidence_coords'])} raw confidence sites | {path}",
              flush=True)


if __name__ == "__main__":
    main()
