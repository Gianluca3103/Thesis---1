"""Score one frozen detector across matched clean, faulty and completed scans."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from pcdet.datasets.vod.kitti_object_eval_python.eval import get_official_eval_result


CONDITIONS = ("clean", "faulty", "reconstructed")
CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("easy", "moderate", "hard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--clean-infos", type=Path, required=True)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition}-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.clean_infos.open("rb") as source:
        infos = pickle.load(source)
    ids = [str(item["point_cloud"]["lidar_idx"]) for item in infos]
    if len(ids) != len(set(ids)):
        raise ValueError("Validation infos repeat frame IDs")
    predictions = {}
    for condition in CONDITIONS:
        with getattr(args, f"{condition}_results").open("rb") as source:
            predictions[condition] = pickle.load(source)
        if [str(item["frame_id"]) for item in predictions[condition]] != ids:
            raise ValueError(f"{condition} prediction IDs/order differ from clean labels")
    faults = []
    for frame_id in ids:
        path = args.export_root / "frames" / f"{frame_id}.json"
        faults.append(json.loads(path.read_text(encoding="utf-8"))["fault"])
    subsets = {"all": list(range(len(ids)))}
    for fault in sorted(set(faults)):
        subsets[fault] = [i for i, name in enumerate(faults) if name == fault]
    rows = []
    for subset, indices in subsets.items():
        ground_truth = [infos[i]["annos"] for i in indices]
        for condition in CONDITIONS:
            selected = [predictions[condition][i] for i in indices]
            _, scores = get_official_eval_result(ground_truth, selected, list(CLASSES))
            row = {"subset": subset, "condition": condition, "frames": len(indices)}
            for name in CLASSES:
                for difficulty in DIFFICULTIES:
                    key = f"{name}_3d/{difficulty}_R40"
                    row[key] = float(scores[key])
            rows.append(row)
            print(subset, condition, {name: row[f"{name}_3d/moderate_R40"]
                                     for name in CLASSES}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
