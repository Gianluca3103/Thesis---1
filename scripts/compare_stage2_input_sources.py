"""Compare Gaussian-start and faulty-latent-start Stage II reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("iou", "f1", "iou_0p2m", "f1_0p2m")


def compare(gaussian: dict, faulty_start: dict) -> dict:
    if (gaussian.get("sampler", "gaussian_ddim") != "gaussian_ddim" or
            faulty_start.get("sampler") != "faulty_latent_one_step"):
        raise ValueError("Expected Gaussian-start and faulty-latent-start reports")
    result = {"gaussian_checkpoint_sha256": gaussian["checkpoint_sha256"],
              "faulty_start_checkpoint_sha256":
                  faulty_start["checkpoint_sha256"],
              "comparison": {}}
    for split, group in (("validation", "all_validation"), ("test", "test")):
        old = gaussian[split]["overall"][group]
        new = faulty_start[split]["overall"][group]
        if (old["faulty"]["frames"] != new["faulty"]["frames"] or
                any(old["faulty"][key] != new["faulty"][key]
                    for key in ("exact_tp", "predicted", "eligible_target"))):
            raise ValueError(f"{split} frame set or faulty baseline differs")
        result["comparison"][split] = {
            "frames": old["faulty"]["frames"],
            "gaussian_threshold": gaussian["sweep"]["threshold"],
            "faulty_start_threshold": faulty_start["sweep"]["threshold"],
            "faulty_baseline": {key: old["faulty"][key] for key in METRICS},
            "gaussian_completed": {key: old["completed"][key] for key in METRICS},
            "faulty_start_completed": {
                key: new["completed"][key] for key in METRICS},
            "faulty_start_minus_gaussian": {
                key: new["completed"][key] - old["completed"][key]
                for key in METRICS}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-report", type=Path, required=True)
    parser.add_argument("--faulty-start-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.gaussian_report.read_text()),
                     json.loads(args.faulty_start_report.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    for split, row in report["comparison"].items():
        print(f"{split}: {row['frames']} frames")
        for key in METRICS:
            print(f"  {key}: Gaussian {row['gaussian_completed'][key]:.2%}; "
                  f"faulty-start {row['faulty_start_completed'][key]:.2%}; "
                  f"delta {row['faulty_start_minus_gaussian'][key] * 100:+.2f} pp")
    print(f"Report: {args.output_json}")


if __name__ == "__main__":
    main()
