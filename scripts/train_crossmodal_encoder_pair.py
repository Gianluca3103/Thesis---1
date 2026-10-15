"""Train the independent radar and LiDAR encoders concurrently with two live bars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm import tqdm


def _stop(processes: dict[str, subprocess.Popen]) -> None:
    for process in processes.values():
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()


def _tail(path: Path, limit: int = 30) -> str:
    if not path.is_file():
        return "(no log file)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])


def _metrics_after(path: Path, after_epoch: int) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A writer may still be appending the final line.
        if int(record["epoch"]) > after_epoch:
            records.append(record)
    return records


def main() -> None:
    def stop_on_signal(signum, _frame):
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_on_signal)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--radar-variant", required=True)
    parser.add_argument("--online-augmentation-config", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-val-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        parser.error("batch size and epochs must be positive; workers must be nonnegative")

    processes: dict[str, subprocess.Popen] = {}
    progress_files: dict[str, Path] = {}
    logs: dict[str, Path] = {}
    metrics_files: dict[str, Path] = {}
    reported_epoch: dict[str, int] = {}
    commands: dict[str, list[str]] = {}
    for modality in ("radar", "lidar"):
        root = args.output_root / modality
        root.mkdir(parents=True, exist_ok=True)
        last = root / "last.ckpt"
        if (root / "metrics.jsonl").exists() and (not last.is_file() or last.stat().st_size == 0):
            parser.error(f"{root} has metrics but no usable last.ckpt; preserve and inspect it")
        progress = root / "progress.json"
        progress.unlink(missing_ok=True)
        command = [
            sys.executable, "-u", "-m", "models.radar_lidar_crossmodal_v1.pretrain_encoder",
            "--vod-root", str(args.vod_root), "--radar-variant", args.radar_variant,
            "--online-augmentation-config", str(args.online_augmentation_config),
            "--output-root", str(root), "--modality", modality,
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--full-val-every", str(args.full_val_every),
            "--num-workers", str(args.num_workers), "--device", args.device,
            "--no-tensorboard", "--progress-file", str(progress), "--quiet-progress",
        ]
        if args.data_cache_root is not None:
            command.extend(("--data-cache-root", str(args.data_cache_root)))
        if last.is_file() and last.stat().st_size:
            command.extend(("--resume", str(last)))
            print(f"{modality} encoder: resuming from {last}", flush=True)
        else:
            print(f"{modality} encoder: starting from epoch 1", flush=True)
        progress_files[modality] = progress
        logs[modality] = root / "console.log"
        metrics_files[modality] = root / "metrics.jsonl"
        reported_epoch[modality] = max(
            (int(record["epoch"]) for record in _metrics_after(metrics_files[modality], -1)),
            default=0,
        )
        commands[modality] = command
    try:
        for modality, command in commands.items():
            with logs[modality].open("a", encoding="utf-8") as stream:
                processes[modality] = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
    except BaseException:
        _stop(processes)
        raise

    try:
        bars = {
            modality: tqdm(total=1, desc=f"{modality} starting", position=index,
                           leave=False, dynamic_ncols=True, unit="batch")
            for index, modality in enumerate(processes)
        }
    except BaseException:
        _stop(processes)
        raise
    current_epoch: dict[str, int | None] = {name: None for name in processes}
    current_phase: dict[str, str | None] = {name: None for name in processes}
    try:
        while True:
            for modality, process in processes.items():
                path = progress_files[modality]
                status = None
                if path.is_file():
                    try:
                        status = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pass
                if status is not None:
                    epoch = int(status["epoch"])
                    phase = status["phase"]
                    bar = bars[modality]
                    if phase == "complete":
                        if current_epoch[modality] != epoch:
                            current_epoch[modality] = epoch
                            bar.reset(total=1)
                        bar.n = bar.total
                        bar.refresh()
                    else:
                        total = max(int(status["total"]), 1)
                        if current_epoch[modality] != epoch:
                            current_epoch[modality] = epoch
                            bar.reset(total=total)
                        if current_phase[modality] != phase:
                            current_phase[modality] = phase
                            bar.set_description(f"{modality} {epoch:03d}/{args.epochs:03d} {phase}")
                        completed = min(max(int(status["step"]), 0), total)
                        if completed != bar.n:
                            bar.update(completed - bar.n)
                for record in _metrics_after(metrics_files[modality], reported_epoch[modality]):
                    record_epoch = int(record["epoch"])
                    tqdm.write(
                        f"{modality} {record_epoch:03d}/{args.epochs:03d} | "
                        f"train {record['train']['loss_total']} | "
                        f"val {record['val']['loss_total']} "
                        f"({record['validation_scope']})", file=sys.stderr
                    )
                    reported_epoch[modality] = record_epoch
                    if (status is not None and status.get("phase") == "complete"
                            and int(status["epoch"]) == record_epoch and status.get("warning")):
                        tqdm.write(f"{modality} warning: {status['warning']}",
                                   file=sys.stderr)
                if process.poll() is not None and process.returncode != 0:
                    raise RuntimeError(
                        f"{modality} encoder exited {process.returncode}. "
                        f"Last lines of {logs[modality]}:\n{_tail(logs[modality])}"
                    )
            if all(process.poll() is not None for process in processes.values()):
                break
            time.sleep(0.2)
        for modality in processes:
            selected = args.output_root / modality / "best_encoder.pth"
            if not selected.is_file() or selected.stat().st_size == 0:
                raise RuntimeError(f"{modality} finished without {selected}")
        tqdm.write("Both encoder pretraining runs completed.", file=sys.stderr)
    except BaseException:
        _stop(processes)
        raise
    finally:
        for bar in bars.values():
            bar.close()


if __name__ == "__main__":
    main()
