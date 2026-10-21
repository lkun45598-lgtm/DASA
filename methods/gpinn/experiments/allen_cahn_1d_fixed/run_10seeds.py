"""Run the fixed 1D Allen-Cahn gPINN+RAR solver automatically for seeds 0 through 9."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch


METRICS = (
    "relative_l2",
    "max_abs",
    "rmse",
    "periodic_value_autograd_max_abs",
    "periodic_derivative_max_abs",
    "heldout_total",
    "heldout_pde",
    "heldout_gradient",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending seed range {token!r}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    if not result:
        raise ValueError("at least one seed is required")
    if len(set(result)) != len(result):
        raise ValueError("seed list contains duplicates")
    return tuple(result)


def detected_gpu_ids() -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        identifiers = tuple(value.strip() for value in visible.split(",") if value.strip())
        if identifiers:
            return identifiers
    if torch.cuda.is_available():
        return tuple(str(index) for index in range(torch.cuda.device_count()))
    return ()


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    if value.lower() in {"cpu", "none"}:
        return ()
    if value.lower() == "auto":
        return detected_gpu_ids()
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("--gpus requires auto or a comma-separated GPU list")
    return result


def is_complete(output_root: Path, seed: int, mode: str) -> bool:
    path = output_root / f"seed_{seed}" / "metrics.json"
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        return metrics.get("seed") == seed and metrics.get("mode") == mode
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_seed(
    seed: int,
    device: str,
    gpu_id: str | None,
    args: argparse.Namespace,
    runner_args: list[str],
) -> tuple[int, int]:
    seed_dir = args.output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run.py")),
        "--mode",
        args.mode,
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-root",
        str(args.output_root),
        "--reference-root",
        str(args.reference_root),
        *runner_args,
    ]
    if not args.force and (seed_dir / "last.pt").exists():
        command.append("--resume")
    environment = os.environ.copy()
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    with (seed_dir / "launcher.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return seed, completed.returncode


def write_summary(output_root: Path, seeds: tuple[int, ...]) -> dict:
    rows: list[dict] = []
    missing: list[int] = []
    for seed in seeds:
        path = output_root / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            missing.append(seed)
            continue
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        rows.append(row)
    aggregate: dict[str, float | int | str | list[int]] = {
        "requested_seeds": list(seeds),
        "completed_seeds": [int(row["seed"]) for row in rows],
        "missing_seeds": missing,
        "completed_runs": len(rows),
    }
    for name in METRICS:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if len(values):
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            )
    payload = {"aggregate": aggregate, "runs": rows}
    with (output_root / "summary_10seeds.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    if rows:
        fields = ["seed", *METRICS, "device", "global_adam_steps", "rar_anchors"]
        with (output_root / "summary_10seeds.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: int(row["seed"])))
    with (output_root / "summary_10seeds.md").open("w", encoding="utf-8") as handle:
        handle.write("# 1D Allen-Cahn gPINN+RAR 十种子汇总\n\n")
        handle.write(f"完成：{len(rows)}/{len(seeds)}；缺失：{missing or '无'}。\n\n")
        handle.write("| 指标 | mean ± std |\n|---|---:|\n")
        for name in METRICS:
            mean = aggregate.get(f"{name}_mean")
            std = aggregate.get(f"{name}_std")
            if mean is not None:
                handle.write(f"| {name} | {mean:.6e} ± {std:.6e} |\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="default: 0-9")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--gpus", default="auto", help="auto, cpu, or e.g. 0,1,2,3"
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="parallel workers used only when CUDA is unavailable",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/allen_cahn_1d_fixed_10seeds"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/allen_cahn_1d_fixed_reference_data"),
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun completed seeds from scratch"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args, runner_args = build_parser().parse_known_args(argv)
    args.output_root = args.output_root.resolve()
    args.reference_root = args.reference_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    gpu_ids = parse_gpu_ids(args.gpus)
    if args.cpu_workers < 1:
        raise ValueError("--cpu-workers must be positive")
    pending = [
        seed
        for seed in seeds
        if args.force or not is_complete(args.output_root, seed, args.mode)
    ]
    skipped = [seed for seed in seeds if seed not in pending]
    print(
        json.dumps(
            {
                "seeds": list(seeds),
                "pending": pending,
                "skipped_complete": skipped,
                "gpu_ids": list(gpu_ids),
                "mode": args.mode,
            },
            indent=2,
        ),
        flush=True,
    )
    task_queue: queue.Queue[int] = queue.Queue()
    for seed in pending:
        task_queue.put(seed)
    print_lock = threading.Lock()
    failures: list[tuple[int, int]] = []

    if gpu_ids:
        workers: tuple[tuple[str, str | None], ...] = tuple(
            ("cuda", gpu_id) for gpu_id in gpu_ids
        )
    else:
        workers = tuple(("cpu", None) for _ in range(args.cpu_workers))

    def worker(device: str, gpu_id: str | None) -> None:
        while True:
            try:
                seed = task_queue.get_nowait()
            except queue.Empty:
                return
            label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            with print_lock:
                print(f"[{label}] seed {seed} started", flush=True)
            result = run_seed(seed, device, gpu_id, args, runner_args)
            if result[1] != 0:
                failures.append(result)
            with print_lock:
                status = "completed" if result[1] == 0 else f"failed ({result[1]})"
                print(f"[{label}] seed {seed} {status}", flush=True)
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=worker_args, daemon=False)
        for worker_args in workers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = write_summary(args.output_root, seeds)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(f"failed seeds: {failures}")


if __name__ == "__main__":
    main()
