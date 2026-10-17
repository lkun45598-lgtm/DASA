#!/usr/bin/env python3
"""Aggregate wave-equation rebuttal runs into the three requested tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SENSITIVITY_ROWS = (
    ("Sub-network architecture", "3 x 5", "sensitivity_arch_3x5"),
    ("Sub-network architecture", "5 x 10", "sensitivity_baseline"),
    ("Sub-network architecture", "7 x 20", "sensitivity_arch_7x20"),
    ("Learning rates (eta_b, eta_s)", "(5e-4, 5e-4)", "sensitivity_lr_both_5e-4"),
    ("Learning rates (eta_b, eta_s)", "(1e-3, 5e-4)", "sensitivity_lr_subnet_5e-4"),
    ("Learning rates (eta_b, eta_s)", "(1e-3, 1e-3)", "sensitivity_baseline"),
    ("Learning rates (eta_b, eta_s)", "(1e-3, 2e-3)", "sensitivity_lr_subnet_2e-3"),
    ("Learning rates (eta_b, eta_s)", "(2e-3, 2e-3)", "sensitivity_lr_both_2e-3"),
    ("Update ratio", "1:5", "sensitivity_ratio_1to5"),
    ("Update ratio", "1:1", "sensitivity_baseline"),
    ("Update ratio", "5:1", "sensitivity_ratio_5to1"),
)

STAGE_ROWS = (
    ("Stage 1", "(2500, 2000, 50000, 50000)", "stage_length_stage1_2500"),
    ("Baseline", "(5000, 2000, 50000, 50000)", "stage_length_baseline"),
    ("Stage 1", "(10000, 2000, 50000, 50000)", "stage_length_stage1_10000"),
    ("Stage 2", "(5000, 1000, 50000, 50000)", "stage_length_stage2_1000"),
    ("Stage 2", "(5000, 4000, 50000, 50000)", "stage_length_stage2_4000"),
    ("Stage 3", "(5000, 2000, 25000, 50000)", "stage_length_stage3_25000"),
    ("Stage 3", "(5000, 2000, 100000, 50000)", "stage_length_stage3_100000"),
    ("Stage 4", "(5000, 2000, 50000, 25000)", "stage_length_stage4_25000"),
    ("Stage 4", "(5000, 2000, 50000, 100000)", "stage_length_stage4_100000"),
)


def load_runs(root: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(root.rglob("metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["metrics_path"] = str(path)
        runs.append(payload)
    return runs


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def scientific_pair(values: Iterable[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.6e} +/- {std:.6e}"


def fixed_pair(values: Iterable[float], scale: float = 1.0) -> str:
    mean, std = mean_std(value * scale for value in values)
    return f"{mean:.3f} +/- {std:.3f}"


def select(runs: Sequence[Dict[str, Any]], experiment_name: str) -> List[Dict[str, Any]]:
    return [run for run in runs if run["experiment_name"] == experiment_name]


def final_metric(run: Dict[str, Any], name: str) -> float:
    return float(run["evaluations"]["final"][name])


def write_raw_csv(runs: Sequence[Dict[str, Any]], output: Path) -> None:
    fields = (
        "category",
        "experiment_name",
        "variant",
        "method",
        "seed",
        "relative_l2",
        "max_pointwise_error",
        "main_time_per_iteration_ms",
        "adam_total_min",
        "training_total_min",
        "total_trainable",
        "gpu_name",
        "metrics_path",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "category": run["category"],
                    "experiment_name": run["experiment_name"],
                    "variant": run["variant"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "relative_l2": f"{final_metric(run, 'relative_l2'):.12e}",
                    "max_pointwise_error": f"{final_metric(run, 'max_pointwise_error'):.12e}",
                    "main_time_per_iteration_ms": f"{run['stage_timings']['stage3']['seconds_per_step'] * 1000.0:.12e}",
                    "adam_total_min": f"{run['timing_summary']['adam_total_s'] / 60.0:.12e}",
                    "training_total_min": f"{run['timing_summary']['training_total_s'] / 60.0:.12e}",
                    "total_trainable": run["parameter_counts"]["total_trainable"],
                    "gpu_name": run["environment"]["gpu_name"],
                    "metrics_path": run["metrics_path"],
                }
            )


def cost_table(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Table 1: Computational cost on the wave equation",
        "",
        "| Method | Runs | Trainable parameters | Main-stage Adam time/iteration (ms) | Total Adam-stage time (min) | End-to-end training time (min) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, experiment in (("PINNs", "cost_pinn"), ("DASA-PINNs", "cost_dasa")):
        selected = select(runs, experiment)
        if not selected:
            lines.append(f"| {label} | 0 | pending | pending | pending | pending |")
            continue
        params = selected[0]["parameter_counts"]["total_trainable"]
        lines.append(
            "| {} | {} | {:,} | {} | {} | {} |".format(
                label,
                len(selected),
                params,
                fixed_pair(
                    (run["stage_timings"]["stage3"]["seconds_per_step"] for run in selected),
                    1000.0,
                ),
                fixed_pair(
                    (run["timing_summary"]["adam_total_s"] for run in selected),
                    1.0 / 60.0,
                ),
                fixed_pair(
                    (run["timing_summary"]["training_total_s"] for run in selected),
                    1.0 / 60.0,
                ),
            )
        )
    return lines


def sensitivity_table(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Table 2: DASA hyperparameter sensitivity on the wave equation",
        "",
        "| Factor | Setting | Runs | Relative L2 error | Maximum pointwise error | Training time (min) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for factor, setting, experiment in SENSITIVITY_ROWS:
        selected = select(runs, experiment)
        if not selected:
            lines.append(f"| {factor} | {setting} | 0 | pending | pending | pending |")
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                factor,
                setting,
                len(selected),
                scientific_pair(final_metric(run, "relative_l2") for run in selected),
                scientific_pair(final_metric(run, "max_pointwise_error") for run in selected),
                fixed_pair(
                    (run["timing_summary"]["training_total_s"] for run in selected),
                    1.0 / 60.0,
                ),
            )
        )
    return lines


def stage_table(runs: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Table 3: Four-stage length sensitivity on the wave equation",
        "",
        "| Varied stage | (N_S1, N_S2, N_S3, N_S4) | Runs | Relative L2 error | Maximum pointwise error |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, schedule, experiment in STAGE_ROWS:
        selected = select(runs, experiment)
        if not selected:
            lines.append(f"| {stage} | {schedule} | 0 | pending | pending |")
            continue
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                stage,
                schedule,
                len(selected),
                scientific_pair(final_metric(run, "relative_l2") for run in selected),
                scientific_pair(final_metric(run, "max_pointwise_error") for run in selected),
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    write_raw_csv(runs, args.out / "all_runs.csv")
    sections = cost_table(runs) + [""] + sensitivity_table(runs) + [""] + stage_table(runs)
    (args.out / "TABLES.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Aggregated {len(runs)} runs into {args.out / 'TABLES.md'}")


if __name__ == "__main__":
    main()
