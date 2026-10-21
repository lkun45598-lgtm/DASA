"""Build CSV and Markdown summaries only from actual metrics files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import argparse

from .config import CASE_NAMES


METRIC_FIELDS = (
    "relative_l2",
    "max_abs",
    "u_relative_l2",
    "u_max_abs",
    "v_relative_l2",
    "v_max_abs",
    "velocity_relative_l2",
    "velocity_max_pointwise_l2",
)


FIELDS = (
    "case",
    "seed",
    "network",
    "n_residual",
    "n_boundary",
    "n_initial",
    "adam_iterations",
    "lbfgs_iterations",
    *METRIC_FIELDS,
    "reference_type",
    "full_training_completed",
    "result_directory",
)


def collect(output_root: Path, mode: str) -> list[dict]:
    rows: list[dict] = []
    mode_root = output_root / mode
    if not mode_root.exists():
        return rows
    for metrics_path in sorted(mode_root.glob("*/seed_*/metrics.json")):
        result_dir = metrics_path.parent
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (result_dir / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        sampling, optimizer = config["sampling"], config["optimizer"]
        row = {
            "case": config["case"],
            "seed": config["seed"],
            "network": "6x128 tanh (float64)",
            "n_residual": sampling["n_residual"],
            "n_boundary": sampling["n_boundary"],
            "n_initial": sampling["n_initial"],
            "adam_iterations": metrics.get(
                "adam_iterations_completed", optimizer["adam_iterations"]
            ),
            "lbfgs_iterations": optimizer["lbfgs_iterations"],
            "reference_type": config["reference_type"],
            "full_training_completed": metrics.get(
                "full_training_completed", False
            ),
            "result_directory": str(result_dir),
        }
        row.update({name: metrics.get(name) for name in METRIC_FIELDS})
        rows.append(row)
    case_order = {name: index for index, name in enumerate(CASE_NAMES)}
    return sorted(rows, key=lambda row: (case_order[row["case"]], int(row["seed"])))


def write_summary(output_root: Path, mode: str) -> tuple[Path, Path]:
    rows = collect(output_root, mode)
    mode_root = output_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    csv_path = mode_root / "results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = mode_root / "results_summary.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# gPINN results ({mode})\n\n")
        handle.write("All values below were read from completed run artifacts.\n\n")
        handle.write(
            "| Case | Seed | Network | Nr/Nb/Nic | Adam/L-BFGS | Relative L2 | Max abs | Reference | Full |\n"
        )
        handle.write("|---|---:|---|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['case']} | {row['seed']} | {row['network']} | "
                f"{row['n_residual']}/{row['n_boundary']}/{row['n_initial']} | "
                f"{row['adam_iterations']}/{row['lbfgs_iterations']} | "
                f"{row['relative_l2']:.8e} | {row['max_abs']:.8e} | "
                f"{row['reference_type']} | {row['full_training_completed']} |\n"
            )
    return csv_path, markdown_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/gpinn_baselines"))
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args(argv)
    csv_path, markdown_path = write_summary(args.output_root, args.mode)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
