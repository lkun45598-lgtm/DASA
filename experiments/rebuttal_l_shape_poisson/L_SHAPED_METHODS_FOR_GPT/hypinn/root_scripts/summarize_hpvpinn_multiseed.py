"""Aggregate the fixed 10-seed hp-VPINN Table 1 runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "table1_hpvpinn_multiseed_20260811"
POISSON_BOUNDARY_PREFINED_ROOT = (
    ROOT / "results" / "table1_hpvpinn_poisson_boundary_prefined_20260811"
)
HELMHOLTZ_1D_FULL_H20_ROOT = (
    ROOT / "results" / "helmholtz1d_full_h20_multiseed_20260811"
)
SEEDS = tuple(range(1234, 1244))
PROBLEMS = (
    "poisson_1d_boundary",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
    "cavity_re100",
    "burgers_1d",
    "allen_cahn_1d",
    "diffusion_reaction_1d",
    "advection_diffusion_1d",
    "poisson_lshape",
    "advection_discontinuous_1d",
)

SEED_1234_RUNS = {
    "poisson_1d_boundary": ROOT / "results/table1_hpvpinn_optimized/poisson_1d_boundary/adam_continuation_lr1e-5_seed_1234",
    "poisson_1d_steep": ROOT / "results/table1_hpvpinn_optimized/poisson_1d_steep/p180_q200_continuation_seed_1234",
    "helmholtz_1d": ROOT / "results/table1_hpvpinn_helmholtz_lshape_p1/helmholtz_1d/strong_p20_q32_seed_1234",
    "poisson_2d_peak": ROOT / "results/table1_hpvpinn_corrected/poisson_2d_peak/seed_1234",
    "poisson_2d_steep": ROOT / "results/table1_hpvpinn_corrected/poisson_2d_steep/p1_p8_q16_continuation_seed_1234",
    "helmholtz_2d": ROOT / "results/table1_hpvpinn_helmholtz_lshape_p1/helmholtz_2d/p1_h8_p5_q10_seed_1234",
    "cavity_re100": ROOT / "results/table1_hpvpinn_corrected/cavity_re100/seed_1234",
    "burgers_1d": ROOT / "results/table1_hpvpinn_burgers_paper_r1_20260810/spatial_refined_p20q80",
    "allen_cahn_1d": ROOT / "results/table1_hpvpinn_allen_spacetime_normalized_20260810/p10q20_h5x8_lr1e4_seed_1234",
    "diffusion_reaction_1d": ROOT / "results/table1_hpvpinn_diffusion_reaction_candidates_20260811/p1_prefined_p10q20_continuation",
    "advection_diffusion_1d": ROOT / "results/table1_hpvpinn_kappa001/advection_diffusion_1d/h4_p5_q10_100k_lr1e-5_seed_1234",
    "poisson_lshape": ROOT / "results/table1_hpvpinn_helmholtz_lshape_p1/poisson_lshape/p1_p8_q16_seed_1234",
    "advection_discontinuous_1d": ROOT / "results/table1_hpvpinn_corrected/advection_discontinuous_1d/seed_1234",
}

POISSON_BOUNDARY_PREFINED_READY = all(
    (POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "final" / "metrics.json").is_file()
    and (POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "final" / "config.json").is_file()
    for seed in SEEDS
)
POISSON_BOUNDARY_P180_READY = all(
    (POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "p180_q200" / "metrics.json").is_file()
    and (POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "p180_q200" / "config.json").is_file()
    for seed in SEEDS
)
HELMHOLTZ_1D_FULL_H20_READY = all(
    (HELMHOLTZ_1D_FULL_H20_ROOT / f"seed_{seed}" / "metrics.json").is_file()
    and (HELMHOLTZ_1D_FULL_H20_ROOT / f"seed_{seed}" / "config.json").is_file()
    for seed in SEEDS
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write interim summaries instead of failing when runs are missing",
    )
    return parser.parse_args()


def run_dir(problem: str, seed: int) -> Path:
    if problem == "poisson_1d_boundary" and POISSON_BOUNDARY_P180_READY:
        return POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "p180_q200"
    if problem == "poisson_1d_boundary" and POISSON_BOUNDARY_PREFINED_READY:
        return POISSON_BOUNDARY_PREFINED_ROOT / f"seed_{seed}" / "final"
    if problem == "helmholtz_1d" and HELMHOLTZ_1D_FULL_H20_READY:
        return HELMHOLTZ_1D_FULL_H20_ROOT / f"seed_{seed}"
    if seed == 1234:
        return SEED_1234_RUNS[problem]
    return RESULT_ROOT / problem / f"seed_{seed}" / "final"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_rows: list[dict[str, object]] = []
    missing: list[str] = []

    for problem in PROBLEMS:
        for seed in SEEDS:
            directory = run_dir(problem, seed)
            metrics_path = directory / "metrics.json"
            config_path = directory / "config.json"
            if not metrics_path.is_file() or not config_path.is_file():
                missing.append(f"{problem}/seed_{seed}")
                continue
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
            with config_path.open(encoding="utf-8") as file:
                config = json.load(file)
            if metrics["problem"] != problem or int(metrics["seed"]) != seed:
                raise ValueError(f"metric identity mismatch in {directory}")
            if config.get("hidden_layers") != 6 or config.get("width") != 128:
                raise ValueError(f"network is not 6x128 in {directory}")
            raw_rows.append(
                {
                    "problem": problem,
                    "seed": seed,
                    "relative_l2_error": float(metrics["relative_l2_error"]),
                    "max_pointwise_error": float(metrics["max_pointwise_error"]),
                    "run_dir": str(directory.resolve()),
                }
            )

    if missing and not args.allow_incomplete:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing {len(missing)} runs; first missing: {preview}")

    summary_rows: list[dict[str, object]] = []
    for problem in PROBLEMS:
        rows = [row for row in raw_rows if row["problem"] == problem]
        l2_values = [float(row["relative_l2_error"]) for row in rows]
        max_values = [float(row["max_pointwise_error"]) for row in rows]
        summary_rows.append(
            {
                "problem": problem,
                "completed_runs": len(rows),
                "expected_runs": len(SEEDS),
                "relative_l2_mean": statistics.mean(l2_values) if l2_values else "",
                "relative_l2_sample_variance": statistics.variance(l2_values) if len(l2_values) > 1 else "",
                "relative_l2_sample_std": statistics.stdev(l2_values) if len(l2_values) > 1 else "",
                "max_error_mean": statistics.mean(max_values) if max_values else "",
                "max_error_sample_variance": statistics.variance(max_values) if len(max_values) > 1 else "",
                "max_error_sample_std": statistics.stdev(max_values) if len(max_values) > 1 else "",
            }
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULT_ROOT / "raw_metrics.csv",
        raw_rows,
        ["problem", "seed", "relative_l2_error", "max_pointwise_error", "run_dir"],
    )
    summary_fields = [
        "problem",
        "completed_runs",
        "expected_runs",
        "relative_l2_mean",
        "relative_l2_sample_variance",
        "relative_l2_sample_std",
        "max_error_mean",
        "max_error_sample_variance",
        "max_error_sample_std",
    ]
    write_csv(RESULT_ROOT / "summary.csv", summary_rows, summary_fields)

    def format_value(row: dict[str, object], key: str) -> str:
        value = row[key]
        return "" if value == "" else f"{float(value):.6e}"

    def format_mean_pm_variance(
        row: dict[str, object], mean_key: str, variance_key: str
    ) -> str:
        mean = format_value(row, mean_key)
        variance = format_value(row, variance_key)
        return f"{mean} ± {variance}" if mean and variance else ""

    table_rows = [
        {
            "problem": row["problem"],
            "completed_runs": row["completed_runs"],
            "expected_runs": row["expected_runs"],
            "relative_l2_mean_pm_sample_variance": format_mean_pm_variance(
                row, "relative_l2_mean", "relative_l2_sample_variance"
            ),
            "max_error_mean_pm_sample_variance": format_mean_pm_variance(
                row, "max_error_mean", "max_error_sample_variance"
            ),
        }
        for row in summary_rows
    ]
    table_fields = [
        "problem",
        "completed_runs",
        "expected_runs",
        "relative_l2_mean_pm_sample_variance",
        "max_error_mean_pm_sample_variance",
    ]
    write_csv(
        RESULT_ROOT / "table1_mean_pm_sample_variance.csv", table_rows, table_fields
    )

    lines = [
        "# hp-VPINN 10-Seed Summary",
        "",
        "The displayed uncertainty is the sample variance (ddof=1), not the standard deviation.",
        "",
        "| Problem | Runs | Relative L2 mean ± variance | Max error mean ± variance |",
        "|---|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['problem']} | {row['completed_runs']}/{row['expected_runs']} | "
            f"{row['relative_l2_mean_pm_sample_variance'] or '-'} | "
            f"{row['max_error_mean_pm_sample_variance'] or '-'} |"
        )
    if missing:
        lines.extend(["", f"Incomplete: {len(missing)} runs are still missing."])
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"completed={len(raw_rows)}/{len(PROBLEMS) * len(SEEDS)} missing={len(missing)}")
    print(RESULT_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
