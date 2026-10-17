"""Build reproducible CSV and Markdown summaries for all hp-VPINN runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY_DIR = RESULTS / "experiment_summary"

PROBLEM_ORDER = (
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

DISPLAY_NAMES = {
    "poisson_1d_boundary": "1D Poisson boundary",
    "poisson_1d_steep": "1D Poisson steep",
    "helmholtz_1d": "1D Helmholtz",
    "poisson_2d_peak": "2D Poisson peak",
    "poisson_2d_steep": "2D Poisson steep",
    "helmholtz_2d": "2D Helmholtz",
    "cavity_re100": "Cavity Re=100",
    "burgers_1d": "1D Burgers",
    "allen_cahn_1d": "1D Allen-Cahn",
    "diffusion_reaction_1d": "1D diffusion-reaction",
    "advection_diffusion_1d": "1D advection-diffusion",
    "poisson_lshape": "L-shaped Poisson",
    "advection_discontinuous_1d": "1D discontinuous advection",
}

TREES = (
    ("historical_baseline", RESULTS / "table1_hpvpinn"),
    ("corrected_baseline", RESULTS / "table1_hpvpinn_corrected"),
    ("optimized_reference", RESULTS / "table1_hpvpinn_optimized"),
    ("ade_kappa_0.01", RESULTS / "table1_hpvpinn_kappa001"),
    ("helmholtz_lshape_high_order", RESULTS / "table1_hpvpinn_helmholtz_lshape_p1"),
)

# These historical runs used the pre-fix 2D tensor-product ordering.
TWO_COORDINATE_PROBLEMS = {
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
}

REFERENCE_MAX_ERROR = {
    "poisson_1d_boundary": 1.0e-3,
    "poisson_1d_steep": 1.0e-4,
    "poisson_2d_steep": 1.0e-3,
    "advection_diffusion_1d": 1.0e-1,
    "poisson_lshape": 2.0e-2,
}

SELECTED_RUNS = {
    "poisson_1d_boundary": RESULTS / "table1_hpvpinn_optimized/poisson_1d_boundary/adam_continuation_lr1e-5_seed_1234",
    "poisson_1d_steep": RESULTS / "table1_hpvpinn_optimized/poisson_1d_steep/p180_q200_continuation_seed_1234",
    "helmholtz_1d": RESULTS / "table1_hpvpinn_helmholtz_lshape_p1/helmholtz_1d/strong_p20_q32_seed_1234",
    "poisson_2d_peak": RESULTS / "table1_hpvpinn_corrected/poisson_2d_peak/seed_1234",
    "poisson_2d_steep": RESULTS / "table1_hpvpinn_corrected/poisson_2d_steep/p1_p8_q16_continuation_seed_1234",
    "helmholtz_2d": RESULTS / "table1_hpvpinn_helmholtz_lshape_p1/helmholtz_2d/p1_h8_p5_q10_seed_1234",
    "cavity_re100": RESULTS / "table1_hpvpinn_corrected/cavity_re100/seed_1234",
    "burgers_1d": RESULTS / "table1_hpvpinn_corrected/burgers_1d/seed_1234",
    "allen_cahn_1d": RESULTS / "table1_hpvpinn_corrected/allen_cahn_1d/seed_1234",
    "diffusion_reaction_1d": RESULTS / "table1_hpvpinn_corrected/diffusion_reaction_1d/seed_1234",
    "advection_diffusion_1d": RESULTS / "table1_hpvpinn_kappa001/advection_diffusion_1d/h4_p5_q10_100k_lr1e-5_seed_1234",
    "poisson_lshape": RESULTS / "table1_hpvpinn_helmholtz_lshape_p1/poisson_lshape/p1_p8_q16_seed_1234",
    "advection_discontinuous_1d": RESULTS / "table1_hpvpinn_corrected/advection_discontinuous_1d/seed_1234",
}

NEEDS_ACCURACY_VALIDATION = {"helmholtz_1d"}


def load_run(run_dir: Path, group: str) -> dict[str, object]:
    with (run_dir / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    with (run_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    problem = metrics["problem"]
    kappa = config.get("ade_kappa")
    if problem == "advection_diffusion_1d" and kappa is None:
        kappa = 0.1 / math.pi
    variational_form = config.get("variational_form", "unknown")
    if str(variational_form).startswith("local strong-residual"):
        variational_form = "strong"

    validity = "valid_fixed_endpoint"
    if group == "historical_baseline" and problem in TWO_COORDINATE_PROBLEMS:
        validity = "historical_needs_rerun_after_2d_ordering_fix"
    elif group == "historical_baseline":
        validity = "historical_1d_baseline"
    elif group == "corrected_baseline":
        validity = "corrected_fixed_endpoint"
    elif group == "helmholtz_lshape_high_order":
        validity = "high_order_fixed_endpoint"

    return {
        "group": group,
        "problem": problem,
        "display_name": DISPLAY_NAMES[problem],
        "variant": run_dir.name,
        "seed": metrics["seed"],
        "network": f"{config.get('hidden_layers', 6)}x{config.get('width', 128)} modified MLP",
        "dtype": config.get("dtype", "float64"),
        "variational_form": variational_form,
        "mesh_variant": config.get("mesh_variant", "historical"),
        "n_elements": config.get("n_elements", ""),
        "n_test": config.get("n_test", ""),
        "n_quad": config.get("n_quad", ""),
        "learning_rate": config.get("learning_rate", ""),
        "kappa": "" if kappa is None else kappa,
        "stage_iterations": int(metrics["iterations"]),
        "relative_l2_error": float(metrics["relative_l2_error"]),
        "max_pointwise_error": float(metrics["max_pointwise_error"]),
        "final_loss": float(metrics["final_loss"]),
        "reference_method": metrics.get("evaluation", {}).get("reference_method", "analytic_or_published_data"),
        "validity": validity,
        "initial_checkpoint": config.get("initial_checkpoint"),
        "run_dir": str(run_dir.resolve()),
    }


def cumulative_iterations(run: dict[str, object], by_dir: dict[Path, dict[str, object]], stack: set[Path] | None = None) -> int:
    source = run.get("initial_checkpoint")
    if not source:
        return int(run["stage_iterations"])
    source_dir = Path(str(source)).resolve().parent
    if source_dir not in by_dir:
        return int(run["stage_iterations"])
    stack = set() if stack is None else stack
    current_dir = Path(str(run["run_dir"])).resolve()
    if current_dir in stack:
        raise RuntimeError(f"Checkpoint cycle at {current_dir}")
    return int(run["stage_iterations"]) + cumulative_iterations(by_dir[source_dir], by_dir, stack | {current_dir})


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sci(value: object) -> str:
    return f"{float(value):.6e}"


def main() -> None:
    rows: list[dict[str, object]] = []
    for group, tree in TREES:
        for metrics_path in sorted(tree.glob("*/*/metrics.json")):
            rows.append(load_run(metrics_path.parent, group))

    by_dir = {Path(str(row["run_dir"])).resolve(): row for row in rows}
    for row in rows:
        row["cumulative_iterations"] = cumulative_iterations(row, by_dir)
        row["selected_for_current_table"] = Path(str(row["run_dir"])).resolve() == SELECTED_RUNS[row["problem"]].resolve()

    order = {name: index for index, name in enumerate(PROBLEM_ORDER)}
    rows.sort(key=lambda row: (order[str(row["problem"])], str(row["group"]), str(row["variant"])))

    selected = [row for row in rows if row["selected_for_current_table"]]
    selected.sort(key=lambda row: order[str(row["problem"])])
    for row in selected:
        anchor = REFERENCE_MAX_ERROR.get(str(row["problem"]))
        row["reference_max_error"] = "" if anchor is None else anchor
        row["meets_reference_anchor"] = "" if anchor is None else float(row["max_pointwise_error"]) <= anchor
        if str(row["validity"]).startswith("historical_needs_rerun"):
            row["table_status"] = "needs_rerun_after_2d_ordering_fix"
        elif str(row["problem"]) in NEEDS_ACCURACY_VALIDATION:
            row["table_status"] = "q(x)_assumption_needs_confirmation"
        else:
            row["table_status"] = "current_fixed_endpoint_candidate"

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "group", "problem", "display_name", "variant", "seed", "network", "dtype",
        "variational_form", "mesh_variant", "n_elements", "n_test", "n_quad", "learning_rate",
        "kappa", "stage_iterations", "cumulative_iterations", "relative_l2_error",
        "max_pointwise_error", "final_loss", "reference_method", "validity",
        "selected_for_current_table", "initial_checkpoint", "run_dir",
    ]
    selected_fields = [
        "problem", "display_name", "variant", "relative_l2_error", "max_pointwise_error",
        "reference_max_error", "meets_reference_anchor", "cumulative_iterations", "variational_form",
        "n_elements", "n_test", "n_quad", "learning_rate", "kappa", "table_status", "run_dir",
    ]
    write_csv(SUMMARY_DIR / "all_runs.csv", rows, all_fields)
    write_csv(SUMMARY_DIR / "current_table.csv", selected, selected_fields)

    lines = [
        "# hp-VPINN Experiment Summary",
        "",
        "Snapshot: 2026-08-10. All runs use seed 1234, float64, and a 6 x 128 modified MLP.",
        "Metrics are fixed-endpoint Relative L2 and maximum pointwise error.",
        "",
        "## Current 13-Equation Table",
        "",
        "| Equation | Variant | Relative L2 | Max error | Max-error anchor | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in selected:
        anchor = row["reference_max_error"]
        anchor_text = "-" if anchor == "" else sci(anchor)
        lines.append(
            f"| {row['display_name']} | `{row['variant']}` | {sci(row['relative_l2_error'])} | "
            f"{sci(row['max_pointwise_error'])} | {anchor_text} | `{row['table_status']}` |"
        )

    group_titles = {
        "historical_baseline": "Historical Unified Baseline (13 Runs)",
        "corrected_baseline": "Tensor-Ordering Corrected Runs",
        "optimized_reference": "Reference and Optimization Variants (13 Runs)",
        "ade_kappa_0.01": "ADE kappa=0.01 Variants (8 Runs)",
        "helmholtz_lshape_high_order": "Helmholtz and L-shaped High-Order Candidates",
    }
    for group, title in group_titles.items():
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Equation | Variant | Form | Elements | p / Q | LR | kappa | Relative L2 | Max error | Cumulative iterations | Validity |",
                "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            if row["group"] != group:
                continue
            kappa_text = "-" if row["kappa"] == "" else sci(row["kappa"])
            lines.append(
                f"| {row['display_name']} | `{row['variant']}` | {row['variational_form']} | "
                f"{row['n_elements']} | {row['n_test']} / {row['n_quad']} | {row['learning_rate']} | "
                f"{kappa_text} | {sci(row['relative_l2_error'])} | {sci(row['max_pointwise_error'])} | "
                f"{row['cumulative_iterations']} | `{row['validity']}` |"
            )

    lines.extend(
        [
            "",
            "## Reference-Anchor Check",
            "",
            "| Equation | Reference max error | Current max error | Meets anchor |",
            "|---|---:|---:|---|",
        ]
    )
    for row in selected:
        if row["reference_max_error"] == "":
            continue
        lines.append(
            f"| {row['display_name']} | {sci(row['reference_max_error'])} | "
            f"{sci(row['max_pointwise_error'])} | {'yes' if row['meets_reference_anchor'] else 'no'} |"
        )

    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[str(row["group"])] = group_counts.get(str(row["group"]), 0) + 1
    lines.extend(
        [
            "",
            "## Run Inventory",
            "",
            f"- Historical unified baseline: {group_counts.get('historical_baseline', 0)} runs.",
            f"- Tensor-ordering corrected runs: {group_counts.get('corrected_baseline', 0)} runs.",
            f"- Reference/optimization variants: {group_counts.get('optimized_reference', 0)} runs.",
            f"- ADE kappa=0.01 variants: {group_counts.get('ade_kappa_0.01', 0)} runs.",
            f"- Helmholtz/L-shaped high-order candidates: {group_counts.get('helmholtz_lshape_high_order', 0)} runs.",
            f"- Total fixed-endpoint results: {len(rows)} runs.",
            "",
            "## Validity Notes",
            "",
            "- ADE selected for the current table uses kappa=0.01. Its independent evaluation reference is a 16,384-cell Crank-Nicolson FDM solution; reference data are not used in training.",
            "- Older ADE results under `table1_hpvpinn` and `table1_hpvpinn_optimized` use kappa=0.1/pi and are retained only as a different-PDE record.",
            "- Historical two-coordinate baseline runs were produced before the local quadrature/test tensor ordering fix. Their values are preserved in `all_runs.csv`, but rows marked `needs_rerun_after_2d_ordering_fix` are not final comparison results.",
            "- Selected corrected runs use the verified local ordering: x is the outer quadrature index and y varies fastest. Their training configurations match the historical baseline or stated continuation lineage.",
            "- The selected L-shaped Poisson result is the fixed-endpoint P1 p8/q16 high-order candidate on the official 35-element mesh; the published-code strong p5/q10 configuration remains in the corrected baseline group.",
            "- The selected 1D Poisson steep run improves the maximum error to 8.807974e-4 but does not reach the supplied 1e-4 anchor.",
            "- `cumulative_iterations` follows the full initialization-checkpoint lineage; it is more informative than the stage-local iteration field for continuation runs.",
            "",
            "## Files",
            "",
            "- `all_runs.csv`: every fixed-endpoint run and its full configuration metadata.",
            "- `current_table.csv`: one current selected row per equation, including validity and anchor status.",
        ]
    )
    (SUMMARY_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} runs and {len(selected)} selected rows to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
