"""Aggregate repeated runs into per-case mean and sample-standard-deviation tables."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import CASE_NAMES
from .seeds import DEFAULT_SEED_SPEC, canonical_seed_spec, parse_seed_spec
from .summary import METRIC_FIELDS, collect


CASE_DISPLAY_NAMES = {
    "poisson_1d_boundary_layer": "1D Poisson（boundary-layer）",
    "poisson_1d_steep": "1D Poisson（steep）",
    "helmholtz_1d": "1D Helmholtz",
    "poisson_2d_peak": "2D Poisson（peak）",
    "poisson_2d_steep": "2D Poisson（steep）",
    "helmholtz_2d": "2D Helmholtz",
    "lid_driven_cavity": "Lid-driven cavity（Re = 100）",
    "burgers_1d": "1D Burgers",
    "allen_cahn_1d": "1D Allen–Cahn（D = 1e-4）",
    "diffusion_reaction_1d": "1D Diffusion–reaction",
    "advection_diffusion_1d": "1D Advection–diffusion",
    "poisson_lshape": "L-shaped Poisson",
    "advection_discontinuous_1d": "1D Advection（discontinuous）",
}


METRIC_LABELS = {
    "relative_l2": "相对 L2 误差",
    "max_abs": "最大逐点误差",
    "u_relative_l2": "u 分量相对 L2",
    "u_max_abs": "u 分量最大误差",
    "v_relative_l2": "v 分量相对 L2",
    "v_max_abs": "v 分量最大误差",
    "velocity_relative_l2": "联合速度相对 L2",
    "velocity_max_pointwise_l2": "最大逐点速度二范数",
}


BASE_STAT_FIELDS = (
    "index",
    "case",
    "case_name",
    "seeds",
    "completed_runs",
    "network",
    "n_residual",
    "n_boundary",
    "n_initial",
    "adam_iterations",
    "lbfgs_iterations",
)

STAT_FIELDS = (
    *BASE_STAT_FIELDS,
    *(field for metric in METRIC_FIELDS for field in (
        f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"
    )),
    "reference_type",
    "all_full_training_completed",
)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    finite = [float(value) for value in values]
    if not finite:
        raise ValueError("Cannot aggregate an empty metric")
    if not all(math.isfinite(value) for value in finite):
        raise FloatingPointError("A metric contains NaN or Inf")
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.6e} ± {std:.6e}"


def aggregate_rows(
    rows: list[dict[str, Any]],
    expected_seeds: tuple[int, ...] | None = None,
    require_complete: bool = False,
    case_names: tuple[str, ...] = CASE_NAMES,
) -> list[dict[str, Any]]:
    """Return one statistically audited row per equation.

    Standard deviations are sample standard deviations (N-1 denominator).
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case"])].append(row)

    result: list[dict[str, Any]] = []
    unknown = sorted(set(case_names) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"Unknown case names: {unknown}")
    if len(set(case_names)) != len(case_names):
        raise ValueError("case_names contains duplicates")

    for index, case in enumerate(case_names, start=1):
        case_rows = sorted(grouped.get(case, []), key=lambda row: int(row["seed"]))
        if not case_rows:
            raise FileNotFoundError(f"No result rows for {case}")
        actual_seeds = tuple(int(row["seed"]) for row in case_rows)
        if len(set(actual_seeds)) != len(actual_seeds):
            raise ValueError(f"{case}: duplicate seed result")
        if expected_seeds is not None and actual_seeds != tuple(expected_seeds):
            missing = sorted(set(expected_seeds) - set(actual_seeds))
            extra = sorted(set(actual_seeds) - set(expected_seeds))
            raise ValueError(
                f"{case}: seed mismatch; missing={missing}, extra={extra}, "
                f"actual={list(actual_seeds)}"
            )
        if require_complete and not all(
            bool(row["full_training_completed"]) for row in case_rows
        ):
            incomplete = [
                int(row["seed"])
                for row in case_rows
                if not bool(row["full_training_completed"])
            ]
            raise ValueError(f"{case}: incomplete full seeds {incomplete}")

        constant_fields = (
            "network",
            "n_residual",
            "n_boundary",
            "n_initial",
            "adam_iterations",
            "lbfgs_iterations",
            "reference_type",
        )
        for field in constant_fields:
            values = {str(row[field]) for row in case_rows}
            if len(values) != 1:
                raise ValueError(f"{case}: inconsistent {field}: {sorted(values)}")

        row: dict[str, Any] = {
            "index": index,
            "case": case,
            "case_name": CASE_DISPLAY_NAMES[case],
            "seeds": canonical_seed_spec(list(actual_seeds)),
            "completed_runs": len(case_rows),
            "network": case_rows[0]["network"],
            "n_residual": case_rows[0]["n_residual"],
            "n_boundary": case_rows[0]["n_boundary"],
            "n_initial": case_rows[0]["n_initial"],
            "adam_iterations": case_rows[0]["adam_iterations"],
            "lbfgs_iterations": case_rows[0]["lbfgs_iterations"],
            "reference_type": case_rows[0]["reference_type"],
            "all_full_training_completed": all(
                bool(item["full_training_completed"]) for item in case_rows
            ),
        }
        for metric in METRIC_FIELDS:
            values = [item[metric] for item in case_rows if item.get(metric) is not None]
            if not values:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_mean_std"] = ""
                continue
            if len(values) != len(case_rows):
                raise ValueError(f"{case}: metric {metric} is missing for some seeds")
            mean, std = _mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = _format_mean_std(mean, std)
        result.append(row)
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# gPINN 十种子统计\n\n")
        handle.write(
            "均值和标准差仅来自保存的完整实验；标准差为样本标准差（ddof=1）。\n\n"
        )
        handle.write(
            "| 算例 | 种子 | 完成数 | 相对 L2（mean ± std） | 最大误差（mean ± std） | 参考解 | 完整 |\n"
        )
        handle.write("|---|---|---:|---:|---:|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['case_name']} | {row['seeds']} | {row['completed_runs']} | "
                f"{row['relative_l2_mean_std']} | {row['max_abs_mean_std']} | "
                f"{row['reference_type']} | {row['all_full_training_completed']} |\n"
            )


def _write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as error:  # pragma: no cover - exercised by server preflight
        raise RuntimeError(
            "XLSX export requires openpyxl; install requirements.txt"
        ) from error

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "十种子汇总"
    headers = [
        "序号",
        "算例标识",
        "算例名称",
        "随机种子",
        "完成次数",
        *(f"{METRIC_LABELS[metric]}（mean ± std）" for metric in METRIC_FIELDS),
        "参考解类型",
        "10次完整训练完成",
    ]
    sheet.append(headers)
    for row in rows:
        sheet.append(
            [
                row["index"],
                row["case"],
                row["case_name"],
                row["seeds"],
                row["completed_runs"],
                *(row[f"{metric}_mean_std"] for metric in METRIC_FIELDS),
                row["reference_type"],
                "是" if row["all_full_training_completed"] else "否",
            ]
        )

    header_fill = PatternFill("solid", fgColor="1F4E78")
    band_fill = PatternFill("solid", fgColor="C6E7F5")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_index in range(2, sheet.max_row + 1):
        if row_index % 2 == 0:
            for cell in sheet[row_index]:
                cell.fill = band_fill
        for cell in sheet[row_index]:
            cell.alignment = Alignment(vertical="center")
        for column in range(1, sheet.max_column + 1):
            if column not in (2, 3, sheet.max_column - 1):
                sheet.cell(row_index, column).alignment = Alignment(
                    horizontal="center", vertical="center"
                )

    widths = {
        "A": 8,
        "B": 32,
        "C": 34,
        "D": 23,
        "E": 10,
        "F": 27,
        "G": 27,
        "H": 27,
        "I": 27,
        "J": 27,
        "K": 27,
        "L": 29,
        "M": 33,
        "N": 40,
        "O": 18,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.row_dimensions[1].height = 42
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False
    sheet.auto_filter.ref = sheet.dimensions
    table = Table(displayName="GPINNMeanStdSummary", ref=sheet.dimensions)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=False,
        showColumnStripes=False,
    )
    sheet.add_table(table)
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    workbook.save(path)


def write_statistics(
    output_root: Path,
    mode: str,
    expected_seeds: tuple[int, ...] | None,
    require_complete: bool,
    case_names: tuple[str, ...] = CASE_NAMES,
) -> tuple[Path, Path, Path]:
    mode_root = output_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    rows = aggregate_rows(
        collect(output_root, mode),
        expected_seeds=expected_seeds,
        require_complete=require_complete,
        case_names=case_names,
    )
    csv_path = mode_root / "results_mean_std.csv"
    markdown_path = mode_root / "results_mean_std.md"
    xlsx_path = mode_root / "results_mean_std.xlsx"
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    _write_xlsx(rows, xlsx_path)
    return csv_path, markdown_path, xlsx_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/gpinn_baselines_10seeds"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", default=DEFAULT_SEED_SPEC)
    parser.add_argument(
        "--cases",
        help="Comma-separated case subset; the default is all 13 equations.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    case_names = (
        tuple(item.strip() for item in args.cases.split(",") if item.strip())
        if args.cases
        else CASE_NAMES
    )
    if not case_names:
        parser.error("--cases must contain at least one case")
    paths = write_statistics(
        args.output_root,
        args.mode,
        parse_seed_spec(args.seeds),
        require_complete=not args.allow_incomplete,
        case_names=case_names,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
