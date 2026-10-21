#!/usr/bin/env python3
"""Aggregate four-stage iteration-length sensitivity results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(root: Path, exclusions):
    records = []
    for path in sorted(root.rglob("metrics.json")):
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        key = (item["experiment_name"], int(item["seed"]))
        if key in exclusions:
            continue
        item["_path"] = str(path)
        records.append(item)
    return records


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def std(values):
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sci(value):
    return "NA" if value is None or math.isnan(value) else f"{value:.6e}"


def aggregate(records, evaluation_label, category):
    grouped = defaultdict(list)
    for record in records:
        if record.get("category") == category:
            grouped[record["variant"]].append(record)

    rows = []
    for variant, items in sorted(grouped.items()):
        first = items[0]
        opt = first["config"]["optimization"]
        evaluation = [item["evaluations"][evaluation_label] for item in items]
        rows.append({
            "axis": first.get("sensitivity_axis"),
            "variant": variant,
            "stage1_steps": int(opt["warmup_steps"]),
            "stage2_steps": int(opt["subnet_pretrain_steps"]),
            "stage3_steps": int(opt["main_steps"]),
            "stage4_steps": int(opt["finetune_steps"]),
            "n_seeds": len(items),
            "relative_l2_mean": mean(x["relative_l2"] for x in evaluation),
            "relative_l2_std": std(x["relative_l2"] for x in evaluation),
            "max_pointwise_error_mean": mean(x["max_abs_error"] for x in evaluation),
            "max_pointwise_error_std": std(x["max_abs_error"] for x in evaluation),
        })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "axis", "variant", "stage1_steps", "stage2_steps", "stage3_steps", "stage4_steps",
        "n_seeds", "relative_l2_mean", "relative_l2_std",
        "max_pointwise_error_mean", "max_pointwise_error_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows, evaluation_label):
    evaluation_description = (
        "after the common L-BFGS refinement"
        if evaluation_label == "final"
        else "after Stage 4"
    )
    lines = [
        "# Stage-length sensitivity: L-shaped Poisson",
        "",
        f"The four stage lengths are varied one factor at a time. All other training settings are fixed, including the Stage 4 starting learning rate (5.904900e-04). Network parameters and Adam moment states are retained at the Stage 3--4 transition. Errors are evaluated {evaluation_description} and reported as mean +/- sample standard deviation over completed seeds.",
        "",
        "| Axis | Variant | (S1, S2, S3, S4) | Seeds | Relative L2 error | Maximum pointwise error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        schedule = f"({row['stage1_steps']}, {row['stage2_steps']}, {row['stage3_steps']}, {row['stage4_steps']})"
        lines.append(
            f"| {row['axis']} | {row['variant']} | {schedule} | {row['n_seeds']} | "
            f"{sci(row['relative_l2_mean'])} +/- {sci(row['relative_l2_std'])} | "
            f"{sci(row['max_pointwise_error_mean'])} +/- {sci(row['max_pointwise_error_std'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--evaluation-label", choices=("after_finetune", "final"), default="after_finetune"
    )
    parser.add_argument("--category", default="stage_length")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="EXPERIMENT:SEED",
        help="Exclude one run from the aggregate; may be repeated.",
    )
    args = parser.parse_args()
    exclusions = set()
    for value in args.exclude:
        experiment, separator, seed = value.rpartition(":")
        if not separator or not experiment:
            parser.error(f"invalid --exclude value: {value!r}")
        exclusions.add((experiment, int(seed)))
    rows = aggregate(load_records(args.results, exclusions), args.evaluation_label, args.category)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "stage_length_summary.csv", rows)
    write_markdown(args.out / "stage_length_summary.md", rows, args.evaluation_label)
    print(f"completed configurations: {len(rows)}")
    print(f"completed runs: {sum(row['n_seeds'] for row in rows)}")
    print(args.out / "stage_length_summary.md")


if __name__ == "__main__":
    main()
