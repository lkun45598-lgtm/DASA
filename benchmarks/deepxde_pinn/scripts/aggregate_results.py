"""Aggregate completed baseline runs using the two requested metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = (
    "case",
    "seed",
    "l2_relative_error",
    "maximum_pointwise_error",
)
METRIC_COLUMNS = COLUMNS[2:]


def _formatted(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row["case"],
        "seed": row["seed"],
        **{column: f"{float(row[column]):.6e}" for column in METRIC_COLUMNS},
    }


def aggregate_results(
    root: Path, csv_path: Path, markdown_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metric_path in sorted(root.glob("**/metrics.json")):
        row = json.loads(metric_path.read_text(encoding="utf-8"))
        if set(row) != set(COLUMNS):
            raise ValueError(f"unexpected metric fields in {metric_path}: {sorted(row)}")
        rows.append({column: row[column] for column in COLUMNS})
    rows.sort(key=lambda row: (row["case"], row["seed"]))

    for failure_path in sorted(root.glob("**/failure.json")):
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failures.append({"path": str(failure_path), **failure})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_formatted(row) for row in rows)

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(COLUMNS) + " |\n")
        handle.write("|" + "|".join("---" for _ in COLUMNS) + "|\n")
        for row in rows:
            formatted = _formatted(row)
            handle.write(
                "| " + " | ".join(str(formatted[column]) for column in COLUMNS) + " |\n"
            )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args()
    rows, failures = aggregate_results(
        arguments.root,
        arguments.csv or arguments.root / "results.csv",
        arguments.markdown or arguments.root / "results.md",
    )
    print(f"completed={len(rows)} failed={len(failures)}")
    for failure in failures:
        print(f"FAILED {failure.get('case', failure['path'])}: {failure.get('error', '')}")


if __name__ == "__main__":
    main()
