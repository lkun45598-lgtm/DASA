"""Aggregate completed runs without introducing additional metric columns."""

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


def aggregate_results(
    root: Path, csv_path: Path, markdown_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metric_path in sorted(root.glob("*/metrics.json")):
        with metric_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if set(row) != set(COLUMNS):
            raise ValueError(f"unexpected metric fields in {metric_path}: {sorted(row)}")
        rows.append({column: row[column] for column in COLUMNS})
    rows.sort(key=lambda row: (row["case"], row["seed"]))

    for failure_path in sorted(root.glob("*/failure.json")):
        with failure_path.open("r", encoding="utf-8") as handle:
            failure = json.load(handle)
        failures.append({"path": str(failure_path), **failure})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(COLUMNS) + " |\n")
        handle.write("|" + "|".join("---" for _ in COLUMNS) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |\n")
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args()
    csv_path = arguments.csv or arguments.root / "results.csv"
    markdown_path = arguments.markdown or arguments.root / "results.md"
    rows, failures = aggregate_results(arguments.root, csv_path, markdown_path)
    print(f"completed={len(rows)} failed={len(failures)}")
    for failure in failures:
        print(f"FAILED {failure.get('case', failure['path'])}: {failure.get('error', '')}")


if __name__ == "__main__":
    main()
