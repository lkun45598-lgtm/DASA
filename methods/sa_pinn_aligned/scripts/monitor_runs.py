"""Read-only status monitor for concurrent benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from ..dasa_sa.config import list_case_ids


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_status(root: Path, cases: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for case_id in cases:
        run = root / case_id
        metrics = run / "metrics.json"
        failure = run / "failure.json"
        progress = run / "progress.jsonl"
        if metrics.is_file():
            statuses[case_id] = {"status": "complete", "metrics": _read_json(metrics)}
        elif failure.is_file():
            statuses[case_id] = {"status": "failed", "failure": _read_json(failure)}
        elif progress.is_file():
            lines = progress.read_text(encoding="utf-8").splitlines()
            statuses[case_id] = {
                "status": "running",
                "last_progress": json.loads(lines[-1]) if lines else None,
            }
        else:
            statuses[case_id] = {"status": "pending"}
    return statuses


def _print_status(statuses: dict[str, dict[str, Any]]) -> None:
    print(json.dumps(statuses, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    arguments = parser.parse_args()
    while True:
        _print_status(collect_status(arguments.root, list_case_ids()))
        if not arguments.watch:
            break
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
