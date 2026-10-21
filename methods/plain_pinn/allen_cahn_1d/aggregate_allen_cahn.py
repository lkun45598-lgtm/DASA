"""Aggregate the two approved Allen-Cahn metrics across completed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()

    rows: list[tuple[int, float, float]] = []
    for path in sorted(arguments.root.glob("seed_*/allen_cahn_1d/*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        seed = int(payload["metadata"]["seed"])
        rows.append(
            (
                seed,
                float(metrics["l2_relative_error"]),
                float(metrics["max_pointwise_error"]),
            )
        )
    rows.sort()
    if not rows:
        raise SystemExit(f"no completed metrics under {arguments.root}")

    values = np.asarray([(row[1], row[2]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise SystemExit("non-finite metric found")
    for seed, relative_l2, maximum in rows:
        print(f"seed={seed} l2={relative_l2:.6e} max={maximum:.6e}")
    print(
        f"mean_std l2={values[:, 0].mean():.6e} +/- "
        f"{values[:, 0].std(ddof=0):.6e}"
    )
    print(
        f"mean_std max={values[:, 1].mean():.6e} +/- "
        f"{values[:, 1].std(ddof=0):.6e}"
    )
    print(f"count={len(rows)}")


if __name__ == "__main__":
    main()
