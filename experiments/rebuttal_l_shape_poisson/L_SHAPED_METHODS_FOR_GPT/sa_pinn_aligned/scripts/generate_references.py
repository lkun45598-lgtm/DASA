"""Generate and convergence-check the L-shaped Poisson reference field."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..dasa_sa.config import SUITE_ROOT, load_case_config
from ..dasa_sa.grids import lshape_grid
from ..dasa_sa.reference_solvers import (
    solve_lshape_reference,
)
from ..dasa_sa.references import compare_refinements, write_reference


REFERENCE_SPECS: dict[str, dict[str, Any]] = {
    "lshape_poisson": {
        "grid_levels": [101, 201, 401],
        "tolerances": {"l2_relative_error": 5e-4, "maximum_pointwise_error": 5e-3},
    },
}


def _solve(case_id: str, level: int, points):
    if case_id == "lshape_poisson":
        return solve_lshape_reference(level, points), {"method": "five_point_sparse_direct"}
    raise KeyError(f"unknown numerical reference case: {case_id}")


def generate_case(case_id: str, *, root: Path | None = None) -> dict[str, Any]:
    if case_id not in REFERENCE_SPECS:
        raise KeyError(f"unknown numerical reference case: {case_id}")
    config = load_case_config(case_id)
    points = lshape_grid()

    spec = REFERENCE_SPECS[case_id]
    level_values = []
    solver_runs = []
    for level in spec["grid_levels"]:
        print(f"[{case_id}] solving reference grid {level}", flush=True)
        values, solver_metadata = _solve(case_id, level, points)
        level_values.append(values)
        solver_runs.append({"grid_points": level, **solver_metadata})
    convergence = compare_refinements(level_values[-2], level_values[-1])
    destination = Path(root) if root is not None else Path(SUITE_ROOT) / "references"
    return write_reference(
        case_id,
        points,
        level_values[-1],
        root=destination,
        generator_path=Path(__file__),
        equation_parameters=config.parameters,
        solver={"runs": solver_runs},
        grid_levels=spec["grid_levels"],
        tolerances=spec["tolerances"],
        convergence=convergence,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--case", choices=sorted(REFERENCE_SPECS))
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    cases = sorted(REFERENCE_SPECS) if arguments.all else [arguments.case]
    for case_id in cases:
        manifest = generate_case(case_id, root=arguments.root)
        print(
            f"[{case_id}] frozen with convergence {manifest['convergence']}", flush=True
        )


if __name__ == "__main__":
    main()
