"""Run one DeepXDE classical PINN baseline case."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from benchmarks.dasa_aligned_sa.dasa_sa.config import list_case_ids, load_case_config
from benchmarks.deepxde_pinn.artifacts import atomic_write_json
from benchmarks.deepxde_pinn.problems import build_problem
from benchmarks.deepxde_pinn.trainer import TRAINING_PROFILES, Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=list_case_ids())
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--lbfgs-steps", type=int)
    parser.add_argument(
        "--training-profile", choices=TRAINING_PROFILES, default="case_config"
    )
    parser.add_argument("--residual-points", type=int)
    parser.add_argument("--boundary-points", type=int)
    parser.add_argument("--initial-points", type=int)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_case_config(arguments.case)
        sampling_overrides = {
            name: value
            for name, value in {
                "residual": arguments.residual_points,
                "boundary": arguments.boundary_points,
                "initial": arguments.initial_points,
            }.items()
            if value is not None
        }
        problem = build_problem(
            config, seed=arguments.seed, sampling_overrides=sampling_overrides
        )
        result = Trainer(
            config, problem, seed=arguments.seed, output_dir=arguments.output
        ).run(
            adam_steps=arguments.adam_steps,
            lbfgs_steps=arguments.lbfgs_steps,
            training_profile=arguments.training_profile,
        )
        print(result, flush=True)
    except Exception as error:
        atomic_write_json(
            arguments.output / "failure.json",
            {
                "case": arguments.case,
                "seed": arguments.seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
