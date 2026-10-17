"""Run one DASA-aligned SA-PINNs benchmark case."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import tensorflow as tf

from ..dasa_sa.artifacts import atomic_write_json
from ..dasa_sa.config import list_case_ids, load_case_config
from ..dasa_sa.problems import build_problem
from ..dasa_sa.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=list_case_ids())
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--lbfgs-steps", type=int)
    return parser


def configure_tensorflow_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    try:
        configure_tensorflow_memory_growth()
        config = load_case_config(arguments.case)
        problem = build_problem(config)
        trainer = Trainer(
            config, problem, seed=arguments.seed, output_dir=arguments.output
        )
        result = trainer.run(
            adam_steps=arguments.adam_steps, lbfgs_steps=arguments.lbfgs_steps
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
