"""Run all thirteen baselines sequentially and update the aggregate summary."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .config import CASE_NAMES
from .aggregate import write_statistics
from .run import build_parser, run_one
from .seeds import DEFAULT_SEED_SPEC, parse_seed_spec
from .summary import write_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="Legacy single-seed mode")
    seed_group.add_argument(
        "--seeds",
        default=DEFAULT_SEED_SPEC,
        help="Comma-separated seeds and ranges; default: 0-9",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="results/gpinn_baselines_10seeds")
    parser.add_argument("--reference-root", default="data/reference_data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    seeds = (args.seed,) if args.seed is not None else parse_seed_spec(args.seeds)
    single_parser = build_parser()
    for seed in seeds:
        for case in CASE_NAMES:
            command = [
                "--case",
                case,
                "--mode",
                args.mode,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output-root",
                args.output_root,
                "--reference-root",
                args.reference_root,
                "--skip-summary",
            ]
            if args.resume:
                command.append("--resume")
            run_one(single_parser.parse_args(command))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output_root = Path(args.output_root)
    write_summary(output_root, args.mode)
    if args.mode == "full":
        write_statistics(
            output_root,
            args.mode,
            expected_seeds=tuple(seeds),
            require_complete=True,
        )


if __name__ == "__main__":
    main()
