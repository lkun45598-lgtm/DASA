"""Unified command-line entry point for one pure gPINN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .cases import create_case
from .config import CASE_NAMES, build_config
from .device import select_device
from .network import MLP
from .summary import write_summary
from .trainer import Trainer, evaluate_checkpoint


def _number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    mapping = {
        "n_residual": ("sampling", "n_residual"),
        "n_boundary": ("sampling", "n_boundary"),
        "n_initial": ("sampling", "n_initial"),
        "adam_iterations": ("optimizer", "adam_iterations"),
        "lbfgs_iterations": ("optimizer", "lbfgs_iterations"),
        "adam_batch_size": ("optimizer", "adam_batch_size"),
        "learning_rate": ("optimizer", "learning_rate"),
        "pde_weight": ("loss_weights", "pde"),
        "gradient_weight": ("loss_weights", "gradient"),
        "boundary_weight": ("loss_weights", "boundary"),
        "initial_weight": ("loss_weights", "initial"),
        "periodic_value_weight": ("loss_weights", "periodic_value"),
        "periodic_derivative_weight": ("loss_weights", "periodic_derivative"),
    }
    for argument, path in mapping.items():
        value = getattr(args, argument, None)
        if value is not None:
            config[path[0]][path[1]] = value
    for item in args.parameter or []:
        if "=" not in item:
            raise ValueError(f"--parameter expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        config["parameters"][name] = _number(value)


def result_directory(output_root: Path, mode: str, case: str, seed: int) -> Path:
    return output_root / mode / case / f"seed_{seed}"


def run_one(args: argparse.Namespace) -> dict:
    config = build_config(args.case, args.mode, args.seed)
    apply_overrides(config, args)
    device = select_device(args.device)
    torch.set_default_dtype(torch.float64)
    case = create_case(config)
    model = MLP(
        config["input_dim"],
        config["output_dim"],
        config["lower"],
        config["upper"],
        hidden_layers=config["network"]["hidden_layers"],
        width=config["network"]["width"],
    ).to(device=device, dtype=torch.float64)
    output_transform = getattr(case, "output_transform", None)
    if output_transform is not None:
        model.set_output_transform(output_transform)
    output_root = Path(args.output_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    directory = result_directory(output_root, args.mode, args.case, args.seed)
    if args.mode == "test":
        checkpoint = (
            Path(args.checkpoint).resolve()
            if args.checkpoint
            else result_directory(output_root, "full", args.case, args.seed) / "last.pt"
        )
        metrics = evaluate_checkpoint(
            model, case, config, directory, reference_root, device, checkpoint
        )
    else:
        resume_path = None
        if args.resume is not None:
            candidate = (
                directory / "last.pt"
                if args.resume == "auto"
                else Path(args.resume).resolve()
            )
            resume_path = candidate if candidate.exists() else None
        trainer = Trainer(
            model,
            case,
            config,
            directory,
            reference_root,
            device,
            resume_path=resume_path,
        )
        metrics = trainer.train()
    if not args.skip_summary:
        write_summary(output_root, args.mode)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASE_NAMES)
    parser.add_argument("--mode", choices=("smoke", "full", "test"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output-root", default="results/gpinn_baselines")
    parser.add_argument("--reference-root", default="data/reference_data")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Defer aggregate summary writing (used by parallel multi-GPU orchestration).",
    )
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--checkpoint")
    parser.add_argument("--n-residual", type=int)
    parser.add_argument("--n-boundary", type=int)
    parser.add_argument("--n-initial", type=int)
    parser.add_argument("--adam-iterations", type=int)
    parser.add_argument("--lbfgs-iterations", type=int)
    parser.add_argument("--adam-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--pde-weight", type=float)
    parser.add_argument("--gradient-weight", type=float)
    parser.add_argument("--boundary-weight", type=float)
    parser.add_argument("--initial-weight", type=float)
    parser.add_argument("--periodic-value-weight", type=float)
    parser.add_argument("--periodic-derivative-weight", type=float)
    parser.add_argument(
        "--parameter",
        action="append",
        help="Override a centralized case parameter, e.g. --parameter diffusivity=0.01",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    run_one(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
