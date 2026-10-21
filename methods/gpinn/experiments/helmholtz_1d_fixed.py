"""Stable one-file solver and 10-seed launcher for 1D Helmholtz.

Problem
-------
    -u_xx + q*u = f,  x in [0, 1]
    u(0) = u(1) = 0
    f = ((m*pi)^2 + q) * sin(m*pi*x), with m=20 by default.

The old tanh implementation mixed a high-frequency target, a badly scaled
residual-gradient loss, soft boundary penalties, and no L-BFGS stage.  This
version uses a SIREN network, exact Dirichlet output transform, dimensionless
pointwise gPINN residuals, and a spectral inverse-operator preconditioner.  The
preconditioner is computed from the PDE residual; the analytical solution is
used only for final error measurement.

Running this file without --worker-seed automatically launches seeds 0-9 and
uses one process per visible CUDA GPU.  It also supports CPU smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    mode: str = "full"
    q: float = 1.0
    mode_number: int = 20
    hidden_layers: int = 3
    width: int = 64
    omega0: float = 30.0
    train_points: int = 512
    validation_points: int = 2048
    spectral_modes: int = 64
    validation_spectral_modes: int = 128
    gradient_weight: float = 1.0e-2
    spectral_weight: float = 1.0
    adam_iterations: int = 5000
    learning_rate: float = 1.0e-4
    lbfgs_iterations: int = 200
    lbfgs_history_size: int = 100
    validation_every: int = 250
    log_every: int = 100
    checkpoint_every: int = 500
    test_points: int = 2001

    def apply_smoke(self) -> None:
        self.hidden_layers = 2
        self.width = 24
        self.train_points = 64
        self.validation_points = 128
        self.spectral_modes = 24
        self.validation_spectral_modes = 32
        self.adam_iterations = 2
        self.lbfgs_iterations = 0
        self.validation_every = 1
        self.log_every = 1
        self.checkpoint_every = 2
        self.test_points = 201

    def validate(self) -> None:
        positive_ints = (
            "mode_number",
            "hidden_layers",
            "width",
            "train_points",
            "validation_points",
            "spectral_modes",
            "validation_spectral_modes",
            "adam_iterations",
            "validation_every",
            "log_every",
            "checkpoint_every",
            "test_points",
        )
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.lbfgs_iterations < 0:
            raise ValueError("lbfgs_iterations must be non-negative")
        if self.q <= 0.0:
            raise ValueError("q must be positive for this coercive Helmholtz benchmark")
        if self.omega0 <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("omega0 and learning_rate must be positive")
        if self.gradient_weight < 0.0 or self.spectral_weight <= 0.0:
            raise ValueError("gradient_weight must be non-negative and spectral_weight positive")
        if self.spectral_modes < self.mode_number:
            raise ValueError("spectral_modes must cover mode_number")
        if self.validation_spectral_modes < self.mode_number:
            raise ValueError("validation_spectral_modes must cover mode_number")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    name = requested.lower()
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "mps":
        raise RuntimeError("float64 higher derivatives are not reliable on MPS; use CPU or CUDA")
    probe = torch.tensor([[0.2]], dtype=torch.float64, device=device, requires_grad=True)
    value = torch.sin(probe)
    for _ in range(3):
        value = torch.autograd.grad(
            value, probe, torch.ones_like(value), create_graph=True
        )[0]
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{device} failed the float64 third-derivative probe")
    return device


class Siren(nn.Module):
    """High-frequency sine network with exact homogeneous Dirichlet values."""

    def __init__(self, hidden_layers: int, width: int, omega0: float) -> None:
        super().__init__()
        self.omega0 = float(omega0)
        layers: list[nn.Linear] = [nn.Linear(1, width)]
        layers.extend(nn.Linear(width, width) for _ in range(hidden_layers - 1))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(width, 1)
        self.reset_parameters()
        self.to(dtype=torch.float64)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.hidden[0].weight.uniform_(-1.0, 1.0)
            self.hidden[0].bias.uniform_(-1.0, 1.0)
            for layer in self.hidden[1:]:
                bound = math.sqrt(6.0 / layer.in_features)
                layer.weight.uniform_(-bound, bound)
                layer.bias.uniform_(-bound, bound)
            bound = math.sqrt(6.0 / self.output.in_features)
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.sin(self.omega0 * self.hidden[0](2.0 * x - 1.0))
        for layer in self.hidden[1:]:
            z = torch.sin(layer(z))
        return self.output(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw(x)
        zero = torch.zeros((1, 1), dtype=x.dtype, device=x.device)
        one = torch.ones((1, 1), dtype=x.dtype, device=x.device)
        # Subtract the linear endpoint interpolant.  This retains the SIREN's
        # direct high-frequency representation while imposing u(0)=u(1)=0 exactly.
        return raw - (1.0 - x) * self.raw(zero) - x * self.raw(one)


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def midpoint_grid(count: int, device: torch.device) -> torch.Tensor:
    values = (torch.arange(count, dtype=torch.float64, device=device) + 0.5) / count
    return values[:, None]


def objective_terms(
    model: nn.Module,
    x: torch.Tensor,
    config: Config,
    spectral_modes: int,
) -> dict[str, torch.Tensor]:
    u = model(x)
    ux = gradient(u, x)
    uxx = gradient(ux, x)
    wave_number = config.mode_number * math.pi
    target_eigenvalue = wave_number**2 + config.q
    forcing = target_eigenvalue * torch.sin(wave_number * x)
    raw_residual = -uxx + config.q * u - forcing

    # Make the target-mode PDE and derivative residuals O(1), instead of the
    # old O(1e7) and O(1e10) squared losses.
    residual = raw_residual / target_eigenvalue
    residual_x = gradient(residual, x) / wave_number
    pde = torch.mean(residual**2)
    residual_gradient = torch.mean(residual_x**2)

    # For L=-d2/dx2+q and Dirichlet boundaries, sin(n*pi*x) are eigenfunctions.
    # If r=L(u)-f, then the solution-error coefficient is r_n/lambda_n.  Penalizing
    # these coefficients preconditions the ill-conditioned high-frequency solve
    # without inserting the analytical solution into the network output.
    modes = torch.arange(
        1, spectral_modes + 1, dtype=x.dtype, device=x.device
    )[None, :]
    basis = torch.sin(math.pi * x.detach() * modes)
    eigenvalues = (math.pi * modes) ** 2 + config.q
    residual_coefficients = (
        2.0 * (basis.T @ raw_residual) / len(x) / eigenvalues.T
    )
    spectral = torch.sum(residual_coefficients**2)
    total = (
        pde
        + config.gradient_weight * residual_gradient
        + config.spectral_weight * spectral
    )
    return {
        "total": total,
        "pde": pde,
        "gradient": residual_gradient,
        "spectral": spectral,
    }


def evaluate_objective(
    model: nn.Module,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    x = midpoint_grid(config.validation_points, device).requires_grad_(True)
    with torch.enable_grad():
        terms = objective_terms(
            model, x, config, config.validation_spectral_modes
        )
    return {name: float(value.detach()) for name, value in terms.items()}


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            tensor = torch.as_tensor(
                x[start : start + 8192], dtype=torch.float64, device=device
            )
            result.append(model(tensor).cpu().numpy())
    return np.concatenate(result, axis=0)


LOG_FIELDS = (
    "phase",
    "step",
    "total",
    "pde",
    "gradient",
    "spectral",
    "validation_total",
    "validation_pde",
    "validation_gradient",
    "validation_spectral",
    "lr",
    "wall_seconds",
)


class Trainer:
    def __init__(
        self,
        model: Siren,
        config: Config,
        seed: int,
        device: torch.device,
        result_dir: Path,
        resume: bool,
    ) -> None:
        self.model = model
        self.config = config
        self.seed = seed
        self.device = device
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.adam_step = 0
        self.phase = "adam"
        self.best_validation = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.started = time.perf_counter()
        if resume:
            self.load_last()

    def payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "adam_step": self.adam_step,
            "phase": self.phase,
            "best_validation": self.best_validation,
            "config": asdict(self.config),
            "seed": self.seed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def save_last(self) -> None:
        torch.save(self.payload(), self.result_dir / "last.pt")

    def load_last(self) -> None:
        path = self.result_dir / "last.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("config") != asdict(self.config) or int(state.get("seed")) != self.seed:
            raise ValueError(f"checkpoint configuration mismatch: {path}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.adam_step = int(state["adam_step"])
        self.phase = str(state["phase"])
        self.best_validation = float(state["best_validation"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda_rng_state_all"]]
            )
        best_path = self.result_dir / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            self.best_state = best["model"]

    def record_best(self, validation: dict[str, float]) -> None:
        if validation["total"] >= self.best_validation:
            return
        self.best_validation = validation["total"]
        self.best_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        torch.save(
            {
                "model": self.best_state,
                "validation": validation,
                "step": self.adam_step,
                "config": asdict(self.config),
                "seed": self.seed,
            },
            self.result_dir / "best.pt",
        )

    def write_row(
        self,
        writer: csv.DictWriter,
        phase: str,
        terms: dict[str, float],
        validation: dict[str, float] | None,
    ) -> None:
        writer.writerow(
            {
                "phase": phase,
                "step": self.adam_step,
                "total": terms["total"],
                "pde": terms["pde"],
                "gradient": terms["gradient"],
                "spectral": terms["spectral"],
                "validation_total": "" if validation is None else validation["total"],
                "validation_pde": "" if validation is None else validation["pde"],
                "validation_gradient": ""
                if validation is None
                else validation["gradient"],
                "validation_spectral": ""
                if validation is None
                else validation["spectral"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "wall_seconds": time.perf_counter() - self.started,
            }
        )

    def train(self) -> dict[str, float | int]:
        if self.phase == "complete":
            validation = evaluate_objective(self.model, self.config, self.device)
            return {
                "adam_iterations_completed": self.adam_step,
                "best_validation_total": self.best_validation,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        log_path = self.result_dir / "training_log.csv"
        append = self.adam_step > 0 and log_path.exists()
        with log_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not append:
                writer.writeheader()
            if self.phase == "adam":
                while self.adam_step < self.config.adam_iterations:
                    x = midpoint_grid(self.config.train_points, self.device).requires_grad_(True)
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    tensors = objective_terms(
                        self.model, x, self.config, self.config.spectral_modes
                    )
                    if not torch.isfinite(tensors["total"]):
                        raise FloatingPointError(
                            f"non-finite loss at Adam step {self.adam_step + 1}"
                        )
                    tensors["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
                    self.optimizer.step()
                    self.adam_step += 1
                    terms = {
                        name: float(value.detach()) for name, value in tensors.items()
                    }
                    validation = None
                    if (
                        self.adam_step == 1
                        or self.adam_step % self.config.validation_every == 0
                        or self.adam_step == self.config.adam_iterations
                    ):
                        validation = evaluate_objective(
                            self.model, self.config, self.device
                        )
                        self.record_best(validation)
                    if (
                        self.adam_step == 1
                        or self.adam_step % self.config.log_every == 0
                        or validation is not None
                    ):
                        self.write_row(writer, "adam", terms, validation)
                        handle.flush()
                    if (
                        self.adam_step % self.config.checkpoint_every == 0
                        or self.adam_step == self.config.adam_iterations
                    ):
                        self.save_last()
                self.phase = "lbfgs"
                self.save_last()

            if self.phase == "lbfgs" and self.config.lbfgs_iterations > 0:
                if self.best_state is not None:
                    self.model.load_state_dict(self.best_state)
                x = midpoint_grid(self.config.train_points, self.device).requires_grad_(True)
                lbfgs = torch.optim.LBFGS(
                    self.model.parameters(),
                    max_iter=self.config.lbfgs_iterations,
                    max_eval=max(1, int(1.25 * self.config.lbfgs_iterations)),
                    history_size=self.config.lbfgs_history_size,
                    tolerance_grad=1.0e-12,
                    tolerance_change=1.0e-14,
                    line_search_fn="strong_wolfe",
                )
                closures = 0
                last_terms: dict[str, float] = {}

                def closure() -> torch.Tensor:
                    nonlocal closures, last_terms
                    lbfgs.zero_grad(set_to_none=True)
                    tensors = objective_terms(
                        self.model, x, self.config, self.config.spectral_modes
                    )
                    tensors["total"].backward()
                    closures += 1
                    last_terms = {
                        name: float(value.detach()) for name, value in tensors.items()
                    }
                    return tensors["total"]

                lbfgs.step(closure)
                validation = evaluate_objective(
                    self.model, self.config, self.device
                )
                self.record_best(validation)
                self.write_row(writer, "lbfgs", last_terms, validation)
                handle.flush()
            self.phase = "complete"
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            self.save_last()
        validation = evaluate_objective(self.model, self.config, self.device)
        return {
            "adam_iterations_completed": self.adam_step,
            "best_validation_total": self.best_validation,
            **{f"validation_{key}": value for key, value in validation.items()},
        }


def plot_outputs(
    result_dir: Path,
    x: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.8), constrained_layout=True)
    axes[0].plot(x[:, 0], reference[:, 0], label="Exact", linewidth=1.4)
    axes[0].plot(
        x[:, 0], prediction[:, 0], "--", label="SIREN-gPINN", linewidth=1.1
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(x[:, 0], np.maximum(np.abs(prediction[:, 0] - reference[:, 0]), 1e-18))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(alpha=0.25)
    fig.savefig(result_dir / "solution_error.png", dpi=180)
    plt.close(fig)

    steps: list[int] = []
    losses: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    with (result_dir / "training_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            losses.append(float(row["total"]))
            if row["validation_total"]:
                validation_steps.append(int(row["step"]))
                validation_losses.append(float(row["validation_total"]))
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.semilogy(steps, losses, label="training")
    axis.semilogy(validation_steps, validation_losses, "o-", label="held-out")
    axis.set_xlabel("Adam step")
    axis.set_ylabel("Preconditioned gPINN objective")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def run_one(seed: int, config: Config, output_root: Path, device_name: str, resume: bool) -> dict:
    set_seed(seed)
    torch.set_default_dtype(torch.float64)
    device = select_device(device_name)
    result_dir = output_root / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**asdict(config), "seed": seed}, handle, indent=2, sort_keys=True)
    model = Siren(config.hidden_layers, config.width, config.omega0).to(device)
    trainer = Trainer(model, config, seed, device, result_dir, resume)
    training = trainer.train()

    x = np.linspace(0.0, 1.0, config.test_points, dtype=np.float64)[:, None]
    reference = np.sin(config.mode_number * np.pi * x)
    prediction = predict(model, x, device)
    error = prediction - reference
    boundary = predict(model, np.asarray([[0.0], [1.0]]), device)
    metrics = {
        "case": "helmholtz_1d",
        "method": "SIREN gPINN with exact boundary and spectral residual preconditioning",
        "seed": seed,
        "mode": config.mode,
        "device": str(device),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "max_abs": float(np.max(np.abs(error))),
        "boundary_max_abs": float(np.max(np.abs(boundary))),
        "test_points": len(x),
        "full_training_completed": config.mode == "full",
        **training,
    }
    np.savez_compressed(
        result_dir / "prediction.npz",
        x=x,
        prediction=prediction,
        reference=reference,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    plot_outputs(result_dir, x, prediction, reference)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def parse_seeds(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            if end < start:
                raise ValueError(f"invalid seed range {token}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return tuple(result)


def gpu_ids(value: str) -> tuple[str, ...]:
    if value.lower() in {"cpu", "none"}:
        return ()
    if value.lower() != "auto":
        result = tuple(item.strip() for item in value.split(",") if item.strip())
        if not result:
            raise ValueError("--gpus must be auto, cpu, or a comma-separated list")
        return result
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        result = tuple(item.strip() for item in visible.split(",") if item.strip())
        if result:
            return result
    if torch.cuda.is_available():
        return tuple(str(index) for index in range(torch.cuda.device_count()))
    return ()


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config(mode=args.mode)
    if args.mode == "smoke":
        config.apply_smoke()
    for name in asdict(config):
        value = getattr(args, name, None)
        if value is not None and name != "mode":
            setattr(config, name, value)
    config.validate()
    return config


def completed(output_root: Path, seed: int, mode: str) -> bool:
    path = output_root / f"seed_{seed}" / "metrics.json"
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        return int(metrics["seed"]) == seed and metrics["mode"] == mode
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def worker_command(
    args: argparse.Namespace,
    config: Config,
    seed: int,
    device: str,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-seed",
        str(seed),
        "--device",
        device,
        "--output-root",
        str(args.output_root),
        "--mode",
        config.mode,
    ]
    for name, value in asdict(config).items():
        if name == "mode":
            continue
        command.extend((f"--{name.replace('_', '-')}", str(value)))
    if resume:
        command.append("--resume")
    return command


def summarize(output_root: Path, seeds: tuple[int, ...]) -> dict:
    runs: list[dict] = []
    for seed in seeds:
        path = output_root / f"seed_{seed}" / "metrics.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                runs.append(json.load(handle))
    runs.sort(key=lambda item: int(item["seed"]))
    summary: dict[str, object] = {
        "requested_seeds": list(seeds),
        "completed_seeds": [int(run["seed"]) for run in runs],
        "completed_runs": len(runs),
        "missing_seeds": sorted(set(seeds) - {int(run["seed"]) for run in runs}),
    }
    for name in ("relative_l2", "max_abs"):
        values = np.asarray([float(run[name]) for run in runs], dtype=np.float64)
        if len(values):
            average = float(np.mean(values))
            deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{name}_mean"] = average
            summary[f"{name}_std"] = deviation
            summary[f"{name}_mean_std"] = f"{average:.6e} ± {deviation:.6e}"
    payload = {"summary": summary, "runs": runs}
    with (output_root / "summary_10seeds.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (output_root / "summary_10seeds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "relative_l2", "max_abs", "boundary_max_abs")
        )
        writer.writeheader()
        for run in runs:
            writer.writerow({name: run[name] for name in writer.fieldnames})
    with (output_root / "summary_10seeds.md").open("w", encoding="utf-8") as handle:
        handle.write("# 1D Helmholtz 十种子结果\n\n")
        handle.write(f"完成：{len(runs)}/{len(seeds)}。\n\n")
        handle.write(f"- Relative L2: {summary.get('relative_l2_mean_std', 'N/A')}\n")
        handle.write(f"- Max pointwise error: {summary.get('max_abs_mean_std', 'N/A')}\n")
    return payload


def launch(args: argparse.Namespace, config: Config) -> None:
    seeds = parse_seeds(args.seeds)
    devices = gpu_ids(args.gpus)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = [
        seed
        for seed in seeds
        if args.force or not completed(args.output_root, seed, config.mode)
    ]
    print(
        json.dumps(
            {
                "seeds": list(seeds),
                "pending": pending,
                "skipped": [seed for seed in seeds if seed not in pending],
                "gpus": list(devices),
                "output_root": str(args.output_root),
            },
            indent=2,
        ),
        flush=True,
    )
    tasks: queue.Queue[int] = queue.Queue()
    for seed in pending:
        tasks.put(seed)
    failures: list[tuple[int, int]] = []
    print_lock = threading.Lock()
    workers = (
        tuple(("cuda", identifier) for identifier in devices)
        if devices
        else tuple(("cpu", None) for _ in range(args.cpu_workers))
    )

    def work(device: str, identifier: str | None) -> None:
        while True:
            try:
                seed = tasks.get_nowait()
            except queue.Empty:
                return
            result_dir = args.output_root / f"seed_{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            resume = not args.force and (result_dir / "last.pt").exists()
            command = worker_command(args, config, seed, device, resume)
            environment = os.environ.copy()
            if identifier is not None:
                environment["CUDA_VISIBLE_DEVICES"] = identifier
            label = f"GPU {identifier}" if identifier is not None else "CPU"
            with print_lock:
                print(f"[{label}] seed {seed} started", flush=True)
            with (result_dir / "launcher.log").open("a", encoding="utf-8") as log:
                log.write("\nCOMMAND: " + " ".join(command) + "\n")
                log.flush()
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            if process.returncode:
                failures.append((seed, process.returncode))
            with print_lock:
                status = "completed" if process.returncode == 0 else f"failed ({process.returncode})"
                print(f"[{label}] seed {seed} {status}", flush=True)
            tasks.task_done()

    threads = [threading.Thread(target=work, args=worker, daemon=False) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    payload = summarize(args.output_root, seeds)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(f"failed seeds: {failures}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--gpus", default="auto", help="auto, cpu, or 0,1,...")
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results/helmholtz_1d_fixed_10seeds",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--device", default="auto")
    int_fields = (
        "mode_number",
        "hidden_layers",
        "width",
        "train_points",
        "validation_points",
        "spectral_modes",
        "validation_spectral_modes",
        "adam_iterations",
        "lbfgs_iterations",
        "lbfgs_history_size",
        "validation_every",
        "log_every",
        "checkpoint_every",
        "test_points",
    )
    float_fields = (
        "q",
        "omega0",
        "gradient_weight",
        "spectral_weight",
        "learning_rate",
    )
    for name in int_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    for name in float_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_workers < 1:
        raise ValueError("cpu_workers must be positive")
    config = config_from_args(args)
    args.output_root = args.output_root.resolve()
    if args.worker_seed is not None:
        run_one(
            args.worker_seed,
            config,
            args.output_root,
            args.device,
            args.resume,
        )
    else:
        launch(args, config)


if __name__ == "__main__":
    main()
