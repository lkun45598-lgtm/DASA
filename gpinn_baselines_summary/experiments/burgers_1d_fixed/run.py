"""Train only the 1D viscous Burgers gPINN with residual-adaptive refinement.

The original multi-case runner repeatedly optimized one fixed set of 1500
collocation points.  A 6x128 network can drive that discrete objective close to
zero while remaining badly wrong between the points.  This standalone runner
uses fresh interior samples during Adam, residual-adaptive anchors (RAR), and a
fixed held-out residual set for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from torch import nn


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"
    mode: str = "full"
    viscosity: float = 0.01 / math.pi
    hidden_layers: int = 6
    width: int = 128
    gradient_weight: float = 1.0e-4
    batch_size: int = 2048
    anchor_batch_size: int = 512
    warmup_steps: int = 20_000
    rar_cycles: int = 10
    rar_steps: int = 3_000
    candidate_points: int = 50_000
    candidate_chunk_size: int = 4096
    anchors_per_cycle: int = 100
    min_anchor_distance: float = 2.0e-3
    learning_rate: float = 1.0e-3
    lr_decay_rate: float = 0.9
    lr_decay_steps: int = 5_000
    lbfgs_iterations: int = 500
    lbfgs_points: int = 8192
    lbfgs_history_size: int = 100
    validation_points: int = 4096
    validation_every: int = 1000
    log_every: int = 100
    checkpoint_every: int = 1000
    test_nx: int = 257
    test_nt: int = 101
    reference_nx: int = 2049
    reference_dt: float = 2.5e-4

    def apply_smoke_defaults(self) -> None:
        self.hidden_layers = 2
        self.width = 24
        self.batch_size = 32
        self.anchor_batch_size = 8
        self.warmup_steps = 2
        self.rar_cycles = 1
        self.rar_steps = 2
        self.candidate_points = 128
        self.candidate_chunk_size = 64
        self.anchors_per_cycle = 8
        self.lbfgs_iterations = 0
        self.lbfgs_points = 64
        self.validation_points = 64
        self.validation_every = 1
        self.log_every = 1
        self.checkpoint_every = 2
        self.test_nx = 33
        self.test_nt = 17
        self.reference_nx = 257
        self.reference_dt = 1.0e-3

    def validate(self) -> None:
        integer_fields = (
            "hidden_layers",
            "width",
            "batch_size",
            "warmup_steps",
            "rar_cycles",
            "rar_steps",
            "candidate_points",
            "candidate_chunk_size",
            "anchors_per_cycle",
            "lbfgs_iterations",
            "lbfgs_points",
            "validation_points",
            "validation_every",
            "log_every",
            "checkpoint_every",
            "test_nx",
            "test_nt",
            "reference_nx",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hidden_layers < 1 or self.width < 1 or self.batch_size < 1:
            raise ValueError("network dimensions and batch_size must be positive")
        for name in (
            "candidate_chunk_size",
            "validation_points",
            "validation_every",
            "log_every",
            "checkpoint_every",
            "lr_decay_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.anchor_batch_size < 0:
            raise ValueError("anchor_batch_size must be non-negative")
        if self.rar_cycles and self.candidate_points < self.anchors_per_cycle:
            raise ValueError(
                "candidate_points must be at least anchors_per_cycle when RAR is enabled"
            )
        if self.lbfgs_iterations and self.lbfgs_points < 1:
            raise ValueError("lbfgs_points must be positive when L-BFGS is enabled")
        if self.test_nx < 2 or self.test_nt < 2 or self.reference_nx < 5:
            raise ValueError("test grids need at least 2 points and reference_nx at least 5")
        if self.viscosity <= 0.0 or self.reference_dt <= 0.0:
            raise ValueError("viscosity and reference_dt must be positive")
        if not 0.0 < self.lr_decay_rate <= 1.0:
            raise ValueError("lr_decay_rate must lie in (0, 1]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(name)
    if device.type == "mps":
        raise RuntimeError(
            "Burgers gPINN requires float64 third derivatives; PyTorch MPS does "
            "not support this reliably. Use --device cpu or --device cuda."
        )
    probe = torch.tensor([[0.2]], dtype=torch.float64, device=device, requires_grad=True)
    value = torch.tanh(probe) ** 4
    for _ in range(3):
        value = torch.autograd.grad(
            value, probe, torch.ones_like(value), create_graph=True
        )[0]
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{device} failed the float64 third-derivative probe")
    return device


class BurgersMLP(nn.Module):
    """Tanh MLP with exact constraints and the problem's odd spatial symmetry."""

    def __init__(self, hidden_layers: int, width: int) -> None:
        super().__init__()
        layers: list[nn.Linear] = [nn.Linear(2, width)]
        layers.extend(nn.Linear(width, width) for _ in range(hidden_layers - 1))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(width, 1)
        self.activation = nn.Tanh()
        for layer in [*self.hidden, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.to(dtype=torch.float64)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0:1]
        t = points[:, 1:2]
        # The unique viscous solution is odd in x.  Antisymmetrizing an
        # otherwise unrestricted network pins the thin shock at x=0 without
        # sacrificing the ability to represent its steep local profile.
        normalized_t = 2.0 * t - 1.0

        def unconstrained(spatial: torch.Tensor) -> torch.Tensor:
            z = torch.cat((spatial, normalized_t), dim=1)
            for layer in self.hidden:
                z = self.activation(layer(z))
            return self.output(z)

        odd_raw = 0.5 * (unconstrained(x) - unconstrained(-x))
        return (1.0 - x**2) * (1.0 - torch.exp(-t)) * odd_raw - torch.sin(
            torch.pi * x
        )


def _gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def burgers_residual(
    model: nn.Module, points: torch.Tensor, viscosity: float
) -> torch.Tensor:
    u = model(points)
    first = _gradient(u, points)
    ux = first[:, 0:1]
    ut = first[:, 1:2]
    uxx = _gradient(ux, points)[:, 0:1]
    return ut + u * ux - viscosity * uxx


def loss_terms(
    model: nn.Module,
    points: torch.Tensor,
    viscosity: float,
    gradient_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = burgers_residual(model, points, viscosity)
    residual_gradient = _gradient(residual, points)
    pde = torch.mean(residual**2)
    gradient = torch.mean(residual_gradient[:, 0:1] ** 2) + torch.mean(
        residual_gradient[:, 1:2] ** 2
    )
    return pde + gradient_weight * gradient, pde, gradient


def latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((n, 2), dtype=np.float64)
    for dim in range(2):
        strata = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        values[:, dim] = strata[rng.permutation(n)]
    values[:, 0] = 2.0 * values[:, 0] - 1.0
    return values


def make_batch(
    rng: np.random.Generator,
    batch_size: int,
    anchors: np.ndarray,
    anchor_batch_size: int,
) -> np.ndarray:
    n_anchor = min(len(anchors), anchor_batch_size, batch_size)
    n_random = batch_size - n_anchor
    pieces: list[np.ndarray] = []
    if n_random:
        pieces.append(latin_hypercube(n_random, rng))
    if n_anchor:
        indices = rng.choice(len(anchors), n_anchor, replace=False)
        pieces.append(anchors[indices])
    batch = np.concatenate(pieces, axis=0)
    return batch[rng.permutation(len(batch))]


def as_input(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.as_tensor(points, dtype=torch.float64, device=device)
        .detach()
        .requires_grad_(True)
    )


def residual_scores(
    model: nn.Module,
    points: np.ndarray,
    viscosity: float,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.enable_grad():
        for start in range(0, len(points), chunk_size):
            chunk = as_input(points[start : start + chunk_size], device)
            value = burgers_residual(model, chunk, viscosity)
            scores.append(torch.abs(value).detach().cpu().numpy()[:, 0])
    return np.concatenate(scores)


def heldout_terms(
    model: nn.Module,
    points: np.ndarray,
    config: RunConfig,
    device: torch.device,
) -> dict[str, float]:
    sums = {"pde": 0.0, "gradient": 0.0}
    model.eval()
    with torch.enable_grad():
        for start in range(0, len(points), config.candidate_chunk_size):
            array = points[start : start + config.candidate_chunk_size]
            tensor = as_input(array, device)
            _, pde, gradient = loss_terms(
                model, tensor, config.viscosity, config.gradient_weight
            )
            fraction = len(array) / len(points)
            sums["pde"] += fraction * float(pde.detach())
            sums["gradient"] += fraction * float(gradient.detach())
    sums["total"] = sums["pde"] + config.gradient_weight * sums["gradient"]
    return sums


def select_rar_anchors(
    candidates: np.ndarray,
    scores: np.ndarray,
    existing: np.ndarray,
    count: int,
    min_distance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64), {
            "candidate_mean_abs_residual": float(np.mean(scores)),
            "candidate_max_abs_residual": float(np.max(scores)),
            "selected_min_abs_residual": 0.0,
        }
    order = np.argsort(scores)[::-1]
    selected: list[np.ndarray] = []

    def normalize(value: np.ndarray) -> np.ndarray:
        result = value.copy()
        result[..., 0] = (result[..., 0] + 1.0) / 2.0
        return result

    existing_normalized = normalize(existing) if len(existing) else existing
    for index in order:
        point = candidates[index]
        normalized = normalize(point[None, :])[0]
        if len(existing_normalized):
            if np.min(np.linalg.norm(existing_normalized - normalized, axis=1)) < min_distance:
                continue
        if selected:
            selected_normalized = normalize(np.asarray(selected))
            if np.min(np.linalg.norm(selected_normalized - normalized, axis=1)) < min_distance:
                continue
        selected.append(point)
        if len(selected) == count:
            break
    # A very large distance threshold must not silently reduce the requested RAR count.
    if len(selected) < count:
        selected_keys = {tuple(value) for value in selected}
        for index in order:
            point = candidates[index]
            if tuple(point) not in selected_keys:
                selected.append(point)
                selected_keys.add(tuple(point))
            if len(selected) == count:
                break
    result = np.asarray(selected, dtype=np.float64)
    selected_scores = residual_scores_from_candidates(result, candidates, scores)
    details = {
        "candidate_mean_abs_residual": float(np.mean(scores)),
        "candidate_max_abs_residual": float(np.max(scores)),
        "selected_min_abs_residual": float(np.min(selected_scores)),
        "selected_mean_abs_residual": float(np.mean(selected_scores)),
    }
    return result, details


def residual_scores_from_candidates(
    selected: np.ndarray, candidates: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    lookup = {tuple(point): score for point, score in zip(candidates, scores)}
    return np.asarray([lookup[tuple(point)] for point in selected], dtype=np.float64)


class _Tridiagonal:
    def __init__(self, lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray):
        self.lower = lower
        self.diagonal = diagonal
        n = len(diagonal)
        self.cprime = np.empty(n - 1, dtype=np.float64)
        self.denominator = np.empty(n, dtype=np.float64)
        self.denominator[0] = diagonal[0]
        self.cprime[0] = upper[0] / self.denominator[0]
        for index in range(1, n - 1):
            self.denominator[index] = (
                diagonal[index] - lower[index - 1] * self.cprime[index - 1]
            )
            self.cprime[index] = upper[index] / self.denominator[index]
        self.denominator[-1] = (
            diagonal[-1] - lower[-1] * self.cprime[-1]
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        n = len(rhs)
        dprime = np.empty(n, dtype=np.float64)
        dprime[0] = rhs[0] / self.denominator[0]
        for index in range(1, n):
            dprime[index] = (
                rhs[index] - self.lower[index - 1] * dprime[index - 1]
            ) / self.denominator[index]
        result = np.empty(n, dtype=np.float64)
        result[-1] = dprime[-1]
        for index in range(n - 2, -1, -1):
            result[index] = dprime[index] - self.cprime[index] * result[index + 1]
        return result


def burgers_reference_grid(
    nx: int, dt: float, viscosity: float, save_dt: float = 0.0025
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order IMEX-BDF2 finite-difference reference solution."""
    steps = int(round(1.0 / dt))
    stride = int(round(save_dt / dt))
    if nx < 5 or not np.isclose(steps * dt, 1.0):
        raise ValueError("reference grid must cover t in [0, 1] exactly")
    if stride < 1 or not np.isclose(stride * dt, save_dt):
        raise ValueError("save_dt must be an integer multiple of reference_dt")
    if steps % stride != 0:
        raise ValueError("saved reference snapshots must include t=1 exactly")
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    h = x[1] - x[0]
    times = np.arange(0, steps + 1, stride, dtype=np.float64) * dt
    history = np.empty((len(times), nx), dtype=np.float64)
    current = -np.sin(np.pi * x)
    current[[0, -1]] = 0.0
    history[0] = current

    def nonlinear(value: np.ndarray) -> np.ndarray:
        ux = (value[2:] - value[:-2]) / (2.0 * h)
        return -value[1:-1] * ux

    interior = nx - 2
    alpha = viscosity / h**2
    first_solver = _Tridiagonal(
        -alpha * np.ones(interior - 1),
        (1.0 / dt + 2.0 * alpha) * np.ones(interior),
        -alpha * np.ones(interior - 1),
    )
    previous = current.copy()
    following = current.copy()
    following[1:-1] = first_solver.solve(current[1:-1] / dt + nonlinear(current))
    current = following
    save_index = 1
    if stride == 1:
        history[save_index] = current
        save_index += 1

    bdf_solver = _Tridiagonal(
        -alpha * np.ones(interior - 1),
        (3.0 / (2.0 * dt) + 2.0 * alpha) * np.ones(interior),
        -alpha * np.ones(interior - 1),
    )
    nonlinear_previous = nonlinear(previous)
    nonlinear_current = nonlinear(current)
    for step in range(2, steps + 1):
        rhs = (
            4.0 * current[1:-1] / (2.0 * dt)
            - previous[1:-1] / (2.0 * dt)
            + 2.0 * nonlinear_current
            - nonlinear_previous
        )
        following = np.zeros_like(current)
        following[1:-1] = bdf_solver.solve(rhs)
        previous, current = current, following
        nonlinear_previous, nonlinear_current = nonlinear_current, nonlinear(following)
        if step % stride == 0:
            history[save_index] = current
            save_index += 1
    if save_index != len(history):
        raise RuntimeError("reference time history was not filled completely")
    return x, times, history


def ensure_reference(path: Path, config: RunConfig) -> Path:
    expected = {
        "viscosity": config.viscosity,
        "nx": config.reference_nx,
        "dt": config.reference_dt,
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    def matches() -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                metadata = json.loads(str(data["metadata"]))
            return all(
                np.isclose(metadata[key], value) for key, value in expected.items()
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    if matches():
        return path

    # Ten-seed jobs may finish at nearly the same time.  Serialize the one-time
    # reference generation so workers never race while writing the shared NPZ.
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if matches():
            return path
        # Keep snapshots close to 0.0025 apart while guaranteeing that the spacing
        # is an integer number of time steps for user-overridden reference_dt.
        reference_stride = max(1, int(round(2.5e-3 / config.reference_dt)))
        reference_save_dt = reference_stride * config.reference_dt
        x, t, u = burgers_reference_grid(
            config.reference_nx,
            config.reference_dt,
            config.viscosity,
            save_dt=reference_save_dt,
        )
        metadata = {
            **expected,
            "equation": "u_t + u*u_x - viscosity*u_xx = 0",
            "method": "centered finite differences plus second-order IMEX-BDF2",
            "saved_dt": reference_save_dt,
            "boundary": "u(-1,t)=u(1,t)=0",
            "initial": "u(x,0)=-sin(pi*x)",
        }
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            x=x,
            t=t,
            u=u,
            metadata=np.asarray(json.dumps(metadata)),
        )
        os.replace(temporary, path)
    return path


def predict(
    model: nn.Module, points: np.ndarray, device: torch.device, chunk_size: int = 8192
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            tensor = torch.as_tensor(
                points[start : start + chunk_size],
                dtype=torch.float64,
                device=device,
            )
            values.append(model(tensor).cpu().numpy())
    return np.concatenate(values, axis=0)


def evaluate(
    model: nn.Module,
    config: RunConfig,
    output_dir: Path,
    reference_path: Path,
    device: torch.device,
) -> dict[str, float | int | str]:
    x = np.linspace(-1.0, 1.0, config.test_nx, dtype=np.float64)
    t = np.linspace(0.0, 1.0, config.test_nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="xy")
    points = np.column_stack((X.ravel(), T.ravel()))
    prediction = predict(model, points, device)
    with np.load(reference_path) as data:
        interpolator = RegularGridInterpolator((data["t"], data["x"]), data["u"])
        reference = interpolator(np.column_stack((points[:, 1], points[:, 0])))[:, None]
        reference_metadata = str(data["metadata"])
    error = prediction - reference
    prediction_grid = prediction.reshape(T.shape)
    initial_error = np.max(np.abs(prediction_grid[0] + np.sin(np.pi * x)))
    boundary_error = np.max(np.abs(prediction_grid[:, [0, -1]]))
    odd_symmetry_error = np.max(np.abs(prediction_grid + prediction_grid[:, ::-1]))
    metrics: dict[str, float | int | str] = {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "initial_condition_max_abs": float(initial_error),
        "boundary_condition_max_abs": float(boundary_error),
        "odd_symmetry_max_abs": float(odd_symmetry_error),
        "test_points": int(len(points)),
        "reference": str(reference_path.resolve()),
    }
    np.savez_compressed(
        output_dir / "prediction.npz",
        points=points,
        prediction=prediction,
        reference=reference,
        reference_metadata=np.asarray(reference_metadata),
    )
    plot_solution(X, T, prediction_grid, reference.reshape(T.shape), output_dir)
    return metrics


def plot_solution(
    X: np.ndarray,
    T: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.abs(prediction - reference)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    panels = (
        (prediction, "gPINN prediction", "coolwarm"),
        (reference, "Numerical reference", "coolwarm"),
        (error, "Absolute error", "magma"),
    )
    for axis, (values, title, cmap) in zip(axes, panels):
        image = axis.pcolormesh(X, T, values, shading="auto", cmap=cmap)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
    fig.savefig(output_dir / "solution_and_error.png", dpi=180)
    plt.close(fig)


def plot_training(log_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[int] = []
    training: list[float] = []
    validation_steps: list[int] = []
    validation: list[float] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            training.append(float(row["total"]))
            if row["validation_total"]:
                validation_steps.append(int(row["step"]))
                validation.append(float(row["validation_total"]))
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.semilogy(steps, training, label="training batch", linewidth=1.0)
    if validation:
        axis.semilogy(
            validation_steps,
            validation,
            "o-",
            label="fixed held-out points",
            markersize=3,
            linewidth=1.0,
        )
    axis.set_xlabel("Adam step")
    axis.set_ylabel("gPINN objective")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


LOG_FIELDS = (
    "phase",
    "cycle",
    "step",
    "total",
    "pde",
    "gradient",
    "validation_total",
    "validation_pde",
    "validation_gradient",
    "anchors",
    "lr",
    "wall_seconds",
)


class AdaptiveTrainer:
    def __init__(
        self,
        model: BurgersMLP,
        config: RunConfig,
        output_dir: Path,
        device: torch.device,
        resume: bool,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.resume = resume
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.seed)
        self.validation = latin_hypercube(config.validation_points, self.rng)
        self.anchors = np.empty((0, 2), dtype=np.float64)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: config.lr_decay_rate
            ** (step // max(config.lr_decay_steps, 1)),
        )
        self.global_step = 0
        self.stage = "warmup"
        self.cycle = -1
        self.stage_step = 0
        self.best_validation = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.started = time.perf_counter()
        if resume:
            self._load_last()

    def _checkpoint_payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "stage": self.stage,
            "cycle": self.cycle,
            "stage_step": self.stage_step,
            "anchors": self.anchors,
            "validation": self.validation,
            "best_validation": self.best_validation,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "config": asdict(self.config),
        }

    def _save_last(self) -> None:
        torch.save(self._checkpoint_payload(), self.output_dir / "last.pt")

    def _load_last(self) -> None:
        path = self.output_dir / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"cannot resume: {path} does not exist")
        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_config = state.get("config", {})
        incompatible = {
            key: (saved_config.get(key), value)
            for key, value in asdict(self.config).items()
            if key in saved_config and saved_config[key] != value
        }
        if incompatible:
            raise ValueError(f"resume configuration differs: {incompatible}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["global_step"])
        self.stage = str(state["stage"])
        self.cycle = int(state["cycle"])
        self.stage_step = int(state["stage_step"])
        self.anchors = np.asarray(state["anchors"], dtype=np.float64)
        self.validation = np.asarray(state["validation"], dtype=np.float64)
        self.best_validation = float(state["best_validation"])
        self.rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda_rng_state_all"]]
            )
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            self.best_state = best["model"]

    def _record_best(self, validation: dict[str, float]) -> None:
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
                "global_step": self.global_step,
                "anchors": self.anchors,
                "config": asdict(self.config),
            },
            self.output_dir / "best.pt",
        )

    def _write_row(
        self,
        writer: csv.DictWriter,
        phase: str,
        cycle: int,
        total: float,
        pde: float,
        gradient: float,
        validation: dict[str, float] | None,
    ) -> None:
        writer.writerow(
            {
                "phase": phase,
                "cycle": cycle,
                "step": self.global_step,
                "total": total,
                "pde": pde,
                "gradient": gradient,
                "validation_total": "" if validation is None else validation["total"],
                "validation_pde": "" if validation is None else validation["pde"],
                "validation_gradient": ""
                if validation is None
                else validation["gradient"],
                "anchors": len(self.anchors),
                "lr": self.optimizer.param_groups[0]["lr"],
                "wall_seconds": time.perf_counter() - self.started,
            }
        )

    def _adam_stage(
        self,
        writer: csv.DictWriter,
        handle,
        phase: str,
        cycle: int,
        target_steps: int,
    ) -> None:
        last_values = (float("nan"), float("nan"), float("nan"))
        while self.stage_step < target_steps:
            batch = make_batch(
                self.rng,
                self.config.batch_size,
                self.anchors,
                self.config.anchor_batch_size,
            )
            points = as_input(batch, self.device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total, pde, gradient = loss_terms(
                self.model,
                points,
                self.config.viscosity,
                self.config.gradient_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {self.global_step + 1}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            self.stage_step += 1
            last_values = (
                float(total.detach()),
                float(pde.detach()),
                float(gradient.detach()),
            )
            needs_validation = (
                self.global_step == 1
                or self.global_step % self.config.validation_every == 0
                or self.stage_step == target_steps
            )
            validation = None
            if needs_validation:
                validation = heldout_terms(
                    self.model, self.validation, self.config, self.device
                )
                self._record_best(validation)
            needs_log = (
                self.global_step == 1
                or self.global_step % self.config.log_every == 0
                or self.stage_step == target_steps
                or validation is not None
            )
            if needs_log:
                self._write_row(
                    writer,
                    phase,
                    cycle,
                    *last_values,
                    validation,
                )
                handle.flush()
            if (
                self.global_step % self.config.checkpoint_every == 0
                or self.stage_step == target_steps
            ):
                self._save_last()

    def _add_anchors(self, cycle: int) -> dict[str, float | int]:
        candidate_rng = np.random.default_rng(self.config.seed + 10_000 + cycle)
        candidates = latin_hypercube(self.config.candidate_points, candidate_rng)
        scores = residual_scores(
            self.model,
            candidates,
            self.config.viscosity,
            self.device,
            self.config.candidate_chunk_size,
        )
        selected, details = select_rar_anchors(
            candidates,
            scores,
            self.anchors,
            self.config.anchors_per_cycle,
            self.config.min_anchor_distance,
        )
        self.anchors = np.concatenate((self.anchors, selected), axis=0)
        record: dict[str, float | int] = {
            "cycle": cycle,
            "new_anchors": len(selected),
            "total_anchors": len(self.anchors),
            **details,
        }
        with (self.output_dir / "rar_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        np.savez_compressed(self.output_dir / "rar_anchors.npz", points=self.anchors)
        return record

    def _lbfgs(self, writer: csv.DictWriter, handle) -> None:
        if self.config.lbfgs_iterations <= 0:
            return
        base_rng = np.random.default_rng(self.config.seed + 90_000)
        fixed = latin_hypercube(self.config.lbfgs_points, base_rng)
        if len(self.anchors):
            fixed = np.concatenate((fixed, self.anchors), axis=0)
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config.lbfgs_iterations,
            max_eval=max(1, int(self.config.lbfgs_iterations * 1.25)),
            history_size=self.config.lbfgs_history_size,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )
        last = {"total": float("nan"), "pde": float("nan"), "gradient": float("nan")}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            aggregate = {"total": 0.0, "pde": 0.0, "gradient": 0.0}
            for start in range(0, len(fixed), self.config.batch_size):
                array = fixed[start : start + self.config.batch_size]
                points = as_input(array, self.device)
                total, pde, gradient = loss_terms(
                    self.model,
                    points,
                    self.config.viscosity,
                    self.config.gradient_weight,
                )
                fraction = len(array) / len(fixed)
                (fraction * total).backward()
                aggregate["total"] += fraction * float(total.detach())
                aggregate["pde"] += fraction * float(pde.detach())
                aggregate["gradient"] += fraction * float(gradient.detach())
            if not np.isfinite(aggregate["total"]):
                raise FloatingPointError("non-finite L-BFGS objective")
            last.update(aggregate)
            return torch.tensor(aggregate["total"], dtype=torch.float64, device=self.device)

        optimizer.step(closure)
        validation = heldout_terms(self.model, self.validation, self.config, self.device)
        self._record_best(validation)
        self._write_row(
            writer,
            "lbfgs",
            self.config.rar_cycles - 1,
            last["total"],
            last["pde"],
            last["gradient"],
            validation,
        )
        handle.flush()

    def train(self) -> dict[str, float | int | str]:
        if self.stage == "complete":
            final_validation = heldout_terms(
                self.model, self.validation, self.config, self.device
            )
            return {
                "global_adam_steps": self.global_step,
                "rar_cycles": self.config.rar_cycles,
                "rar_anchors": len(self.anchors),
                "heldout_total": final_validation["total"],
                "heldout_pde": final_validation["pde"],
                "heldout_gradient": final_validation["gradient"],
                "best_heldout_total": self.best_validation,
            }
        log_path = self.output_dir / "training_log.csv"
        append = self.resume and log_path.exists()
        if not append:
            rar_path = self.output_dir / "rar_history.jsonl"
            if rar_path.exists():
                rar_path.unlink()
        with log_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not append:
                writer.writeheader()
            if self.stage == "warmup":
                self._adam_stage(
                    writer, handle, "warmup", -1, self.config.warmup_steps
                )
                self.stage = "rar_train"
                self.cycle = 0
                self.stage_step = 0
                if self.config.rar_cycles > 0:
                    self._add_anchors(self.cycle)
                self._save_last()
            while self.stage == "rar_train" and self.cycle < self.config.rar_cycles:
                self._adam_stage(
                    writer,
                    handle,
                    "rar",
                    self.cycle,
                    self.config.rar_steps,
                )
                self.cycle += 1
                self.stage_step = 0
                if self.cycle < self.config.rar_cycles:
                    self._add_anchors(self.cycle)
                    self._save_last()
            self.stage = "lbfgs"
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            self._lbfgs(writer, handle)
            self.stage = "complete"
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            self._save_last()
        final_validation = heldout_terms(
            self.model, self.validation, self.config, self.device
        )
        return {
            "global_adam_steps": self.global_step,
            "rar_cycles": self.config.rar_cycles,
            "rar_anchors": len(self.anchors),
            "heldout_total": final_validation["total"],
            "heldout_pde": final_validation["pde"],
            "heldout_gradient": final_validation["gradient"],
            "best_heldout_total": self.best_validation,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output-root", type=Path, default=Path("results/burgers_1d_fixed"))
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/burgers_1d_fixed_reference_data"),
    )
    parser.add_argument("--resume", action="store_true")
    for name in (
        "hidden_layers",
        "width",
        "batch_size",
        "anchor_batch_size",
        "warmup_steps",
        "rar_cycles",
        "rar_steps",
        "candidate_points",
        "candidate_chunk_size",
        "anchors_per_cycle",
        "lr_decay_steps",
        "lbfgs_iterations",
        "lbfgs_points",
        "lbfgs_history_size",
        "validation_points",
        "validation_every",
        "log_every",
        "checkpoint_every",
        "test_nx",
        "test_nt",
        "reference_nx",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    for name in (
        "viscosity",
        "gradient_weight",
        "min_anchor_distance",
        "learning_rate",
        "lr_decay_rate",
        "reference_dt",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float)
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(seed=args.seed, device=args.device, mode=args.mode)
    if args.mode == "smoke":
        config.apply_smoke_defaults()
    for name, value in vars(args).items():
        if value is not None and hasattr(config, name) and name not in {
            "seed",
            "device",
            "mode",
        }:
            setattr(config, name, value)
    config.validate()
    return config


def run(args: argparse.Namespace) -> dict:
    config = config_from_args(args)
    set_seed(config.seed)
    torch.set_default_dtype(torch.float64)
    device = select_device(config.device)
    output_dir = (args.output_root / f"seed_{config.seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
    model = BurgersMLP(config.hidden_layers, config.width).to(device)
    trainer = AdaptiveTrainer(model, config, output_dir, device, args.resume)
    training_metrics = trainer.train()
    reference_path = ensure_reference(
        (args.reference_root / f"reference_nx{config.reference_nx}.npz").resolve(),
        config,
    )
    metrics = {
        "case": "burgers_1d",
        "seed": config.seed,
        "mode": config.mode,
        "device": str(device),
        **training_metrics,
        **evaluate(model, config, output_dir, reference_path, device),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    plot_training(output_dir / "training_log.csv", output_dir / "training_curve.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
