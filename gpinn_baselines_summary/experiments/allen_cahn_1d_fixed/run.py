"""Train the 1D Allen-Cahn gPINN with residual-adaptive refinement (RAR).

This standalone implementation solves the forward problem with periodic
value/derivative matching and residual ``u_t - D*u_xx + 5*u^3 - 5*u``. It adds the
gradient-enhanced (gPINN) loss and the residual-based adaptive refinement
(RAR) that the paper combines for Allen-Cahn specifically, following this
repository's own ``burgers_1d_fixed/`` precedent for "gPINN baseline + RAR".

Sources consulted:

- Yu, Lu, Meng, Karniadakis, "Gradient-enhanced physics-informed neural
  networks for forward and inverse PDE problems", CMAME 393 (2022) 114823,
  arXiv:2111.02801.
- The paper's official code, ``src/allen_cahn.py`` in
  https://github.com/lu-group/gpinn (DeepXDE/TensorFlow-1).  Its RAR loop is
  reproduced faithfully: 100 cycles, the top 30 highest-residual points out of
  100,000 random candidates added per cycle.  Its physics (D=0.001, a hard
  ``t*(1-x^2)*y + x^2*cos(pi*x)`` output transform, no explicit boundary
  condition list) is NOT reproduced -- see "Deviations" below.
- ``gpinn_baselines/cases/time_dependent.py::AllenCahn1D`` in this repository,
  the already-audited, paper-faithful (equation (20), D=1e-4) but
  intentionally RAR-free gPINN baseline.  Its loss structure (soft initial
  condition, soft periodic value and periodic first-derivative losses, a
  plain MLP with no output transform, gradient enhancement over both x and t)
  is reused here unchanged; this script only adds continuous resampling and
  RAR on top of it.

Deviations from the original ``src/allen_cahn.py`` (both intentional, and
already the position taken by this repository -- see IMPLEMENTATION_REPORT.md
-- for the RAR-free baseline this script extends):

1. Diffusion is fixed to 1e-4, matching the paper's equation (20a).  The
   original script and its ``usol_D_0.001_k_5.mat`` reference both use 0.001,
   which the paper's own text does not support.
2. No hard output transform.  The original script's transform pins
   u(-1,t)=u(1,t)=cos(pi)=-1 for *every* t (since 1-x^2 vanishes at both
   ends regardless of t) and never constrains u_x at all -- convenient, but
   not what a periodic boundary condition actually asserts.  This script
   keeps the soft periodic value and periodic first-derivative losses that
   ``AllenCahn1D`` already uses, on a plain (unconstrained) MLP.
3. The residual/interior points are freshly resampled every Adam step
   (Latin hypercube) instead of reusing one fixed pool, matching the fix
   already applied in ``burgers_1d_fixed`` to stop a 6x128 network from
   memorizing a static point set.
4. The evaluation reference is an independently generated periodic Fourier
   pseudospectral ETDRK4 solution at D=1e-4 (matching
   ``gpinn_baselines/references.py::generate_allen_cahn``), not the
   mismatched-diffusion ``usol_D_0.001_k_5.mat`` file.
5. RAR cycle/step counts are adapted to a fixed, deterministic step budget
   (so runs are reproducible and resumable) instead of the original's
   open-ended ``EarlyStopping(patience=2000)``; see RunConfig defaults below.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    diffusion: float = 1.0e-4
    reaction: float = 5.0
    hidden_layers: int = 6
    width: int = 128
    gradient_weight: float = 1.0e-4
    initial_weight: float = 100.0
    periodic_value_weight: float = 100.0
    periodic_derivative_weight: float = 100.0
    batch_size: int = 2048
    initial_batch_size: int = 512
    periodic_batch_size: int = 512
    anchor_batch_size: int = 512
    warmup_steps: int = 20_000
    rar_cycles: int = 30
    rar_steps: int = 2_000
    candidate_points: int = 100_000
    candidate_chunk_size: int = 4096
    anchors_per_cycle: int = 30
    min_anchor_distance: float = 2.0e-3
    learning_rate: float = 1.0e-3
    lr_decay_rate: float = 0.9
    lr_decay_steps: int = 5_000
    lbfgs_iterations: int = 1_000
    lbfgs_points: int = 8192
    lbfgs_history_size: int = 100
    validation_points: int = 4096
    validation_initial_points: int = 512
    validation_periodic_points: int = 512
    validation_every: int = 1000
    log_every: int = 100
    checkpoint_every: int = 1000
    test_nx: int = 257
    test_nt: int = 201
    reference_nx: int = 1024
    reference_dt: float = 2.5e-4

    def apply_smoke_defaults(self) -> None:
        self.hidden_layers = 2
        self.width = 24
        self.batch_size = 32
        self.initial_batch_size = 8
        self.periodic_batch_size = 8
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
        self.validation_initial_points = 16
        self.validation_periodic_points = 16
        self.validation_every = 1
        self.log_every = 1
        self.checkpoint_every = 2
        self.test_nx = 17
        self.test_nt = 11
        self.reference_nx = 64
        self.reference_dt = 1.0e-2

    def validate(self) -> None:
        integer_fields = (
            "hidden_layers",
            "width",
            "batch_size",
            "initial_batch_size",
            "periodic_batch_size",
            "warmup_steps",
            "rar_cycles",
            "rar_steps",
            "candidate_points",
            "candidate_chunk_size",
            "anchors_per_cycle",
            "lbfgs_iterations",
            "lbfgs_points",
            "validation_points",
            "validation_initial_points",
            "validation_periodic_points",
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
        if self.test_nx < 2 or self.test_nt < 2 or self.reference_nx < 8:
            raise ValueError("test grids need at least 2 points and reference_nx at least 8")
        if self.diffusion <= 0.0 or self.reference_dt <= 0.0:
            raise ValueError("diffusion and reference_dt must be positive")
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
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(name)
    if device.type == "mps":
        raise RuntimeError(
            "Allen-Cahn gPINN requires float64 third derivatives (u_xxx, u_xxt "
            "for the gradient-enhanced loss); PyTorch MPS does not support "
            "this reliably. Use --device cpu or --device cuda."
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


class AllenCahnMLP(nn.Module):
    """Plain tanh MLP; periodic value/derivative matching is a soft loss, not a transform."""

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
        z = points
        for layer in self.hidden:
            z = self.activation(layer(z))
        return self.output(z)


def _gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def allen_cahn_residual(
    model: nn.Module, points: torch.Tensor, diffusion: float, reaction: float
) -> torch.Tensor:
    u = model(points)
    first = _gradient(u, points)
    u_x = first[:, 0:1]
    u_t = first[:, 1:2]
    u_xx = _gradient(u_x, points)[:, 0:1]
    return u_t - diffusion * u_xx + reaction * u**3 - reaction * u


def initial_condition_numpy(x: np.ndarray) -> np.ndarray:
    return x**2 * np.cos(np.pi * x)


def loss_components(
    model: nn.Module,
    residual_points: torch.Tensor,
    initial_points: torch.Tensor,
    initial_values: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    diffusion: float,
    reaction: float,
) -> dict[str, torch.Tensor]:
    residual = allen_cahn_residual(model, residual_points, diffusion, reaction)
    residual_gradient = _gradient(residual, residual_points)
    pde = torch.mean(residual**2)
    gradient = torch.mean(residual_gradient[:, 0:1] ** 2) + torch.mean(
        residual_gradient[:, 1:2] ** 2
    )
    initial_pred = model(initial_points)
    initial = torch.mean((initial_pred - initial_values) ** 2)
    u_left = model(left)
    u_right = model(right)
    periodic_value = torch.mean((u_left - u_right) ** 2)
    ux_left = _gradient(u_left, left)[:, 0:1]
    ux_right = _gradient(u_right, right)[:, 0:1]
    periodic_derivative = torch.mean((ux_left - ux_right) ** 2)
    return {
        "pde": pde,
        "gradient": gradient,
        "initial": initial,
        "periodic_value": periodic_value,
        "periodic_derivative": periodic_derivative,
    }


def weighted_total(terms: dict[str, torch.Tensor], config: RunConfig) -> torch.Tensor:
    return (
        terms["pde"]
        + config.gradient_weight * terms["gradient"]
        + config.initial_weight * terms["initial"]
        + config.periodic_value_weight * terms["periodic_value"]
        + config.periodic_derivative_weight * terms["periodic_derivative"]
    )


def latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    """LHS over (x,t) in [-1,1]x[0,1]."""
    values = np.empty((n, 2), dtype=np.float64)
    for dim in range(2):
        strata = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        values[:, dim] = strata[rng.permutation(n)]
    values[:, 0] = 2.0 * values[:, 0] - 1.0
    return values


def latin_hypercube_1d(
    n: int, rng: np.random.Generator, lower: float = -1.0, upper: float = 1.0
) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    strata = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
    return lower + (upper - lower) * strata[rng.permutation(n)]


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


def sample_initial_batch(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    x = latin_hypercube_1d(n, rng, -1.0, 1.0)
    points = np.column_stack((x, np.zeros_like(x)))
    values = initial_condition_numpy(x)[:, None]
    return points, values


def sample_periodic_batch(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    times = latin_hypercube_1d(n, rng, 0.0, 1.0)
    left = np.column_stack((np.full(len(times), -1.0), times))
    right = np.column_stack((np.full(len(times), 1.0), times))
    return left, right


def as_input(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.as_tensor(points, dtype=torch.float64, device=device)
        .detach()
        .requires_grad_(True)
    )


def residual_scores(
    model: nn.Module,
    points: np.ndarray,
    diffusion: float,
    reaction: float,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.enable_grad():
        for start in range(0, len(points), chunk_size):
            chunk = as_input(points[start : start + chunk_size], device)
            value = allen_cahn_residual(model, chunk, diffusion, reaction)
            scores.append(torch.abs(value).detach().cpu().numpy()[:, 0])
    return np.concatenate(scores)


def residual_scores_from_candidates(
    selected: np.ndarray, candidates: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    lookup = {tuple(point): score for point, score in zip(candidates, scores)}
    return np.asarray([lookup[tuple(point)] for point in selected], dtype=np.float64)


def select_rar_anchors(
    candidates: np.ndarray,
    scores: np.ndarray,
    existing: np.ndarray,
    count: int,
    min_distance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Top-residual candidates, filtered so anchors don't cluster on one point.

    The original gPINN+RAR script (lu-group/gpinn, src/allen_cahn.py) adds
    the raw top-k highest-residual points every cycle with no spacing check,
    which tends to pile many near-duplicate anchors on the same sharp front.
    This mirrors burgers_1d_fixed's fix: a minimum-distance filter (in a
    space that first rescales x from [-1,1] to [0,1] so both axes are
    comparable) spreads anchors across distinct high-residual regions.
    """

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


def allen_cahn_reference_grid(
    n: int, dt: float, diffusion: float, reaction: float, save_dt: float = 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Fourier pseudospectral ETDRK4 (Kassam & Trefethen 2005).

    Matches ``gpinn_baselines/references.py::_allen_cahn_grid`` (same scheme,
    parameterized by diffusion/reaction instead of hardcoded 1e-4/5.0) so the
    evaluation reference here is consistent with the rest of the repository.
    """

    steps = int(round(1.0 / dt))
    stride = int(round(save_dt / dt))
    if n < 8:
        raise ValueError("reference grid needs at least 8 Fourier modes")
    if not np.isclose(steps * dt, 1.0):
        raise ValueError("reference grid must cover t in [0, 1] exactly")
    if stride < 1 or not np.isclose(stride * dt, save_dt):
        raise ValueError("save_dt must be an integer multiple of dt")
    if steps % stride != 0:
        raise ValueError("saved reference snapshots must include t=1 exactly")

    x = np.linspace(-1.0, 1.0, n, endpoint=False, dtype=np.float64)
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=2.0 / n)
    linear = -diffusion * wave**2
    times = np.arange(0, steps + 1, stride, dtype=np.float64) * dt
    history = np.empty((len(times), n), dtype=np.float64)
    value = x**2 * np.cos(np.pi * x)
    spectral = np.fft.fft(value)
    history[0] = value

    E = np.exp(dt * linear)
    E2 = np.exp(dt * linear / 2.0)
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32.0)
    lr = dt * linear[:, None] + roots[None, :]
    Q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
    f1 = dt * np.real(
        np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
    )
    f2 = dt * np.real(np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1))
    f3 = dt * np.real(
        np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
    )

    def nonlinear(vhat: np.ndarray) -> np.ndarray:
        physical = np.fft.ifft(vhat).real
        return np.fft.fft(reaction * physical - reaction * physical**3)

    save_index = 1
    for step in range(1, steps + 1):
        Nv = nonlinear(spectral)
        a = E2 * spectral + Q * Nv
        Na = nonlinear(a)
        b = E2 * spectral + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        spectral = E * spectral + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        if step % stride == 0:
            history[save_index] = np.fft.ifft(spectral).real
            save_index += 1
    if save_index != len(history):
        raise RuntimeError("reference time history was not filled completely")
    return x, times, history


def ensure_reference(path: Path, config: RunConfig) -> Path:
    expected = {
        "diffusion": config.diffusion,
        "reaction": config.reaction,
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

    # Ten-seed jobs may finish at nearly the same time; serialize the
    # one-time reference generation the same way burgers_1d_fixed does.
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if matches():
            return path
        reference_stride = max(1, int(round(5.0e-3 / config.reference_dt)))
        reference_save_dt = reference_stride * config.reference_dt
        x, t, u = allen_cahn_reference_grid(
            config.reference_nx,
            config.reference_dt,
            config.diffusion,
            config.reaction,
            save_dt=reference_save_dt,
        )
        metadata = {
            **expected,
            "equation": "u_t - diffusion*u_xx + reaction*u^3 - reaction*u = 0",
            "method": "periodic Fourier pseudospectral ETDRK4",
            "saved_dt": reference_save_dt,
            "boundary": "periodic: u(-1,t)=u(1,t), u_x(-1,t)=u_x(1,t)",
            "initial": "u(x,0)=x^2*cos(pi*x)",
        }
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary, x=x, t=t, u=u, metadata=np.asarray(json.dumps(metadata))
        )
        os.replace(temporary, path)
    return path


def load_reference(path: Path) -> tuple[RegularGridInterpolator, str]:
    with np.load(path) as data:
        x = data["x"]
        t = data["t"]
        u = data["u"]
        metadata = str(data["metadata"])
    # The Fourier grid is periodic and deliberately excludes x=1; add the
    # equivalent wrap-around column so the PINN test grid may include both
    # -1 and 1 without relaxing the interpolator's bounds check.
    periodic_x = np.concatenate((x, np.asarray([1.0], dtype=np.float64)))
    periodic_u = np.concatenate((u, u[:, 0:1]), axis=1)
    interpolator = RegularGridInterpolator((t, periodic_x), periodic_u, bounds_error=True)
    return interpolator, metadata


def predict(
    model: nn.Module, points: np.ndarray, device: torch.device, chunk_size: int = 8192
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            tensor = torch.as_tensor(
                points[start : start + chunk_size], dtype=torch.float64, device=device
            )
            values.append(model(tensor).cpu().numpy())
    return np.concatenate(values, axis=0)


def boundary_gaps(
    model: nn.Module, nt: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64, device=device)
    left = torch.stack((-torch.ones_like(t), t), dim=1).requires_grad_(True)
    right = torch.stack((torch.ones_like(t), t), dim=1).requires_grad_(True)
    model.eval()
    with torch.enable_grad():
        u_left = model(left)
        u_right = model(right)
        ux_left = _gradient(u_left, left)[:, 0:1]
        ux_right = _gradient(u_right, right)[:, 0:1]
        value_gap = (u_left - u_right).detach().cpu().numpy()[:, 0]
        derivative_gap = (ux_left - ux_right).detach().cpu().numpy()[:, 0]
    return value_gap, derivative_gap


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
    interpolator, reference_metadata = load_reference(reference_path)
    reference = interpolator(np.column_stack((points[:, 1], points[:, 0])))[:, None]
    error = prediction - reference
    prediction_grid = prediction.reshape(T.shape)
    initial_error = float(np.max(np.abs(prediction_grid[0] - initial_condition_numpy(x))))
    periodic_value_grid_error = float(
        np.max(np.abs(prediction_grid[:, 0] - prediction_grid[:, -1]))
    )
    value_gap, derivative_gap = boundary_gaps(model, config.test_nt, device)
    metrics: dict[str, float | int | str] = {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "initial_condition_max_abs": initial_error,
        "periodic_value_grid_max_abs": periodic_value_grid_error,
        "periodic_value_autograd_max_abs": float(np.max(np.abs(value_gap))),
        "periodic_derivative_max_abs": float(np.max(np.abs(derivative_gap))),
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
        (prediction, "gPINN+RAR prediction", "coolwarm"),
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
    "initial",
    "periodic_value",
    "periodic_derivative",
    "validation_total",
    "validation_pde",
    "validation_gradient",
    "validation_initial",
    "validation_periodic_value",
    "validation_periodic_derivative",
    "anchors",
    "lr",
    "wall_seconds",
)


def heldout_terms(
    model: nn.Module,
    validation: dict[str, np.ndarray],
    config: RunConfig,
    device: torch.device,
) -> dict[str, float]:
    sums = {"pde": 0.0, "gradient": 0.0}
    model.eval()
    with torch.enable_grad():
        residual_points = validation["residual"]
        for start in range(0, len(residual_points), config.candidate_chunk_size):
            chunk = as_input(
                residual_points[start : start + config.candidate_chunk_size], device
            )
            residual = allen_cahn_residual(model, chunk, config.diffusion, config.reaction)
            residual_gradient = _gradient(residual, chunk)
            fraction = len(chunk) / len(residual_points)
            sums["pde"] += fraction * float(torch.mean(residual**2).detach())
            sums["gradient"] += fraction * float(
                (
                    torch.mean(residual_gradient[:, 0:1] ** 2)
                    + torch.mean(residual_gradient[:, 1:2] ** 2)
                ).detach()
            )
        initial_points = torch.as_tensor(
            validation["initial_points"], dtype=torch.float64, device=device
        )
        initial_values = torch.as_tensor(
            validation["initial_values"], dtype=torch.float64, device=device
        )
        sums["initial"] = float(
            torch.mean((model(initial_points) - initial_values) ** 2).detach()
        )
        left = as_input(validation["periodic_left"], device)
        right = as_input(validation["periodic_right"], device)
        u_left = model(left)
        u_right = model(right)
        sums["periodic_value"] = float(torch.mean((u_left - u_right) ** 2).detach())
        ux_left = _gradient(u_left, left)[:, 0:1]
        ux_right = _gradient(u_right, right)[:, 0:1]
        sums["periodic_derivative"] = float(
            torch.mean((ux_left - ux_right) ** 2).detach()
        )
    sums["total"] = (
        sums["pde"]
        + config.gradient_weight * sums["gradient"]
        + config.initial_weight * sums["initial"]
        + config.periodic_value_weight * sums["periodic_value"]
        + config.periodic_derivative_weight * sums["periodic_derivative"]
    )
    return sums


class AdaptiveTrainer:
    def __init__(
        self,
        model: AllenCahnMLP,
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
        self.validation_residual = latin_hypercube(config.validation_points, self.rng)
        self.validation_initial_points, self.validation_initial_values = sample_initial_batch(
            config.validation_initial_points, self.rng
        )
        self.validation_periodic_left, self.validation_periodic_right = sample_periodic_batch(
            config.validation_periodic_points, self.rng
        )
        self.anchors = np.empty((0, 2), dtype=np.float64)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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

    def _validation_dict(self) -> dict[str, np.ndarray]:
        return {
            "residual": self.validation_residual,
            "initial_points": self.validation_initial_points,
            "initial_values": self.validation_initial_values,
            "periodic_left": self.validation_periodic_left,
            "periodic_right": self.validation_periodic_right,
        }

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
            "validation_residual": self.validation_residual,
            "validation_initial_points": self.validation_initial_points,
            "validation_initial_values": self.validation_initial_values,
            "validation_periodic_left": self.validation_periodic_left,
            "validation_periodic_right": self.validation_periodic_right,
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
        self.validation_residual = np.asarray(state["validation_residual"], dtype=np.float64)
        self.validation_initial_points = np.asarray(
            state["validation_initial_points"], dtype=np.float64
        )
        self.validation_initial_values = np.asarray(
            state["validation_initial_values"], dtype=np.float64
        )
        self.validation_periodic_left = np.asarray(
            state["validation_periodic_left"], dtype=np.float64
        )
        self.validation_periodic_right = np.asarray(
            state["validation_periodic_right"], dtype=np.float64
        )
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
        values: dict[str, float],
        validation: dict[str, float] | None,
    ) -> None:
        writer.writerow(
            {
                "phase": phase,
                "cycle": cycle,
                "step": self.global_step,
                "total": values["total"],
                "pde": values["pde"],
                "gradient": values["gradient"],
                "initial": values["initial"],
                "periodic_value": values["periodic_value"],
                "periodic_derivative": values["periodic_derivative"],
                "validation_total": "" if validation is None else validation["total"],
                "validation_pde": "" if validation is None else validation["pde"],
                "validation_gradient": ""
                if validation is None
                else validation["gradient"],
                "validation_initial": "" if validation is None else validation["initial"],
                "validation_periodic_value": ""
                if validation is None
                else validation["periodic_value"],
                "validation_periodic_derivative": ""
                if validation is None
                else validation["periodic_derivative"],
                "anchors": len(self.anchors),
                "lr": self.optimizer.param_groups[0]["lr"],
                "wall_seconds": time.perf_counter() - self.started,
            }
        )

    def _sample_step(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        residual_batch = make_batch(
            self.rng, self.config.batch_size, self.anchors, self.config.anchor_batch_size
        )
        initial_points, initial_values = sample_initial_batch(
            self.config.initial_batch_size, self.rng
        )
        left, right = sample_periodic_batch(self.config.periodic_batch_size, self.rng)
        return residual_batch, initial_points, initial_values, left, right

    def _adam_stage(
        self,
        writer: csv.DictWriter,
        handle,
        phase: str,
        cycle: int,
        target_steps: int,
    ) -> None:
        last_values = {
            name: float("nan")
            for name in ("total", "pde", "gradient", "initial", "periodic_value", "periodic_derivative")
        }
        while self.stage_step < target_steps:
            residual_batch, initial_points_np, initial_values_np, left_np, right_np = (
                self._sample_step()
            )
            residual_points = as_input(residual_batch, self.device)
            initial_points = torch.as_tensor(
                initial_points_np, dtype=torch.float64, device=self.device
            )
            initial_values = torch.as_tensor(
                initial_values_np, dtype=torch.float64, device=self.device
            )
            left = as_input(left_np, self.device)
            right = as_input(right_np, self.device)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            terms = loss_components(
                self.model,
                residual_points,
                initial_points,
                initial_values,
                left,
                right,
                self.config.diffusion,
                self.config.reaction,
            )
            total = weighted_total(terms, self.config)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {self.global_step + 1}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            self.stage_step += 1
            last_values = {name: float(value.detach()) for name, value in terms.items()}
            last_values["total"] = float(total.detach())

            needs_validation = (
                self.global_step == 1
                or self.global_step % self.config.validation_every == 0
                or self.stage_step == target_steps
            )
            validation = None
            if needs_validation:
                validation = heldout_terms(
                    self.model, self._validation_dict(), self.config, self.device
                )
                self._record_best(validation)
            needs_log = (
                self.global_step == 1
                or self.global_step % self.config.log_every == 0
                or self.stage_step == target_steps
                or validation is not None
            )
            if needs_log:
                self._write_row(writer, phase, cycle, last_values, validation)
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
            self.config.diffusion,
            self.config.reaction,
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
        fixed_residual = latin_hypercube(self.config.lbfgs_points, base_rng)
        if len(self.anchors):
            fixed_residual = np.concatenate((fixed_residual, self.anchors), axis=0)
        fixed_initial_points, fixed_initial_values = sample_initial_batch(
            self.config.initial_batch_size, base_rng
        )
        fixed_left, fixed_right = sample_periodic_batch(
            self.config.periodic_batch_size, base_rng
        )
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config.lbfgs_iterations,
            max_eval=max(1, int(self.config.lbfgs_iterations * 1.25)),
            history_size=self.config.lbfgs_history_size,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )
        last = {
            name: float("nan")
            for name in ("total", "pde", "gradient", "initial", "periodic_value", "periodic_derivative")
        }

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            aggregate = {"pde": 0.0, "gradient": 0.0}
            for start in range(0, len(fixed_residual), self.config.batch_size):
                array = fixed_residual[start : start + self.config.batch_size]
                points = as_input(array, self.device)
                residual = allen_cahn_residual(
                    self.model, points, self.config.diffusion, self.config.reaction
                )
                residual_gradient = _gradient(residual, points)
                pde = torch.mean(residual**2)
                gradient = torch.mean(residual_gradient[:, 0:1] ** 2) + torch.mean(
                    residual_gradient[:, 1:2] ** 2
                )
                fraction = len(array) / len(fixed_residual)
                (fraction * (pde + self.config.gradient_weight * gradient)).backward()
                aggregate["pde"] += fraction * float(pde.detach())
                aggregate["gradient"] += fraction * float(gradient.detach())
            initial_points = torch.as_tensor(
                fixed_initial_points, dtype=torch.float64, device=self.device
            )
            initial_values = torch.as_tensor(
                fixed_initial_values, dtype=torch.float64, device=self.device
            )
            initial = torch.mean((self.model(initial_points) - initial_values) ** 2)
            left = as_input(fixed_left, self.device)
            right = as_input(fixed_right, self.device)
            u_left = self.model(left)
            u_right = self.model(right)
            periodic_value = torch.mean((u_left - u_right) ** 2)
            ux_left = _gradient(u_left, left)[:, 0:1]
            ux_right = _gradient(u_right, right)[:, 0:1]
            periodic_derivative = torch.mean((ux_left - ux_right) ** 2)
            constraint_loss = (
                self.config.initial_weight * initial
                + self.config.periodic_value_weight * periodic_value
                + self.config.periodic_derivative_weight * periodic_derivative
            )
            constraint_loss.backward()
            total_value = (
                aggregate["pde"]
                + self.config.gradient_weight * aggregate["gradient"]
                + float(constraint_loss.detach())
            )
            if not np.isfinite(total_value):
                raise FloatingPointError("non-finite L-BFGS objective")
            last.update(
                {
                    "total": total_value,
                    "pde": aggregate["pde"],
                    "gradient": aggregate["gradient"],
                    "initial": float(initial.detach()),
                    "periodic_value": float(periodic_value.detach()),
                    "periodic_derivative": float(periodic_derivative.detach()),
                }
            )
            return torch.tensor(total_value, dtype=torch.float64, device=self.device)

        optimizer.step(closure)
        validation = heldout_terms(self.model, self._validation_dict(), self.config, self.device)
        self._record_best(validation)
        self._write_row(writer, "lbfgs", self.config.rar_cycles - 1, last, validation)
        handle.flush()

    def _training_summary(self, final_validation: dict[str, float]) -> dict[str, float | int | str]:
        return {
            "global_adam_steps": self.global_step,
            "rar_cycles": self.config.rar_cycles,
            "rar_anchors": len(self.anchors),
            "heldout_total": final_validation["total"],
            "heldout_pde": final_validation["pde"],
            "heldout_gradient": final_validation["gradient"],
            "heldout_initial": final_validation["initial"],
            "heldout_periodic_value": final_validation["periodic_value"],
            "heldout_periodic_derivative": final_validation["periodic_derivative"],
            "best_heldout_total": self.best_validation,
        }

    def train(self) -> dict[str, float | int | str]:
        if self.stage == "complete":
            final_validation = heldout_terms(
                self.model, self._validation_dict(), self.config, self.device
            )
            return self._training_summary(final_validation)
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
            self.model, self._validation_dict(), self.config, self.device
        )
        return self._training_summary(final_validation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/allen_cahn_1d_fixed")
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/allen_cahn_1d_fixed_reference_data"),
    )
    parser.add_argument("--resume", action="store_true")
    for name in (
        "hidden_layers",
        "width",
        "batch_size",
        "initial_batch_size",
        "periodic_batch_size",
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
        "validation_initial_points",
        "validation_periodic_points",
        "validation_every",
        "log_every",
        "checkpoint_every",
        "test_nx",
        "test_nt",
        "reference_nx",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    for name in (
        "diffusion",
        "reaction",
        "gradient_weight",
        "initial_weight",
        "periodic_value_weight",
        "periodic_derivative_weight",
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
    model = AllenCahnMLP(config.hidden_layers, config.width).to(device)
    trainer = AdaptiveTrainer(model, config, output_dir, device, args.resume)
    training_metrics = trainer.train()
    reference_path = ensure_reference(
        (args.reference_root / f"reference_nx{config.reference_nx}.npz").resolve(),
        config,
    )
    metrics = {
        "case": "allen_cahn_1d",
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
