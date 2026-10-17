"""hp-VPINN solver for the 1D Poisson boundary-layer benchmark."""

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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.special import jacobi, roots_jacobi
from torch import nn


DOMAIN = (-1.0, 1.0)
ELEMENTS = np.asarray([[[-1.0], [-0.98]], [[-0.98], [-0.9]], [[-0.9], [1.0]]])


@dataclass(frozen=True)
class Stage:
    name: str
    n_test: int
    n_quad: int
    iterations: int
    learning_rate: float
    lr_gamma: float


FORMAL_STAGES = (
    Stage("stage1_p60_q80", 60, 80, 40001, 1.0e-3, 0.9),
    Stage("stage2_p60_q80_lr1e-5", 60, 80, 20001, 1.0e-5, 1.0),
    Stage("stage3_p100_q140", 100, 140, 30001, 1.0e-5, 1.0),
    Stage("final_p180_q200", 180, 200, 30001, 1.0e-5, 1.0),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    return 0.1 * torch.sin(5.0 * math.pi * x) + torch.exp(
        (0.01 - (x + 1.0)) / 0.01
    )


def forcing(x: torch.Tensor) -> torch.Tensor:
    exponential = torch.exp((0.01 - (x + 1.0)) / 0.01)
    u_xx = (
        -0.1 * (5.0 * math.pi) ** 2 * torch.sin(5.0 * math.pi * x)
        + exponential / 0.01**2
    )
    return -u_xx


def gauss_lobatto_legendre(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    interior = roots_jacobi(point_count - 2, 1, 1)[0]
    polynomial = jacobi(point_count - 1, 0, 0)
    interior_weights = 2.0 / (
        (point_count - 1) * point_count * polynomial(interior) ** 2
    )
    left_weight = 2.0 / (
        (point_count - 1) * point_count * polynomial(-1.0) ** 2
    )
    right_weight = 2.0 / (
        (point_count - 1) * point_count * polynomial(1.0) ** 2
    )
    points = np.concatenate(([-1.0], interior, [1.0]))
    weights = np.concatenate(([left_weight], interior_weights, [right_weight]))
    return points, weights


def legendre_value(degree: int, x: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    return np.polynomial.legendre.legval(x, coefficients)


def quadrature_and_tests(n_quad: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = gauss_lobatto_legendre(n_quad)
    tests = np.asarray(
        [
            legendre_value(n + 1, points) - legendre_value(n - 1, points)
            for n in range(1, n_test + 1)
        ]
    )
    return points[:, None], tests * weights[None, :]


def map_to_elements(local_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = ELEMENTS[:, 0, :]
    right = ELEMENTS[:, 1, :]
    half_size = (right - left) / 2.0
    center = (right + left) / 2.0
    points = center[:, None, :] + half_size[:, None, :] * local_points[None, :, :]
    return points, half_size[:, 0]


class ModifiedMLP(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.depth = 6
        self.width = 128
        self.encoder_u = nn.Linear(1, self.width, dtype=dtype)
        self.encoder_v = nn.Linear(1, self.width, dtype=dtype)
        self.gates = nn.ModuleList(
            [nn.Linear(1, self.width, dtype=dtype)]
            + [
                nn.Linear(self.width, self.width, dtype=dtype)
                for _ in range(self.depth - 1)
            ]
        )
        self.output = nn.Linear(self.width, 1, dtype=dtype)
        for layer in [self.encoder_u, self.encoder_v, *self.gates, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        # Normalization from [-1, 1] to [-1, 1] is the identity for this problem.
        features = coordinates
        encoder_u = torch.tanh(self.encoder_u(features))
        encoder_v = torch.tanh(self.encoder_v(features))
        hidden = features
        for layer in self.gates:
            gate = torch.tanh(layer(hidden))
            hidden = (1.0 - gate) * encoder_u + gate * encoder_v
        return self.output(hidden)


def gradient(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value,
        coordinates,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]


def strong_residual(model: nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    prediction = model(coordinates)
    u_x = gradient(prediction, coordinates)
    u_xx = gradient(u_x, coordinates)
    return -u_xx - forcing(coordinates)


def variational_loss(
    model: nn.Module,
    points: torch.Tensor,
    weighted_tests: torch.Tensor,
    jacobians: torch.Tensor,
    n_quad: int,
) -> torch.Tensor:
    active_points = points.detach().clone().requires_grad_(True)
    residual = strong_residual(model, active_points).reshape(len(ELEMENTS), n_quad, 1)
    projection = torch.einsum("eqc,tq->etc", residual, weighted_tests)
    projection = projection * jacobians[:, None, None]
    return torch.sum(torch.mean(projection**2, dim=(1, 2)))


def boundary_loss(
    model: nn.Module, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    points = torch.tensor([[-1.0], [1.0]], device=device, dtype=dtype)
    targets = exact_solution(points)
    return torch.mean((model(points) - targets) ** 2)


@torch.no_grad()
def predict(
    model: nn.Module,
    coordinates: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    values = []
    for start in range(0, coordinates.shape[0], 65536):
        points = torch.as_tensor(coordinates[start : start + 65536], device=device, dtype=dtype)
        values.append(model(points).cpu().numpy())
    return np.vstack(values)


def save_checkpoint(
    path: Path,
    iteration: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    history: list[dict[str, float | int]],
    prior_iterations: int,
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "prior_total_iterations": prior_iterations,
            "total_iterations": prior_iterations + iteration + 1,
        },
        path,
    )


def train_stage(
    stage: Stage,
    seed: int,
    output_dir: Path,
    device: torch.device,
    initial_checkpoint: Path | None,
    checkpoint_every: int,
    log_every: int,
) -> Path:
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if (output_dir / "metrics.json").is_file() and checkpoint_path.is_file():
        return checkpoint_path

    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    set_seed(seed)

    local_points, weighted_tests = quadrature_and_tests(stage.n_quad, stage.n_test)
    element_points, jacobians = map_to_elements(local_points)
    points = torch.as_tensor(
        element_points.reshape(-1, 1), device=device, dtype=dtype
    )
    weighted_tests_t = torch.as_tensor(weighted_tests, device=device, dtype=dtype)
    jacobians_t = torch.as_tensor(jacobians, device=device, dtype=dtype)

    audit_points = points.detach().clone().requires_grad_(True)
    exact_residual = strong_residual(exact_solution, audit_points)
    audit = float(torch.max(torch.abs(exact_residual)).detach().cpu())
    if not np.isfinite(audit) or audit > 1.0e-8:
        raise RuntimeError(f"Analytic residual audit failed: {audit:.3e}")

    model = ModifiedMLP(dtype).to(device)
    prior_iterations = 0
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        prior_iterations = int(
            checkpoint.get("total_iterations", int(checkpoint["iteration"]) + 1)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=stage.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=stage.lr_gamma
    )
    history: list[dict[str, float | int]] = []
    config = {
        **asdict(stage),
        "problem": "poisson_1d_boundary",
        "seed": seed,
        "domain": list(DOMAIN),
        "elements": ELEMENTS.tolist(),
        "network": "modified_mlp",
        "hidden_layers": 6,
        "width": 128,
        "dtype": "float64",
        "boundary_weight": 1.0,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "prior_total_iterations": prior_iterations,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    start_time = time.perf_counter()
    for iteration in range(stage.iterations):
        optimizer.zero_grad(set_to_none=True)
        equation_loss = variational_loss(
            model, points, weighted_tests_t, jacobians_t, stage.n_quad
        )
        condition_loss = boundary_loss(model, device, dtype)
        loss = equation_loss + condition_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % log_every == 0 or iteration == stage.iterations - 1:
            record = {
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "variational_loss": float(equation_loss.detach().cpu()),
                "constraint_loss": float(condition_loss.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(
                f"{stage.name} it={iteration:6d} loss={record['loss']:.4e} "
                f"var={record['variational_loss']:.4e} "
                f"boundary={record['constraint_loss']:.4e}",
                flush=True,
            )

        if checkpoint_every > 0 and iteration > 0 and iteration % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                iteration,
                model,
                optimizer,
                scheduler,
                history,
                prior_iterations,
            )

    elapsed = time.perf_counter() - start_time
    coordinates = np.linspace(-1.0, 1.0, 2001)[:, None]
    y_true = exact_solution(torch.as_tensor(coordinates, dtype=torch.float64)).numpy()
    y_pred = predict(model, coordinates, device, dtype)
    errors = np.abs(y_true - y_pred)
    metrics = {
        "problem": "poisson_1d_boundary",
        "seed": seed,
        "stage": stage.name,
        "relative_l2_error": float(
            np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        ),
        "max_pointwise_error": float(np.max(errors)),
        "iterations": stage.iterations,
        "total_iterations_including_initial_checkpoint": (
            prior_iterations + stage.iterations
        ),
        "evaluation_points": coordinates.shape[0],
        "elapsed_seconds": elapsed,
        "final_loss": history[-1]["loss"],
        "final_variational_loss": history[-1]["variational_loss"],
        "final_constraint_loss": history[-1]["constraint_loss"],
        "exact_strong_residual_max": audit,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output_dir / "predictions.npz",
        coordinates=coordinates,
        y_true=y_true,
        y_pred=y_pred,
        absolute_error=errors,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    save_checkpoint(
        checkpoint_path,
        stage.iterations - 1,
        model,
        optimizer,
        scheduler,
        history,
        prior_iterations,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    stages = FORMAL_STAGES
    if args.smoke:
        stages = tuple(
            Stage(stage.name, 4, 8, 2, stage.learning_rate, stage.lr_gamma)
            for stage in FORMAL_STAGES
        )

    seed_dir = args.output_dir / f"seed_{args.seed}"
    checkpoint: Path | None = None
    for stage in stages:
        checkpoint = train_stage(
            stage,
            args.seed,
            seed_dir / stage.name,
            device,
            checkpoint,
            args.checkpoint_every,
            args.log_every,
        )


if __name__ == "__main__":
    main()

