"""Single-seed hp-VPINN reproduction for the 1D steep Poisson benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights, Jacobi


DTYPE = torch.float64


def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def max_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def exact_solution(x: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(8.0 * np.pi * x) + np.tanh(80.0 * x)


def forcing(x: np.ndarray) -> np.ndarray:
    omega = 8.0 * np.pi
    r = 80.0
    u_xx = (
        -0.1 * omega**2 * np.sin(omega * x)
        - 2.0 * r**2 * np.tanh(r * x) / np.cosh(r * x) ** 2
    )
    return -u_xx


class SineMLP(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features, dtype=DTYPE)
            for in_features, out_features in zip(layers[:-1], layers[1:])
        )
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))
        return self.layers[-1](x)


def first_and_second_derivatives(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = model(x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    return u, u_x, u_xx


def make_test_functions(n_test: int, xi: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [Jacobi(n + 1, 0, 0, xi) - Jacobi(n - 1, 0, 0, xi) for n in range(1, n_test + 1)]
    )
    return values.reshape(n_test, -1)


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=DTYPE, device=device).reshape(-1, 1)
        return model(x_tensor).cpu().numpy()


def evaluate_grid(
    model: nn.Module, step: float, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    count = int(round(2.0 / step)) + 1
    x = np.linspace(-1.0, 1.0, count, dtype=np.float64).reshape(-1, 1)
    y_true = exact_solution(x)
    y_pred = predict(model, x, device)
    return (
        x,
        y_true,
        y_pred,
        l2_relative_error(y_true, y_pred),
        max_pointwise_error(y_true, y_pred),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=1001)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-elements", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=60)
    parser.add_argument("--n-quad", type=int, default=80)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/poisson_1d_steep/seed_1234"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xi, weights = GaussLobattoJacobiWeights(args.n_quad, 0, 0)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1, 1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    test_functions = make_test_functions(args.n_test, xi)
    if args.n_elements == 3:
        # Adaptive grid used by the original hp-VPINN Poisson-1D script.
        grid = np.asarray([-1.0, -0.1, 0.1, 1.0], dtype=np.float64)
    else:
        grid = np.linspace(-1.0, 1.0, args.n_elements + 1, dtype=np.float64)

    element_data: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
    ] = []
    exact_projected_residual_max = 0.0
    for left, right in zip(grid[:-1], grid[1:]):
        jacobian = (right - left) / 2.0
        x_element = left + jacobian * (xi + 1.0)
        f_element = forcing(x_element)
        rhs = jacobian * np.sum(
            weights.reshape(1, -1)
            * f_element.reshape(1, -1)
            * test_functions.reshape(args.n_test, -1),
            axis=1,
            keepdims=True,
        )

        # For the manufactured solution, -u_xx equals f pointwise.
        exact_lhs = jacobian * np.sum(
            weights.reshape(1, -1)
            * f_element.reshape(1, -1)
            * test_functions.reshape(args.n_test, -1),
            axis=1,
            keepdims=True,
        )
        exact_projected_residual_max = max(
            exact_projected_residual_max, float(np.max(np.abs(exact_lhs - rhs)))
        )

        element_data.append(
            (
                torch.as_tensor(x_element, dtype=DTYPE, device=device),
                torch.as_tensor(weights, dtype=DTYPE, device=device),
                torch.as_tensor(test_functions, dtype=DTYPE, device=device),
                torch.as_tensor(rhs, dtype=DTYPE, device=device),
                jacobian,
            )
        )

    boundary_x_np = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    boundary_u_np = exact_solution(boundary_x_np)
    boundary_x = torch.as_tensor(boundary_x_np, dtype=DTYPE, device=device)
    boundary_u = torch.as_tensor(boundary_u_np, dtype=DTYPE, device=device)

    model = SineMLP([1, 20, 20, 20, 20, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []
    evaluation_history: list[dict[str, float | int]] = []
    start_iteration = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        start_iteration = int(checkpoint["iteration"]) + 1
        history = checkpoint.get("history", [])
        evaluation_history = checkpoint.get("evaluation_history", [])
        print(f"Resuming from iteration {start_iteration}")
    start = time.perf_counter()

    for iteration in range(start_iteration, args.iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_variational = torch.zeros((), dtype=DTYPE, device=device)
        for (
            x_element_base,
            w_element,
            tests_element,
            rhs_element,
            jacobian,
        ) in element_data:
            x_element = x_element_base.detach().clone().requires_grad_(True)
            _, _, u_xx = first_and_second_derivatives(model, x_element)
            lhs = -((tests_element * (w_element * u_xx).T).sum(dim=1, keepdim=True))
            residual = jacobian * lhs - rhs_element
            loss_variational = loss_variational + torch.mean(residual.square())

        boundary_prediction = model(boundary_x)
        loss_boundary = torch.mean((boundary_u - boundary_prediction).square())
        loss = args.boundary_weight * loss_boundary + loss_variational
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0 or iteration == args.iterations - 1:
            record = {
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "loss_boundary": float(loss_boundary.detach().cpu()),
                "loss_variational": float(loss_variational.detach().cpu()),
            }
            history.append(record)
            if iteration % 100 == 0 or iteration == args.iterations - 1:
                print(
                    f"It: {iteration:6d}, Loss: {record['loss']:.3e}, "
                    f"Lossb: {record['loss_boundary']:.3e}, "
                    f"Lossv: {record['loss_variational']:.3e}"
                )

        if args.eval_every > 0 and (
            iteration % args.eval_every == 0 or iteration == args.iterations - 1
        ):
            _, _, _, eval_l2, eval_max = evaluate_grid(model, 0.001, device)
            evaluation = {
                "iteration": iteration,
                "l2_relative_error": eval_l2,
                "max_pointwise_error": eval_max,
            }
            evaluation_history.append(evaluation)
            print(
                f"Eval: {iteration:6d}, Relative L2: {eval_l2:.3e}, "
                f"Max error: {eval_max:.3e}"
            )

        if args.checkpoint_every > 0 and iteration > 0 and (
            iteration % args.checkpoint_every == 0
        ):
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "evaluation_history": evaluation_history,
                },
                output_dir / "checkpoint_latest.pt",
            )

    elapsed_seconds = time.perf_counter() - start
    x_test, y_true, y_pred, relative_l2, max_error = evaluate_grid(model, 0.001, device)
    _, _, _, relative_l2_coarse, _ = evaluate_grid(model, 0.002, device)
    _, _, _, relative_l2_fine, _ = evaluate_grid(model, 0.0005, device)
    error = np.abs(y_true - y_pred)
    max_error_index = int(np.argmax(error))
    boundary_prediction_np = predict(model, boundary_x_np, device)
    boundary_max_error = max_pointwise_error(boundary_u_np, boundary_prediction_np)

    metrics = {
        "benchmark": "1D Poisson steep",
        "equation": "-u_xx = f on [-1, 1]",
        "exact_solution": "0.1*sin(8*pi*x) + tanh(80*x)",
        "seed": args.seed,
        "device": str(device),
        "dtype": "float64",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "n_elements": args.n_elements,
        "element_grid": grid.tolist(),
        "n_test_functions_per_element": args.n_test,
        "n_quadrature_points_per_element": args.n_quad,
        "boundary_weight": args.boundary_weight,
        "network_layers": [1, 20, 20, 20, 20, 1],
        "activation": "sin",
        "variational_form": 1,
        "test_grid_step": 0.001,
        "test_grid_size": int(x_test.size),
        "l2_relative_error": relative_l2,
        "max_pointwise_error": max_error,
        "max_pointwise_error_x": float(x_test[max_error_index, 0]),
        "boundary_max_error": boundary_max_error,
        "l2_relative_error_step_0.002": relative_l2_coarse,
        "l2_relative_error_step_0.0005": relative_l2_fine,
        "exact_projected_residual_max": exact_projected_residual_max,
        "final_loss": history[-1]["loss"],
        "final_boundary_loss": history[-1]["loss_boundary"],
        "final_variational_loss": history[-1]["loss_variational"],
        "elapsed_seconds": elapsed_seconds,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "evaluation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=evaluation_history[0].keys())
        writer.writeheader()
        writer.writerows(evaluation_history)
    np.savez(
        output_dir / "predictions.npz",
        x=x_test,
        y_true=y_true,
        y_pred=y_pred,
        absolute_error=error,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x_test, y_true, label="exact", linewidth=1.5)
    axes[0].plot(x_test, y_pred, "--", label="hp-VPINN", linewidth=1.2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(x_test, np.maximum(error, np.finfo(float).tiny))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("absolute error")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_and_error.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
