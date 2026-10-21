import argparse
from pathlib import Path

import numpy as np
import torch

from pinn_baselines.common_modified_mlp_corrected import (
    ModifiedMLP,
    TrainConfig,
    compile_and_train,
    dde,
    evaluate_errors,
    experiment_dir,
    positive_int,
    save_metrics,
    save_predictions,
    set_seed,
)


A1 = 1
A2 = 4
WAVE_NUMBER = 1.0
DOMAIN = (-1.0, 1.0)


def exact_solution(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    return np.sin(A1 * np.pi * x0) * np.sin(A2 * np.pi * x1)


def forcing(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    solution = torch.sin(A1 * torch.pi * x0) * torch.sin(A2 * torch.pi * x1)
    return (
        -(A1 * torch.pi) ** 2
        - (A2 * torch.pi) ** 2
        + WAVE_NUMBER**2
    ) * solution


def mesh_points(grid_size: int):
    coordinates = np.linspace(*DOMAIN, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def boundary_points(points_per_edge: int):
    lower, upper = DOMAIN
    edge = np.linspace(lower, upper, points_per_edge, endpoint=False, dtype=np.float64)
    reverse_edge = np.linspace(
        upper, lower, points_per_edge, endpoint=False, dtype=np.float64
    )
    bottom = np.column_stack((edge, np.full(points_per_edge, lower)))
    right = np.column_stack((np.full(points_per_edge, upper), edge))
    top = np.column_stack((reverse_edge, np.full(points_per_edge, upper)))
    left = np.column_stack((np.full(points_per_edge, lower), reverse_edge))
    return np.vstack((bottom, right, top, left))


def build_problem(residual_grid_size: int, boundary_points_per_edge: int):
    lower, upper = DOMAIN
    geom = dde.geometry.Rectangle([lower, lower], [upper, upper])

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return dy_xx + dy_yy + WAVE_NUMBER**2 * y - forcing(x)

    residual_x = mesh_points(residual_grid_size)
    boundary_x = boundary_points(boundary_points_per_edge)
    bc = dde.icbc.PointSetBC(boundary_x, exact_solution(boundary_x))
    data = dde.data.PDE(
        geom,
        pde,
        [bc],
        num_domain=0,
        num_boundary=0,
        train_distribution="uniform",
        anchors=residual_x,
        solution=exact_solution,
        num_test=10000,
    )
    net = ModifiedMLP(input_dim=2, output_dim=1, depth=6, width=128)
    return dde.Model(data, net)


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 2D Helmholtz experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--residual-grid-size", type=positive_int, default=101)
    parser.add_argument("--boundary-points-per-edge", type=positive_int, default=50)
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=1000)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--test-grid-size", type=positive_int, default=201)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(args.residual_grid_size, args.boundary_points_per_edge)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(model, train_config, loss_weights=[1.0, 100.0])

    test_x = mesh_points(args.test_grid_size)
    errors = evaluate_errors(model, test_x, exact_solution)
    run_dir = experiment_dir(args.output_dir, "helmholtz_2d", "a1_1_a2_4_k_1")
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    save_predictions(predictions_path, test_x, exact_solution, model)
    save_metrics(
        metrics_path,
        errors,
        metadata={
            "equation": "2D Helmholtz",
            "paper_equations": ["Eq. 14", "Eq. 15", "Eq. 16"],
            "domain": [list(DOMAIN), list(DOMAIN)],
            "parameters": {
                "a1": A1,
                "a2": A2,
                "k": WAVE_NUMBER,
                "source": "confirmed experiment setting",
                "status": "confirmed",
            },
            "exact_solution": "sin(a1*pi*x) * sin(a2*pi*y)",
            "num_domain": args.residual_grid_size**2,
            "residual_grid": [args.residual_grid_size, args.residual_grid_size],
            "num_boundary": 4 * args.boundary_points_per_edge,
            "boundary_points_per_edge": args.boundary_points_per_edge,
            "domain_sampling": "uniform mesh including endpoints",
            "boundary_sampling": (
                f"uniform, {args.boundary_points_per_edge} points per edge"
            ),
            "test_grid_size": args.test_grid_size,
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "boundary": 100.0},
            "boundary_loss": {
                "reduction": "mean",
                "formula": "100 * mean(boundary point squared errors)",
                "paper_reference": "Eq. 3 and Appendix D, 2D Helmholtz loss",
            },
            "backbone": {
                "type": "modified_mlp",
                "version": "modified_mlp_corrected",
                "formula_reference": "DASA-PINNs reference [21], Eq. 4.1-4.5",
                "hidden_layer_count": 6,
                "gate_update_count": 5,
                "parameter_count": sum(p.numel() for p in model.net.parameters()),
                "depth": 6,
                "width": 128,
                "activation": "tanh",
                "initializer": "glorot normal",
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "budget_source": (
                    "Classical PINN mapping of paper NS3=10000 and NS4=10000 "
                    "to one standard Adam phase."
                ),
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
                "lbfgs_source": "Paper Section 3.1.4 specifies 1000 iterations.",
            },
            "seed": args.seed,
        },
    )
    print("boundary_loss=paper Eq. 3 mean with Appendix D weight wb=100")
    print(f"l2_relative_error={errors['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={errors['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
