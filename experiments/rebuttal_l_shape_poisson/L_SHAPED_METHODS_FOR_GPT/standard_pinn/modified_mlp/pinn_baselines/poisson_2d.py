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


PEAK_A = 10
PEAK_SCALE = float(2 ** (4 * PEAK_A))


def peak_exact(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    return (
        PEAK_SCALE
        * x0**PEAK_A
        * (1.0 - x0) ** PEAK_A
        * x1**PEAK_A
        * (1.0 - x1) ** PEAK_A
    )


def peak_forcing(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    a = PEAK_A
    px = x0**a * (1.0 - x0) ** a
    pxx = (
        a * (a - 1) * x0 ** (a - 2) * (1.0 - x0) ** a
        - 2.0 * a**2 * x0 ** (a - 1) * (1.0 - x0) ** (a - 1)
        + a * (a - 1) * x0**a * (1.0 - x0) ** (a - 2)
    )
    py = x1**a * (1.0 - x1) ** a
    pyy = (
        a * (a - 1) * x1 ** (a - 2) * (1.0 - x1) ** a
        - 2.0 * a**2 * x1 ** (a - 1) * (1.0 - x1) ** (a - 1)
        + a * (a - 1) * x1**a * (1.0 - x1) ** (a - 2)
    )
    return -PEAK_SCALE * (pxx * py + px * pyy)


def steep_exact(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    return (0.1 * np.sin(2.0 * np.pi * x0) + np.tanh(10.0 * x0)) * np.sin(
        2.0 * np.pi * x1
    )


def steep_forcing(x):
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    sin_x = torch.sin(2.0 * torch.pi * x0)
    sin_y = torch.sin(2.0 * torch.pi * x1)
    tanh_x = torch.tanh(10.0 * x0)
    sech2_x = 1.0 / torch.cosh(10.0 * x0) ** 2
    amplitude = 0.1 * sin_x + tanh_x
    minus_amplitude_xx = 0.4 * torch.pi**2 * sin_x + 200.0 * sech2_x * tanh_x
    return (minus_amplitude_xx + 4.0 * torch.pi**2 * amplitude) * sin_y


CASES = {
    "peak": (peak_exact, peak_forcing),
    "steep": (steep_exact, steep_forcing),
}

DOMAINS = {
    "peak": (0.0, 1.0),
    "steep": (-1.0, 1.0),
}


def uniform_perimeter_points(num_points: int, lower: float, upper: float):
    s = np.arange(num_points, dtype=np.float64) * (4.0 / num_points)
    unit_points = np.empty((num_points, 2), dtype=np.float64)
    bottom = s < 1.0
    right = np.logical_and(s >= 1.0, s < 2.0)
    top = np.logical_and(s >= 2.0, s < 3.0)
    left = s >= 3.0
    unit_points[bottom] = np.column_stack((s[bottom], np.zeros(np.sum(bottom))))
    unit_points[right] = np.column_stack((np.ones(np.sum(right)), s[right] - 1.0))
    unit_points[top] = np.column_stack((3.0 - s[top], np.ones(np.sum(top))))
    unit_points[left] = np.column_stack((np.zeros(np.sum(left)), 4.0 - s[left]))
    return lower + (upper - lower) * unit_points


def build_problem(case: str, num_domain: int, num_boundary: int):
    exact, forcing = CASES[case]
    lower, upper = DOMAINS[case]
    geom = dde.geometry.Rectangle([lower, lower], [upper, upper])

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return -(dy_xx + dy_yy) - forcing(x)

    boundary_x = uniform_perimeter_points(num_boundary, lower, upper)
    bc = dde.icbc.PointSetBC(boundary_x, exact(boundary_x))
    data = dde.data.PDE(
        geom,
        pde,
        [bc],
        num_domain=num_domain,
        num_boundary=0,
        train_distribution="uniform",
        solution=exact,
        num_test=10000,
    )
    net = ModifiedMLP(input_dim=2, output_dim=1, depth=6, width=128)
    return dde.Model(data, net), exact


def test_grid(case: str, grid_size: int):
    lower, upper = DOMAINS[case]
    coordinates = np.linspace(lower, upper, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 2D Poisson experiments."
    )
    parser.add_argument("--case", choices=CASES.keys(), default="peak")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=2500)
    parser.add_argument("--num-boundary", type=positive_int, default=50)
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=1000)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--test-grid-size", type=positive_int, default=201)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model, exact = build_problem(args.case, args.num_domain, args.num_boundary)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(model, train_config, loss_weights=[1.0, 100.0])

    test_x = test_grid(args.case, args.test_grid_size)
    errors = evaluate_errors(model, test_x, exact)
    run_dir = experiment_dir(args.output_dir, "poisson_2d", args.case)
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    save_predictions(predictions_path, test_x, exact, model)
    save_metrics(
        metrics_path,
        errors,
        metadata={
            "equation": "2D Poisson",
            "paper_equation": "Eq. 13",
            "case": args.case,
            "domain": [list(DOMAINS[args.case]), list(DOMAINS[args.case])],
            "exact_solution": (
                "2^(4a) * x^a * (1-x)^a * y^a * (1-y)^a, a=10"
                if args.case == "peak"
                else "(0.1*sin(2*pi*x) + tanh(10*x)) * sin(2*pi*y)"
            ),
            "paper_consistency_note": (
                "Confirmed correction: the printed peak formula accidentally omits "
                "y^a; the complete formula includes y^a."
                if args.case == "peak"
                else "Confirmed setup: the steep case uses [-1,1]^2, as shown "
                "in Figure 3b."
            ),
            "setup_status": "confirmed",
            "num_domain": args.num_domain,
            "num_boundary": args.num_boundary,
            "domain_sampling": "uniform 50x50 grid" if args.num_domain == 2500 else "uniform grid",
            "boundary_sampling": "uniform around perimeter",
            "test_grid_size": args.test_grid_size,
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "boundary": 100.0},
            "boundary_loss": {
                "reduction": "mean",
                "formula": "100 * mean(boundary point squared errors)",
                "paper_reference": "Eq. 3 and Appendix D, Eq. 55",
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
                "lbfgs_source": "Paper Section 3.1.3 specifies 1000 epochs.",
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
