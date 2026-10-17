import argparse
from pathlib import Path

import numpy as np
import torch

from plain_pinn_baselines.common import (
    PlainFNN,
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


def boundary_layer_exact(x):
    return 0.1 * np.sin(5 * np.pi * x[:, 0:1]) + np.exp((0.01 - (x[:, 0:1] + 1)) / 0.01)


def boundary_layer_forcing(x):
    x0 = x[:, 0:1]
    return (
        2.5 * torch.pi**2 * torch.sin(5 * torch.pi * x0)
        - 10000.0 * torch.exp((0.01 - (x0 + 1.0)) / 0.01)
    )


def steep_exact(x):
    return 0.1 * np.sin(8 * np.pi * x[:, 0:1]) + np.tanh(80 * x[:, 0:1])


def steep_forcing(x):
    x0 = x[:, 0:1]
    z = 80.0 * x0
    sech2 = 1.0 / torch.cosh(z) ** 2
    return 6.4 * torch.pi**2 * torch.sin(8 * torch.pi * x0) + 12800.0 * sech2 * torch.tanh(z)


CASES = {
    "boundary-layer": (boundary_layer_exact, boundary_layer_forcing),
    "steep": (steep_exact, steep_forcing),
}

def sum_squared_error(y_true, y_pred):
    """Sum endpoint errors exactly as written in Eq. 53 of the paper."""
    return dde.backend.reduce_sum(dde.backend.square(y_true - y_pred))


def build_problem(case: str, num_domain: int):
    exact, forcing = CASES[case]
    geom = dde.geometry.Interval(-1.0, 1.0)

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return -dy_xx - forcing(x)

    def boundary(_, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, exact, boundary)
    data = dde.data.PDE(
        geom,
        pde,
        [bc],
        num_domain=num_domain,
        num_boundary=2,
        train_distribution="LHS",
        solution=exact,
        num_test=1000,
    )
    net = PlainFNN(input_dim=1, output_dim=1, depth=6, width=128)
    model = dde.Model(data, net)
    return model, exact


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 1D Poisson experiments."
    )
    parser.add_argument("--case", choices=CASES.keys(), default="boundary-layer")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=500)
    parser.add_argument("--adam-iterations", type=positive_int, default=40000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=0)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plain_fnn"))
    args = parser.parse_args()

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model, exact = build_problem(args.case, args.num_domain)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(
        model,
        train_config,
        loss_weights=[1.0, 100.0],
        loss=["MSE", sum_squared_error],
    )

    test_x = np.linspace(-1.0, 1.0, 2000, dtype=np.float64)[:, None]
    errors = evaluate_errors(model, test_x, exact)
    run_dir = experiment_dir(args.output_dir, "poisson_1d", args.case)
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    save_predictions(predictions_path, test_x, exact, model)
    save_metrics(
        metrics_path,
        errors,
        metadata={
            "equation": "1D Poisson",
            "case": args.case,
            "num_domain": args.num_domain,
            "num_boundary": 2,
            "train_distribution": "LHS",
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "boundary": 100.0},
            "boundary_loss": {
                "reduction": "sum",
                "formula": "100 * (e_left^2 + e_right^2)",
                "paper_reference": "Appendix D, Eq. 53",
            },
            "backbone": {
                "type": "plain_fnn",
                "version": "plain_fnn_original_pinn",
                "gated_modified_mlp": False,
                "source": "Original PINNs ordinary fully connected network family",
                "hidden_layer_count": 6,
                "parameter_count": sum(p.numel() for p in model.net.parameters()),
                "depth": 6,
                "width": 128,
                "activation": "tanh",
                "initializer": "glorot normal",
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
            },
            "seed": args.seed,
        },
    )
    print("boundary_loss=paper Eq. 53 endpoint sum")
    print(f"l2_relative_error={errors['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={errors['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
