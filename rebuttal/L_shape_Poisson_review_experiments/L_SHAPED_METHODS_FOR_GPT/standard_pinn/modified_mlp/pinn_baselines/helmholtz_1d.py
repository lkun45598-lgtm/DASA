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


FREQUENCY = 20.0 * np.pi
Q_VALUE = -(FREQUENCY**2)
OBSERVATION_X = 1.0 / 40.0
OBSERVATION_VALUE = 1.0
OBSERVATION_WEIGHT = 100.0


def exact_solution(x):
    return np.sin(FREQUENCY * x[:, 0:1])


def coefficient(x):
    return Q_VALUE * torch.ones_like(x[:, 0:1])


def forcing(x):
    return torch.zeros_like(x[:, 0:1])


def build_problem(num_domain: int):
    geom = dde.geometry.Interval(0.0, 1.0)

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return -dy_xx + coefficient(x) * y - forcing(x)

    def boundary(_, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, exact_solution, boundary)
    observation = dde.icbc.PointSetBC(
        np.asarray([[OBSERVATION_X]], dtype=np.float64),
        np.asarray([[OBSERVATION_VALUE]], dtype=np.float64),
    )
    data = dde.data.PDE(
        geom,
        pde,
        [bc, observation],
        num_domain=num_domain,
        num_boundary=2,
        train_distribution="LHS",
        solution=exact_solution,
        num_test=1000,
    )
    net = ModifiedMLP(input_dim=1, output_dim=1, depth=6, width=128)
    return dde.Model(data, net)


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 1D Helmholtz experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=500)
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=0)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(args.num_domain)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(
        model,
        train_config,
        loss_weights=[1.0, 100.0, OBSERVATION_WEIGHT],
    )

    test_x = np.linspace(0.0, 1.0, 2000, dtype=np.float64)[:, None]
    errors = evaluate_errors(model, test_x, exact_solution)
    observation_prediction = float(
        model.predict(np.asarray([[OBSERVATION_X]], dtype=np.float64))[0, 0]
    )
    errors["interior_observation_absolute_error"] = abs(
        observation_prediction - OBSERVATION_VALUE
    )
    run_dir = experiment_dir(
        args.output_dir,
        "helmholtz_1d",
        "sin20pi_q_minus_400pi2_f_0_observation_x_0.025_u_1",
    )
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    save_predictions(predictions_path, test_x, exact_solution, model)
    save_metrics(
        metrics_path,
        errors,
        metadata={
            "equation": "1D Helmholtz",
            "paper_equation": "Eq. 12",
            "implemented_equation": "-u_xx + q*u = f",
            "domain": [0.0, 1.0],
            "exact_solution": "sin(20*pi*x)",
            "forcing": "f(x)=0",
            "coefficient": {
                "q_x": Q_VALUE,
                "symbolic": "-(20*pi)^2 = -400*pi^2",
                "source": "confirmed experiment setting",
            },
            "sign_convention_note": (
                "The implementation follows PDF Eq. 12 exactly. With "
                "q=-(20*pi)^2 and u=sin(20*pi*x), f=0."
            ),
            "interior_observation": {
                "x": OBSERVATION_X,
                "value": OBSERVATION_VALUE,
                "formula": "u(1/40)=1",
                "weight": OBSERVATION_WEIGHT,
                "purpose": "fix the eigenfunction amplitude A=1",
                "source": (
                    "Added after the non-uniqueness diagnosis; this condition is not "
                    "printed in the DASA-PINNs paper."
                ),
            },
            "well_posedness_note": (
                "With homogeneous Dirichlet boundaries and q=-(20*pi)^2, the problem "
                "without an observation is non-unique: zero and A*sin(20*pi*x) are all "
                "valid solutions. The added condition u(1/40)=1 fixes A=1 and excludes "
                "the zero solution."
            ),
            "num_domain": args.num_domain,
            "num_boundary": 2,
            "num_interior_observations": 1,
            "train_distribution": "LHS",
            "precision": "float64",
            "loss_weights": {
                "residual": 1.0,
                "boundary": 100.0,
                "interior_observation": OBSERVATION_WEIGHT,
            },
            "boundary_loss": {
                "reduction": "mean",
                "formula": "100 * mean([e_left^2, e_right^2])",
                "paper_reference": "Eq. 3 and Appendix D, Eq. 54",
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
            },
            "seed": args.seed,
        },
    )
    print("interior_observation=u(1/40)=1 fixes the eigenfunction amplitude")
    print("boundary_loss=paper Eq. 3 mean with Appendix D weight wb=100")
    print(
        "interior_observation_absolute_error="
        f"{errors['interior_observation_absolute_error']:.6e}"
    )
    print(f"l2_relative_error={errors['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={errors['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
