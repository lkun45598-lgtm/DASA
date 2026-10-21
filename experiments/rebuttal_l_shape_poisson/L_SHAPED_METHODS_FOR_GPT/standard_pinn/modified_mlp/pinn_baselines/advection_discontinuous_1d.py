import argparse
from pathlib import Path

import numpy as np

from pinn_baselines.common_modified_mlp_corrected import (
    ModifiedMLP,
    TrainConfig,
    compile_and_train,
    dde,
    experiment_dir,
    l2_relative_error,
    max_pointwise_error,
    positive_int,
    save_metrics,
    set_seed,
)


VELOCITY = 1.0
DIFFUSIVITY = 0.0
SPACE_DOMAIN = (-1.5, 1.5)
TIME_DOMAIN = (0.0, 1.0)
PULSE_INTERVAL = (-0.2, 0.2)


def initial_condition(space):
    return np.logical_and(
        space >= PULSE_INTERVAL[0], space <= PULSE_INTERVAL[1]
    ).astype(np.float64)


def exact_solution(points):
    space = points[:, 0:1]
    time = points[:, 1:2]
    return initial_condition(space - VELOCITY * time)


def pde(points, prediction):
    u_x = dde.grad.jacobian(prediction, points, i=0, j=0)
    u_t = dde.grad.jacobian(prediction, points, i=0, j=1)
    return u_t + VELOCITY * u_x


def lhs_points(num_points: int, bounds):
    dimensions = len(bounds)
    points = np.empty((num_points, dimensions), dtype=np.float64)
    for dimension, (lower, upper) in enumerate(bounds):
        coordinate = (np.arange(num_points) + np.random.random(num_points)) / num_points
        np.random.shuffle(coordinate)
        points[:, dimension] = lower + (upper - lower) * coordinate
    return points


def training_points(
    num_domain: int, boundary_points_per_side: int, num_initial: int
):
    residual_x = lhs_points(num_domain, (SPACE_DOMAIN, TIME_DOMAIN))
    boundary_time = lhs_points(boundary_points_per_side, (TIME_DOMAIN,))[:, 0]
    left_x = np.column_stack(
        (np.full(boundary_points_per_side, SPACE_DOMAIN[0]), boundary_time)
    )
    right_x = np.column_stack(
        (np.full(boundary_points_per_side, SPACE_DOMAIN[1]), boundary_time)
    )
    initial_space = lhs_points(num_initial, (SPACE_DOMAIN,))[:, 0:1]
    initial_x = np.column_stack((initial_space[:, 0], np.zeros(num_initial)))
    initial_u = initial_condition(initial_space)
    return residual_x, left_x, right_x, initial_x, initial_u


def build_problem(
    num_domain: int, boundary_points_per_side: int, num_initial: int
):
    geometry = dde.geometry.Interval(*SPACE_DOMAIN)
    time_domain = dde.geometry.TimeDomain(*TIME_DOMAIN)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    residual_x, left_x, right_x, initial_x, initial_u = training_points(
        num_domain, boundary_points_per_side, num_initial
    )
    zeros = np.zeros((boundary_points_per_side, 1), dtype=np.float64)
    left_bc = dde.icbc.PointSetBC(left_x, zeros)
    right_bc = dde.icbc.PointSetBC(right_x, zeros.copy())
    initial_bc = dde.icbc.PointSetBC(initial_x, initial_u)
    data = dde.data.PDE(
        geometry_time,
        pde,
        [left_bc, right_bc, initial_bc],
        num_domain=0,
        num_boundary=0,
        anchors=residual_x,
        num_test=None,
    )
    net = ModifiedMLP(input_dim=2, output_dim=1, depth=6, width=128)
    return dde.Model(data, net)


def evaluation_points(space_grid_size: int, time_grid_size: int):
    space = np.linspace(*SPACE_DOMAIN, space_grid_size, dtype=np.float64)
    time = np.linspace(*TIME_DOMAIN, time_grid_size, dtype=np.float64)
    grid_t, grid_x = np.meshgrid(time, space, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    return points, space, time


def physics_metrics(
    model, grid_size: int, boundary_points_per_side: int, num_initial: int
):
    residual_x, _, _ = evaluation_points(grid_size, grid_size)
    residual = model.predict(residual_x, operator=pde).reshape(-1)

    boundary_time = np.linspace(
        *TIME_DOMAIN, boundary_points_per_side, dtype=np.float64
    )
    left_x = np.column_stack(
        (np.full(boundary_points_per_side, SPACE_DOMAIN[0]), boundary_time)
    )
    right_x = np.column_stack(
        (np.full(boundary_points_per_side, SPACE_DOMAIN[1]), boundary_time)
    )
    left_error = model.predict(left_x).reshape(-1)
    right_error = model.predict(right_x).reshape(-1)
    boundary_error = np.concatenate((left_error, right_error))

    initial_space = np.linspace(*SPACE_DOMAIN, num_initial, dtype=np.float64)
    initial_x = np.column_stack((initial_space, np.zeros(num_initial)))
    initial_error = model.predict(initial_x).reshape(-1) - initial_condition(
        initial_space
    ).reshape(-1)
    return {
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "pde_residual_max": float(np.max(np.abs(residual))),
        "boundary_error_rms": float(np.sqrt(np.mean(boundary_error**2))),
        "boundary_error_max": float(np.max(np.abs(boundary_error))),
        "left_boundary_error_rms": float(np.sqrt(np.mean(left_error**2))),
        "right_boundary_error_rms": float(np.sqrt(np.mean(right_error**2))),
        "initial_error_rms": float(np.sqrt(np.mean(initial_error**2))),
        "initial_error_max": float(np.max(np.abs(initial_error))),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classical PINN baseline for the DASA-PINNs 1D advection experiment "
            "with a discontinuous initial condition."
        )
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=25600)
    parser.add_argument(
        "--boundary-points-per-side", type=positive_int, default=256
    )
    parser.add_argument("--num-initial", type=positive_int, default=512)
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=1000)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--test-space-grid-size", type=positive_int, default=256)
    parser.add_argument("--test-time-grid-size", type=positive_int, default=101)
    parser.add_argument("--physics-grid-size", type=positive_int, default=101)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(
        args.num_domain, args.boundary_points_per_side, args.num_initial
    )
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    loss_weights = [1.0, 100.0, 100.0, 100.0]
    compile_and_train(model, train_config, loss_weights=loss_weights)

    test_x, reference_space, reference_time = evaluation_points(
        args.test_space_grid_size, args.test_time_grid_size
    )
    y_true = exact_solution(test_x)
    y_pred = model.predict(test_x)
    metrics = {
        "l2_relative_error": l2_relative_error(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
    }
    metrics.update(
        physics_metrics(
            model,
            args.physics_grid_size,
            args.boundary_points_per_side,
            args.num_initial,
        )
    )

    run_dir = experiment_dir(
        args.output_dir, "advection_discontinuous_1d", "velocity_1_kappa_0"
    )
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    np.savez(
        predictions_path,
        x=test_x,
        y_true=y_true,
        y_pred=y_pred,
        reference_x=reference_space,
        reference_t=reference_time,
    )
    save_metrics(
        metrics_path,
        metrics,
        metadata={
            "equation": "1D advection with discontinuous initial condition",
            "paper_equations": ["Eq. 26a", "Eq. 27", "Eq. 28"],
            "velocity": VELOCITY,
            "diffusivity": DIFFUSIVITY,
            "space_domain": list(SPACE_DOMAIN),
            "time_domain": list(TIME_DOMAIN),
            "initial_condition": "u0(x)=1 for x in [-0.2,0.2], otherwise 0",
            "boundary_condition": "u(t,-1.5)=u(t,1.5)=0",
            "exact_solution": "u_exact(t,x)=u0(x-t)",
            "num_domain": args.num_domain,
            "num_boundary_per_side": args.boundary_points_per_side,
            "num_boundary_constraints": 2 * args.boundary_points_per_side,
            "num_initial": args.num_initial,
            "point_count_source": (
                "inferred from neighboring time-dependent experiments; Section "
                "3.3.2 does not state Nr, Nbc, or Nic"
            ),
            "sampling": "fixed LHS sets for residual, boundary, and initial points",
            "precision": "float64",
            "loss_weights": {
                "residual": 1.0,
                "left_boundary": 100.0,
                "right_boundary": 100.0,
                "initial": 100.0,
            },
            "loss_reference": (
                "Paper Appendix D Eq. 60: separate mean losses for both spatial "
                "boundaries, wr=1, wb=100, wi=100."
            ),
            "backbone": {
                "type": "modified_mlp",
                "version": "modified_mlp_corrected",
                "formula_reference": "DASA-PINNs reference [21], Eq. 4.1-4.5",
                "hidden_layer_count": 6,
                "gate_update_count": 5,
                "parameter_count": sum(p.numel() for p in model.net.parameters()),
                "input_dim": 2,
                "output_dim": 1,
                "depth": 6,
                "width": 128,
                "activation": "tanh",
                "initializer": "glorot normal",
                "fourier_features": False,
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "budget_source": (
                    "Classical PINN mapping of paper NS3=10000 and NS4=10000 "
                    "to one standard Adam phase; DASA-specific NS1 and NS2 are excluded."
                ),
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
                "lbfgs_source": "Paper Section 3.3.2 specifies 1000 epochs.",
            },
            "boundary_count_note": (
                "Nbc is interpreted per side because Appendix Eq. 60 contains two "
                "independent sums, each normalized by Nbc."
            ),
            "test_grid": [args.test_space_grid_size, args.test_time_grid_size],
            "test_grid_note": "The paper does not state the error-evaluation grid.",
            "paper_classical_pinn_l2": 8.24e-1,
            "seed": args.seed,
        },
    )
    print("loss=paper Appendix D Eq. 60 with wr=1, wb=100, wi=100")
    print(f"l2_relative_error={metrics['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
