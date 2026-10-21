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


REYNOLDS_NUMBER = 100.0
DOMAIN = (0.0, 1.0)
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference_data"
    / "lid_driven_cavity_re100_reference.npz"
)


def mesh_points(grid_size: int):
    coordinates = np.linspace(*DOMAIN, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def boundary_data(points_per_edge: int):
    lower, upper = DOMAIN
    edge = np.linspace(
        lower, upper, points_per_edge + 2, endpoint=True, dtype=np.float64
    )[1:-1]
    bottom = np.column_stack((edge, np.full(points_per_edge, lower)))
    right = np.column_stack((np.full(points_per_edge, upper), edge))
    top = np.column_stack((edge, np.full(points_per_edge, upper)))
    left = np.column_stack((np.full(points_per_edge, lower), edge))
    points = np.vstack((bottom, right, top, left))
    u_target = np.zeros((len(points), 1), dtype=np.float64)
    u_target[2 * points_per_edge : 3 * points_per_edge] = 1.0
    v_target = np.zeros_like(u_target)
    return points, u_target, v_target


def residual_components(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    residual_u = u * u_x + v * u_y + p_x - (u_xx + u_yy) / REYNOLDS_NUMBER
    residual_v = u * v_x + v * v_y + p_y - (v_xx + v_yy) / REYNOLDS_NUMBER
    residual_continuity = u_x + v_y
    return residual_u, residual_v, residual_continuity


def pde(x, y):
    return list(residual_components(x, y))


def residual_vector(x, y):
    return dde.backend.concat(residual_components(x, y), axis=1)


def build_problem(residual_grid_size: int, boundary_points_per_edge: int):
    lower, upper = DOMAIN
    geom = dde.geometry.Rectangle([lower, lower], [upper, upper])
    residual_x = mesh_points(residual_grid_size)
    boundary_x, boundary_u, boundary_v = boundary_data(boundary_points_per_edge)
    bc_u = dde.icbc.PointSetBC(boundary_x, boundary_u, component=0)
    bc_v = dde.icbc.PointSetBC(boundary_x, boundary_v, component=1)
    data = dde.data.PDE(
        geom,
        pde,
        [bc_u, bc_v],
        num_domain=0,
        num_boundary=0,
        train_distribution="uniform",
        anchors=residual_x,
        num_test=10000,
    )
    net = ModifiedMLP(input_dim=2, output_dim=3, depth=6, width=128)
    return dde.Model(data, net)


def reference_metrics(model, reference_path: Path):
    with np.load(reference_path) as reference:
        required = {"x", "u", "v"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                f"Reference file must contain x, u, v; missing: {sorted(missing)}"
            )
        x = np.asarray(reference["x"], dtype=np.float64)
        u_true = np.asarray(reference["u"], dtype=np.float64).reshape(-1, 1)
        v_true = np.asarray(reference["v"], dtype=np.float64).reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Reference x must have shape (N, 2); got {x.shape}")
    if len(x) != len(u_true) or len(x) != len(v_true):
        raise ValueError("Reference x, u, and v must contain the same number of points")

    prediction = model.predict(x)
    velocity_true = np.sqrt(u_true**2 + v_true**2)
    velocity_pred = np.linalg.norm(prediction[:, :2], axis=1, keepdims=True)
    speed_error = velocity_true - velocity_pred
    velocity_components_true = np.hstack((u_true, v_true))
    velocity_components_pred = prediction[:, :2]

    # The project-wide L2 metric is computed on the scalar speed field. Keep the
    # matrix 2-norm used by the supplied NS.py as a separate audit metric.
    reference_grid_size = int(round(np.sqrt(len(x))))
    if reference_grid_size**2 != len(x):
        raise ValueError("Reference points must form a square grid for the paper metric")
    speed_matrix_error = speed_error.reshape(reference_grid_size, reference_grid_size)
    speed_matrix_true = velocity_true.reshape(reference_grid_size, reference_grid_size)
    return {
        "l2_relative_error": l2_relative_error(velocity_true, velocity_pred),
        "max_pointwise_error": max_pointwise_error(velocity_true, velocity_pred),
        "paper_source_speed_matrix_2norm_relative_error": float(
            np.linalg.norm(speed_matrix_error, 2)
            / np.linalg.norm(speed_matrix_true, 2)
        ),
        "speed_l2_relative_error": l2_relative_error(velocity_true, velocity_pred),
        "speed_max_pointwise_error": max_pointwise_error(
            velocity_true, velocity_pred
        ),
        "u_l2_relative_error": l2_relative_error(
            u_true, velocity_components_pred[:, 0:1]
        ),
        "v_l2_relative_error": l2_relative_error(
            v_true, velocity_components_pred[:, 1:2]
        ),
        "u_max_pointwise_error": max_pointwise_error(
            u_true, velocity_components_pred[:, 0:1]
        ),
        "v_max_pointwise_error": max_pointwise_error(
            v_true, velocity_components_pred[:, 1:2]
        ),
        "reference_status": "loaded",
        "reference_path": str(reference_path),
        "reference_point_count": int(len(x)),
    }


def evaluate_physics(model, points: np.ndarray, boundary_points_per_edge: int):
    residuals = model.predict(points, operator=residual_vector)
    boundary_x, boundary_u, boundary_v = boundary_data(boundary_points_per_edge)
    boundary_prediction = model.predict(boundary_x)
    boundary_error = np.hstack(
        (
            boundary_prediction[:, 0:1] - boundary_u,
            boundary_prediction[:, 1:2] - boundary_v,
        )
    )
    metrics = {}
    for index, name in enumerate(("momentum_u", "momentum_v", "continuity")):
        values = residuals[:, index]
        metrics[f"{name}_residual_rms"] = float(np.sqrt(np.mean(values**2)))
        metrics[f"{name}_residual_max"] = float(np.max(np.abs(values)))
    metrics["boundary_velocity_error_rms"] = float(
        np.sqrt(np.mean(boundary_error**2))
    )
    metrics["boundary_velocity_error_max"] = float(np.max(np.abs(boundary_error)))
    return metrics


def save_predictions(path: Path, points: np.ndarray, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(points)
    pressure = prediction[:, 2:3]
    np.savez(
        path,
        x=points,
        u=prediction[:, 0:1],
        v=prediction[:, 1:2],
        p=pressure,
        p_zero_mean=pressure - np.mean(pressure),
        speed=np.linalg.norm(prediction[:, :2], axis=1, keepdims=True),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs lid-driven cavity experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--residual-grid-size", type=positive_int, default=50)
    parser.add_argument("--boundary-points-per-edge", type=positive_int, default=128)
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=1000)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--test-grid-size", type=positive_int, default=101)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
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
    compile_and_train(
        model,
        train_config,
        loss_weights=[1.0, 1.0, 1.0, 100.0, 100.0],
    )

    test_x = mesh_points(args.test_grid_size)
    metrics = reference_metrics(model, args.reference_path)
    metrics.update(evaluate_physics(model, test_x, args.boundary_points_per_edge))
    run_dir = experiment_dir(args.output_dir, "lid_driven_cavity", "re_100")
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    save_predictions(predictions_path, test_x, model)
    save_metrics(
        metrics_path,
        metrics,
        metadata={
            "equation": "Steady 2D incompressible Navier-Stokes",
            "paper_equations": ["Eq. 17", "Eq. 18"],
            "case": "Lid-driven cavity",
            "reynolds_number": REYNOLDS_NUMBER,
            "domain": [list(DOMAIN), list(DOMAIN)],
            "outputs": ["u", "v", "p"],
            "pressure_gauge": (
                "Unconstrained as in the paper; p_zero_mean is also saved for visualization."
            ),
            "num_domain": args.residual_grid_size**2,
            "residual_grid": [args.residual_grid_size, args.residual_grid_size],
            "num_boundary": 4 * args.boundary_points_per_edge,
            "boundary_points_per_edge": args.boundary_points_per_edge,
            "domain_sampling": "uniform mesh including endpoints",
            "boundary_sampling": (
                f"uniform, {args.boundary_points_per_edge} interior points per edge; "
                "corners excluded because the lid velocity is discontinuous there"
            ),
            "test_grid_size": args.test_grid_size,
            "precision": "float64",
            "loss_weights": {
                "momentum_u": 1.0,
                "momentum_v": 1.0,
                "continuity": 1.0,
                "boundary_u": 100.0,
                "boundary_v": 100.0,
            },
            "boundary_loss": {
                "reduction": "mean",
                "paper_reference": "Eq. 3 and Appendix D, lid-driven cavity loss",
            },
            "backbone": {
                "type": "modified_mlp",
                "version": "modified_mlp_corrected",
                "formula_reference": "DASA-PINNs reference [21], Eq. 4.1-4.5",
                "hidden_layer_count": 6,
                "gate_update_count": 5,
                "parameter_count": sum(p.numel() for p in model.net.parameters()),
                "input_dim": 2,
                "output_dim": 3,
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
                "lbfgs_source": "Paper Section 3.1.5 specifies 1000 iterations.",
            },
            "paper_error_note": (
                "Table 1 reports classical PINN 7.53e-1. The paper uses the reference "
                "field from [19]. The supplied project reference is evaluated using the "
                "scalar speed field; the matrix 2-norm used by Lid-driven Cavity/NS.py "
                "is saved separately as paper_source_speed_matrix_2norm_relative_error."
            ),
            "reference_note": (
                "Converted from Lid-driven Cavity/reference_u.csv and reference_v.csv. "
                "The source CSVs store arrays with x as the first index and y as the "
                "second index; they were transposed to match the (y,x) mesh layout."
            ),
            "seed": args.seed,
        },
    )
    print(f"reference_status={metrics['reference_status']}")
    print(f"l2_relative_error={metrics['l2_relative_error']}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
