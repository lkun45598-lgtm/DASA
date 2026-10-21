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


VISCOSITY = 0.01 / np.pi
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "reference_data" / "Burgers.npz"
)


def pde(x, y):
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    return u_t + y * u_x - VISCOSITY * u_xx


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
    boundary_x = np.vstack(
        (
            np.column_stack(
                (np.full(boundary_points_per_side, SPACE_DOMAIN[0]), boundary_time)
            ),
            np.column_stack(
                (np.full(boundary_points_per_side, SPACE_DOMAIN[1]), boundary_time)
            ),
        )
    )
    initial_space = lhs_points(num_initial, (SPACE_DOMAIN,))[:, 0:1]
    initial_x = np.column_stack((initial_space[:, 0], np.zeros(num_initial)))
    initial_u = -np.sin(np.pi * initial_space)
    return residual_x, boundary_x, initial_x, initial_u


def build_problem(
    num_domain: int, boundary_points_per_side: int, num_initial: int
):
    geometry = dde.geometry.Interval(*SPACE_DOMAIN)
    time_domain = dde.geometry.TimeDomain(*TIME_DOMAIN)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    residual_x, boundary_x, initial_x, initial_u = training_points(
        num_domain, boundary_points_per_side, num_initial
    )
    num_boundary = 2 * boundary_points_per_side
    bc = dde.icbc.PointSetBC(boundary_x, np.zeros((num_boundary, 1)))
    ic = dde.icbc.PointSetBC(initial_x, initial_u)
    data = dde.data.PDE(
        geometry_time,
        pde,
        [bc, ic],
        num_domain=0,
        num_boundary=0,
        anchors=residual_x,
        num_test=None,
    )
    net = ModifiedMLP(input_dim=2, output_dim=1, depth=6, width=128)
    return dde.Model(data, net)


def load_reference(path: Path):
    with np.load(path) as reference:
        required = {"t", "x", "usol"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                f"Reference file must contain t, x, usol; missing: {sorted(missing)}"
            )
        time = np.asarray(reference["t"], dtype=np.float64).reshape(-1)
        space = np.asarray(reference["x"], dtype=np.float64).reshape(-1)
        solution = np.asarray(reference["usol"], dtype=np.float64)
    if solution.shape != (len(space), len(time)):
        raise ValueError(
            "Reference usol must have shape (len(x), len(t)); "
            f"got {solution.shape}"
        )
    grid_t, grid_x = np.meshgrid(time, space, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    return points, solution.reshape(-1, 1), time, space


def physics_metrics(
    model, grid_size: int, boundary_points_per_side: int, num_initial: int
):
    space = np.linspace(*SPACE_DOMAIN, grid_size, dtype=np.float64)
    time = np.linspace(*TIME_DOMAIN, grid_size, dtype=np.float64)
    grid_t, grid_x = np.meshgrid(time, space, indexing="xy")
    residual_x = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    residual = model.predict(residual_x, operator=pde).reshape(-1)
    boundary_time = np.linspace(
        *TIME_DOMAIN, boundary_points_per_side, dtype=np.float64
    )
    boundary_x = np.vstack(
        (
            np.column_stack((np.full(len(boundary_time), -1.0), boundary_time)),
            np.column_stack((np.full(len(boundary_time), 1.0), boundary_time)),
        )
    )
    boundary_error = model.predict(boundary_x).reshape(-1)
    initial_space = np.linspace(*SPACE_DOMAIN, num_initial, dtype=np.float64)
    initial_x = np.column_stack((initial_space, np.zeros(num_initial)))
    initial_error = model.predict(initial_x).reshape(-1) + np.sin(
        np.pi * initial_space
    )
    return {
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "pde_residual_max": float(np.max(np.abs(residual))),
        "boundary_error_rms": float(np.sqrt(np.mean(boundary_error**2))),
        "boundary_error_max": float(np.max(np.abs(boundary_error))),
        "initial_error_rms": float(np.sqrt(np.mean(initial_error**2))),
        "initial_error_max": float(np.max(np.abs(initial_error))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 1D Burgers experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=25600)
    parser.add_argument(
        "--boundary-points-per-side", type=positive_int, default=256
    )
    parser.add_argument("--num-initial", type=positive_int, default=512)
    parser.add_argument("--adam-iterations", type=positive_int, default=60000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=0)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--physics-grid-size", type=positive_int, default=101)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    if not args.reference_path.exists():
        raise FileNotFoundError(f"Burgers reference data not found: {args.reference_path}")

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
    compile_and_train(model, train_config, loss_weights=[1.0, 1.0, 1.0])

    reference_x, y_true, reference_t, reference_space = load_reference(
        args.reference_path
    )
    y_pred = model.predict(reference_x)
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
    run_dir = experiment_dir(args.output_dir, "burgers_1d", "nu_0.01_over_pi")
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        predictions_path,
        x=reference_x,
        y_true=y_true,
        y_pred=y_pred,
        reference_t=reference_t,
        reference_x=reference_space,
    )
    save_metrics(
        metrics_path,
        metrics,
        metadata={
            "equation": "1D viscous Burgers",
            "paper_equation": "Eq. 19",
            "viscosity": "0.01/pi",
            "space_domain": list(SPACE_DOMAIN),
            "time_domain": list(TIME_DOMAIN),
            "initial_condition": "u(0,x)=-sin(pi*x)",
            "boundary_condition": "u(t,-1)=u(t,1)=0",
            "num_domain": args.num_domain,
            "num_boundary_per_side": args.boundary_points_per_side,
            "num_boundary_constraints": 2 * args.boundary_points_per_side,
            "num_initial": args.num_initial,
            "sampling": "fixed LHS sets for residual, boundary, and initial points",
            "boundary_sampling": (
                "symmetric paired sampling: both sides use the same LHS time coordinates"
            ),
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "boundary": 1.0, "initial": 1.0},
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
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
            },
            "paper_setup_note": (
                "The current unified time-PDE setting uses Nr=25600, Nbc=256 per "
                "side, and Nic=512; 60000 Adam steps are inferred from Figure 5a."
            ),
            "reference": {
                "path": str(args.reference_path),
                "grid_shape": [len(reference_space), len(reference_t)],
                "time_range": [float(reference_t[0]), float(reference_t[-1])],
                "source_note": (
                    "Standard numerical Burgers reference bundled with vendored DeepXDE; "
                    "equivalence to the paper's Figure 5a reference is not explicitly stated."
                ),
            },
            "paper_classical_pinn_l2": 2.38e-3,
            "seed": args.seed,
        },
    )
    print(f"l2_relative_error={metrics['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
