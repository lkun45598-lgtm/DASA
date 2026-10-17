import argparse
from pathlib import Path

import numpy as np
import torch

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
DIFFUSIVITY = 0.01
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)
FOURIER_ORDER = 8
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference_data"
    / "advection_diffusion_kappa_0.01_series_reference.npz"
)


def initial_condition(space):
    return -np.sin(np.pi * space)


def pde(points, prediction):
    u_x = dde.grad.jacobian(prediction, points, i=0, j=0)
    u_t = dde.grad.jacobian(prediction, points, i=0, j=1)
    u_xx = dde.grad.hessian(prediction, points, i=0, j=0)
    return u_t + VELOCITY * u_x - DIFFUSIVITY * u_xx


def fourier_features(points):
    space = points[:, 0:1]
    time = points[:, 1:2]
    features = [time]
    for wave_number in range(1, FOURIER_ORDER + 1):
        features.extend(
            (
                torch.cos(wave_number * space),
                torch.sin(wave_number * space),
            )
        )
    return torch.cat(features, dim=1)


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
    net = ModifiedMLP(
        input_dim=2,
        output_dim=1,
        depth=6,
        width=128,
        fourier_features=fourier_features,
    )
    return dde.Model(data, net)


def load_reference(path: Path):
    with np.load(path) as reference:
        required = {"x", "t", "u"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                f"Reference file must contain x, t, u; missing: {sorted(missing)}"
            )
        space = np.asarray(reference["x"], dtype=np.float64).reshape(-1)
        time = np.asarray(reference["t"], dtype=np.float64).reshape(-1)
        solution = np.asarray(reference["u"], dtype=np.float64)
        metadata = {
            key: reference[key].item()
            for key in (
                "velocity",
                "diffusivity",
                "num_terms",
                "decimal_precision",
                "source",
            )
            if key in reference.files
        }
    if solution.shape != (len(time), len(space)):
        raise ValueError(
            "Reference u must have shape (len(t),len(x)); "
            f"got {solution.shape}"
        )
    if not np.isclose(space[0], SPACE_DOMAIN[0]) or not np.isclose(
        space[-1], SPACE_DOMAIN[1]
    ):
        raise ValueError("Reference x range does not match [-1,1]")
    if not np.isclose(time[0], TIME_DOMAIN[0]) or not np.isclose(
        time[-1], TIME_DOMAIN[1]
    ):
        raise ValueError("Reference t range does not match [0,1]")
    if "diffusivity" in metadata and not np.isclose(
        metadata["diffusivity"], DIFFUSIVITY
    ):
        raise ValueError("Reference diffusivity does not match kappa=0.01")
    initial_error = np.max(np.abs(solution[0] - initial_condition(space)))
    boundary_error = max(
        np.max(np.abs(solution[:, 0])), np.max(np.abs(solution[:, -1]))
    )
    if initial_error > 1e-12 or boundary_error > 1e-12:
        raise ValueError(
            "Reference violates Eq. 24 initial/boundary data: "
            f"initial={initial_error:.3e}, boundary={boundary_error:.3e}"
        )
    grid_x, grid_t = np.meshgrid(space, time, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    return points, solution.reshape(-1, 1), space, time, metadata


def physics_metrics(
    model, grid_size: int, boundary_points_per_side: int, num_initial: int
):
    space = np.linspace(*SPACE_DOMAIN, grid_size, dtype=np.float64)
    time = np.linspace(*TIME_DOMAIN, grid_size, dtype=np.float64)
    grid_x, grid_t = np.meshgrid(space, time, indexing="xy")
    residual_x = np.column_stack((grid_x.ravel(), grid_t.ravel()))
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
    )
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
            "Classical PINN baseline for the DASA-PINNs 1D "
            "advection-diffusion experiment."
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
    parser.add_argument("--physics-grid-size", type=positive_int, default=101)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    if not args.reference_path.exists():
        raise FileNotFoundError(
            f"Advection-diffusion reference data not found: {args.reference_path}"
        )

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
    compile_and_train(
        model, train_config, loss_weights=[1.0, 100.0, 100.0, 100.0]
    )

    reference_x, y_true, reference_space, reference_time, reference_metadata = (
        load_reference(args.reference_path)
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

    run_dir = experiment_dir(
        args.output_dir,
        "advection_diffusion_1d",
        "velocity_1_kappa_0.01_fourier_order_8",
    )
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    np.savez(
        predictions_path,
        x=reference_x,
        y_true=y_true,
        y_pred=y_pred,
        reference_x=reference_space,
        reference_t=reference_time,
    )
    save_metrics(
        metrics_path,
        metrics,
        metadata={
            "equation": "1D advection-diffusion",
            "paper_equations": ["Eq. 24a", "Eq. 24b", "Eq. 24c"],
            "velocity": VELOCITY,
            "diffusivity": DIFFUSIVITY,
            "diffusivity_source": "Confirmed by the paper author on 2026-08-10.",
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
            "loss_weights": {
                "residual": 1.0,
                "left_boundary": 100.0,
                "right_boundary": 100.0,
                "initial": 100.0,
            },
            "loss_reference": (
                "Paper Appendix D Eq. 58a: wr=1, wb=100, wi=100, with "
                "separate means for the two boundaries."
            ),
            "backbone": {
                "type": "modified_mlp",
                "version": "modified_mlp_corrected",
                "formula_reference": "DASA-PINNs reference [21], Eq. 4.1-4.5",
                "hidden_layer_count": 6,
                "gate_update_count": 5,
                "parameter_count": sum(p.numel() for p in model.net.parameters()),
                "input_dim": 2,
                "feature_dim": 1 + 2 * FOURIER_ORDER,
                "output_dim": 1,
                "depth": 6,
                "width": 128,
                "activation": "tanh",
                "initializer": "glorot normal",
                "fourier_features": {
                    "mapping": "[t, cos(k*x), sin(k*x)] for k=1,...,8",
                    "order": FOURIER_ORDER,
                    "source": (
                        "Confirmed to match the current Diffusion-Reaction mapping; "
                        "the paper only states that other training details are the same."
                    ),
                },
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "budget_source": (
                    "Classical PINN mapping of paper NS3=10000 and NS4=10000 "
                    "to one standard Adam phase; DASA pretraining stages are excluded."
                ),
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
                "lbfgs_source": "Paper Section 3.2.4 specifies 1000 epochs.",
            },
            "reference": {
                "path": str(args.reference_path),
                "grid_shape": [len(reference_time), len(reference_space)],
                "metadata": reference_metadata,
                "source_note": (
                    "1000-term Dirichlet eigenfunction series, following the paper's "
                    "analytical-series description and reference [36]. It is evaluated "
                    "at high precision before conversion to float64."
                ),
            },
            "test_grid_note": (
                "The current reference uses a 256x101 uniform grid; the paper does "
                "not state the Table 1 evaluation grid."
            ),
            "paper_classical_pinn_l2": 5.73e-2,
            "paper_dasa_pinn_l2": 1.18e-3,
            "seed": args.seed,
        },
    )
    print("equation=Eq. 24 with velocity=1 and kappa=0.01")
    print("loss=paper Appendix D Eq. 58a with wr=1, wb=100, wi=100")
    print(f"l2_relative_error={metrics['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
