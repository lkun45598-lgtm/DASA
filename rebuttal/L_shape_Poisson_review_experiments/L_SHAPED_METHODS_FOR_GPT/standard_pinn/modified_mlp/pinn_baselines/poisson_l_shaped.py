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


DOMAIN = (-1.0, 1.0)
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference_data"
    / "l_shaped_poisson_fdm_reference.npz"
)


def pde(points, prediction):
    u_xx = dde.grad.hessian(prediction, points, i=0, j=0)
    u_yy = dde.grad.hessian(prediction, points, i=1, j=1)
    return -(u_xx + u_yy) - 1.0


def lhs_points(num_points: int, bounds):
    dimensions = len(bounds)
    points = np.empty((num_points, dimensions), dtype=np.float64)
    for dimension, (lower, upper) in enumerate(bounds):
        coordinate = (np.arange(num_points) + np.random.random(num_points)) / num_points
        np.random.shuffle(coordinate)
        points[:, dimension] = lower + (upper - lower) * coordinate
    return points


def l_shaped_lhs_points(num_points: int):
    accepted = []
    num_accepted = 0
    while num_accepted < num_points:
        remaining = num_points - num_accepted
        candidates = lhs_points(max(32, int(np.ceil(remaining / 0.7))), (DOMAIN, DOMAIN))
        inside = np.logical_or(candidates[:, 0] <= 0.0, candidates[:, 1] <= 0.0)
        valid = candidates[inside]
        accepted.append(valid[:remaining])
        num_accepted += min(len(valid), remaining)
    return np.vstack(accepted)


def boundary_points(points_per_segment: int):
    vertices = np.asarray(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    segments = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        parameter = (
            np.arange(points_per_segment, dtype=np.float64)
            + np.random.random(points_per_segment)
        ) / points_per_segment
        np.random.shuffle(parameter)
        parameter = parameter[:, None]
        segments.append(start + parameter * (end - start))
    return np.vstack(segments)


def build_problem(num_domain: int, boundary_points_per_segment: int):
    outer = dde.geometry.Rectangle([DOMAIN[0], DOMAIN[0]], [DOMAIN[1], DOMAIN[1]])
    cutout = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    geometry = dde.geometry.CSGDifference(outer, cutout)
    residual_x = l_shaped_lhs_points(num_domain)
    boundary_x = boundary_points(boundary_points_per_segment)
    boundary_condition = dde.icbc.PointSetBC(
        boundary_x, np.zeros((len(boundary_x), 1), dtype=np.float64)
    )
    data = dde.data.PDE(
        geometry,
        pde,
        [boundary_condition],
        num_domain=0,
        num_boundary=0,
        anchors=residual_x,
        num_test=None,
    )
    net = ModifiedMLP(input_dim=2, output_dim=1, depth=6, width=128)
    return dde.Model(data, net)


def load_reference(path: Path):
    with np.load(path) as reference:
        required = {"x", "u"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                f"Reference file must contain x and u; missing: {sorted(missing)}"
            )
        points = np.asarray(reference["x"], dtype=np.float64)
        solution = np.asarray(reference["u"], dtype=np.float64).reshape(-1, 1)
        metadata = {
            key: reference[key].item()
            for key in (
                "solver_points_per_unit",
                "solver_grid_size",
                "solver_spacing",
                "evaluation_stride",
                "evaluation_spacing",
                "cg_iterations",
                "cg_relative_residual",
                "source",
            )
            if key in reference.files
        }
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Reference x must have shape (N,2); got {points.shape}")
    if len(points) != len(solution):
        raise ValueError("Reference x and u must contain the same number of points")
    inside = np.logical_or(points[:, 0] <= 0.0, points[:, 1] <= 0.0)
    if not np.all(inside):
        raise ValueError("Reference contains points outside the L-shaped domain")
    return points, solution, metadata


def physics_metrics(model, num_points: int, boundary_points_per_segment: int):
    residual_x = l_shaped_lhs_points(num_points)
    residual = model.predict(residual_x, operator=pde).reshape(-1)
    boundary_x = boundary_points(boundary_points_per_segment)
    boundary_error = model.predict(boundary_x).reshape(-1)
    return {
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "pde_residual_max": float(np.max(np.abs(residual))),
        "boundary_error_rms": float(np.sqrt(np.mean(boundary_error**2))),
        "boundary_error_max": float(np.max(np.abs(boundary_error))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs L-shaped Poisson experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=10000)
    parser.add_argument(
        "--boundary-points-per-segment", type=positive_int, default=128
    )
    parser.add_argument("--adam-iterations", type=positive_int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=1000)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--physics-num-points", type=positive_int, default=10000)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    if not args.reference_path.exists():
        raise FileNotFoundError(
            f"L-shaped Poisson reference data not found: {args.reference_path}"
        )

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(args.num_domain, args.boundary_points_per_segment)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(model, train_config, loss_weights=[1.0, 100.0])

    reference_x, y_true, reference_metadata = load_reference(args.reference_path)
    y_pred = model.predict(reference_x)
    metrics = {
        "l2_relative_error": l2_relative_error(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
    }
    metrics.update(
        physics_metrics(
            model, args.physics_num_points, args.boundary_points_per_segment
        )
    )

    run_dir = experiment_dir(
        args.output_dir, "poisson_l_shaped", "fdm_reference_h_1_over_256"
    )
    predictions_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "metrics.json"
    np.savez(
        predictions_path,
        x=reference_x,
        y_true=y_true,
        y_pred=y_pred,
    )
    save_metrics(
        metrics_path,
        metrics,
        metadata={
            "equation": "2D Poisson on an L-shaped domain",
            "paper_equations": ["Eq. 25a", "Eq. 25b"],
            "domain": "[-1,1]^2 \\ [0,1]^2",
            "forcing": 1.0,
            "boundary_condition": "u=0 on the complete L-shaped boundary",
            "num_domain": args.num_domain,
            "num_boundary": 8 * args.boundary_points_per_segment,
            "boundary_segments": 8,
            "boundary_points_per_segment": args.boundary_points_per_segment,
            "domain_sampling": "fixed rejection-LHS sampling on the L-shaped domain",
            "boundary_sampling": (
                "independent one-dimensional LHS on each of 8 unit boundary segments"
            ),
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "boundary": 100.0},
            "loss_reference": "Paper Appendix D Eq. 59, wr=1 and wb=100.",
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
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "budget_source": (
                    "Baseline convention: map paper NS3=10000 and NS4=10000 to "
                    "one standard Adam phase. DASA-specific NS1 and NS2 are excluded; "
                    "the paper does not separately state the classical PINN step count."
                ),
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
                "lbfgs_source": "Paper Section 3.3.1 specifies 1000 epochs.",
            },
            "reference": {
                "path": str(args.reference_path),
                "point_count": len(reference_x),
                "metadata": reference_metadata,
                "source_note": (
                    "Generated locally with a five-point finite-difference method because "
                    "FEALPy is unavailable. The paper uses an adaptive FEALPy FEM "
                    "reference, so the numerical fields are not guaranteed identical."
                ),
            },
            "paper_classical_pinn_l2": 9.25e-2,
            "paper_dasa_pinn_l2": 3.46e-3,
            "seed": args.seed,
        },
    )
    print(f"l2_relative_error={metrics['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
