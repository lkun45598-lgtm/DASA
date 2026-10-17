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


DIFFUSIVITY = 0.0001
REACTION_COEFFICIENT = 5.0
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)
FOURIER_ORDER = 8
DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference_data"
    / "allen_cahn_diffusivity_0.0001_spectral_reference.npz"
)


def initial_condition(space):
    return space**2 * np.cos(np.pi * space)


def pde(points, prediction):
    u_t = dde.grad.jacobian(prediction, points, i=0, j=1)
    u_xx = dde.grad.hessian(prediction, points, i=0, j=0)
    return (
        u_t
        - DIFFUSIVITY * u_xx
        + REACTION_COEFFICIENT * prediction**3
        - REACTION_COEFFICIENT * prediction
    )


def periodic_fourier_features(points):
    space = points[:, 0:1]
    time = points[:, 1:2]
    features = [time]
    for wave_number in range(1, FOURIER_ORDER + 1):
        features.extend(
            (
                torch.cos(wave_number * torch.pi * space),
                torch.sin(wave_number * torch.pi * space),
            )
        )
    return torch.cat(features, dim=1)


def spatial_derivative(points, prediction):
    return dde.grad.jacobian(prediction, points, i=0, j=0)


def lhs_points(num_points: int, bounds):
    dimensions = len(bounds)
    points = np.empty((num_points, dimensions), dtype=np.float64)
    for dimension, (lower, upper) in enumerate(bounds):
        coordinate = (np.arange(num_points) + np.random.random(num_points)) / num_points
        np.random.shuffle(coordinate)
        points[:, dimension] = lower + (upper - lower) * coordinate
    return points


def training_points(num_domain: int, num_initial: int):
    residual_x = lhs_points(num_domain, (SPACE_DOMAIN, TIME_DOMAIN))
    initial_space = lhs_points(num_initial, (SPACE_DOMAIN,))[:, 0:1]
    initial_x = np.column_stack((initial_space[:, 0], np.zeros(num_initial)))
    initial_u = initial_condition(initial_space)
    return residual_x, initial_x, initial_u


def build_problem(num_domain: int, num_initial: int):
    geometry = dde.geometry.Interval(*SPACE_DOMAIN)
    time_domain = dde.geometry.TimeDomain(*TIME_DOMAIN)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    residual_x, initial_x, initial_u = training_points(num_domain, num_initial)
    initial_bc = dde.icbc.PointSetBC(initial_x, initial_u)
    data = dde.data.PDE(
        geometry_time,
        pde,
        [initial_bc],
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
        fourier_features=periodic_fourier_features,
    )
    return dde.Model(data, net)


def load_reference(path: Path):
    if path.suffix.lower() == ".mat":
        raise ValueError(
            "The bundled DeepXDE Allen_Cahn.mat corresponds to diffusivity 0.001, "
            "not the paper's diffusivity 0.0001; use the generated NPZ reference."
        )
    with np.load(path) as reference:
        required = {"x", "t", "u", "diffusivity"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(
                f"Reference file must contain {sorted(required)}; "
                f"missing: {sorted(missing)}"
            )
        space = np.asarray(reference["x"], dtype=np.float64).reshape(-1)
        time = np.asarray(reference["t"], dtype=np.float64).reshape(-1)
        solution = np.asarray(reference["u"], dtype=np.float64)
        reference_diffusivity = float(reference["diffusivity"])
        metadata = {
            key: reference[key].item()
            for key in (
                "diffusivity",
                "reaction_coefficient",
                "internal_grid_size",
                "internal_spacing",
                "rtol",
                "atol",
                "maximum_time_step",
                "time_integrator",
                "space_discretization",
                "source",
            )
            if key in reference.files
        }
    if not np.isclose(reference_diffusivity, DIFFUSIVITY, rtol=0.0, atol=1e-15):
        raise ValueError(
            "Reference diffusivity does not match the paper equation: "
            f"reference={reference_diffusivity}, expected={DIFFUSIVITY}"
        )
    expected_shape = (len(time), len(space))
    if solution.shape == (len(space), len(time)):
        solution = solution.T
    if solution.shape != expected_shape:
        raise ValueError(
            f"Reference u must have shape {expected_shape}; got {solution.shape}"
        )
    if not np.isclose(space[0], SPACE_DOMAIN[0]) or not np.isclose(
        space[-1], SPACE_DOMAIN[1]
    ):
        raise ValueError("Reference x range does not match the paper domain [-1,1]")
    if not np.isclose(time[0], TIME_DOMAIN[0]) or not np.isclose(
        time[-1], TIME_DOMAIN[1]
    ):
        raise ValueError("Reference t range does not match the paper domain [0,1]")

    initial_error = np.max(np.abs(solution[0] - initial_condition(space)))
    if initial_error > 1e-10:
        raise ValueError(
            "Reference initial field does not match x^2*cos(pi*x): "
            f"max error {initial_error:.3e}"
        )

    grid_x, grid_t = np.meshgrid(space, time, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    return points, solution.reshape(-1, 1), space, time, metadata


def physics_metrics(model, grid_size: int, num_initial: int):
    space = np.linspace(*SPACE_DOMAIN, grid_size, dtype=np.float64)
    time = np.linspace(*TIME_DOMAIN, grid_size, dtype=np.float64)
    grid_x, grid_t = np.meshgrid(space, time, indexing="xy")
    residual_x = np.column_stack((grid_x.ravel(), grid_t.ravel()))
    residual = model.predict(residual_x, operator=pde).reshape(-1)

    boundary_time = np.linspace(*TIME_DOMAIN, grid_size, dtype=np.float64)
    left_x = np.column_stack((np.full(grid_size, SPACE_DOMAIN[0]), boundary_time))
    right_x = np.column_stack((np.full(grid_size, SPACE_DOMAIN[1]), boundary_time))
    value_jump = model.predict(left_x).reshape(-1) - model.predict(right_x).reshape(-1)
    derivative_jump = model.predict(left_x, operator=spatial_derivative).reshape(
        -1
    ) - model.predict(right_x, operator=spatial_derivative).reshape(-1)

    initial_space = np.linspace(*SPACE_DOMAIN, num_initial, dtype=np.float64)
    initial_x = np.column_stack((initial_space, np.zeros(num_initial)))
    initial_error = model.predict(initial_x).reshape(-1) - initial_condition(
        initial_space
    )
    return {
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "pde_residual_max": float(np.max(np.abs(residual))),
        "periodic_value_jump_rms": float(np.sqrt(np.mean(value_jump**2))),
        "periodic_value_jump_max": float(np.max(np.abs(value_jump))),
        "periodic_derivative_jump_rms": float(
            np.sqrt(np.mean(derivative_jump**2))
        ),
        "periodic_derivative_jump_max": float(np.max(np.abs(derivative_jump))),
        "initial_error_rms": float(np.sqrt(np.mean(initial_error**2))),
        "initial_error_max": float(np.max(np.abs(initial_error))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classical PINN baseline for the DASA-PINNs 1D Allen-Cahn experiment."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-domain", type=positive_int, default=25600)
    parser.add_argument("--num-initial", type=positive_int, default=512)
    parser.add_argument("--adam-iterations", type=positive_int, default=300000)
    parser.add_argument("--lbfgs-iterations", type=positive_int, default=0)
    parser.add_argument("--display-every", type=positive_int, default=1000)
    parser.add_argument("--physics-grid-size", type=positive_int, default=101)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pinn_baselines"))
    args = parser.parse_args()

    if not args.reference_path.exists():
        raise FileNotFoundError(f"Allen-Cahn reference data not found: {args.reference_path}")

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(args.num_domain, args.num_initial)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    compile_and_train(model, train_config, loss_weights=[1.0, 100.0])

    reference_x, y_true, reference_space, reference_time, reference_metadata = (
        load_reference(args.reference_path)
    )
    y_pred = model.predict(reference_x)
    metrics = {
        "l2_relative_error": l2_relative_error(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
    }
    metrics.update(physics_metrics(model, args.physics_grid_size, args.num_initial))

    run_dir = experiment_dir(
        args.output_dir,
        "allen_cahn_1d",
        "diffusivity_0.0001_reaction_5_fourier_order_8_spectral_reference",
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
            "equation": "1D Allen-Cahn",
            "paper_equations": ["Eq. 20a", "Eq. 20b", "Eq. 20c", "Eq. 20d"],
            "diffusivity": DIFFUSIVITY,
            "reaction_coefficient": REACTION_COEFFICIENT,
            "space_domain": list(SPACE_DOMAIN),
            "time_domain": list(TIME_DOMAIN),
            "initial_condition": "u(0,x)=x^2*cos(pi*x)",
            "boundary_condition": (
                "u(t,-1)=u(t,1), ux(t,-1)=ux(t,1), enforced by Fourier features"
            ),
            "num_domain": args.num_domain,
            "num_boundary": 0,
            "num_initial": args.num_initial,
            "sampling": "fixed LHS sets for residual and initial points",
            "sampling_note": (
                "The paper states Nr and Nic but does not explicitly name the sampling "
                "method in Section 3.2.2; LHS follows neighboring time-dependent cases."
            ),
            "precision": "float64",
            "loss_weights": {"residual": 1.0, "initial": 100.0},
            "loss_reference": (
                "Paper Appendix D Eq. 56; no boundary loss because the Fourier "
                "embedding enforces both periodic conditions exactly."
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
                    "mapping": "[t, cos(k*pi*x), sin(k*pi*x)] for k=1,...,8",
                    "order": FOURIER_ORDER,
                    "source": (
                        "Temporary project setting. The DASA paper states Fourier feature "
                        "embedding but does not print its order; reference [21] uses m=10."
                    ),
                },
            },
            "optimizer": {
                "adam_iterations": args.adam_iterations,
                "budget_source": (
                    "Confirmed experiment setting: 300000 Adam iterations for the "
                    "classical PINN training budget."
                ),
                "budget_status": "confirmed",
                "learning_rate": train_config.lr,
                "decay_steps": train_config.decay_steps,
                "decay_rate": train_config.decay_rate,
                "lbfgs_iterations": args.lbfgs_iterations,
                "lbfgs_source": "Section 3.2.2 does not specify L-BFGS for this experiment.",
            },
            "reference": {
                "path": str(args.reference_path),
                "format": "NPZ",
                "grid_shape": [len(reference_time), len(reference_space)],
                "metadata": reference_metadata,
                "source_note": (
                    "Generated locally using periodic Fourier pseudospectral spatial "
                    "discretization and SciPy DOP853 for diffusivity 0.0001. The "
                    "previous bundled DeepXDE MAT field uses diffusivity 0.001 and is "
                    "not compatible with paper Eq. 20."
                ),
            },
            "paper_classical_pinn_l2": 4.02e-2,
            "seed": args.seed,
        },
    )
    print("loss=paper Appendix D Eq. 56 with wr=1, wi=100; periodic BC is hard-enforced")
    print(f"l2_relative_error={metrics['l2_relative_error']:.6e}")
    print(f"max_pointwise_error={metrics['max_pointwise_error']:.6e}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
