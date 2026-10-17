import argparse
import re
from pathlib import Path

import numpy as np
import torch

from plain_pinn_baselines.common import (
    PlainFNN,
    TrainConfig,
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
CHECKPOINT_PATTERN = re.compile(r"^adam_step_(\d+)\.pt$")


def latest_checkpoint(checkpoint_dir: Path):
    candidates = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("adam_step_*.pt"):
            match = CHECKPOINT_PATTERN.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])


def load_torch_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class AdamCheckpoint(dde.callbacks.Callback):
    def __init__(self, checkpoint_dir: Path, period: int, signature: dict):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.period = period
        self.signature = signature
        self.last_saved_step = None

    def _save(self):
        step = int(self.model.train_state.step)
        if step == self.last_saved_step:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"adam_step_{step:09d}.pt"
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        scheduler_state = (
            self.model.lr_scheduler.state_dict()
            if self.model.lr_scheduler is not None
            else None
        )
        payload = {
            "format_version": 1,
            "stage": "adam",
            "step": step,
            "epoch": int(self.model.train_state.iteration),
            "model_state_dict": self.model.net.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
            "lr_scheduler_state_dict": scheduler_state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "signature": self.signature,
        }
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
        self.last_saved_step = step
        print(f"Saved Adam checkpoint: {checkpoint_path}")

    def on_epoch_end(self):
        if self.model.train_state.step % self.period == 0:
            self._save()

    def on_train_end(self):
        self._save()


def checkpoint_signature(args, train_config: TrainConfig):
    return {
        "equation": "allen_cahn_1d",
        "seed": args.seed,
        "num_domain": args.num_domain,
        "num_initial": args.num_initial,
        "diffusivity": DIFFUSIVITY,
        "reaction_coefficient": REACTION_COEFFICIENT,
        "fourier_order": FOURIER_ORDER,
        "network": {
            "type": "plain_fnn",
            "depth": 6,
            "width": 128,
            "activation": "tanh",
            "initializer": "Glorot normal",
        },
        "optimizer": {
            "type": "adam",
            "learning_rate": train_config.lr,
            "decay_steps": train_config.decay_steps,
            "decay_rate": train_config.decay_rate,
        },
        "loss_weights": [1.0, 100.0],
        "precision": "float64",
    }


def restore_adam_checkpoint(model, checkpoint_path: Path, signature: dict):
    device = next(model.net.parameters()).device
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if checkpoint.get("format_version") != 1 or checkpoint.get("stage") != "adam":
        raise ValueError(f"Unsupported Allen-Cahn checkpoint: {checkpoint_path}")
    if checkpoint.get("signature") != signature:
        raise ValueError(
            "Checkpoint configuration does not match this Allen-Cahn run: "
            f"{checkpoint_path}"
        )

    filename_match = CHECKPOINT_PATTERN.match(checkpoint_path.name)
    completed_step = int(checkpoint["step"])
    if filename_match is None or int(filename_match.group(1)) != completed_step:
        raise ValueError(f"Checkpoint filename and saved step disagree: {checkpoint_path}")

    model.net.load_state_dict(checkpoint["model_state_dict"])
    model.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("lr_scheduler_state_dict")
    if model.lr_scheduler is not None and scheduler_state is not None:
        model.lr_scheduler.load_state_dict(scheduler_state)
    model.train_state.step = completed_step
    model.train_state.iteration = int(checkpoint.get("epoch", completed_step))

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    return completed_step


def train_with_checkpoints(
    model,
    train_config: TrainConfig,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume_mode: str,
    signature: dict,
):
    model.compile(
        "adam",
        lr=train_config.lr,
        decay=("step", train_config.decay_steps, train_config.decay_rate),
        loss="MSE",
        loss_weights=[1.0, 100.0],
    )

    latest_step, resume_path = latest_checkpoint(checkpoint_dir)
    if resume_mode == "required" and resume_path is None:
        raise FileNotFoundError(f"No checkpoint found in required mode: {checkpoint_dir}")
    if resume_mode == "never" and resume_path is not None:
        raise FileExistsError(
            "Checkpoint directory is not empty in never-resume mode; use a new "
            f"OUTPUT_DIR or remove it explicitly: {checkpoint_dir}"
        )

    resumed_step = 0
    if resume_mode in {"auto", "required"} and resume_path is not None:
        resumed_step = restore_adam_checkpoint(model, resume_path, signature)
        if resumed_step != latest_step:
            raise ValueError(f"Latest checkpoint step mismatch: {resume_path}")
        print(f"Resuming Adam from step {resumed_step}: {resume_path}")

    if resumed_step > train_config.adam_iterations:
        raise ValueError(
            f"Checkpoint step {resumed_step} exceeds target Adam iterations "
            f"{train_config.adam_iterations}"
        )
    if train_config.lbfgs_iterations > 0 and resumed_step > 0:
        raise ValueError("L-BFGS restart is not supported by the Adam checkpoint format")

    remaining_iterations = train_config.adam_iterations - resumed_step
    checkpoint_callback = AdamCheckpoint(
        checkpoint_dir=checkpoint_dir,
        period=checkpoint_every,
        signature=signature,
    )
    if remaining_iterations > 0:
        losshistory, train_state = model.train(
            iterations=remaining_iterations,
            display_every=train_config.display_every,
            callbacks=[checkpoint_callback],
        )
    else:
        print(
            f"Adam target already reached at step {resumed_step}; skipping Adam training."
        )
        losshistory, train_state = model.losshistory, model.train_state

    if train_config.lbfgs_iterations > 0:
        dde.optimizers.set_LBFGS_options(maxiter=train_config.lbfgs_iterations)
        model.compile("L-BFGS", loss="MSE", loss_weights=[1.0, 100.0])
        losshistory, train_state = model.train(
            display_every=train_config.display_every
        )

    return losshistory, train_state, {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_every": checkpoint_every,
        "resume_mode": resume_mode,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resumed_step": resumed_step,
        "adam_iterations_this_process": remaining_iterations,
        "adam_target_iterations": train_config.adam_iterations,
    }


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
    net = PlainFNN(
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plain_fnn"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=positive_int, default=10000)
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "required", "never"),
        default="auto",
    )
    args = parser.parse_args()

    if not args.reference_path.exists():
        raise FileNotFoundError(f"Allen-Cahn reference data not found: {args.reference_path}")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than zero")

    dde.config.set_default_float("float64")
    set_seed(args.seed)
    model = build_problem(args.num_domain, args.num_initial)
    train_config = TrainConfig(
        adam_iterations=args.adam_iterations,
        lbfgs_iterations=args.lbfgs_iterations,
        display_every=args.display_every,
    )
    checkpoint_dir = args.checkpoint_dir or args.output_dir / "checkpoints"
    signature = checkpoint_signature(args, train_config)
    _, _, checkpoint_metadata = train_with_checkpoints(
        model=model,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_mode=args.resume_mode,
        signature=signature,
    )

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
                "type": "plain_fnn",
                "version": "plain_fnn_original_pinn",
                "gated_modified_mlp": False,
                "source": "Original PINNs ordinary fully connected network family",
                "hidden_layer_count": 6,
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
            "checkpointing": checkpoint_metadata,
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
