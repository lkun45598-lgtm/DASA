"""Central configuration for architecture, losses, sampling and optimization."""

from __future__ import annotations

import copy
from typing import Any


NETWORK = {
    "hidden_layers": 6,
    "width": 128,
    "activation": "tanh",
    "initializer": "xavier_normal",
    "dtype": "float64",
}

BASE_LOSSES = {
    "pde": 1.0,
    "gradient": 0.01,
    "boundary": 100.0,
    "initial": 100.0,
    "periodic_value": 100.0,
    "periodic_derivative": 100.0,
    "pressure_gauge": 1.0,
}


def _case(
    *,
    input_dim: int,
    output_dim: int,
    lower: list[float],
    upper: list[float],
    n_residual: int,
    n_boundary: int,
    n_initial: int,
    adam_iterations: int,
    lbfgs_iterations: int,
    learning_rate: float = 1.0e-3,
    lr_decay_rate: float = 0.9,
    lr_decay_steps: int = 2000,
    gradient_weight: float = 0.01,
    lbfgs_history_size: int = 50,
    lbfgs_tolerance_grad: float = 1.0e-10,
    lbfgs_tolerance_change: float = 1.0e-12,
    reference_type: str = "analytic",
    parameters: dict[str, Any] | None = None,
    test_shape: list[int] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    losses = dict(BASE_LOSSES)
    losses["gradient"] = gradient_weight
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "lower": lower,
        "upper": upper,
        "network": dict(NETWORK),
        "sampling": {
            "n_residual": n_residual,
            "n_boundary": n_boundary,
            "n_initial": n_initial,
            "distribution": "latin_hypercube",
        },
        "optimizer": {
            "adam_iterations": adam_iterations,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
            "adam_batch_size": min(n_residual, 2048),
            "lbfgs_iterations": lbfgs_iterations,
            "lbfgs_history_size": lbfgs_history_size,
            "lbfgs_tolerance_grad": lbfgs_tolerance_grad,
            "lbfgs_tolerance_change": lbfgs_tolerance_change,
            "checkpoint_every": 1000,
            "log_every": 100,
        },
        "loss_weights": losses,
        "parameters": parameters or {},
        "reference_type": reference_type,
        "test_shape": test_shape or ([2001] if input_dim == 1 else [201, 201]),
        "notes": notes or [],
    }


CASE_DEFAULTS: dict[str, dict[str, Any]] = {
    "poisson_1d_boundary_layer": _case(
        input_dim=1,
        output_dim=1,
        lower=[-1.0],
        upper=[1.0],
        n_residual=500,
        n_boundary=0,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=0,
        lr_decay_rate=1.0,
        gradient_weight=0.01,
        test_shape=[2001],
        notes=[
            "Equation (10) with analytical solution (11a). Original gPINN-style output transformation imposes both Dirichlet values exactly; pure gPINN, no RAR."
        ],
    ),
    "poisson_1d_steep": _case(
        input_dim=1,
        output_dim=1,
        lower=[-1.0],
        upper=[1.0],
        n_residual=500,
        n_boundary=0,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=0,
        lr_decay_rate=1.0,
        gradient_weight=0.01,
        test_shape=[2001],
        notes=[
            "Equation (10) with analytical solution (11b). Original gPINN-style output transformation imposes both Dirichlet values exactly; pure gPINN, no RAR."
        ],
    ),
    "helmholtz_1d": _case(
        input_dim=1,
        output_dim=1,
        lower=[0.0],
        upper=[1.0],
        n_residual=500,
        n_boundary=2,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=0,
        gradient_weight=0.01,
        parameters={"q": 1.0},
        test_shape=[2001],
        notes=[
            "Equation (12). The paper does not define q(x); q=1 is an explicit configurable manufactured-solution default."
        ],
    ),
    "poisson_2d_peak": _case(
        input_dim=2,
        output_dim=1,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=2500,
        n_boundary=50,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-5,
        parameters={"a": 10},
        test_shape=[201, 201],
        notes=[
            "Equation (13), first solution. The printed formula omits y^a; the symmetric intended form (16*x*y*(1-x)*(1-y))^a is confirmed by the original gPINN file and Figure 3."
        ],
    ),
    "poisson_2d_steep": _case(
        input_dim=2,
        output_dim=1,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=2500,
        n_boundary=50,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-5,
        test_shape=[201, 201],
        notes=["Equation (13), second analytical solution."],
    ),
    "helmholtz_2d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        n_residual=10201,
        n_boundary=200,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-5,
        parameters={"a1": 1.0, "a2": 4.0, "k": 1.0},
        test_shape=[201, 201],
        notes=[
            "Equations (14)-(16). The paper leaves a1, a2 and k unspecified; (1,4,1) is a configurable standard manufactured-solution default."
        ],
    ),
    "lid_driven_cavity": _case(
        input_dim=2,
        output_dim=3,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=2500,
        n_boundary=512,
        n_initial=0,
        adam_iterations=25000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-4,
        parameters={"reynolds": 100.0},
        reference_type="numerical_streamfunction_vorticity",
        test_shape=[65, 65],
        notes=["Equations (17)-(18); pressure is gauge-fixed by its sampled mean."],
    ),
    "burgers_1d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=1500,
        n_boundary=0,
        n_initial=0,
        adam_iterations=20000,
        lbfgs_iterations=15000,
        lr_decay_rate=1.0,
        gradient_weight=1.0e-4,
        lbfgs_history_size=100,
        lbfgs_tolerance_grad=1.0e-8,
        lbfgs_tolerance_change=0.0,
        parameters={"viscosity": 0.01 / 3.141592653589793},
        reference_type="numerical_imex_fd",
        test_shape=[257, 101],
        notes=[
            "Equation (19). Yu et al.'s original gPINN protocol is retained: 1500 interior points, exact output-transform initial/boundary constraints, constant Adam lr=1e-3 for 20000 iterations, then a PyTorch L-BFGS equivalent of the repository's unconstrained L-BFGS-B stage. Only the common benchmark network is changed to 6x128 float64."
        ],
    ),
    "allen_cahn_1d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=25600,
        n_boundary=512,
        n_initial=512,
        adam_iterations=20000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-4,
        parameters={"diffusion": 1.0e-4, "reaction": 5.0},
        reference_type="numerical_fourier_etdrk4",
        test_shape=[257, 201],
        notes=[
            "Equation (20), with diffusion fixed to 1e-4. Periodic value and first-derivative losses are both explicit."
        ],
    ),
    "diffusion_reaction_1d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-3.141592653589793, 0.0],
        upper=[3.141592653589793, 1.0],
        n_residual=25600,
        n_boundary=256,
        n_initial=512,
        adam_iterations=100000,
        lbfgs_iterations=1000,
        learning_rate=1.0e-4,
        gradient_weight=0.1,
        parameters={"diffusion": 1.0},
        test_shape=[257, 201],
        notes=["Equations (21)-(23)."],
    ),
    "advection_diffusion_1d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.0, 0.0],
        upper=[1.0, 1.0],
        n_residual=25600,
        n_boundary=256,
        n_initial=512,
        adam_iterations=100000,
        lbfgs_iterations=1000,
        learning_rate=1.0e-4,
        gradient_weight=0.1,
        parameters={"velocity": 1.0, "diffusivity": 0.01, "series_terms": 1000},
        test_shape=[257, 201],
        notes=[
            "Equation (24). The paper does not give kappa; kappa=0.01 is an explicit configurable advection-dominated default."
        ],
    ),
    "poisson_lshape": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        n_residual=10000,
        n_boundary=1024,
        n_initial=0,
        adam_iterations=25000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-5,
        reference_type="numerical_finite_difference",
        test_shape=[161, 161],
        notes=["Equation (25) on [-1,1]^2 \\ [0,1]^2."],
    ),
    "advection_discontinuous_1d": _case(
        input_dim=2,
        output_dim=1,
        lower=[-1.5, 0.0],
        upper=[1.5, 1.0],
        n_residual=25600,
        n_boundary=256,
        n_initial=512,
        adam_iterations=25000,
        lbfgs_iterations=1000,
        gradient_weight=1.0e-4,
        parameters={"velocity": 1.0, "diffusivity": 0.0},
        test_shape=[301, 201],
        notes=[
            "Equations (26)-(28). Point counts are not reported for this case; the temporal-case counts 25600/256/512 are an explicit configurable default."
        ],
    ),
}

CASE_NAMES = tuple(CASE_DEFAULTS)


def build_config(case_name: str, mode: str, seed: int) -> dict[str, Any]:
    if case_name not in CASE_DEFAULTS:
        raise KeyError(f"Unknown case {case_name!r}; choose from {', '.join(CASE_NAMES)}")
    config = copy.deepcopy(CASE_DEFAULTS[case_name])
    config.update({"case": case_name, "mode": mode, "seed": int(seed)})
    if mode == "smoke":
        sampling = config["sampling"]
        sampling["n_residual"] = min(64, sampling["n_residual"])
        sampling["n_boundary"] = min(32, sampling["n_boundary"])
        sampling["n_initial"] = min(32, sampling["n_initial"])
        config["optimizer"]["adam_iterations"] = 2
        config["optimizer"]["adam_batch_size"] = sampling["n_residual"]
        config["optimizer"]["lbfgs_iterations"] = 0
        config["optimizer"]["log_every"] = 1
        config["optimizer"]["checkpoint_every"] = 2
        if config["input_dim"] == 1:
            config["test_shape"] = [101]
        else:
            config["test_shape"] = [33, 17] if "_1d" in case_name else [25, 25]
    elif mode not in {"full", "test"}:
        raise ValueError("mode must be smoke, full, or test")
    return config
