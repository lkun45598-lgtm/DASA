import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


DIFFUSIVITY = 0.0001
REACTION_COEFFICIENT = 5.0
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "allen_cahn_diffusivity_0.0001_spectral_reference.npz"
)


def initial_condition(space):
    return space**2 * np.cos(np.pi * space)


def solve_reference(
    diffusivity: float,
    internal_grid_size: int,
    output_space_grid_size: int,
    output_time_grid_size: int,
    rtol: float,
    atol: float,
):
    output_intervals = output_space_grid_size - 1
    if output_intervals <= 0:
        raise ValueError("output_space_grid_size must be at least 2")
    if internal_grid_size % output_intervals != 0:
        raise ValueError(
            "internal_grid_size must be divisible by output_space_grid_size - 1"
        )

    domain_length = SPACE_DOMAIN[1] - SPACE_DOMAIN[0]
    spacing = domain_length / internal_grid_size
    internal_space = np.linspace(
        *SPACE_DOMAIN, internal_grid_size, endpoint=False, dtype=np.float64
    )
    time = np.linspace(*TIME_DOMAIN, output_time_grid_size, dtype=np.float64)
    angular_wave_numbers = 2.0 * np.pi * np.fft.fftfreq(
        internal_grid_size, d=spacing
    )
    laplacian_multiplier = -(angular_wave_numbers**2)
    maximum_diffusive_decay = diffusivity * np.max(angular_wave_numbers**2)
    maximum_time_step = min(0.005, 4.0 / maximum_diffusive_decay)

    def right_hand_side(_, solution):
        laplacian = np.fft.ifft(
            laplacian_multiplier * np.fft.fft(solution)
        ).real
        return (
            diffusivity * laplacian
            + REACTION_COEFFICIENT * solution
            - REACTION_COEFFICIENT * solution**3
        )

    integration = solve_ivp(
        right_hand_side,
        TIME_DOMAIN,
        initial_condition(internal_space),
        method="DOP853",
        t_eval=time,
        rtol=rtol,
        atol=atol,
        first_step=min(1e-4, maximum_time_step),
        max_step=maximum_time_step,
    )
    if not integration.success:
        raise RuntimeError(f"Time integration failed: {integration.message}")

    stride = internal_grid_size // output_intervals
    periodic_solution = integration.y.T[:, ::stride]
    solution = np.column_stack((periodic_solution, periodic_solution[:, 0]))
    space = np.linspace(*SPACE_DOMAIN, output_space_grid_size, dtype=np.float64)
    return {
        "x": space,
        "t": time,
        "u": solution,
        "diffusivity": np.asarray(diffusivity),
        "reaction_coefficient": np.asarray(REACTION_COEFFICIENT),
        "internal_grid_size": np.asarray(internal_grid_size),
        "internal_spacing": np.asarray(spacing),
        "rtol": np.asarray(rtol),
        "atol": np.asarray(atol),
        "maximum_time_step": np.asarray(maximum_time_step),
        "time_integrator": np.asarray("scipy.solve_ivp DOP853"),
        "space_discretization": np.asarray("periodic Fourier pseudospectral"),
        "source": np.asarray("generated locally for the DASA-PINNs paper Eq. 20"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a periodic spectral reference for the DASA-PINNs "
            "Allen-Cahn equation."
        )
    )
    parser.add_argument("--diffusivity", type=float, default=DIFFUSIVITY)
    parser.add_argument("--internal-grid-size", type=int, default=6400)
    parser.add_argument("--output-space-grid-size", type=int, default=201)
    parser.add_argument("--output-time-grid-size", type=int, default=101)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    if min(
        args.diffusivity,
        args.internal_grid_size,
        args.output_space_grid_size,
        args.output_time_grid_size,
        args.rtol,
        args.atol,
    ) <= 0:
        raise ValueError("All numerical arguments must be positive")

    reference = solve_reference(
        args.diffusivity,
        args.internal_grid_size,
        args.output_space_grid_size,
        args.output_time_grid_size,
        args.rtol,
        args.atol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **reference)
    solution = reference["u"]
    print(f"reference={args.output}")
    print(f"grid_shape={solution.shape}")
    print(f"diffusivity={float(reference['diffusivity']):.12g}")
    print(f"solution_min={solution.min():.12e}")
    print(f"solution_max={solution.max():.12e}")


if __name__ == "__main__":
    main()
