import argparse
from pathlib import Path

import mpmath as mp
import numpy as np


VELOCITY = 1.0
DIFFUSIVITY = 0.01
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "advection_diffusion_kappa_0.01_series_reference.npz"
)


def initial_condition(space):
    return -np.sin(np.pi * space)


def series_reference(space, time, num_terms: int, decimal_precision: int):
    """Evaluate the Dirichlet eigenfunction series in high precision.

    The exponential substitution used for advection-diffusion is ill-conditioned
    in float64 at kappa=0.01. High-precision summation prevents cancellation near
    the outflow boundary from corrupting the reference field.
    """
    mp.mp.dps = decimal_precision
    velocity = mp.mpf(str(VELOCITY))
    diffusivity = mp.mpf(str(DIFFUSIVITY))
    domain_length = mp.mpf(str(SPACE_DOMAIN[1] - SPACE_DOMAIN[0]))
    alpha = velocity / (2 * diffusivity)
    temporal_shift = velocity**2 / (4 * diffusivity)
    pi = mp.pi

    wave_numbers = [n * pi / domain_length for n in range(1, num_terms + 1)]
    exp_alpha = mp.exp(alpha)
    exp_minus_two_alpha = mp.exp(-2 * alpha)
    coefficients = []
    decay_rates = []
    for n, wave_number in enumerate(wave_numbers, start=1):
        parity_factor = 1 - ((-1) ** n) * exp_minus_two_alpha
        denominator = (
            (alpha**2 + (pi - wave_number) ** 2)
            * (alpha**2 + (pi + wave_number) ** 2)
        )
        coefficients.append(
            exp_alpha
            * 2
            * alpha
            * pi
            * wave_number
            * parity_factor
            / denominator
        )
        decay_rates.append(diffusivity * wave_number**2 + temporal_shift)

    interior_space = [mp.mpf(str(value)) for value in space[1:-1]]
    sine_basis = [
        [
            mp.sin(wave_number * (coordinate - SPACE_DOMAIN[0]))
            for wave_number in wave_numbers
        ]
        for coordinate in interior_space
    ]
    spatial_factors = [mp.exp(alpha * coordinate) for coordinate in interior_space]

    solution = np.zeros((len(time), len(space)), dtype=np.float64)
    solution[0] = initial_condition(space)
    solution[0, 0] = 0.0
    solution[0, -1] = 0.0
    for time_index, time_value in enumerate(time[1:], start=1):
        current_time = mp.mpf(str(time_value))
        time_coefficients = [
            coefficient * mp.exp(-decay_rate * current_time)
            for coefficient, decay_rate in zip(coefficients, decay_rates)
        ]
        for space_index, (basis, spatial_factor) in enumerate(
            zip(sine_basis, spatial_factors), start=1
        ):
            solution[time_index, space_index] = float(
                spatial_factor * mp.fdot(basis, time_coefficients)
            )
        if time_index % 10 == 0 or time_index == len(time) - 1:
            print(f"evaluated_time_index={time_index}/{len(time) - 1}")
    return solution


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the 1000-term analytical reference for the paper's "
            "1D advection-diffusion equation."
        )
    )
    parser.add_argument("--space-grid-size", type=int, default=256)
    parser.add_argument("--time-grid-size", type=int, default=101)
    parser.add_argument("--num-terms", type=int, default=1000)
    parser.add_argument("--decimal-precision", type=int, default=80)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    if min(
        args.space_grid_size,
        args.time_grid_size,
        args.num_terms,
        args.decimal_precision,
    ) <= 0:
        raise ValueError("All grid, series, and precision arguments must be positive")

    space = np.linspace(*SPACE_DOMAIN, args.space_grid_size, dtype=np.float64)
    time = np.linspace(*TIME_DOMAIN, args.time_grid_size, dtype=np.float64)
    solution = series_reference(
        space, time, args.num_terms, args.decimal_precision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        x=space,
        t=time,
        u=solution,
        velocity=np.asarray(VELOCITY),
        diffusivity=np.asarray(DIFFUSIVITY),
        num_terms=np.asarray(args.num_terms),
        decimal_precision=np.asarray(args.decimal_precision),
        source=np.asarray(
            "Dirichlet eigenfunction series for Eq. 24; paper cites reference [36]"
        ),
    )
    print(f"reference={args.output}")
    print(f"grid_shape={solution.shape}")
    print(f"solution_min={solution.min():.12e}")
    print(f"solution_max={solution.max():.12e}")


if __name__ == "__main__":
    main()
