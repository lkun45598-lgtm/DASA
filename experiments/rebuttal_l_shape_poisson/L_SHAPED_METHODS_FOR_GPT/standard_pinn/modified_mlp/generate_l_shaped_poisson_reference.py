import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "reference_data"
    / "l_shaped_poisson_fdm_reference.npz"
)


def solve_reference(points_per_unit: int, evaluation_stride: int, rtol: float):
    if points_per_unit % evaluation_stride != 0:
        raise ValueError("points_per_unit must be divisible by evaluation_stride")

    grid_size = 2 * points_per_unit + 1
    spacing = 1.0 / points_per_unit
    coordinates = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    domain = np.logical_or(grid_x <= 0.0, grid_y <= 0.0)

    boundary = np.zeros_like(domain)
    boundary[0, :] = domain[0, :]
    boundary[-1, :] = domain[-1, :]
    boundary[:, 0] = domain[:, 0]
    boundary[:, -1] = domain[:, -1]
    zero_index = points_per_unit
    boundary[zero_index:, zero_index] = True
    boundary[zero_index, zero_index:] = True
    boundary &= domain
    interior = domain & ~boundary

    indices = np.full((grid_size, grid_size), -1, dtype=np.int64)
    indices[interior] = np.arange(np.count_nonzero(interior), dtype=np.int64)
    interior_coordinates = np.argwhere(interior)
    num_unknowns = len(interior_coordinates)

    rows = [np.arange(num_unknowns, dtype=np.int64)]
    columns = [np.arange(num_unknowns, dtype=np.int64)]
    values = [np.full(num_unknowns, 4.0, dtype=np.float64)]
    for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_rows = interior_coordinates[:, 0] + row_offset
        neighbor_columns = interior_coordinates[:, 1] + column_offset
        neighbor_is_interior = interior[neighbor_rows, neighbor_columns]
        rows.append(
            indices[
                interior_coordinates[neighbor_is_interior, 0],
                interior_coordinates[neighbor_is_interior, 1],
            ]
        )
        columns.append(
            indices[
                neighbor_rows[neighbor_is_interior],
                neighbor_columns[neighbor_is_interior],
            ]
        )
        values.append(
            np.full(np.count_nonzero(neighbor_is_interior), -1.0, dtype=np.float64)
        )

    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(num_unknowns, num_unknowns),
    ).tocsr()
    right_hand_side = np.full(num_unknowns, spacing**2, dtype=np.float64)
    iterations = [0]

    def count_iteration(_):
        iterations[0] += 1

    interior_solution, info = cg(
        matrix,
        right_hand_side,
        rtol=rtol,
        atol=0.0,
        maxiter=10000,
        callback=count_iteration,
    )
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient solver failed with info={info}")

    relative_residual = float(
        np.linalg.norm(matrix @ interior_solution - right_hand_side)
        / np.linalg.norm(right_hand_side)
    )
    full_solution = np.zeros((grid_size, grid_size), dtype=np.float64)
    full_solution[interior] = interior_solution

    evaluation_slice = slice(None, None, evaluation_stride)
    evaluation_coordinates = coordinates[evaluation_slice]
    evaluation_solution = full_solution[evaluation_slice, evaluation_slice]
    evaluation_domain = domain[evaluation_slice, evaluation_slice]
    evaluation_x, evaluation_y = np.meshgrid(
        evaluation_coordinates, evaluation_coordinates, indexing="xy"
    )
    points = np.column_stack(
        (evaluation_x[evaluation_domain], evaluation_y[evaluation_domain])
    )
    values_at_points = evaluation_solution[evaluation_domain, None]
    return {
        "x": points,
        "u": values_at_points,
        "reference_x": evaluation_coordinates,
        "reference_y": evaluation_coordinates,
        "solver_points_per_unit": np.asarray(points_per_unit),
        "solver_grid_size": np.asarray(grid_size),
        "solver_spacing": np.asarray(spacing),
        "evaluation_stride": np.asarray(evaluation_stride),
        "evaluation_spacing": np.asarray(spacing * evaluation_stride),
        "cg_iterations": np.asarray(iterations[0]),
        "cg_relative_residual": np.asarray(relative_residual),
        "source": np.asarray("five-point finite difference generated locally"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a finite-difference reference for L-shaped Poisson."
    )
    parser.add_argument("--points-per-unit", type=int, default=256)
    parser.add_argument("--evaluation-stride", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    reference = solve_reference(
        args.points_per_unit, args.evaluation_stride, args.rtol
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **reference)
    print(f"reference={args.output}")
    print(f"points={len(reference['x'])}")
    print(f"solver_grid_size={int(reference['solver_grid_size'])}")
    print(f"cg_iterations={int(reference['cg_iterations'])}")
    print(f"cg_relative_residual={float(reference['cg_relative_residual']):.6e}")


if __name__ == "__main__":
    main()
