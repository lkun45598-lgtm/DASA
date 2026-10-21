"""Finite-difference solver for the L-shaped Poisson reference."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def solve_lshape_reference(
    grid_points: int, evaluation_points: np.ndarray
) -> np.ndarray:
    if grid_points % 2 == 0:
        raise ValueError("L-shaped grid must have an odd number of points so zero is represented")
    points = np.asarray(evaluation_points, dtype=np.float64)
    axis = np.linspace(-1.0, 1.0, grid_points, dtype=np.float64)
    spacing = axis[1] - axis[0]
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    interior = (
        (xx > -1.0)
        & (xx < 1.0)
        & (yy > -1.0)
        & (yy < 1.0)
        & ~((xx >= 0.0) & (yy >= 0.0))
    )
    indices = -np.ones((grid_points, grid_points), dtype=np.int64)
    indices[interior] = np.arange(np.count_nonzero(interior))
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row, column in np.argwhere(interior):
        equation = int(indices[row, column])
        rows.append(equation)
        columns.append(equation)
        data.append(4.0)
        for neighbor_row, neighbor_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            neighbor = int(indices[neighbor_row, neighbor_column])
            if neighbor >= 0:
                rows.append(equation)
                columns.append(neighbor)
                data.append(-1.0)
    matrix = csr_matrix((data, (rows, columns)), shape=(np.count_nonzero(interior),) * 2)
    right_hand_side = np.full(np.count_nonzero(interior), spacing**2, dtype=np.float64)
    solution = spsolve(matrix, right_hand_side)
    field = np.zeros_like(xx)
    field[interior] = solution
    interpolator = RegularGridInterpolator(
        (axis, axis), field, bounds_error=True
    )
    # Array axes are (y, x), while benchmark coordinates are ordered (x, y).
    return interpolator(points[:, [1, 0]])[:, None]
