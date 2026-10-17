"""Deterministic evaluation grids shared by problems and references."""

from __future__ import annotations

import numpy as np


def static_1d_grid(domain: tuple[float, float]) -> np.ndarray:
    return np.linspace(domain[0], domain[1], 10001, dtype=np.float64)[:, None]


def regular_2d_grid(
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    x = np.linspace(*domain[0], 401, dtype=np.float64)
    y = np.linspace(*domain[1], 401, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def time_1d_grid(
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    t = np.linspace(*domain[0], 201, dtype=np.float64)
    x = np.linspace(*domain[1], 512, dtype=np.float64)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    return np.column_stack((tt.ravel(), xx.ravel()))


def lshape_grid() -> np.ndarray:
    points = regular_2d_grid(((-1.0, 1.0), (-1.0, 1.0)))
    removed = (points[:, 0] > 0.0) & (points[:, 1] > 0.0)
    return points[~removed]
