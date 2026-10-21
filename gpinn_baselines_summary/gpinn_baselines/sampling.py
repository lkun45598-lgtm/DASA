"""Deterministic Latin-hypercube and boundary samplers."""

from __future__ import annotations

import numpy as np


def lhs(n: int, lower: tuple[float, ...], upper: tuple[float, ...], rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((0, len(lower)), dtype=np.float64)
    d = len(lower)
    result = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        strata = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        result[:, j] = strata[rng.permutation(n)]
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return lo + (hi - lo) * result


def interval_boundary(n: int, left: float, right: float) -> np.ndarray:
    if n <= 1:
        return np.asarray([[left]], dtype=np.float64)
    return np.linspace(left, right, n, dtype=np.float64)[:, None]


def rectangle_boundary(
    n: int,
    lower: tuple[float, float],
    upper: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a total of n points, distributed as evenly as possible over four edges."""
    counts = np.full(4, n // 4, dtype=int)
    counts[: n % 4] += 1
    x0, y0 = lower
    x1, y1 = upper
    edges: list[np.ndarray] = []
    for edge, count in enumerate(counts):
        if count == 0:
            continue
        s = lhs(count, (0.0,), (1.0,), rng)[:, 0]
        if edge == 0:
            edges.append(np.column_stack((x0 + (x1 - x0) * s, np.full(count, y0))))
        elif edge == 1:
            edges.append(np.column_stack((x0 + (x1 - x0) * s, np.full(count, y1))))
        elif edge == 2:
            edges.append(np.column_stack((np.full(count, x0), y0 + (y1 - y0) * s)))
        else:
            edges.append(np.column_stack((np.full(count, x1), y0 + (y1 - y0) * s)))
    return np.concatenate(edges, axis=0).astype(np.float64)


def time_interval_boundary(
    n: int,
    x_left: float,
    x_right: float,
    t0: float,
    t1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = (n // 2, n - n // 2)
    left_t = lhs(counts[0], (t0,), (t1,), rng)[:, 0]
    right_t = lhs(counts[1], (t0,), (t1,), rng)[:, 0]
    left = np.column_stack((np.full(counts[0], x_left), left_t))
    right = np.column_stack((np.full(counts[1], x_right), right_t))
    return np.concatenate((left, right), axis=0)


def lshape_interior(n: int, rng: np.random.Generator) -> np.ndarray:
    accepted: list[np.ndarray] = []
    count = 0
    while count < n:
        proposal = lhs(max(2 * (n - count), 32), (-1.0, -1.0), (1.0, 1.0), rng)
        keep = ~((proposal[:, 0] > 0.0) & (proposal[:, 1] > 0.0))
        chunk = proposal[keep]
        accepted.append(chunk)
        count += len(chunk)
    return np.concatenate(accepted, axis=0)[:n]


def lshape_boundary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Eight equal-length boundary segments of [-1,1]^2 \\ [0,1]^2."""
    counts = np.full(8, n // 8, dtype=int)
    counts[: n % 8] += 1
    segments = [
        ((-1.0, -1.0), (0.0, -1.0)),
        ((0.0, -1.0), (1.0, -1.0)),
        ((1.0, -1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (-1.0, 1.0)),
        ((-1.0, 1.0), (-1.0, 0.0)),
        ((-1.0, 0.0), (-1.0, -1.0)),
    ]
    values: list[np.ndarray] = []
    for count, (start, end) in zip(counts, segments):
        s = lhs(int(count), (0.0,), (1.0,), rng)[:, 0]
        a, b = np.asarray(start), np.asarray(end)
        values.append(a + (b - a) * s[:, None])
    return np.concatenate(values, axis=0).astype(np.float64)
