"""DeepXDE definition for Poisson's equation on the L-shaped domain."""

from __future__ import annotations

import deepxde as dde
import numpy as np

from benchmarks.dasa_aligned_sa.dasa_sa.config import SUITE_ROOT
from benchmarks.dasa_aligned_sa.dasa_sa.grids import lshape_grid

from .base import BaseProblem, latin_hypercube


def _lshape_boundary(count: int) -> np.ndarray:
    counts = [count // 8 + (index < count % 8) for index in range(8)]
    segments: list[np.ndarray] = []
    for index, segment_count in enumerate(counts):
        if not segment_count:
            continue
        unit = (np.arange(segment_count, dtype=np.float64) + 0.5) / segment_count
        negative = unit - 1.0
        if index == 0:
            points = np.column_stack((np.full(segment_count, -1.0), negative))
        elif index == 1:
            points = np.column_stack((np.full(segment_count, -1.0), unit))
        elif index == 2:
            points = np.column_stack((negative, np.full(segment_count, -1.0)))
        elif index == 3:
            points = np.column_stack((unit, np.full(segment_count, -1.0)))
        elif index == 4:
            points = np.column_stack((negative, np.full(segment_count, 1.0)))
        elif index == 5:
            points = np.column_stack((np.full(segment_count, 1.0), negative))
        elif index == 6:
            points = np.column_stack((np.zeros(segment_count), unit))
        else:
            points = np.column_stack((unit, np.zeros(segment_count)))
        segments.append(points)
    return np.vstack(segments)


class LShapeProblem(BaseProblem):
    def pde(self, points, values):
        laplacian = dde.grad.hessian(values, points, component=0, i=0, j=0)
        laplacian += dde.grad.hessian(values, points, component=0, i=1, j=1)
        return -laplacian - self.config.parameters["forcing"]

    def build_data(self):
        vertices = [(-1.0, -1.0), (1.0, -1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0)]
        geometry = dde.geometry.Polygon(vertices)
        accepted: list[np.ndarray] = []
        collected = 0
        batch = 0
        target = self.training_counts["residual"]
        while collected < target:
            candidates = latin_hypercube(
                self.config.domain, max(2048, target - collected), self.seed + batch
            )
            valid = candidates[
                ~((candidates[:, 0] >= 0.0) & (candidates[:, 1] >= 0.0))
            ]
            accepted.append(valid)
            collected += valid.shape[0]
            batch += 1
        residual = np.vstack(accepted)[:target]
        boundary = _lshape_boundary(self.training_counts["boundary"])
        condition = dde.icbc.PointSetBC(
            boundary, np.zeros((boundary.shape[0], 1), dtype=np.float64), component=0
        )
        return self.make_data(geometry, residual, [(condition, "boundary_values")])

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = lshape_grid()
        with np.load(SUITE_ROOT / "references" / "lshape_poisson.npz") as reference:
            stored_points = np.asarray(reference["points"], dtype=np.float64)
            values = np.asarray(reference["values"], dtype=np.float64)
        if not np.array_equal(stored_points, points):
            raise ValueError("reference coordinates do not match the L-shaped evaluation grid")
        return points, values
