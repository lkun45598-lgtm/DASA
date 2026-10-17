"""DeepXDE definition for the steady lid-driven cavity benchmark."""

from __future__ import annotations

import deepxde as dde
import numpy as np

from benchmarks.dasa_aligned_sa.dasa_sa.config import SUITE_ROOT

from .base import BaseProblem, latin_hypercube


def _cavity_boundary(count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = [count // 4 + (index < count % 4) for index in range(4)]
    edges: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for edge, edge_count in enumerate(counts):
        coordinate = (np.arange(edge_count, dtype=np.float64) + 0.5) / edge_count
        if edge == 0:
            points = np.column_stack((coordinate, np.zeros(edge_count)))
        elif edge == 1:
            points = np.column_stack((coordinate, np.ones(edge_count)))
        elif edge == 2:
            points = np.column_stack((np.zeros(edge_count), coordinate))
        else:
            points = np.column_stack((np.ones(edge_count), coordinate))
        values = np.zeros((edge_count, 2), dtype=np.float64)
        if edge == 1:
            values[:, 0] = 1.0
        edges.append(points)
        targets.append(values)
    return np.vstack(edges), np.vstack(targets)


class CavityProblem(BaseProblem):
    def pde(self, points, values):
        u, v, pressure = values[:, 0:1], values[:, 1:2], values[:, 2:3]
        u_x = dde.grad.jacobian(values, points, i=0, j=0)
        u_y = dde.grad.jacobian(values, points, i=0, j=1)
        v_x = dde.grad.jacobian(values, points, i=1, j=0)
        v_y = dde.grad.jacobian(values, points, i=1, j=1)
        p_x = dde.grad.jacobian(values, points, i=2, j=0)
        p_y = dde.grad.jacobian(values, points, i=2, j=1)
        u_xx = dde.grad.hessian(values, points, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(values, points, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(values, points, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(values, points, component=1, i=1, j=1)
        inverse_reynolds = 1.0 / self.config.parameters["reynolds"]
        return [
            u * u_x + v * u_y + p_x - inverse_reynolds * (u_xx + u_yy),
            u * v_x + v * v_y + p_y - inverse_reynolds * (v_xx + v_yy),
            u_x + v_y,
        ]

    def build_data(self):
        geometry = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
        if self.config.sampling.residual_grid and "residual" not in self.sampling_overrides:
            nx, ny = self.config.sampling.residual_grid
            x = np.linspace(0.0, 1.0, nx + 2, dtype=np.float64)[1:-1]
            y = np.linspace(0.0, 1.0, ny + 2, dtype=np.float64)[1:-1]
            xx, yy = np.meshgrid(x, y, indexing="xy")
            residual = np.column_stack((xx.ravel(), yy.ravel()))
        else:
            residual = latin_hypercube(
                self.config.domain, self.training_counts["residual"], self.seed
            )

        boundary, velocity = _cavity_boundary(self.training_counts["boundary"])
        velocity_bc = dde.icbc.PointSetBC(boundary, velocity, component=[0, 1])
        gauge = dde.icbc.PointSetBC(
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([[0.0]], dtype=np.float64),
            component=2,
        )
        return self.make_data(
            geometry,
            residual,
            [(velocity_bc, "boundary_velocity"), (gauge, "gauge_pressure")],
            residual_components=3,
        )

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        values = np.loadtxt(SUITE_ROOT / "data" / "cavity_reference_u.csv", delimiter=",")
        if values.shape != (100, 100):
            raise ValueError("invalid cavity reference dataset shape")
        return points, values.reshape(-1, 1)
