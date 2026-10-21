"""DeepXDE definitions for the one-dimensional time-dependent PDEs."""

from __future__ import annotations

from pathlib import Path

import deepxde as dde
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from methods.sa_pinn_aligned.dasa_sa.config import SUITE_ROOT
from methods.sa_pinn_aligned.dasa_sa.grids import time_1d_grid

from .base import BaseProblem, latin_hypercube


PROJECT_ROOT = SUITE_ROOT.parents[1]
ORIGINAL_DATASETS: dict[str, tuple[Path, str, str]] = {
    "burgers": (PROJECT_ROOT / "methods" / "sa_pinn" / "burgers" / "burgers_shock.mat", "t", "usol"),
    "allen_cahn": (PROJECT_ROOT / "methods" / "sa_pinn" / "allen_cahn" / "AC.mat", "tt", "uu"),
}


class PairedPointSetBC:
    """Value or derivative equality on two equally sized point sets."""

    def __init__(self, left: np.ndarray, right: np.ndarray, *, derivative_order: int) -> None:
        if left.shape != right.shape:
            raise ValueError("paired boundary point sets must have equal shape")
        if derivative_order not in {0, 1}:
            raise ValueError("only value and first-derivative periodicity are supported")
        self.points = np.asarray(np.vstack((left, right)), dtype=np.float32)
        self.derivative_order = derivative_order

    def collocation_points(self, _):
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        del X, aux_var
        middle = beg + (end - beg) // 2
        if self.derivative_order == 0:
            values = outputs[:, 0:1]
        else:
            values = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        return values[beg:middle] - values[middle:end]


class Time1DProblem(BaseProblem):
    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        case_id = self.config.case_id
        if case_id in {"burgers", "advection_diffusion"}:
            return -np.sin(np.pi * x)
        if case_id == "allen_cahn":
            return x**2 * np.cos(np.pi * x)
        if case_id == "diffusion_reaction":
            result = np.zeros_like(x)
            for frequency in range(1, 5):
                result += np.sin(frequency * x) / frequency
            return result + np.sin(8.0 * x) / 8.0
        if case_id == "discontinuous_advection":
            return (np.abs(x) <= self.config.parameters["pulse_half_width"]).astype(
                np.float64
            )
        raise ValueError(f"unsupported time case: {case_id}")

    def _diffusion_reaction_forcing(self, points: torch.Tensor) -> torch.Tensor:
        t, x = points[:, 0:1], points[:, 1:2]
        return torch.exp(-t) * (
            1.5 * torch.sin(2.0 * x)
            + (8.0 / 3.0) * torch.sin(3.0 * x)
            + (15.0 / 4.0) * torch.sin(4.0 * x)
            + (63.0 / 8.0) * torch.sin(8.0 * x)
        )

    def pde(self, points, values):
        u_t = dde.grad.jacobian(values, points, i=0, j=0)
        u_x = dde.grad.jacobian(values, points, i=0, j=1)
        u_xx = dde.grad.hessian(values, points, component=0, i=1, j=1)
        parameters = self.config.parameters
        case_id = self.config.case_id
        if case_id == "burgers":
            return u_t + values * u_x - parameters["viscosity"] * u_xx
        if case_id == "allen_cahn":
            return (
                u_t
                - parameters["diffusion"] * u_xx
                + parameters["reaction"] * values**3
                - parameters["reaction"] * values
            )
        if case_id == "diffusion_reaction":
            return u_t - parameters["diffusion"] * u_xx - self._diffusion_reaction_forcing(
                points
            )
        if case_id in {"advection_diffusion", "discontinuous_advection"}:
            return u_t + parameters["velocity"] * u_x - parameters["kappa"] * u_xx
        raise ValueError(f"unsupported time case: {case_id}")

    def build_data(self):
        residual = latin_hypercube(
            self.config.domain, self.training_counts["residual"], self.seed
        )
        t_domain, x_domain = self.config.domain
        boundary_conditions: list[tuple[object, str]] = []

        per_side = self.training_counts["boundary"]
        if per_side:
            boundary_t = latin_hypercube((t_domain,), per_side, self.seed + 2)[:, 0]
            left = np.column_stack((boundary_t, np.full(per_side, x_domain[0])))
            right = np.column_stack((boundary_t, np.full(per_side, x_domain[1])))
            if self.config.case_id == "allen_cahn":
                boundary_conditions.extend(
                    (
                        (
                            PairedPointSetBC(left, right, derivative_order=0),
                            "boundary_periodic_value",
                        ),
                        (
                            PairedPointSetBC(left, right, derivative_order=1),
                            "boundary_periodic_derivative",
                        ),
                    )
                )
            else:
                boundary = np.vstack((left, right))
                zeros = np.zeros((boundary.shape[0], 1), dtype=np.float64)
                boundary_conditions.append(
                    (dde.icbc.PointSetBC(boundary, zeros, component=0), "boundary_values")
                )

        initial_count = self.training_counts["initial"]
        if initial_count:
            initial_x = latin_hypercube((x_domain,), initial_count, self.seed + 1)
            initial_points = np.column_stack(
                (np.full(initial_count, t_domain[0]), initial_x[:, 0])
            )
            boundary_conditions.append(
                (
                    dde.icbc.PointSetBC(
                        initial_points, self.initial_numpy(initial_x), component=0
                    ),
                    "initial_values",
                )
            )

        geometry = dde.geometry.Rectangle(
            [t_domain[0], x_domain[0]], [t_domain[1], x_domain[1]]
        )
        return self.make_data(geometry, residual, boundary_conditions)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        t, x = points[:, 0:1], points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "diffusion_reaction":
            return np.exp(-t) * self.initial_numpy(x)
        if case_id == "advection_diffusion":
            return self._advection_diffusion_reference(points)
        if case_id == "discontinuous_advection":
            return self.initial_numpy(x - self.config.parameters["velocity"] * t)
        raise ValueError(f"{case_id} uses numerical reference truth")

    def _advection_diffusion_reference(self, points: np.ndarray) -> np.ndarray:
        velocity = self.config.parameters["velocity"]
        kappa = self.config.parameters["kappa"]
        spatial_points = 2049
        x_grid = np.linspace(-1.0, 1.0, spatial_points, dtype=np.float64)
        spacing = x_grid[1] - x_grid[0]
        interior_size = spatial_points - 2

        rows: list[int] = []
        columns: list[int] = []
        entries: list[float] = []
        weight_cache: dict[tuple[int, ...], np.ndarray] = {}
        for grid_index in range(1, spatial_points - 1):
            if grid_index == 1:
                stencil = np.arange(0, 6)
            elif grid_index == spatial_points - 2:
                stencil = np.arange(spatial_points - 6, spatial_points)
            else:
                stencil = np.arange(grid_index - 2, grid_index + 3)
            offsets = tuple(int(value) for value in stencil - grid_index)
            if offsets not in weight_cache:
                vandermonde = np.asarray(offsets, dtype=np.float64)[None, :] ** np.arange(
                    len(offsets), dtype=np.float64
                )[:, None]
                first_target = np.zeros(len(offsets), dtype=np.float64)
                first_target[1] = 1.0
                second_target = np.zeros(len(offsets), dtype=np.float64)
                second_target[2] = 2.0
                first = np.linalg.solve(vandermonde, first_target) / spacing
                second = np.linalg.solve(vandermonde, second_target) / spacing**2
                weight_cache[offsets] = kappa * second - velocity * first
            for column, weight in zip(stencil, weight_cache[offsets]):
                if 0 < column < spatial_points - 1:
                    rows.append(grid_index - 1)
                    columns.append(int(column - 1))
                    entries.append(float(weight))
        operator = csr_matrix(
            (entries, (rows, columns)), shape=(interior_size, interior_size)
        )

        initial = -np.sin(np.pi * x_grid[1:-1])
        times = np.unique(points[:, 0])
        interior_solutions = expm_multiply(
            operator,
            initial,
            start=0.0,
            stop=float(times[-1]),
            num=times.size,
            endpoint=True,
        )
        solutions = np.zeros((times.size, spatial_points), dtype=np.float64)
        solutions[:, 1:-1] = interior_solutions
        result = np.empty((points.shape[0], 1), dtype=np.float64)
        for time_index, time in enumerate(times):
            selected = points[:, 0] == time
            result[selected, 0] = CubicSpline(x_grid, solutions[time_index])(
                points[selected, 1]
            )
        result[points[:, 0] == 0.0] = self.initial_numpy(
            points[points[:, 0] == 0.0, 1:2]
        )
        result[np.isin(points[:, 1], (-1.0, 1.0))] = 0.0
        return result

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.case_id in ORIGINAL_DATASETS:
            path, time_key, value_key = ORIGINAL_DATASETS[self.config.case_id]
            data = loadmat(path)
            x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
            t = np.asarray(data[time_key], dtype=np.float64).reshape(-1)
            field = np.asarray(data[value_key], dtype=np.float64)
            if field.shape != (x.size, t.size):
                raise ValueError(f"invalid original dataset shape for {self.config.case_id}")
            tt, xx = np.meshgrid(t, x, indexing="ij")
            return np.column_stack((tt.ravel(), xx.ravel())), field.T.reshape(-1, 1)
        points = time_1d_grid(self.config.domain)
        return points, self.exact_numpy(points)
