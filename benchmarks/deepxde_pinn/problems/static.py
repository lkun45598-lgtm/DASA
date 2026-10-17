"""DeepXDE definitions for the regular one- and two-dimensional static PDEs."""

from __future__ import annotations

import math

import deepxde as dde
import numpy as np
import torch

from benchmarks.dasa_aligned_sa.dasa_sa.grids import regular_2d_grid, static_1d_grid

from .base import BaseProblem, latin_hypercube


def _edge_points(
    domain: tuple[tuple[float, float], tuple[float, float]], count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = [count // 4 + (index < count % 4) for index in range(4)]
    x_low, x_high = domain[0]
    y_low, y_high = domain[1]
    return np.vstack(
        (
            np.column_stack((rng.uniform(x_low, x_high, counts[0]), np.full(counts[0], y_low))),
            np.column_stack((rng.uniform(x_low, x_high, counts[1]), np.full(counts[1], y_high))),
            np.column_stack((np.full(counts[2], x_low), rng.uniform(y_low, y_high, counts[2]))),
            np.column_stack((np.full(counts[3], x_high), rng.uniform(y_low, y_high, counts[3]))),
        )
    )


class Static1DProblem(BaseProblem):
    def exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0:1]
        case_id = self.config.case_id
        if case_id == "poisson_1d_boundary":
            epsilon = self.config.parameters["layer_scale"]
            frequency = self.config.parameters["frequency"]
            return 0.1 * torch.sin(frequency * math.pi * x) + torch.exp(
                (epsilon - (x + 1.0)) / epsilon
            )
        if case_id == "poisson_1d_steep":
            frequency = self.config.parameters["frequency"]
            steepness = self.config.parameters["steepness"]
            return 0.1 * torch.sin(frequency * math.pi * x) + torch.tanh(steepness * x)
        if case_id == "helmholtz_1d":
            return torch.sin(self.config.parameters["frequency"] * math.pi * x)
        raise ValueError(f"unsupported static 1D case: {case_id}")

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)[:, 0:1]
        case_id = self.config.case_id
        if case_id == "poisson_1d_boundary":
            epsilon = self.config.parameters["layer_scale"]
            frequency = self.config.parameters["frequency"]
            return 0.1 * np.sin(frequency * np.pi * x) + np.exp(
                (epsilon - (x + 1.0)) / epsilon
            )
        if case_id == "poisson_1d_steep":
            frequency = self.config.parameters["frequency"]
            steepness = self.config.parameters["steepness"]
            return 0.1 * np.sin(frequency * np.pi * x) + np.tanh(steepness * x)
        if case_id == "helmholtz_1d":
            return np.sin(self.config.parameters["frequency"] * np.pi * x)
        raise ValueError(f"unsupported static 1D case: {case_id}")

    def forcing_torch(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0:1]
        case_id = self.config.case_id
        if case_id == "poisson_1d_boundary":
            epsilon = self.config.parameters["layer_scale"]
            omega = self.config.parameters["frequency"] * math.pi
            exponential = torch.exp((epsilon - (x + 1.0)) / epsilon)
            return 0.1 * omega**2 * torch.sin(omega * x) - exponential / epsilon**2
        if case_id == "poisson_1d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            tanh = torch.tanh(steepness * x)
            return 0.1 * omega**2 * torch.sin(omega * x) + (
                2.0 * steepness**2 * (1.0 - tanh.square()) * tanh
            )
        if case_id == "helmholtz_1d":
            omega = self.config.parameters["frequency"] * math.pi
            q_value = self.config.parameters["q"]
            return (omega**2 + q_value) * torch.sin(omega * x)
        raise ValueError(f"unsupported static 1D case: {case_id}")

    def pde(self, points, values):
        second = dde.grad.hessian(values, points, component=0, i=0, j=0)
        if self.config.case_id == "helmholtz_1d":
            operator = -second + self.config.parameters["q"] * values
        else:
            operator = -second
        return operator - self.forcing_torch(points)

    def build_data(self):
        low, high = self.config.domain[0]
        geometry = dde.geometry.Interval(low, high)
        residual = latin_hypercube(
            self.config.domain, self.training_counts["residual"], self.seed
        )
        boundary = np.array([[low], [high]], dtype=np.float64)
        condition = dde.icbc.PointSetBC(boundary, self.exact_numpy(boundary), component=0)
        if self.config.case_id == "helmholtz_1d":
            self.metadata["scientific_status"] = "amplitude_underdetermined"
        return self.make_data(geometry, residual, [(condition, "boundary_values")])

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = static_1d_grid(self.config.domain[0])
        return points, self.exact_numpy(points)


class Static2DProblem(BaseProblem):
    def exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0:1], points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            return 2.0 ** (4.0 * a) * x**a * y**a * (1.0 - x) ** a * (1.0 - y) ** a
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            return (0.1 * torch.sin(omega * x) + torch.tanh(steepness * x)) * torch.sin(
                omega * y
            )
        if case_id == "helmholtz_2d":
            return torch.sin(self.config.parameters["a1"] * math.pi * x) * torch.sin(
                self.config.parameters["a2"] * math.pi * y
            )
        raise ValueError(f"unsupported static 2D case: {case_id}")

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        x, y = points[:, 0:1], points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            return 2.0 ** (4.0 * a) * x**a * y**a * (1.0 - x) ** a * (1.0 - y) ** a
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * np.pi
            steepness = self.config.parameters["steepness"]
            return (0.1 * np.sin(omega * x) + np.tanh(steepness * x)) * np.sin(omega * y)
        if case_id == "helmholtz_2d":
            return np.sin(self.config.parameters["a1"] * np.pi * x) * np.sin(
                self.config.parameters["a2"] * np.pi * y
            )
        raise ValueError(f"unsupported static 2D case: {case_id}")

    @staticmethod
    def _peak_second(z: torch.Tensor, a: float) -> torch.Tensor:
        return a * (
            (a - 1.0) * z ** (a - 2.0) * (1.0 - z) ** (a - 2.0) * (1.0 - 2.0 * z) ** 2
            - 2.0 * z ** (a - 1.0) * (1.0 - z) ** (a - 1.0)
        )

    def forcing_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0:1], points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            scale = 2.0 ** (4.0 * a)
            hx, hy = x**a * (1.0 - x) ** a, y**a * (1.0 - y) ** a
            return -scale * (self._peak_second(x, a) * hy + hx * self._peak_second(y, a))
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            tanh = torch.tanh(steepness * x)
            x_part = 0.1 * torch.sin(omega * x) + tanh
            x_second = -0.1 * omega**2 * torch.sin(omega * x) - (
                2.0 * steepness**2 * (1.0 - tanh.square()) * tanh
            )
            y_part = torch.sin(omega * y)
            return -(x_second * y_part - omega**2 * x_part * y_part)
        if case_id == "helmholtz_2d":
            a1, a2, k = (
                self.config.parameters["a1"],
                self.config.parameters["a2"],
                self.config.parameters["k"],
            )
            return (-(a1 * math.pi) ** 2 - (a2 * math.pi) ** 2 + k**2) * self.exact_torch(
                points
            )
        raise ValueError(f"unsupported static 2D case: {case_id}")

    def pde(self, points, values):
        laplacian = dde.grad.hessian(values, points, component=0, i=0, j=0) + dde.grad.hessian(
            values, points, component=0, i=1, j=1
        )
        if self.config.case_id == "helmholtz_2d":
            operator = laplacian + self.config.parameters["k"] ** 2 * values
        else:
            operator = -laplacian
        return operator - self.forcing_torch(points)

    def build_data(self):
        geometry = dde.geometry.Rectangle(
            [axis[0] for axis in self.config.domain],
            [axis[1] for axis in self.config.domain],
        )
        residual_count = self.training_counts["residual"]
        if self.config.sampling.residual_grid and "residual" not in self.sampling_overrides:
            nx, ny = self.config.sampling.residual_grid
            x = np.linspace(*self.config.domain[0], nx, dtype=np.float64)
            y = np.linspace(*self.config.domain[1], ny, dtype=np.float64)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            residual = np.column_stack((xx.ravel(), yy.ravel()))
        else:
            residual = latin_hypercube(self.config.domain, residual_count, self.seed)
        boundary = _edge_points(
            self.config.domain, self.training_counts["boundary"], self.seed + 1
        )
        condition = dde.icbc.PointSetBC(boundary, self.exact_numpy(boundary), component=0)
        return self.make_data(geometry, residual, [(condition, "boundary_values")])

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = regular_2d_grid(self.config.domain)
        return points, self.exact_numpy(points)

