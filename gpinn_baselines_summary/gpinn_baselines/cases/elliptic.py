"""Equations (10)-(16) and the L-shaped Poisson equation (25)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..autodiff import laplacian, second
from ..sampling import lshape_boundary, lshape_interior
from .base import AnalyticDirichlet1D, AnalyticDirichlet2D, PDECase


class HardConstrainedPoisson1D(AnalyticDirichlet1D):
    """Use the original gPINN Poisson-style exact Dirichlet transform."""

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        # Boundary values are satisfied identically by output_transform.
        return PDECase.sample_training(self, rng)

    def output_transform(
        self, points: torch.Tensor, raw_output: torch.Tensor
    ) -> torch.Tensor:
        x = points[:, 0:1]
        lower, upper = self.lower[0], self.upper[0]
        span = upper - lower
        fraction = (x - lower) / span
        left = self._exact_torch(torch.full_like(points, lower))
        right = self._exact_torch(torch.full_like(points, upper))
        boundary_interpolant = (1.0 - fraction) * left + fraction * right
        correction = torch.tanh(x - lower) * torch.tanh(upper - x) * raw_output
        return boundary_interpolant + correction


class Poisson1DBoundaryLayer(HardConstrainedPoisson1D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0:1]
        return 0.1 * torch.sin(5.0 * torch.pi * x) + torch.exp(
            (0.01 - (x + 1.0)) / 0.01
        )

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0:1]
        return 0.1 * np.sin(5.0 * np.pi * x) + np.exp((0.01 - (x + 1.0)) / 0.01)

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        x = points[:, 0:1]
        exponential = torch.exp((0.01 - (x + 1.0)) / 0.01)
        forcing = 0.1 * (5.0 * torch.pi) ** 2 * torch.sin(5.0 * torch.pi * x) - 1.0e4 * exponential
        return {"poisson": -second(u, points, 0) - forcing}


class Poisson1DSteep(HardConstrainedPoisson1D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0:1]
        return 0.1 * torch.sin(8.0 * torch.pi * x) + torch.tanh(80.0 * x)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x = points[:, 0:1]
        return 0.1 * np.sin(8.0 * np.pi * x) + np.tanh(80.0 * x)

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        x = points[:, 0:1]
        tanh = torch.tanh(80.0 * x)
        sech2 = 1.0 - tanh**2
        forcing = 0.1 * (8.0 * torch.pi) ** 2 * torch.sin(8.0 * torch.pi * x) + 12800.0 * tanh * sech2
        return {"poisson": -second(u, points, 0) - forcing}


class Helmholtz1D(AnalyticDirichlet1D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        return torch.sin(20.0 * torch.pi * points[:, 0:1])

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        return np.sin(20.0 * np.pi * points[:, 0:1])

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        x = points[:, 0:1]
        q = float(self.parameters["q"])
        forcing = ((20.0 * torch.pi) ** 2 + q) * torch.sin(20.0 * torch.pi * x)
        return {"helmholtz": -second(u, points, 0) + q * u - forcing}


class Poisson2DPeak(AnalyticDirichlet2D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0:1], points[:, 1:2]
        a = int(self.parameters["a"])
        return (16.0 * x * y * (1.0 - x) * (1.0 - y)) ** a

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0:1], points[:, 1:2]
        a = int(self.parameters["a"])
        return (16.0 * x * y * (1.0 - x) * (1.0 - y)) ** a

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        exact = self._exact_torch(points)
        forcing = -laplacian(exact, points, (0, 1))
        return {"poisson": -laplacian(u, points, (0, 1)) - forcing}


class Poisson2DSteep(AnalyticDirichlet2D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0:1], points[:, 1:2]
        return (0.1 * torch.sin(2.0 * torch.pi * x) + torch.tanh(10.0 * x)) * torch.sin(2.0 * torch.pi * y)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0:1], points[:, 1:2]
        return (0.1 * np.sin(2.0 * np.pi * x) + np.tanh(10.0 * x)) * np.sin(2.0 * np.pi * y)

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        exact = self._exact_torch(points)
        forcing = -laplacian(exact, points, (0, 1))
        return {"poisson": -laplacian(u, points, (0, 1)) - forcing}


class Helmholtz2D(AnalyticDirichlet2D):
    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0:1], points[:, 1:2]
        a1, a2 = float(self.parameters["a1"]), float(self.parameters["a2"])
        return torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0:1], points[:, 1:2]
        a1, a2 = float(self.parameters["a1"]), float(self.parameters["a2"])
        return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        exact = self._exact_torch(points)
        k = float(self.parameters["k"])
        q = laplacian(exact, points, (0, 1)) + k**2 * exact
        return {"helmholtz": laplacian(u, points, (0, 1)) + k**2 * u - q}


class PoissonLShape(PDECase):
    coordinate_names = ("x", "y")

    def sample_residual(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return lshape_interior(n, rng)

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        boundary = lshape_boundary(self.config["sampling"]["n_boundary"], rng)
        values["boundary_points"] = boundary
        values["boundary_values"] = np.zeros((len(boundary), 1), dtype=np.float64)
        return values

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        return {"poisson": -laplacian(u, points, (0, 1)) - 1.0}

    def make_test_points(self) -> np.ndarray:
        points = super().make_test_points()
        keep = ~((points[:, 0] > 0.0) & (points[:, 1] > 0.0))
        return points[keep]
