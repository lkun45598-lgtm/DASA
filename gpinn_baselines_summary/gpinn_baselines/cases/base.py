"""Common case interface and loss assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch

from ..autodiff import residual_gradients
from ..sampling import lhs, rectangle_boundary


class PDECase(ABC):
    coordinate_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ("u",)

    def __init__(self, config: dict) -> None:
        self.config = config
        self.lower = tuple(float(v) for v in config["lower"])
        self.upper = tuple(float(v) for v in config["upper"])
        self.parameters = config["parameters"]

    @abstractmethod
    def residuals(
        self, model: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def residual_gradient_dims(self) -> tuple[int, ...]:
        return tuple(range(len(self.lower)))

    def sample_residual(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return lhs(n, self.lower, self.upper, rng)

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n = self.config["sampling"]
        return {"residual": self.sample_residual(n["n_residual"], rng)}

    def constraint_terms(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        if "boundary_points" in batch:
            prediction = model(batch["boundary_points"])
            target = batch["boundary_values"]
            terms["boundary"] = torch.mean((prediction - target) ** 2)
        if "initial_points" in batch:
            prediction = model(batch["initial_points"])
            target = batch["initial_values"]
            terms["initial"] = torch.mean((prediction - target) ** 2)
        return terms

    def physics_loss_terms(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        residual_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual = self.residuals(model, residual_points)
        pde_loss = torch.stack(
            [torch.mean(value**2) for value in residual.values()]
        ).mean()
        enhanced = residual_gradients(
            residual, residual_points, self.residual_gradient_dims()
        )
        gradient_loss = torch.stack(
            [torch.mean(value**2) for value in enhanced.values()]
        ).mean()
        return {"pde": pde_loss, "gradient": gradient_loss}

    def raw_loss_terms(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        residual_points: torch.Tensor,
        constraint_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        terms = self.physics_loss_terms(model, residual_points)
        terms.update(self.constraint_terms(model, constraint_batch))
        return terms

    def make_test_points(self) -> np.ndarray:
        shape = self.config["test_shape"]
        axes = [
            np.linspace(lo, hi, count, dtype=np.float64)
            for lo, hi, count in zip(self.lower, self.upper, shape)
        ]
        if len(axes) == 1:
            return axes[0][:, None]
        grids = np.meshgrid(*axes, indexing="xy")
        return np.column_stack([grid.ravel() for grid in grids])

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} uses a numerical reference")

    def reference_values(self, points: np.ndarray) -> np.ndarray:
        return self.exact_numpy(points)

    @property
    def reference_type(self) -> str:
        return self.config["reference_type"]


class AnalyticDirichlet1D(PDECase):
    coordinate_names = ("x",)

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        boundary = np.asarray([[self.lower[0]], [self.upper[0]]], dtype=np.float64)
        values["boundary_points"] = boundary
        values["boundary_values"] = self.exact_numpy(boundary)
        return values


class AnalyticDirichlet2D(PDECase):
    coordinate_names = ("x", "y")

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        boundary = rectangle_boundary(
            self.config["sampling"]["n_boundary"],
            (self.lower[0], self.lower[1]),
            (self.upper[0], self.upper[1]),
            rng,
        )
        values["boundary_points"] = boundary
        values["boundary_values"] = self.exact_numpy(boundary)
        return values
