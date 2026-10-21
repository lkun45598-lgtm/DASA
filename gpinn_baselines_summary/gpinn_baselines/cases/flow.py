"""Steady incompressible lid-driven cavity, equations (17)-(18)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..autodiff import partial, second
from ..sampling import rectangle_boundary
from .base import PDECase


class LidDrivenCavity(PDECase):
    coordinate_names = ("x", "y")
    output_names = ("u", "v", "p")

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        boundary = rectangle_boundary(
            self.config["sampling"]["n_boundary"],
            (0.0, 0.0),
            (1.0, 1.0),
            rng,
        )
        target = np.zeros((len(boundary), 2), dtype=np.float64)
        target[np.isclose(boundary[:, 1], 1.0), 0] = 1.0
        values["boundary_points"] = boundary
        values["boundary_values"] = target
        values["gauge_points"] = values["residual"][: min(512, len(values["residual"]))].copy()
        return values

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = model(points)
        u, v, p = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
        ux, uy = partial(u, points, 0), partial(u, points, 1)
        vx, vy = partial(v, points, 0), partial(v, points, 1)
        px, py = partial(p, points, 0), partial(p, points, 1)
        inv_re = 1.0 / float(self.parameters["reynolds"])
        momentum_u = u * ux + v * uy + px - inv_re * (
            second(u, points, 0) + second(u, points, 1)
        )
        momentum_v = u * vx + v * vy + py - inv_re * (
            second(v, points, 0) + second(v, points, 1)
        )
        continuity = ux + vy
        return {
            "momentum_u": momentum_u,
            "momentum_v": momentum_v,
            "continuity": continuity,
        }

    def constraint_terms(self, model: Callable, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        boundary_prediction = model(batch["boundary_points"])[:, :2]
        terms = {
            "boundary": torch.mean(
                (boundary_prediction - batch["boundary_values"]) ** 2
            )
        }
        pressure = model(batch["gauge_points"])[:, 2:3]
        terms["pressure_gauge"] = torch.mean(pressure) ** 2
        return terms
