"""Shared DeepXDE problem interface and deterministic sampling helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import deepxde as dde
import numpy as np
from scipy.stats import qmc

from benchmarks.dasa_aligned_sa.dasa_sa.config import CaseConfig


def latin_hypercube(
    domain: tuple[tuple[float, float], ...], count: int, seed: int
) -> np.ndarray:
    unit = qmc.LatinHypercube(d=len(domain), seed=seed).random(count)
    lower = np.asarray([axis[0] for axis in domain], dtype=np.float64)
    upper = np.asarray([axis[1] for axis in domain], dtype=np.float64)
    return qmc.scale(unit, lower, upper)


class BaseProblem(ABC):
    prediction_component = 0

    def __init__(
        self,
        config: CaseConfig,
        *,
        seed: int,
        sampling_overrides: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.sampling_overrides = dict(sampling_overrides or {})
        self.training_counts = {
            "residual": self._count("residual", config.sampling.residual),
            "boundary": self._count("boundary", config.sampling.boundary),
            "initial": self._count("initial", config.sampling.initial),
        }
        self.metadata: dict[str, Any] = {"provenance": dict(config.provenance)}
        self.loss_names: tuple[str, ...] = ()
        self.loss_weights: tuple[float, ...] = ()
        self.data = self.build_data()

    def _count(self, name: str, configured: int) -> int:
        value = int(self.sampling_overrides.get(name, configured))
        if value < 0:
            raise ValueError(f"sampling override {name} must be nonnegative")
        return value

    def make_data(
        self,
        geometry: dde.geometry.Geometry,
        residual_points: np.ndarray,
        boundary_conditions: list[tuple[Any, str]],
        *,
        residual_components: int = 1,
    ) -> dde.data.PDE:
        weights = self.config.optimizer.loss_weights
        self.loss_names = tuple(
            [f"residual_{index}" for index in range(residual_components)]
            + [name for _, name in boundary_conditions]
        )
        category_weights = {
            "boundary": weights["boundary"],
            "initial": weights["initial"],
            "gauge": self.config.parameters.get("pressure_gauge_weight", 1.0),
        }
        self.loss_weights = tuple(
            [weights["residual"]] * residual_components
            + [category_weights[name.split("_")[0]] for _, name in boundary_conditions]
        )
        return dde.data.PDE(
            geometry,
            self.pde,
            [condition for condition, _ in boundary_conditions],
            num_domain=0,
            num_boundary=0,
            train_distribution="pseudo",
            anchors=np.asarray(residual_points, dtype=np.float32),
        )

    @abstractmethod
    def build_data(self) -> dde.data.PDE:
        raise NotImplementedError

    @abstractmethod
    def pde(self, points, values):
        raise NotImplementedError

    @abstractmethod
    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def prediction_values(self, values: np.ndarray) -> np.ndarray:
        return values[:, self.prediction_component : self.prediction_component + 1]


EvaluationFunction = Callable[[], tuple[np.ndarray, np.ndarray]]

