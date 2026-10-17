"""Poisson benchmark on the L-shaped domain."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import SUITE_ROOT
from ..grids import lshape_grid
from ..references import load_reference
from .base import PDEProblem, TrainingData, latin_hypercube


def _lshape_boundary() -> np.ndarray:
    coordinate = -1.0 + (np.arange(128, dtype=np.float64) + 0.5) / 128.0
    positive = (np.arange(128, dtype=np.float64) + 0.5) / 128.0
    return np.vstack(
        (
            np.column_stack((np.full(128, -1.0), coordinate)),
            np.column_stack((np.full(128, -1.0), positive)),
            np.column_stack((coordinate, np.full(128, -1.0))),
            np.column_stack((positive, np.full(128, -1.0))),
            np.column_stack((coordinate, np.full(128, 1.0))),
            np.column_stack((np.full(128, 1.0), coordinate)),
            np.column_stack((np.zeros(128), positive)),
            np.column_stack((positive, np.zeros(128))),
        )
    )


class LShapeProblem(PDEProblem):
    def sample_training(self, seed: int) -> TrainingData:
        accepted: list[np.ndarray] = []
        collected = 0
        batch = 0
        while collected < self.config.sampling.residual:
            candidates = latin_hypercube(
                self.config.domain,
                max(2048, self.config.sampling.residual - collected),
                seed + batch,
            )
            valid = candidates[
                ~((candidates[:, 0] >= 0.0) & (candidates[:, 1] >= 0.0))
            ]
            accepted.append(valid)
            collected += valid.shape[0]
            batch += 1
        residual = np.vstack(accepted)[: self.config.sampling.residual]
        boundary = _lshape_boundary()
        return TrainingData(
            residual=residual,
            boundary_points=boundary,
            boundary_values=np.zeros((boundary.shape[0], 1), dtype=np.float64),
        )

    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points)
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(points)
            with tf.GradientTape() as inner:
                inner.watch(points)
                values = model(points)
            first = inner.gradient(
                values, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            first_x = first[:, 0:1]
            first_y = first[:, 1:2]
        second_x = outer.gradient(
            first_x, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 0:1]
        second_y = outer.gradient(
            first_y, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 1:2]
        del outer
        return -(second_x + second_y) - tf.cast(1.0, values.dtype)

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = lshape_grid()
        stored_points, values, _ = load_reference(
            "lshape_poisson", root=SUITE_ROOT / "references"
        )
        if not np.array_equal(stored_points, points):
            raise ValueError("reference coordinates do not match the L-shaped evaluation grid")
        return points, values
