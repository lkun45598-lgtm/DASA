"""Regular two-dimensional Poisson and Helmholtz benchmark definitions."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from ..grids import regular_2d_grid
from .base import PDEProblem, TrainingData, latin_hypercube


def _edge_points(
    domain: tuple[tuple[float, float], tuple[float, float]], count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = [count // 4 + (index < count % 4) for index in range(4)]
    x_low, x_high = domain[0]
    y_low, y_high = domain[1]
    bottom_x = rng.uniform(x_low, x_high, counts[0])
    top_x = rng.uniform(x_low, x_high, counts[1])
    left_y = rng.uniform(y_low, y_high, counts[2])
    right_y = rng.uniform(y_low, y_high, counts[3])
    return np.vstack(
        (
            np.column_stack((bottom_x, np.full(counts[0], y_low))),
            np.column_stack((top_x, np.full(counts[1], y_high))),
            np.column_stack((np.full(counts[2], x_low), left_y)),
            np.column_stack((np.full(counts[3], x_high), right_y)),
        )
    )


class Static2DProblem(PDEProblem):
    def exact_tf(self, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        x = points[:, 0:1]
        y = points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            return 2.0 ** (4.0 * a) * (
                x**a * y**a * (1.0 - x) ** a * (1.0 - y) ** a
            )
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            return (0.1 * tf.sin(omega * x) + tf.tanh(steepness * x)) * tf.sin(
                omega * y
            )
        if case_id == "helmholtz_2d":
            a1 = self.config.parameters["a1"]
            a2 = self.config.parameters["a2"]
            return tf.sin(a1 * math.pi * x) * tf.sin(a2 * math.pi * y)
        raise ValueError(f"unsupported static 2D case: {case_id}")

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        x = points[:, 0:1]
        y = points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            return 2.0 ** (4.0 * a) * (
                x**a * y**a * (1.0 - x) ** a * (1.0 - y) ** a
            )
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * np.pi
            steepness = self.config.parameters["steepness"]
            return (0.1 * np.sin(omega * x) + np.tanh(steepness * x)) * np.sin(
                omega * y
            )
        if case_id == "helmholtz_2d":
            a1 = self.config.parameters["a1"]
            a2 = self.config.parameters["a2"]
            return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)
        raise ValueError(f"unsupported static 2D case: {case_id}")

    @staticmethod
    def _peak_second(z: tf.Tensor, a: float) -> tf.Tensor:
        return a * (
            (a - 1.0)
            * z ** (a - 2.0)
            * (1.0 - z) ** (a - 2.0)
            * (1.0 - 2.0 * z) ** 2
            - 2.0 * z ** (a - 1.0) * (1.0 - z) ** (a - 1.0)
        )

    def forcing_tf(self, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        x = points[:, 0:1]
        y = points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "poisson_2d_peak":
            a = self.config.parameters["a"]
            scale = 2.0 ** (4.0 * a)
            hx = x**a * (1.0 - x) ** a
            hy = y**a * (1.0 - y) ** a
            return -scale * (self._peak_second(x, a) * hy + hx * self._peak_second(y, a))
        if case_id == "poisson_2d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            tanh = tf.tanh(steepness * x)
            x_part = 0.1 * tf.sin(omega * x) + tanh
            x_second = -0.1 * omega**2 * tf.sin(omega * x) - (
                2.0 * steepness**2 * (1.0 - tf.square(tanh)) * tanh
            )
            y_part = tf.sin(omega * y)
            return -(x_second * y_part - omega**2 * x_part * y_part)
        if case_id == "helmholtz_2d":
            exact = self.exact_tf(points)
            a1 = self.config.parameters["a1"]
            a2 = self.config.parameters["a2"]
            k = self.config.parameters["k"]
            return (-(a1 * math.pi) ** 2 - (a2 * math.pi) ** 2 + k**2) * exact
        raise ValueError(f"unsupported static 2D case: {case_id}")

    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(points)
            with tf.GradientTape() as inner:
                inner.watch(points)
                values = model(points)
            first = inner.gradient(values, points)
            first_x = first[:, 0:1]
            first_y = first[:, 1:2]
        second_x = outer.gradient(first_x, points)[:, 0:1]
        second_y = outer.gradient(first_y, points)[:, 1:2]
        del outer
        laplacian = second_x + second_y
        if self.config.case_id == "helmholtz_2d":
            operator = laplacian + self.config.parameters["k"] ** 2 * values
        else:
            operator = -laplacian
        return operator - self.forcing_tf(points)

    def sample_training(self, seed: int) -> TrainingData:
        if self.config.sampling.residual_grid:
            nx, ny = self.config.sampling.residual_grid
            x = np.linspace(*self.config.domain[0], nx, dtype=np.float64)
            y = np.linspace(*self.config.domain[1], ny, dtype=np.float64)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            residual = np.column_stack((xx.ravel(), yy.ravel()))
        else:
            residual = latin_hypercube(
                self.config.domain, self.config.sampling.residual, seed
            )
        boundary_points = _edge_points(
            self.config.domain, self.config.sampling.boundary, seed + 1
        )
        return TrainingData(
            residual=residual,
            boundary_points=boundary_points,
            boundary_values=self.exact_numpy(boundary_points),
        )

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = regular_2d_grid(self.config.domain)
        return points, self.exact_numpy(points)
