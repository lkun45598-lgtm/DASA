"""One-dimensional Poisson and Helmholtz benchmark definitions."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from ..grids import static_1d_grid
from .base import PDEProblem, TrainingData, latin_hypercube


class Static1DProblem(PDEProblem):
    def boundary_loss(self, prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        squared_error = tf.square(prediction - target)
        if self.config.case_id in {"poisson_1d_boundary", "poisson_1d_steep"}:
            return tf.reduce_sum(squared_error)
        return tf.reduce_mean(squared_error)

    def exact_tf(self, points: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(points, dtype=tf.float32)[:, 0:1]
        case_id = self.config.case_id
        if case_id == "poisson_1d_boundary":
            epsilon = self.config.parameters["layer_scale"]
            frequency = self.config.parameters["frequency"]
            return 0.1 * tf.sin(frequency * math.pi * x) + tf.exp(
                (epsilon - (x + 1.0)) / epsilon
            )
        if case_id == "poisson_1d_steep":
            frequency = self.config.parameters["frequency"]
            steepness = self.config.parameters["steepness"]
            return 0.1 * tf.sin(frequency * math.pi * x) + tf.tanh(steepness * x)
        if case_id == "helmholtz_1d":
            return tf.sin(self.config.parameters["frequency"] * math.pi * x)
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

    def forcing_tf(self, points: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(points, dtype=tf.float32)[:, 0:1]
        case_id = self.config.case_id
        if case_id == "poisson_1d_boundary":
            epsilon = self.config.parameters["layer_scale"]
            omega = self.config.parameters["frequency"] * math.pi
            exponential = tf.exp((epsilon - (x + 1.0)) / epsilon)
            return 0.1 * omega**2 * tf.sin(omega * x) - exponential / epsilon**2
        if case_id == "poisson_1d_steep":
            omega = self.config.parameters["frequency"] * math.pi
            steepness = self.config.parameters["steepness"]
            tanh = tf.tanh(steepness * x)
            sech_squared = 1.0 - tf.square(tanh)
            return 0.1 * omega**2 * tf.sin(omega * x) + (
                2.0 * steepness**2 * sech_squared * tanh
            )
        if case_id == "helmholtz_1d":
            omega = self.config.parameters["frequency"] * math.pi
            q_value = self.config.parameters["q"]
            return (omega**2 + q_value) * tf.sin(omega * x)
        raise ValueError(f"unsupported static 1D case: {case_id}")

    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points, dtype=tf.float32)
        with tf.GradientTape() as outer:
            outer.watch(points)
            with tf.GradientTape() as inner:
                inner.watch(points)
                values = model(points)
            first = inner.gradient(values, points)
        second = outer.gradient(first, points)[:, 0:1]
        if self.config.case_id == "helmholtz_1d":
            operator = -second + self.config.parameters["q"] * values
        else:
            operator = -second
        return operator - self.forcing_tf(points)

    def sample_training(self, seed: int) -> TrainingData:
        domain = self.config.domain
        residual = latin_hypercube(domain, self.config.sampling.residual, seed)
        boundary_points = np.array([[domain[0][0]], [domain[0][1]]], dtype=np.float64)
        return TrainingData(
            residual=residual,
            boundary_points=boundary_points,
            boundary_values=self.exact_numpy(boundary_points),
        )

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        points = static_1d_grid(self.config.domain[0])
        return points, self.exact_numpy(points)
