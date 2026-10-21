"""Steady two-dimensional lid-driven cavity benchmark."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import SUITE_ROOT
from .base import PDEProblem, TrainingData


class CavityProblem(PDEProblem):
    def sample_training(self, seed: int) -> TrainingData:
        del seed
        x = np.linspace(0.0, 1.0, 52, dtype=np.float64)[1:-1]
        y = np.linspace(0.0, 1.0, 52, dtype=np.float64)[1:-1]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        residual = np.column_stack((xx.ravel(), yy.ravel()))

        edge_coordinate = (np.arange(128, dtype=np.float64) + 0.5) / 128.0
        bottom = np.column_stack((edge_coordinate, np.zeros(128)))
        top = np.column_stack((edge_coordinate, np.ones(128)))
        left = np.column_stack((np.zeros(128), edge_coordinate))
        right = np.column_stack((np.ones(128), edge_coordinate))
        boundary_points = np.vstack((bottom, top, left, right))
        boundary_values = np.zeros((512, 2), dtype=np.float64)
        boundary_values[128:256, 0] = 1.0
        return TrainingData(
            residual=residual,
            boundary_points=boundary_points,
            boundary_values=boundary_values,
            gauge_points=np.array([[0.5, 0.5]], dtype=np.float64),
            gauge_values=np.array([[0.0]], dtype=np.float64),
        )

    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points)
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(points)
            with tf.GradientTape(persistent=True) as inner:
                inner.watch(points)
                fields = model(points)
                u = fields[:, 0:1]
                v = fields[:, 1:2]
                pressure = fields[:, 2:3]
            grad_u = inner.gradient(
                u, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            grad_v = inner.gradient(
                v, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            grad_p = inner.gradient(
                pressure, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            del inner
            u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
            v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        u_xx = outer.gradient(
            u_x, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 0:1]
        u_yy = outer.gradient(
            u_y, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 1:2]
        v_xx = outer.gradient(
            v_x, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 0:1]
        v_yy = outer.gradient(
            v_y, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 1:2]
        del outer

        inverse_re = tf.cast(1.0 / self.config.parameters["reynolds"], fields.dtype)
        momentum_u = (
            u * u_x + v * u_y + grad_p[:, 0:1] - inverse_re * (u_xx + u_yy)
        )
        momentum_v = (
            u * v_x + v * v_y + grad_p[:, 1:2] - inverse_re * (v_xx + v_yy)
        )
        continuity = u_x + v_y
        return tf.concat((momentum_u, momentum_v, continuity), axis=1)

    def boundary_prediction(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        return model(points)[:, 0:2]

    def extra_loss(self, model: tf.keras.Model, data: TrainingData) -> tf.Tensor:
        if data.gauge_points is None or data.gauge_values is None:
            raise ValueError("cavity training data is missing the pressure gauge")
        points = tf.convert_to_tensor(data.gauge_points, dtype=tf.float32)
        target = tf.convert_to_tensor(data.gauge_values, dtype=tf.float32)
        pressure = model(points)[:, 2:3]
        weight = self.config.parameters["pressure_gauge_weight"]
        return tf.cast(weight, pressure.dtype) * tf.reduce_mean(tf.square(pressure - target))

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        values = np.loadtxt(
            SUITE_ROOT / "data" / "cavity_reference_u.csv", delimiter=","
        )
        if values.shape != (100, 100):
            raise ValueError("invalid cavity reference dataset shape")
        return points, values.reshape(-1, 1)
