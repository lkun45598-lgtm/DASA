"""Problem interface and deterministic sampling helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import tensorflow as tf

from ..config import CaseConfig


@dataclass(frozen=True)
class TrainingData:
    residual: np.ndarray
    boundary_points: np.ndarray | None = None
    boundary_values: np.ndarray | None = None
    initial_points: np.ndarray | None = None
    initial_values: np.ndarray | None = None
    gauge_points: np.ndarray | None = None
    gauge_values: np.ndarray | None = None


def latin_hypercube(
    domain: tuple[tuple[float, float], ...], count: int, seed: int
) -> np.ndarray:
    unit = qmc.LatinHypercube(d=len(domain), seed=seed).random(count)
    lower = np.asarray([axis[0] for axis in domain], dtype=np.float64)
    upper = np.asarray([axis[1] for axis in domain], dtype=np.float64)
    return qmc.scale(unit, lower, upper)


class PDEProblem(ABC):
    def __init__(self, config: CaseConfig) -> None:
        self.config = config

    @abstractmethod
    def sample_training(self, seed: int) -> TrainingData:
        raise NotImplementedError

    @abstractmethod
    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def boundary_prediction(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        return model(points)

    def boundary_loss(self, prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(prediction - target))

    def initial_prediction(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        return model(points)

    def extra_loss(self, model: tf.keras.Model, data: TrainingData) -> tf.Tensor:
        del model, data
        return tf.constant(0.0, dtype=tf.float32)
