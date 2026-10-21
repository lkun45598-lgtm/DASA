"""The two benchmark metrics requested for every experiment."""

from __future__ import annotations

import numpy as np


def _paired_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {truth.shape} and {prediction.shape}"
        )
    return truth, prediction


def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _paired_arrays(y_true, y_pred)
    return float(np.linalg.norm(truth - prediction) / np.linalg.norm(truth))


def maximum_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _paired_arrays(y_true, y_pred)
    return float(np.max(np.abs(truth - prediction)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "l2_relative_error": l2_relative_error(y_true, y_pred),
        "maximum_pointwise_error": maximum_pointwise_error(y_true, y_pred),
    }
