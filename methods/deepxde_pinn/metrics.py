"""The two metrics shared with the aligned SA-PINN experiments."""

from __future__ import annotations

import numpy as np


def _matching_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"metric arrays must have the same shape: {truth.shape} != {prediction.shape}"
        )
    return truth, prediction


def relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _matching_arrays(y_true, y_pred)
    denominator = np.linalg.norm(truth)
    if denominator == 0.0:
        raise ValueError("relative L2 error is undefined for an all-zero reference")
    return float(np.linalg.norm(truth - prediction) / denominator)


def maximum_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _matching_arrays(y_true, y_pred)
    return float(np.max(np.abs(truth - prediction)))

