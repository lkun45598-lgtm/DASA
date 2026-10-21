"""Uniform error metrics and saved-data reproducibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0.0:
        raise ZeroDivisionError("relative L2 is undefined for an all-zero reference")
    return float(np.linalg.norm((prediction - reference).ravel()) / denominator)


def max_absolute(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(prediction - reference)))


def scalar_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "relative_l2": relative_l2(prediction, reference),
        "max_abs": max_absolute(prediction, reference),
    }


def cavity_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    predicted_velocity = prediction[:, :2]
    error = predicted_velocity - reference[:, :2]
    joint_denominator = np.sqrt(np.sum(reference[:, 0] ** 2 + reference[:, 1] ** 2))
    joint_error = np.sqrt(np.sum(error[:, 0] ** 2 + error[:, 1] ** 2))
    pointwise = np.sqrt(np.sum(error**2, axis=1))
    return {
        "u_relative_l2": relative_l2(prediction[:, 0], reference[:, 0]),
        "u_max_abs": max_absolute(prediction[:, 0], reference[:, 0]),
        "v_relative_l2": relative_l2(prediction[:, 1], reference[:, 1]),
        "v_max_abs": max_absolute(prediction[:, 1], reference[:, 1]),
        "velocity_relative_l2": float(joint_error / joint_denominator),
        "velocity_max_pointwise_l2": float(np.max(pointwise)),
        "relative_l2": float(joint_error / joint_denominator),
        "max_abs": float(np.max(pointwise)),
        "pressure_predicted_mean": float(np.mean(prediction[:, 2])),
    }


def assert_finite_arrays(**arrays: np.ndarray) -> None:
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")


def recompute_saved_metrics(result_dir: Path) -> dict[str, float]:
    with np.load(result_dir / "prediction.npz") as prediction_data:
        prediction = prediction_data["prediction"]
    with np.load(result_dir / "reference.npz") as reference_data:
        reference = reference_data["reference"]
    with (result_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config["case"] == "lid_driven_cavity":
        return cavity_metrics(prediction, reference)
    return scalar_metrics(prediction[:, :1], reference[:, :1])
