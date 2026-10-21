"""SciPy L-BFGS bridge that optimizes only explicitly listed TensorFlow variables."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
import tensorflow as tf


def flatten_variables(variables: Sequence[tf.Variable]) -> np.ndarray:
    if not variables:
        raise ValueError("at least one variable is required")
    return np.concatenate([variable.numpy().reshape(-1) for variable in variables]).astype(
        np.float64
    )


def assign_flat_variables(values: np.ndarray, variables: Sequence[tf.Variable]) -> None:
    values = np.asarray(values)
    offset = 0
    for variable in variables:
        size = int(np.prod(variable.shape))
        block = values[offset : offset + size].reshape(variable.shape)
        variable.assign(tf.cast(block, variable.dtype))
        offset += size
    if offset != values.size:
        raise ValueError("flat parameter vector has the wrong size")


def minimize_lbfgs(
    loss_function: Callable[[], tf.Tensor],
    variables: Sequence[tf.Variable],
    *,
    max_iterations: int,
) -> OptimizeResult:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    variables = list(variables)

    def value_and_gradient(flat_values: np.ndarray) -> tuple[float, np.ndarray]:
        assign_flat_variables(flat_values, variables)
        with tf.GradientTape() as tape:
            loss = loss_function()
        gradients = tape.gradient(loss, variables)
        if any(gradient is None for gradient in gradients):
            raise ValueError("L-BFGS loss is disconnected from a model variable")
        flat_gradient = np.concatenate(
            [gradient.numpy().reshape(-1) for gradient in gradients]
        ).astype(np.float64)
        return float(loss.numpy().item()), flat_gradient

    result = minimize(
        value_and_gradient,
        flatten_variables(variables),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "maxls": 50, "ftol": 1e-12},
    )
    assign_flat_variables(result.x, variables)
    return result
