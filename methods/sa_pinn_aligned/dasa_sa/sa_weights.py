"""Direct trainable point masks for the SA-PINNs minimax objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tensorflow as tf


def create_point_masks(counts: Mapping[str, int], seed: int) -> dict[str, tf.Variable]:
    masks: dict[str, tf.Variable] = {}
    for index, (name, count) in enumerate(sorted(counts.items())):
        if count <= 0:
            raise ValueError(f"mask count for {name} must be positive")
        initial = tf.random.stateless_uniform(
            (count, 1),
            seed=tf.constant([seed, index], dtype=tf.int32),
            dtype=tf.float32,
        )
        masks[name] = tf.Variable(initial, trainable=True, name=f"{name}_sa_mask")
    return masks


def minimax_gradients(
    tape: tf.GradientTape,
    loss: tf.Tensor,
    model_variables: Sequence[tf.Variable],
    mask_variables: Sequence[tf.Variable],
) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
    model_gradients = tape.gradient(loss, model_variables)
    raw_mask_gradients = tape.gradient(loss, mask_variables)
    if any(gradient is None for gradient in model_gradients):
        raise ValueError("loss is disconnected from one or more model variables")
    if any(gradient is None for gradient in raw_mask_gradients):
        raise ValueError("loss is disconnected from one or more SA masks")
    return list(model_gradients), [-gradient for gradient in raw_mask_gradients]
