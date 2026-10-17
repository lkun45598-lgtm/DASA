"""Modified MLP backbone and deterministic periodic feature embeddings."""

from __future__ import annotations

import math
from typing import Sequence

import tensorflow as tf


class ModifiedMLP(tf.keras.Model):
    """Six-hidden-layer modified MLP used by all aligned cases."""

    def __init__(
        self,
        input_bounds: Sequence[Sequence[float]],
        output_dim: int,
        *,
        width: int = 128,
        hidden_layers: int = 6,
        fourier: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__(name="modified_mlp")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        if fourier not in {"none", "periodic_unit", "periodic_pi"}:
            raise ValueError(f"unsupported Fourier embedding: {fourier}")
        bounds = tf.convert_to_tensor(input_bounds, dtype=tf.float32)
        if bounds.shape.rank != 2 or bounds.shape[1] != 2:
            raise ValueError("input_bounds must have shape (input_dim, 2)")

        self.input_bounds = bounds
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.hidden_depth = int(hidden_layers)
        self.fourier = fourier

        def dense(units: int, offset: int, activation: str | None = "tanh") -> tf.keras.layers.Dense:
            return tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed + offset),
                bias_initializer="zeros",
            )

        self.u_encoder = dense(width, 1)
        self.v_encoder = dense(width, 2)
        self.initial_hidden = dense(width, 3)
        self.mixing_layers = [
            dense(width, 4 + index) for index in range(hidden_layers - 1)
        ]
        self.output_layer = dense(output_dim, 4 + hidden_layers, activation=None)

    def embed(self, inputs: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(inputs, dtype=tf.float32)
        lower = self.input_bounds[:, 0]
        upper = self.input_bounds[:, 1]
        normalized = 2.0 * (values - lower) / (upper - lower) - 1.0
        if self.fourier == "none":
            return normalized
        if values.shape.rank != 2 or values.shape[-1] != 2:
            raise ValueError("periodic embeddings require inputs ordered as (t, x)")
        x = values[:, -1:]
        if self.fourier == "periodic_unit":
            angle = math.pi * x
        else:
            angle = x
        return tf.concat((normalized[:, :1], tf.cos(angle), tf.sin(angle)), axis=1)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        features = self.embed(inputs)
        u_branch = self.u_encoder(features)
        v_branch = self.v_encoder(features)
        hidden = self.initial_hidden(features)
        for layer in self.mixing_layers:
            gate = layer(hidden)
            hidden = (1.0 - gate) * u_branch + gate * v_branch
        return self.output_layer(hidden)
