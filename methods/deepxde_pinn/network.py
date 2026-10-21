"""DeepXDE-compatible classical fully connected PINN network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from deepxde.nn.pytorch.nn import NN


class ClassicalMLP(NN):
    def __init__(
        self,
        input_bounds: Sequence[Sequence[float]],
        output_dim: int,
        *,
        width: int = 128,
        hidden_layers: int = 6,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")

        bounds = torch.as_tensor(input_bounds, dtype=torch.float32)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("input_bounds must have shape (input_dim, 2)")
        if torch.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("each input upper bound must exceed its lower bound")

        self.register_buffer("lower", bounds[:, 0])
        self.register_buffer("upper", bounds[:, 1])
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.hidden_depth = int(hidden_layers)

        input_dim = int(bounds.shape[0])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            layers = [torch.nn.Linear(input_dim, width)]
            layers.extend(
                torch.nn.Linear(width, width) for _ in range(hidden_layers - 1)
            )
            self.hidden_layers = torch.nn.ModuleList(layers)
            self.output_layer = torch.nn.Linear(width, output_dim)
            self._initialize()

    def _initialize(self) -> None:
        for layer in (*self.hidden_layers, self.output_layer):
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        return 2.0 * (inputs - self.lower) / (self.upper - self.lower) - 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_inputs = inputs
        features = self.normalize(inputs)
        if self._input_transform is not None:
            features = self._input_transform(features)

        hidden = features
        for layer in self.hidden_layers:
            hidden = torch.tanh(layer(hidden))
        outputs = self.output_layer(hidden)

        if self._output_transform is not None:
            outputs = self._output_transform(raw_inputs, outputs)
        return outputs
