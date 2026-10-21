"""The common 6 x 128 tanh network used by every baseline."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class MLP(nn.Module):
    """Fully-connected tanh network with affine input normalization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lower: list[float] | tuple[float, ...],
        upper: list[float] | tuple[float, ...],
        hidden_layers: int = 6,
        width: int = 128,
    ) -> None:
        super().__init__()
        if hidden_layers != 6 or width != 128:
            raise ValueError(
                "The benchmark contract requires exactly 6 hidden layers of width 128."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.width = width
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float64))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float64))

        layers: list[nn.Linear] = [nn.Linear(input_dim, width)]
        layers.extend(nn.Linear(width, width) for _ in range(hidden_layers - 1))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(width, output_dim)
        self.activation = nn.Tanh()
        self._output_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self.reset_parameters()
        # The benchmark contract is float64 even if the caller did not alter
        # PyTorch's process-wide default dtype.
        self.to(dtype=torch.float64)

    def reset_parameters(self) -> None:
        for layer in [*self.hidden, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_output_transform(
        self, transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        """Install a case-specific hard-constraint transform."""

        self._output_transform = transform

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (inputs - self.lower) / (self.upper - self.lower) - 1.0
        for layer in self.hidden:
            z = self.activation(layer(z))
        output = self.output(z)
        if self._output_transform is not None:
            output = self._output_transform(inputs, output)
        return output


def architecture_description(model: MLP) -> str:
    return f"{model.input_dim}-[128x6]-{model.output_dim}, tanh, Xavier-normal"
