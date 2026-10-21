"""Plain 6x128 tanh network and the unscaled gPINN objective."""

from __future__ import annotations

import torch
from torch import nn

from .cases import PoissonCase


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiate while retaining the graph for third derivatives."""

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


class MLP(nn.Module):
    """The required 6-hidden-layer, width-128, float64 tanh network.

    The Dirichlet transform follows the original gPINN 1D Poisson code:
    a linear lift supplies the non-zero endpoint values and the product of two
    tanh factors makes the trainable correction vanish at both endpoints.
    """

    def __init__(self, case: PoissonCase) -> None:
        super().__init__()
        self.case = case
        self.lower = float(case.lower)
        self.upper = float(case.upper)
        self.hidden_layers = 6
        self.width = 128
        self.hidden = nn.ModuleList(
            [nn.Linear(1, self.width)]
            + [nn.Linear(self.width, self.width) for _ in range(5)]
        )
        self.output = nn.Linear(self.width, 1)
        for layer in [*self.hidden, self.output]:
            # Official poisson_1d.py uses "Glorot uniform".  This is the
            # native-PyTorch equivalent; only the layer count/width differ.
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.to(dtype=torch.float64)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        value = x
        for layer in self.hidden:
            value = torch.tanh(layer(value))
        return self.output(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fraction = (x - self.lower) / (self.upper - self.lower)
        endpoints = torch.tensor(
            [[self.lower], [self.upper]], dtype=x.dtype, device=x.device
        )
        values = self.case.exact(endpoints)
        lift = (1.0 - fraction) * values[0:1] + fraction * values[1:2]
        correction = (
            torch.tanh(x - self.lower)
            * torch.tanh(self.upper - x)
            * self.raw(x)
        )
        return lift + correction


def pde_residual(model: MLP, x: torch.Tensor) -> torch.Tensor:
    """Return the raw PDE residual ``r=-u_xx-f``."""

    u = model(x)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)
    return -u_xx - model.case.forcing(x)


def residual_pair(model: MLP, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``r=-u_xx-f`` and the gradient enhancement ``r_x``."""

    residual = pde_residual(model, x)
    return residual, derivative(residual, x)


def objective_terms(
    model: MLP,
    x: torch.Tensor,
    gradient_weight: float,
) -> dict[str, torch.Tensor]:
    """Original raw gPINN loss: MSE(r) + weight*MSE(r_x)."""

    residual, residual_x = residual_pair(model, x)
    pde = torch.mean(residual**2)
    gradient = torch.mean(residual_x**2)
    return {
        "total": pde + float(gradient_weight) * gradient,
        "pde": pde,
        "gradient": gradient,
    }
