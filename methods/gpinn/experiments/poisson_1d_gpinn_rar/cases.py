"""The two paper 1D Poisson benchmarks, equation (10) with solutions (11a)/(11b).

Both cases solve

    -u''(x) = f(x),   x in [-1, 1],   u(-1) = u_L,  u(1) = u_R

on a manufactured solution.  ``forcing`` and ``forcing_x`` are the analytical
source term and its derivative; ``forcing_x`` is what the gPINN residual-gradient
loss needs and is written out in closed form so no extra autograd pass through
the source is required.  ``tests`` verifies both against automatic
differentiation of ``exact``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class PoissonCase:
    """Interface shared by the two benchmarks."""

    key: str
    description: str
    lower: float = -1.0
    upper: float = 1.0

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forcing_x(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def exact_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.exact(torch.as_tensor(x, dtype=torch.float64)).numpy()

    def boundary_values(self, device: torch.device) -> torch.Tensor:
        ends = torch.as_tensor(
            [[self.lower], [self.upper]], dtype=torch.float64, device=device
        )
        return self.exact(ends)


class BoundaryLayer(PoissonCase):
    """Solution (11a): a smooth wave plus a 0.01-wide layer at the left wall."""

    key = "boundary_layer"
    description = "u = 0.1*sin(5*pi*x) + exp((0.01 - (x+1))/0.01)"

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.sin(5.0 * math.pi * x) + torch.exp(
            (0.01 - (x + 1.0)) / 0.01
        )

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        layer = torch.exp((0.01 - (x + 1.0)) / 0.01)
        return 0.1 * (5.0 * math.pi) ** 2 * torch.sin(5.0 * math.pi * x) - 1.0e4 * layer

    def forcing_x(self, x: torch.Tensor) -> torch.Tensor:
        layer = torch.exp((0.01 - (x + 1.0)) / 0.01)
        return 0.1 * (5.0 * math.pi) ** 3 * torch.cos(5.0 * math.pi * x) + 1.0e6 * layer


class Steep(PoissonCase):
    """Solution (11b): a smooth wave plus a 1/80-wide interior front at x = 0."""

    key = "steep"
    description = "u = 0.1*sin(8*pi*x) + tanh(80*x)"

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.sin(8.0 * math.pi * x) + torch.tanh(80.0 * x)

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        tanh = torch.tanh(80.0 * x)
        sech2 = 1.0 - tanh**2
        return (
            0.1 * (8.0 * math.pi) ** 2 * torch.sin(8.0 * math.pi * x)
            + 12800.0 * tanh * sech2
        )

    def forcing_x(self, x: torch.Tensor) -> torch.Tensor:
        tanh = torch.tanh(80.0 * x)
        sech2 = 1.0 - tanh**2
        return 0.1 * (8.0 * math.pi) ** 3 * torch.cos(8.0 * math.pi * x) + 1.024e6 * (
            sech2**2 - 2.0 * tanh**2 * sech2
        )


CASES: dict[str, type[PoissonCase]] = {
    BoundaryLayer.key: BoundaryLayer,
    Steep.key: Steep,
}


def build_case(key: str) -> PoissonCase:
    if key not in CASES:
        raise ValueError(f"unknown case {key!r}; choose from {sorted(CASES)}")
    return CASES[key]()
