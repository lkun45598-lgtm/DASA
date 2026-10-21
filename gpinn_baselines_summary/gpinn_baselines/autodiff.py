"""Small, explicit PyTorch automatic-differentiation helpers."""

from __future__ import annotations

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return dy/dx while retaining the graph for higher derivatives."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def partial(y: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    return gradient(y, x)[:, dim : dim + 1]


def second(y: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    return partial(partial(y, x, dim), x, dim)


def mixed(
    y: torch.Tensor, x: torch.Tensor, first_dim: int, second_dim: int
) -> torch.Tensor:
    return partial(partial(y, x, first_dim), x, second_dim)


def laplacian(y: torch.Tensor, x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return sum((second(y, x, dim) for dim in dims), torch.zeros_like(y))


def residual_gradients(
    residuals: dict[str, torch.Tensor],
    coordinates: torch.Tensor,
    dims: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiate every physical residual with respect to every requested input."""
    if dims is None:
        dims = tuple(range(coordinates.shape[1]))
    terms: dict[str, torch.Tensor] = {}
    for name, value in residuals.items():
        full_gradient = gradient(value, coordinates)
        for dim in dims:
            terms[f"{name}_d{dim}"] = full_gradient[:, dim : dim + 1]
    return terms
