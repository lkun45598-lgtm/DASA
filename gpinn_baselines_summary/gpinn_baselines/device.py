"""Float64-aware accelerator selection with a high-order AD probe."""

from __future__ import annotations

import warnings

import torch


def _third_derivative_probe(device: torch.device) -> None:
    x = torch.tensor([[0.25]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tanh(x) ** 4
    for _ in range(3):
        y = torch.autograd.grad(
            y,
            x,
            torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
    if not torch.isfinite(y).all():
        raise RuntimeError("non-finite high-order derivative")


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            candidate = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            candidate = torch.device("mps")
        else:
            candidate = torch.device("cpu")
    else:
        candidate = torch.device(requested)

    # PyTorch MPS does not provide reliable float64/high-order AD support.
    if candidate.type == "mps":
        warnings.warn("MPS cannot satisfy the float64 gPINN contract; using CPU.")
        return torch.device("cpu")
    try:
        _third_derivative_probe(candidate)
        return candidate
    except Exception as exc:  # pragma: no cover - depends on accelerator build
        if candidate.type != "cpu":
            warnings.warn(
                f"{candidate} failed the float64 high-order AD probe ({exc}); using CPU."
            )
            _third_derivative_probe(torch.device("cpu"))
            return torch.device("cpu")
        raise
