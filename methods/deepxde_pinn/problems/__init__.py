"""Factory for DeepXDE problem definitions."""

from __future__ import annotations

from methods.sa_pinn_aligned.dasa_sa.config import CaseConfig

from .base import BaseProblem
from .cavity import CavityProblem
from .lshape import LShapeProblem
from .static import Static1DProblem, Static2DProblem
from .time import Time1DProblem


def build_problem(
    config: CaseConfig,
    *,
    seed: int,
    sampling_overrides: dict[str, int] | None = None,
) -> BaseProblem:
    arguments = {
        "seed": seed,
        "sampling_overrides": sampling_overrides,
    }
    if config.family == "static_1d":
        return Static1DProblem(config, **arguments)
    if config.family == "static_2d":
        return Static2DProblem(config, **arguments)
    if config.family == "time_1d":
        return Time1DProblem(config, **arguments)
    if config.family == "cavity":
        return CavityProblem(config, **arguments)
    if config.family == "lshape":
        return LShapeProblem(config, **arguments)
    raise ValueError(f"unsupported problem family: {config.family}")


__all__ = ["BaseProblem", "build_problem"]
