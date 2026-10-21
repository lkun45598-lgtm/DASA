"""Factory for PDE benchmark definitions."""

from __future__ import annotations

from ..config import CaseConfig
from .base import PDEProblem, TrainingData
from .cavity import CavityProblem
from .lshape import LShapeProblem
from .static_1d import Static1DProblem
from .static_2d import Static2DProblem
from .time_1d import Time1DProblem


def build_problem(config: CaseConfig) -> PDEProblem:
    if config.family == "static_1d":
        return Static1DProblem(config)
    if config.family == "static_2d":
        return Static2DProblem(config)
    if config.family == "time_1d":
        return Time1DProblem(config)
    if config.family == "cavity":
        return CavityProblem(config)
    if config.family == "lshape":
        return LShapeProblem(config)
    raise ValueError(f"unsupported problem family: {config.family}")


__all__ = ["PDEProblem", "TrainingData", "build_problem"]
