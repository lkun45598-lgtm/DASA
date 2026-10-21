"""Pure PyTorch gradient-enhanced PINN baselines for the 13 forward PDEs."""

from .config import CASE_NAMES, build_config

__all__ = ["CASE_NAMES", "MLP", "build_config"]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "MLP":
        from .network import MLP

        return MLP
    raise AttributeError(name)
