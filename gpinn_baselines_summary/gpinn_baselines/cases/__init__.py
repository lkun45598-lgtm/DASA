"""Problem registry for all thirteen forward equations."""

from __future__ import annotations

from .elliptic import (
    Helmholtz1D,
    Helmholtz2D,
    Poisson1DBoundaryLayer,
    Poisson1DSteep,
    Poisson2DPeak,
    Poisson2DSteep,
    PoissonLShape,
)
from .flow import LidDrivenCavity
from .time_dependent import (
    AdvectionDiffusion1D,
    AdvectionDiscontinuous1D,
    AllenCahn1D,
    Burgers1D,
    DiffusionReaction1D,
)


REGISTRY = {
    "poisson_1d_boundary_layer": Poisson1DBoundaryLayer,
    "poisson_1d_steep": Poisson1DSteep,
    "helmholtz_1d": Helmholtz1D,
    "poisson_2d_peak": Poisson2DPeak,
    "poisson_2d_steep": Poisson2DSteep,
    "helmholtz_2d": Helmholtz2D,
    "lid_driven_cavity": LidDrivenCavity,
    "burgers_1d": Burgers1D,
    "allen_cahn_1d": AllenCahn1D,
    "diffusion_reaction_1d": DiffusionReaction1D,
    "advection_diffusion_1d": AdvectionDiffusion1D,
    "poisson_lshape": PoissonLShape,
    "advection_discontinuous_1d": AdvectionDiscontinuous1D,
}


def create_case(config):
    return REGISTRY[config["case"]](config)


__all__ = ["REGISTRY", "create_case"]
