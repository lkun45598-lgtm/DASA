"""Unified hp-VPINN experiments for the 13 DASA-PINNs Table 1 benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.io import loadmat
from scipy.linalg import solve_banded
from torch import nn

from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights


ROOT = Path(__file__).resolve().parent
PINNS_DATA = ROOT / "references" / "PINNs"
HP_DATA = ROOT / "references" / "hp-VPINNs"
GRADIENT_DATA = ROOT / "references" / "GradientPathologiesPINNs"


PROBLEM_ORDER = (
    "poisson_1d_boundary",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
    "cavity_re100",
    "burgers_1d",
    "allen_cahn_1d",
    "diffusion_reaction_1d",
    "advection_diffusion_1d",
    "poisson_lshape",
    "advection_discontinuous_1d",
)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    title: str
    domain: tuple[tuple[float, float], ...]
    iterations: int
    n_test: int = 5
    n_quad: int = 10
    boundary_weight: float = 100.0
    initial_weight: float = 100.0
    learning_rate: float = 1.0e-3
    lr_step: int = 2000
    lr_gamma: float = 0.9
    output_dim: int = 1
    fourier_dim: int | None = None
    fourier_modes: int = 0
    ade_kappa: float | None = None
    evidence: str = "DASA paper"
    assumption: str | None = None


SPECS = {
    "poisson_1d_boundary": ProblemSpec(
        "poisson_1d_boundary", "1D Poisson boundary-layer", ((-1.0, 1.0),), 40001,
        n_test=60, n_quad=80, boundary_weight=1.0,
        evidence="DASA Eq. (10)-(11a); hp-VPINNs Eq. (4.10)",
    ),
    "poisson_1d_steep": ProblemSpec(
        "poisson_1d_steep", "1D Poisson steep", ((-1.0, 1.0),), 40001,
        n_test=60, n_quad=80, boundary_weight=1.0,
        evidence="DASA Eq. (10)-(11b); hp-VPINNs Eq. (4.9)",
    ),
    "helmholtz_1d": ProblemSpec(
        "helmholtz_1d", "1D Helmholtz", ((0.0, 1.0),), 21001,
        n_test=8, n_quad=16,
        evidence="DASA Eq. (12)",
        assumption="DASA omits q(x); q(x)=1 is used and f is manufactured from u=sin(20*pi*x).",
    ),
    "poisson_2d_peak": ProblemSpec(
        "poisson_2d_peak", "2D Poisson peak", ((0.0, 1.0), (0.0, 1.0)), 21001,
        evidence="DASA Eq. (13), u=2^(4a)x^a(1-x)^a y^a(1-y)^a, a=10",
    ),
    "poisson_2d_steep": ProblemSpec(
        "poisson_2d_steep", "2D Poisson steep", ((-1.0, 1.0), (-1.0, 1.0)), 21001,
        evidence="DASA Fig. 3b axes and hp-VPINNs Eq. (5.12); resolves DASA domain typo",
    ),
    "helmholtz_2d": ProblemSpec(
        "helmholtz_2d", "2D Helmholtz", ((-1.0, 1.0), (-1.0, 1.0)), 21001,
        evidence="DASA Eq. (14)-(16) and cited GradientPathologiesPINNs: a1=1,a2=4,k^2=1",
    ),
    "cavity_re100": ProblemSpec(
        "cavity_re100", "Lid-driven cavity Re=100", ((0.0, 1.0), (0.0, 1.0)), 25001,
        output_dim=3,
        evidence="DASA Eq. (17)-(18); cited GradientPathologiesPINNs FDM 100x100 reference",
    ),
    "burgers_1d": ProblemSpec(
        "burgers_1d", "1D Burgers", ((-1.0, 1.0), (0.0, 1.0)), 40001,
        evidence="DASA Eq. (19); Raissi PINNs burgers_shock.mat reference",
    ),
    "allen_cahn_1d": ProblemSpec(
        "allen_cahn_1d", "1D Allen-Cahn", ((-1.0, 1.0), (0.0, 1.0)), 60001,
        fourier_dim=0, fourier_modes=4,
        evidence="DASA Eq. (20); Raissi PINNs AC.mat reference; periodic Fourier encoding",
    ),
    "diffusion_reaction_1d": ProblemSpec(
        "diffusion_reaction_1d", "1D diffusion-reaction", ((-math.pi, math.pi), (0.0, 1.0)), 21001,
        fourier_dim=0, fourier_modes=8,
        evidence="DASA Eq. (21)-(23)",
    ),
    "advection_diffusion_1d": ProblemSpec(
        "advection_diffusion_1d", "1D advection-diffusion", ((-1.0, 1.0), (0.0, 1.0)), 21001,
        ade_kappa=0.01,
        evidence="DASA Eq. (24), v=1,kappa=0.01; stable Crank-Nicolson FDM evaluation reference",
    ),
    "poisson_lshape": ProblemSpec(
        "poisson_lshape", "L-shaped Poisson", ((-1.0, 1.0), (-1.0, 1.0)), 28001,
        evidence="DASA Eq. (25); hp-VPINNs published SEM reference arrays",
    ),
    "advection_discontinuous_1d": ProblemSpec(
        "advection_discontinuous_1d", "1D discontinuous advection", ((-1.5, 1.5), (0.0, 1.0)), 26001,
        evidence="DASA Eq. (26)-(28), v=1,kappa=0",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def max_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def legendre_value(degree: int, x: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    return np.polynomial.legendre.legval(x, coefficients)


def legendre_derivative_value(degree: int, x: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    derivative = np.polynomial.legendre.legder(coefficients)
    return np.polynomial.legendre.legval(x, derivative)


def chebyshev_value(degree: int, x: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    return np.polynomial.chebyshev.chebval(x, coefficients)


def chebyshev_derivative_value(degree: int, x: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    derivative = np.polynomial.chebyshev.chebder(coefficients)
    return np.polynomial.chebyshev.chebval(x, derivative)


def clenshaw_curtis_rule(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights for unweighted integration over [-1, 1]."""
    if point_count < 2:
        raise ValueError("Clenshaw-Curtis quadrature requires at least two points")
    interval_count = point_count - 1
    theta = math.pi * np.arange(point_count, dtype=np.float64) / interval_count
    points = np.cos(theta)
    weights = np.zeros(point_count, dtype=np.float64)
    interior = np.arange(1, interval_count)
    values = np.ones(interval_count - 1, dtype=np.float64)
    if interval_count % 2 == 0:
        weights[0] = 1.0 / (interval_count**2 - 1.0)
        weights[-1] = weights[0]
        for k in range(1, interval_count // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
        values -= np.cos(interval_count * theta[interior]) / (interval_count**2 - 1.0)
    else:
        weights[0] = 1.0 / interval_count**2
        weights[-1] = weights[0]
        for k in range(1, (interval_count + 1) // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
    weights[interior] = 2.0 * values / interval_count
    return points[::-1].copy(), weights[::-1].copy()


def quadrature_and_tests(
    n_quad: int,
    n_test: int,
    dimension: int,
    full_second_dimension: bool = False,
    test_basis: str = "legendre_bubble",
    quadrature_rule: str = "gauss_lobatto_legendre",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if quadrature_rule == "gauss_lobatto_legendre":
        points, weights = GaussLobattoJacobiWeights(n_quad, 0, 0)
        points = np.asarray(points, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    elif quadrature_rule == "clenshaw_curtis":
        points, weights = clenshaw_curtis_rule(n_quad)
    else:
        raise ValueError(f"Unsupported quadrature rule: {quadrature_rule}")

    if test_basis == "legendre_bubble":
        tests = np.asarray(
            [legendre_value(n + 1, points) - legendre_value(n - 1, points) for n in range(1, n_test + 1)]
        )
        test_derivatives = np.asarray(
            [
                legendre_derivative_value(n + 1, points)
                - legendre_derivative_value(n - 1, points)
                for n in range(1, n_test + 1)
            ]
        )
    elif test_basis == "legendre_full":
        tests = np.asarray([legendre_value(n, points) for n in range(n_test)])
        test_derivatives = np.asarray(
            [legendre_derivative_value(n, points) for n in range(n_test)]
        )
    elif test_basis == "chebyshev_full":
        tests = np.asarray([chebyshev_value(n, points) for n in range(n_test)])
        test_derivatives = np.asarray(
            [chebyshev_derivative_value(n, points) for n in range(n_test)]
        )
    else:
        raise ValueError(f"Unsupported test basis: {test_basis}")
    if dimension == 1:
        return points[:, None], tests * weights[None, :], test_derivatives * weights[None, :], None

    second_tests = tests
    second_test_derivatives = test_derivatives
    if full_second_dimension and test_basis == "legendre_bubble":
        second_tests = np.asarray(
            [legendre_value(n, points) for n in range(n_test)]
        )
        second_test_derivatives = np.asarray(
            [legendre_derivative_value(n, points) for n in range(n_test)]
        )

    # Keep the tensor-product ordering consistent with the ``q,r`` axes in
    # the einsum below: x is the outer index and y varies fastest.
    xx, yy = np.meshgrid(points, points, indexing="ij")
    local_points = np.column_stack((xx.ravel(), yy.ravel()))
    weighted = np.einsum("iq,jr,q,r->ijqr", tests, second_tests, weights, weights)
    weighted_x = np.einsum(
        "iq,jr,q,r->ijqr", test_derivatives, second_tests, weights, weights
    )
    weighted_y = np.einsum(
        "iq,jr,q,r->ijqr", tests, second_test_derivatives, weights, weights
    )
    return (
        local_points,
        weighted.reshape(n_test * n_test, n_quad * n_quad),
        weighted_x.reshape(n_test * n_test, n_quad * n_quad),
        weighted_y.reshape(n_test * n_test, n_quad * n_quad),
    )


def regular_elements(x_edges: np.ndarray, y_edges: np.ndarray | None = None) -> np.ndarray:
    if y_edges is None:
        return np.asarray([[[left], [right]] for left, right in zip(x_edges[:-1], x_edges[1:])])
    elements = []
    for bottom, top in zip(y_edges[:-1], y_edges[1:]):
        for left, right in zip(x_edges[:-1], x_edges[1:]):
            elements.append([[left, bottom], [right, top]])
    return np.asarray(elements, dtype=np.float64)


def official_lshape_35_elements() -> np.ndarray:
    """The 35-element graded mesh from the official hp-VPINN example."""
    vertices = np.asarray(
        [
            [-1, -1], [-0.75, -1], [-0.5, -1], [-0.25, -1], [0.0, -1], [0.25, -1], [1, -1],
            [-1, -0.75], [-0.75, -0.75], [-0.5, -0.75], [-0.25, -0.75], [0.0, -0.75], [0.25, -0.75],
            [-1, -0.5], [-0.75, -0.5], [-0.5, -0.5], [-0.25, -0.5], [0.0, -0.5], [0.25, -0.5],
            [-1, -0.25], [-0.75, -0.25], [-0.5, -0.25], [-0.25, -0.25], [-0.125, -0.25],
            [0.0, -0.25], [0.125, -0.25], [0.25, -0.25],
            [-0.25, -0.125], [-0.125, -0.125], [0.0, -0.125], [0.125, -0.125], [0.25, -0.125],
            [-1, 0.0], [-0.75, 0.0], [-0.5, 0.0], [-0.25, 0.0], [-0.125, 0.0], [0.0, 0.0],
            [0.125, 0.0], [0.25, 0.0], [1.0, 0.0],
            [-0.25, 0.125], [-0.125, 0.125], [0.0, 0.125],
            [-1, 0.25], [-0.75, 0.25], [-0.5, 0.25], [-0.25, 0.25], [-0.125, 0.25], [0.0, 0.25],
            [-1, 1], [0, 1],
        ],
        dtype=np.float64,
    )
    element_vertex_indices = (
        (0, 1, 7, 8), (1, 2, 8, 9), (2, 3, 9, 10), (3, 4, 10, 11), (4, 5, 11, 12),
        (7, 8, 13, 14), (8, 9, 14, 15), (9, 10, 15, 16), (10, 11, 16, 17), (11, 12, 17, 18),
        (13, 14, 19, 20), (14, 15, 20, 21), (15, 16, 21, 22), (16, 17, 22, 24),
        (17, 18, 24, 26), (19, 20, 32, 33), (20, 21, 33, 34), (21, 22, 34, 35),
        (32, 33, 44, 45), (33, 34, 45, 46), (34, 35, 46, 47), (22, 23, 27, 28),
        (23, 24, 28, 29), (24, 25, 29, 30), (25, 26, 30, 31), (27, 28, 35, 36),
        (28, 29, 36, 37), (29, 30, 37, 38), (30, 31, 38, 39), (35, 36, 41, 42),
        (36, 37, 42, 43), (41, 42, 47, 48), (42, 43, 48, 49), (5, 6, 39, 40),
        (44, 49, 50, 51),
    )
    return np.asarray(
        [[vertices[indices[0]], vertices[indices[3]]] for indices in element_vertex_indices],
        dtype=np.float64,
    )


def build_elements(name: str, mesh_variant: str = "baseline") -> np.ndarray:
    if name == "poisson_lshape" and mesh_variant == "official":
        return official_lshape_35_elements()
    if name == "advection_diffusion_1d" and mesh_variant == "official":
        # Figure 13 uses the finest reported 4 x 4 uniform space-time mesh.
        return regular_elements(np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    if name == "advection_diffusion_1d" and mesh_variant == "refined":
        return regular_elements(np.linspace(-1.0, 1.0, 9), np.linspace(0.0, 1.0, 9))
    if name == "poisson_1d_boundary":
        return regular_elements(np.asarray([-1.0, -0.98, -0.9, 1.0]))
    if name == "poisson_1d_steep":
        return regular_elements(np.asarray([-1.0, -0.1, 0.1, 1.0]))
    if name == "helmholtz_1d":
        element_count = 20 if mesh_variant == "refined" else 10
        return regular_elements(np.linspace(0.0, 1.0, element_count + 1))
    if name == "poisson_2d_peak":
        edges = np.linspace(0.0, 1.0, 5)
        return regular_elements(edges, edges)
    if name == "poisson_2d_steep":
        edges = np.linspace(-1.0, 1.0, 9)
        return regular_elements(edges, edges)
    if name in {"helmholtz_2d", "cavity_re100"}:
        low, high = SPECS[name].domain[0]
        edge_count = 9 if name == "helmholtz_2d" and mesh_variant == "refined" else 5
        edges = np.linspace(low, high, edge_count)
        return regular_elements(edges, edges)
    if name == "burgers_1d":
        if mesh_variant == "paper":
            return regular_elements(np.asarray([-1.0, 1.0]), np.asarray([0.0, 1.0]))
        if mesh_variant == "paper_figure_fine":
            return regular_elements(
                np.asarray([-1.0, -0.1, 0.1, 1.0]), np.asarray([0.0, 1.0])
            )
        edge_count = 9 if mesh_variant == "refined" else 5
        return regular_elements(
            np.linspace(-1.0, 1.0, edge_count), np.linspace(0.0, 1.0, edge_count)
        )
    if name == "allen_cahn_1d":
        if mesh_variant == "paper":
            return regular_elements(np.asarray([-1.0, 1.0]), np.asarray([0.0, 1.0]))
        if mesh_variant == "causal_refined":
            return regular_elements(
                np.linspace(-1.0, 1.0, 9), np.linspace(0.0, 1.0, 17)
            )
        x_edge_count = 9 if mesh_variant == "refined" else 5
        t_edge_count = 9 if mesh_variant == "refined" else 5
        return regular_elements(
            np.linspace(-1.0, 1.0, x_edge_count), np.linspace(0.0, 1.0, t_edge_count)
        )
    if name == "diffusion_reaction_1d":
        if mesh_variant == "paper":
            return regular_elements(
                np.asarray([-math.pi, math.pi]), np.asarray([0.0, 1.0])
            )
        return regular_elements(np.linspace(-math.pi, math.pi, 9), np.linspace(0.0, 1.0, 5))
    if name == "advection_diffusion_1d":
        return regular_elements(
            np.asarray([-1.0, -0.5, 0.0, 0.5, 0.9, 0.98, 1.0]), np.linspace(0.0, 1.0, 5)
        )
    if name == "poisson_lshape":
        edges = np.linspace(-1.0, 1.0, 5)
        elements = regular_elements(edges, edges)
        return np.asarray([e for e in elements if not (e[0, 0] >= 0.0 and e[0, 1] >= 0.0)])
    if name == "advection_discontinuous_1d":
        return regular_elements(np.linspace(-1.5, 1.5, 13), np.linspace(0.0, 1.0, 5))
    raise KeyError(name)


def map_quadrature(elements: np.ndarray, local_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = elements[:, 0, :]
    right = elements[:, 1, :]
    half = (right - left) / 2.0
    center = (right + left) / 2.0
    points = center[:, None, :] + half[:, None, :] * local_points[None, :, :]
    jacobian = np.prod(half, axis=1)
    return points, jacobian


class ModifiedMLP(nn.Module):
    def __init__(
        self,
        spec: ProblemSpec,
        dtype: torch.dtype,
        coordinate_transform: str = "standard",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.coordinate_transform = coordinate_transform
        self.depth = 6
        self.width = 128
        feature_dim = len(spec.domain)
        if spec.fourier_dim is not None:
            feature_dim = feature_dim - 1 + 2 * spec.fourier_modes
        self.encoder_u = nn.Linear(feature_dim, self.width, dtype=dtype)
        self.encoder_v = nn.Linear(feature_dim, self.width, dtype=dtype)
        self.gates = nn.ModuleList(
            [nn.Linear(feature_dim, self.width, dtype=dtype)]
            + [nn.Linear(self.width, self.width, dtype=dtype) for _ in range(self.depth - 1)]
        )
        self.output = nn.Linear(self.width, spec.output_dim, dtype=dtype)
        for layer in [self.encoder_u, self.encoder_v, *self.gates, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.coordinate_transform == "characteristic":
            x = coordinates[:, 0:1]
            t = coordinates[:, 1:2]
            characteristic = x - t
            # (x - t, t) is an invertible affine reparameterization of (x, t).
            characteristic_normalized = (characteristic + 0.5) / 2.0
            time_normalized = 2.0 * t - 1.0
            return torch.cat((characteristic_normalized, time_normalized), dim=1)
        normalized = []
        periodic = self.spec.fourier_dim
        for index, (low, high) in enumerate(self.spec.domain):
            if index == periodic:
                phase = 2.0 * math.pi * (coordinates[:, index:index + 1] - low) / (high - low)
                for mode in range(1, self.spec.fourier_modes + 1):
                    normalized.extend((torch.sin(mode * phase), torch.cos(mode * phase)))
            else:
                normalized.append(2.0 * (coordinates[:, index:index + 1] - low) / (high - low) - 1.0)
        return torch.cat(normalized, dim=1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = self.features(coordinates)
        encoder_u = torch.tanh(self.encoder_u(x))
        encoder_v = torch.tanh(self.encoder_v(x))
        hidden = x
        for layer in self.gates:
            gate = torch.tanh(layer(hidden))
            hidden = (1.0 - gate) * encoder_u + gate * encoder_v
        return self.output(hidden)


class BurgersConstrainedTrial(nn.Module):
    """Embed the Burgers initial and Dirichlet conditions in the trial space."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        return -torch.sin(math.pi * x) + t * (1.0 - x**2) * self.base(coordinates)


def component_gradient(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value, coordinates, grad_outputs=torch.ones_like(value), create_graph=True, retain_graph=True
    )[0]


def first_second(
    value: torch.Tensor, coordinates: torch.Tensor, dimension: int
) -> tuple[torch.Tensor, torch.Tensor]:
    first_all = component_gradient(value, coordinates)
    first = first_all[:, dimension:dimension + 1]
    second_all = component_gradient(first, coordinates)
    return first, second_all[:, dimension:dimension + 1]


def exact_torch(name: str, z: torch.Tensor) -> torch.Tensor:
    x = z[:, 0:1]
    if name == "poisson_1d_boundary":
        return 0.1 * torch.sin(5.0 * math.pi * x) + torch.exp((0.01 - (x + 1.0)) / 0.01)
    if name == "poisson_1d_steep":
        return 0.1 * torch.sin(8.0 * math.pi * x) + torch.tanh(80.0 * x)
    if name == "helmholtz_1d":
        return torch.sin(20.0 * math.pi * x)
    if name == "poisson_2d_peak":
        y = z[:, 1:2]
        a = 10
        return float(2 ** (4 * a)) * x**a * (1.0 - x) ** a * y**a * (1.0 - y) ** a
    if name == "poisson_2d_steep":
        y = z[:, 1:2]
        return (0.1 * torch.sin(2.0 * math.pi * x) + torch.tanh(10.0 * x)) * torch.sin(2.0 * math.pi * y)
    if name == "helmholtz_2d":
        y = z[:, 1:2]
        return torch.sin(math.pi * x) * torch.sin(4.0 * math.pi * y)
    if name == "diffusion_reaction_1d":
        t = z[:, 1:2]
        spatial = sum(torch.sin(k * x) / k for k in range(1, 5)) + torch.sin(8.0 * x) / 8.0
        return torch.exp(-t) * spatial
    if name == "advection_discontinuous_1d":
        t = z[:, 1:2]
        shifted = x - t
        return ((shifted >= -0.2) & (shifted <= 0.2)).to(z.dtype)
    raise KeyError(f"No analytic torch solution for {name}")


def forcing_torch(name: str, z: torch.Tensor) -> torch.Tensor:
    x = z[:, 0:1]
    if name == "poisson_1d_boundary":
        exponential = torch.exp((0.01 - (x + 1.0)) / 0.01)
        u_xx = -0.1 * (5.0 * math.pi) ** 2 * torch.sin(5.0 * math.pi * x) + exponential / 0.01**2
        return -u_xx
    if name == "poisson_1d_steep":
        sech_sq = 1.0 / torch.cosh(80.0 * x) ** 2
        u_xx = -0.1 * (8.0 * math.pi) ** 2 * torch.sin(8.0 * math.pi * x) - 2.0 * 80.0**2 * torch.tanh(80.0 * x) * sech_sq
        return -u_xx
    if name == "helmholtz_1d":
        return ((20.0 * math.pi) ** 2 + 1.0) * torch.sin(20.0 * math.pi * x)
    if name == "poisson_2d_peak":
        y = z[:, 1:2]
        a = 10
        scale = float(2 ** (4 * a))

        def base(s: torch.Tensor) -> torch.Tensor:
            return s**a * (1.0 - s) ** a

        def second(s: torch.Tensor) -> torch.Tensor:
            return (
                a * (a - 1) * s ** (a - 2) * (1.0 - s) ** a
                - 2 * a * a * s ** (a - 1) * (1.0 - s) ** (a - 1)
                + a * (a - 1) * s**a * (1.0 - s) ** (a - 2)
            )

        return -scale * (second(x) * base(y) + base(x) * second(y))
    if name == "poisson_2d_steep":
        y = z[:, 1:2]
        g = 0.1 * torch.sin(2.0 * math.pi * x) + torch.tanh(10.0 * x)
        g_xx = -0.1 * (2.0 * math.pi) ** 2 * torch.sin(2.0 * math.pi * x)
        g_xx -= 200.0 * torch.tanh(10.0 * x) / torch.cosh(10.0 * x) ** 2
        laplace = g_xx * torch.sin(2.0 * math.pi * y) - (2.0 * math.pi) ** 2 * g * torch.sin(2.0 * math.pi * y)
        return -laplace
    if name == "helmholtz_2d":
        u = exact_torch(name, z)
        return (-(math.pi**2) - (4.0 * math.pi) ** 2 + 1.0) * u
    if name == "diffusion_reaction_1d":
        t = z[:, 1:2]
        reaction = (
            1.5 * torch.sin(2.0 * x)
            + (8.0 / 3.0) * torch.sin(3.0 * x)
            + (15.0 / 4.0) * torch.sin(4.0 * x)
            + (63.0 / 8.0) * torch.sin(8.0 * x)
        )
        return torch.exp(-t) * reaction
    raise KeyError(name)


def strong_residual(
    name: str, model: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, ade_kappa: float | None = None
) -> torch.Tensor:
    output = model(z)
    u = output[:, 0:1]
    if name.startswith("poisson_1d"):
        _, u_xx = first_second(u, z, 0)
        return -u_xx - forcing_torch(name, z)
    if name == "helmholtz_1d":
        _, u_xx = first_second(u, z, 0)
        return -u_xx + u - forcing_torch(name, z)
    if name in {"poisson_2d_peak", "poisson_2d_steep", "poisson_lshape"}:
        _, u_xx = first_second(u, z, 0)
        _, u_yy = first_second(u, z, 1)
        forcing = torch.ones_like(u) if name == "poisson_lshape" else forcing_torch(name, z)
        return -(u_xx + u_yy) - forcing
    if name == "helmholtz_2d":
        _, u_xx = first_second(u, z, 0)
        _, u_yy = first_second(u, z, 1)
        return u_xx + u_yy + u - forcing_torch(name, z)
    if name == "cavity_re100":
        v = output[:, 1:2]
        p = output[:, 2:3]
        u_grad = component_gradient(u, z)
        v_grad = component_gradient(v, z)
        p_grad = component_gradient(p, z)
        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
        v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]
        u_xx = component_gradient(u_x, z)[:, 0:1]
        u_yy = component_gradient(u_y, z)[:, 1:2]
        v_xx = component_gradient(v_x, z)[:, 0:1]
        v_yy = component_gradient(v_y, z)[:, 1:2]
        ru = u * u_x + v * u_y + p_grad[:, 0:1] - (u_xx + u_yy) / 100.0
        rv = u * v_x + v * v_y + p_grad[:, 1:2] - (v_xx + v_yy) / 100.0
        continuity = u_x + v_y
        return torch.cat((ru, rv, continuity), dim=1)
    if name == "burgers_1d":
        u_grad = component_gradient(u, z)
        u_x, u_t = u_grad[:, 0:1], u_grad[:, 1:2]
        u_xx = component_gradient(u_x, z)[:, 0:1]
        return u_t + u * u_x - (0.01 / math.pi) * u_xx
    if name == "allen_cahn_1d":
        u_grad = component_gradient(u, z)
        u_t = u_grad[:, 1:2]
        u_xx = component_gradient(u_grad[:, 0:1], z)[:, 0:1]
        return u_t - 0.0001 * u_xx + 5.0 * u**3 - 5.0 * u
    if name == "diffusion_reaction_1d":
        u_grad = component_gradient(u, z)
        u_xx = component_gradient(u_grad[:, 0:1], z)[:, 0:1]
        return u_grad[:, 1:2] - u_xx - forcing_torch(name, z)
    if name == "advection_diffusion_1d":
        if ade_kappa is None:
            ade_kappa = 0.01
        u_grad = component_gradient(u, z)
        u_xx = component_gradient(u_grad[:, 0:1], z)[:, 0:1]
        return u_grad[:, 1:2] + u_grad[:, 0:1] - ade_kappa * u_xx
    if name == "advection_discontinuous_1d":
        u_grad = component_gradient(u, z)
        return u_grad[:, 1:2] + u_grad[:, 0:1]
    raise KeyError(name)


def variational_projection(
    name: str,
    variational_form: str,
    model: nn.Module,
    points: torch.Tensor,
    weighted_tests: torch.Tensor,
    weighted_test_x: torch.Tensor | None,
    weighted_test_y: torch.Tensor | None,
    half_sizes: torch.Tensor,
    element_count: int,
    points_per_element: int,
    ade_kappa: float | None = None,
    integrate_time_by_parts: bool = False,
    time_boundary_points: tuple[torch.Tensor, torch.Tensor] | None = None,
    weighted_space_tests: torch.Tensor | None = None,
    time_test_boundary_values: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return element-wise hp-VPINN projections for the selected formulation."""
    if variational_form in {"strong", "r1"}:
        residual = strong_residual(name, model, points, ade_kappa)
        residual = residual.reshape(element_count, points_per_element, -1)
        projection = torch.einsum("eqc,tq->etc", residual, weighted_tests)
        jacobians = torch.prod(half_sizes, dim=1)
        return projection * jacobians[:, None, None]

    if variational_form != "p1":
        raise ValueError(f"Unsupported variational form: {variational_form}")

    if name == "helmholtz_1d":
        if weighted_test_x is None:
            raise ValueError("1D p1 requires test-function derivatives")
        output = model(points)
        u = output[:, 0:1]
        ux = component_gradient(u, points)[:, 0:1].reshape(
            element_count, points_per_element, 1
        )
        u = u.reshape(element_count, points_per_element, 1)
        forcing = forcing_torch(name, points).reshape(
            element_count, points_per_element, 1
        )
        hx = half_sizes[:, 0]
        diffusion = torch.einsum("eqc,tq->etc", ux, weighted_test_x)
        reaction_rhs = torch.einsum("eqc,tq->etc", u - forcing, weighted_tests)
        return diffusion + hx[:, None, None] * reaction_rhs

    if weighted_test_x is None or weighted_test_y is None:
        raise ValueError("2D p1 requires tensor-product test-function derivatives")

    if name in {"poisson_2d_peak", "poisson_2d_steep", "poisson_lshape"}:
        output = model(points)
        u = output[:, 0:1]
        gradient = component_gradient(u, points)
        ux = gradient[:, 0:1].reshape(element_count, points_per_element, 1)
        uy = gradient[:, 1:2].reshape(element_count, points_per_element, 1)
        forcing = torch.ones_like(u) if name == "poisson_lshape" else forcing_torch(name, points)
        forcing = forcing.reshape(element_count, points_per_element, 1)
        hx = half_sizes[:, 0]
        hy = half_sizes[:, 1]
        x_part = torch.einsum("eqc,tq->etc", ux, weighted_test_x) * hy[:, None, None]
        y_part = torch.einsum("eqc,tq->etc", uy, weighted_test_y) * hx[:, None, None]
        rhs = torch.einsum("eqc,tq->etc", forcing, weighted_tests)
        rhs = rhs * (hx * hy)[:, None, None]
        return x_part + y_part - rhs

    if name == "helmholtz_2d":
        output = model(points)
        u = output[:, 0:1]
        gradient = component_gradient(u, points)
        ux = gradient[:, 0:1].reshape(element_count, points_per_element, 1)
        uy = gradient[:, 1:2].reshape(element_count, points_per_element, 1)
        u = u.reshape(element_count, points_per_element, 1)
        forcing = forcing_torch(name, points).reshape(
            element_count, points_per_element, 1
        )
        hx = half_sizes[:, 0]
        hy = half_sizes[:, 1]
        x_part = torch.einsum("eqc,tq->etc", ux, weighted_test_x) * hy[:, None, None]
        y_part = torch.einsum("eqc,tq->etc", uy, weighted_test_y) * hx[:, None, None]
        reaction_rhs = torch.einsum("eqc,tq->etc", u - forcing, weighted_tests)
        reaction_rhs = reaction_rhs * (hx * hy)[:, None, None]
        return -x_part - y_part + reaction_rhs

    if name in {"burgers_1d", "allen_cahn_1d"}:
        output = model(points)
        u = output[:, 0:1]
        gradient = component_gradient(u, points)
        ux = gradient[:, 0:1].reshape(element_count, points_per_element, 1)
        ut = gradient[:, 1:2].reshape(element_count, points_per_element, 1)
        u = u.reshape(element_count, points_per_element, 1)
        hx = half_sizes[:, 0]
        ht = half_sizes[:, 1]
        if name == "burgers_1d":
            non_temporal = u * ux
            diffusivity = 0.01 / math.pi
        else:
            non_temporal = 5.0 * u**3 - 5.0 * u
            diffusivity = 0.0001
        if integrate_time_by_parts:
            if (
                time_boundary_points is None
                or weighted_space_tests is None
                or time_test_boundary_values is None
            ):
                raise ValueError("Time P1 requires temporal boundary quadrature data")
            bottom_points, top_points = time_boundary_points
            time_test_bottom, time_test_top = time_test_boundary_values
            u_bottom = model(bottom_points).reshape(element_count, -1, 1)
            u_top = model(top_points).reshape(element_count, -1, 1)
            bottom_space_projection = torch.einsum(
                "eqc,iq->eic", u_bottom, weighted_space_tests
            )
            top_space_projection = torch.einsum(
                "eqc,iq->eic", u_top, weighted_space_tests
            )
            time_boundary = (
                torch.einsum("eic,j->eijc", top_space_projection, time_test_top)
                - torch.einsum("eic,j->eijc", bottom_space_projection, time_test_bottom)
            ).reshape(element_count, -1, 1)
            time_volume = torch.einsum("eqc,tq->etc", u, weighted_test_y)
            time_part = hx[:, None, None] * (time_boundary - time_volume)
            volume = torch.einsum("eqc,tq->etc", non_temporal, weighted_tests)
            volume = (hx * ht)[:, None, None] * volume + time_part
        else:
            volume = torch.einsum("eqc,tq->etc", ut + non_temporal, weighted_tests)
            volume = (hx * ht)[:, None, None] * volume
        diffusion = torch.einsum("eqc,tq->etc", ux, weighted_test_x)
        return volume + diffusivity * ht[:, None, None] * diffusion

    if name == "diffusion_reaction_1d":
        output = model(points)
        u = output[:, 0:1]
        gradient = component_gradient(u, points)
        ux = gradient[:, 0:1].reshape(element_count, points_per_element, 1)
        ut = gradient[:, 1:2].reshape(element_count, points_per_element, 1)
        u = u.reshape(element_count, points_per_element, 1)
        forcing = forcing_torch(name, points).reshape(
            element_count, points_per_element, 1
        )
        hx = half_sizes[:, 0]
        ht = half_sizes[:, 1]
        if integrate_time_by_parts:
            if (
                time_boundary_points is None
                or weighted_space_tests is None
                or time_test_boundary_values is None
            ):
                raise ValueError("Time P1 requires temporal boundary quadrature data")
            bottom_points, top_points = time_boundary_points
            time_test_bottom, time_test_top = time_test_boundary_values
            u_bottom = model(bottom_points).reshape(element_count, -1, 1)
            u_top = model(top_points).reshape(element_count, -1, 1)
            bottom_space_projection = torch.einsum(
                "eqc,iq->eic", u_bottom, weighted_space_tests
            )
            top_space_projection = torch.einsum(
                "eqc,iq->eic", u_top, weighted_space_tests
            )
            time_boundary = (
                torch.einsum("eic,j->eijc", top_space_projection, time_test_top)
                - torch.einsum("eic,j->eijc", bottom_space_projection, time_test_bottom)
            ).reshape(element_count, -1, 1)
            time_volume = torch.einsum("eqc,tq->etc", u, weighted_test_y)
            reaction = torch.einsum("eqc,tq->etc", -forcing, weighted_tests)
            time_reaction = (
                hx[:, None, None] * (time_boundary - time_volume)
                + (hx * ht)[:, None, None] * reaction
            )
        else:
            time_reaction = (hx * ht)[:, None, None] * torch.einsum(
                "eqc,tq->etc", ut - forcing, weighted_tests
            )
        diffusion = torch.einsum("eqc,tq->etc", ux, weighted_test_x)
        return time_reaction + ht[:, None, None] * diffusion

    if name == "advection_diffusion_1d":
        if ade_kappa is None:
            ade_kappa = 0.01
        output = model(points)
        u = output[:, 0:1]
        gradient = component_gradient(u, points)
        ux = gradient[:, 0:1].reshape(element_count, points_per_element, 1)
        ut = gradient[:, 1:2].reshape(element_count, points_per_element, 1)
        hx = half_sizes[:, 0]
        ht = half_sizes[:, 1]
        advective = torch.einsum("eqc,tq->etc", ut + ux, weighted_tests)
        diffusive = torch.einsum("eqc,tq->etc", ux, weighted_test_x)
        return (hx * ht)[:, None, None] * advective + ade_kappa * ht[:, None, None] * diffusive

    raise ValueError(f"p1 is not implemented for {name}")


def tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def edge_points(start: tuple[float, float], end: tuple[float, float], count: int) -> np.ndarray:
    fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
    start_array = np.asarray(start)
    return start_array[None, :] + fractions[:, None] * (np.asarray(end) - start_array)[None, :]


def build_constraints(
    spec: ProblemSpec,
    device: torch.device,
    dtype: torch.dtype,
    aggregation: str = "separate",
    allen_soft_periodic: bool = False,
    initial_sampling: str = "uniform",
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    name = spec.name
    groups: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    if len(spec.domain) == 1:
        points = np.asarray([[spec.domain[0][0]], [spec.domain[0][1]]])
        values = exact_torch(name, torch.as_tensor(points, dtype=torch.float64)).numpy()
        groups.append((points, values, spec.boundary_weight, "boundary"))
    elif name in {"poisson_2d_peak", "poisson_2d_steep", "helmholtz_2d"}:
        (xl, xr), (yl, yr) = spec.domain
        count = 50
        points = np.vstack(
            (edge_points((xl, yl), (xr, yl), count), edge_points((xr, yl), (xr, yr), count),
             edge_points((xr, yr), (xl, yr), count), edge_points((xl, yr), (xl, yl), count))
        )
        values = exact_torch(name, torch.as_tensor(points, dtype=torch.float64)).numpy()
        groups.append((points, values, spec.boundary_weight, "boundary"))
    elif name == "cavity_re100":
        count = 128
        top = edge_points((0.0, 1.0), (1.0, 1.0), count)
        walls = np.vstack((edge_points((0.0, 0.0), (1.0, 0.0), count),
                           edge_points((0.0, 0.0), (0.0, 1.0), count),
                           edge_points((1.0, 0.0), (1.0, 1.0), count)))
        groups.append((top, np.tile([[1.0, 0.0]], (count, 1)), spec.boundary_weight, "moving_lid"))
        groups.append((walls, np.zeros((3 * count, 2)), spec.boundary_weight, "stationary_walls"))
    elif name == "poisson_lshape":
        count = 128
        segments = (
            ((-1.0, -1.0), (1.0, -1.0), 2 * count), ((-1.0, -1.0), (-1.0, 1.0), 2 * count),
            ((-1.0, 1.0), (0.0, 1.0), count), ((1.0, -1.0), (1.0, 0.0), count),
            ((0.0, 0.0), (1.0, 0.0), count), ((0.0, 0.0), (0.0, 1.0), count),
        )
        points = np.vstack([edge_points(start, end, number) for start, end, number in segments])
        groups.append((points, np.zeros((points.shape[0], 1)), spec.boundary_weight, "boundary"))
    else:
        (xl, xr), (tl, tr) = spec.domain
        boundary_count = 256 if name != "advection_discontinuous_1d" else 256
        initial_count = 512
        times = (np.arange(boundary_count)[:, None] + 0.5) / boundary_count * (tr - tl) + tl
        if name != "allen_cahn_1d":
            boundary_points = np.vstack((np.column_stack((np.full(boundary_count, xl), times[:, 0])),
                                         np.column_stack((np.full(boundary_count, xr), times[:, 0]))))
            groups.append((boundary_points, np.zeros((2 * boundary_count, 1)), spec.boundary_weight, "boundary"))
        elif allen_soft_periodic:
            periodic_points = np.vstack(
                (
                    np.column_stack((np.full(boundary_count, xl), times[:, 0])),
                    np.column_stack((np.full(boundary_count, xr), times[:, 0])),
                )
            )
            groups.append(
                (
                    periodic_points,
                    np.zeros((2 * boundary_count, 1)),
                    spec.boundary_weight,
                    "periodic",
                )
            )
        if name == "advection_discontinuous_1d" and initial_sampling == "include_interfaces":
            xs = np.concatenate((np.linspace(xl, xr, 510), [-0.2, 0.2]))[:, None]
        elif name == "advection_discontinuous_1d" and initial_sampling == "interface_refined":
            uniform_x = np.linspace(xl, xr, 382)
            offsets = np.geomspace(1.0e-5, 5.0e-2, 32)
            interface_x = np.concatenate(
                (
                    np.asarray([-0.2, 0.2]),
                    -0.2 - offsets,
                    -0.2 + offsets,
                    0.2 - offsets,
                    0.2 + offsets,
                )
            )
            xs = np.concatenate((uniform_x, interface_x))[:, None]
            if xs.shape[0] != initial_count:
                raise RuntimeError("Interface-refined initial set must retain 512 points")
        else:
            xs = np.linspace(xl, xr, initial_count)[:, None]
        initial_points = np.column_stack((xs[:, 0], np.full(initial_count, tl)))
        if name in {"burgers_1d", "advection_diffusion_1d"}:
            initial_values = -np.sin(math.pi * xs)
        elif name == "allen_cahn_1d":
            initial_values = xs**2 * np.cos(math.pi * xs)
        elif name == "diffusion_reaction_1d":
            initial_values = sum(np.sin(k * xs) / k for k in range(1, 5)) + np.sin(8.0 * xs) / 8.0
        elif name == "advection_discontinuous_1d":
            initial_values = ((xs >= -0.2) & (xs <= 0.2)).astype(np.float64)
        else:
            raise KeyError(name)
        groups.append((initial_points, initial_values, spec.initial_weight, "initial"))

    if aggregation == "combined":
        weights = {weight for _, _, weight, _ in groups}
        if len(weights) != 1:
            raise ValueError("Combined constraints must all have the same weight")
        groups = [
            (
                np.vstack([points for points, _, _, _ in groups]),
                np.vstack([values for _, values, _, _ in groups]),
                weights.pop(),
                "boundary_initial",
            )
        ]
    elif aggregation != "separate":
        raise ValueError(f"Unsupported constraint aggregation: {aggregation}")

    return [(tensor(x, device, dtype), tensor(y, device, dtype), weight, label) for x, y, weight, label in groups]


def constraint_loss(
    name: str, model: nn.Module, groups: list[tuple[torch.Tensor, torch.Tensor, float, str]]
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total = torch.zeros((), device=device, dtype=dtype)
    parts: dict[str, float] = {}
    for points, targets, weight, label in groups:
        if label == "periodic":
            active_points = points.detach().clone().requires_grad_(True)
            prediction = model(active_points)
            gradient = component_gradient(prediction, active_points)[:, 0:1]
            half = active_points.shape[0] // 2
            value_mse = torch.mean((prediction[:half] - prediction[half:]) ** 2)
            derivative_mse = torch.mean((gradient[:half] - gradient[half:]) ** 2)
            total = total + weight * (value_mse + derivative_mse)
            parts["periodic_value"] = float(value_mse.detach().cpu())
            parts["periodic_derivative"] = float(derivative_mse.detach().cpu())
            continue
        prediction = model(points)
        if name == "cavity_re100":
            prediction = prediction[:, :2]
        mse = torch.mean((prediction - targets) ** 2)
        total = total + weight * mse
        parts[label] = float(mse.detach().cpu())
    if name == "cavity_re100":
        interior = torch.full((1, 2), 0.5, device=device, dtype=dtype)
        pressure_gauge = model(interior)[0, 2] ** 2
        total = total + pressure_gauge
        parts["pressure_gauge"] = float(pressure_gauge.detach().cpu())
    return total, parts


def ade_reference_fdm(
    coordinates: np.ndarray, diffusivity: float, max_dt: float = 1.0e-3, n_cells: int = 16384
) -> np.ndarray:
    """Stable independent ADE reference for advection-dominated parameter sets.

    The closed-form series suffers catastrophic cancellation for kappa=0.01
    near x=1.  This Crank-Nicolson solve is used only after training to
    evaluate error; it never contributes to the hp-VPINN loss.
    """
    velocity = 1.0
    grid = np.linspace(-1.0, 1.0, n_cells + 1, dtype=np.float64)
    dx = grid[1] - grid[0]
    interior = grid[1:-1]
    solution = -np.sin(math.pi * interior)
    lower_l = diffusivity / dx**2 + velocity / (2.0 * dx)
    diagonal_l = -2.0 * diffusivity / dx**2
    upper_l = diffusivity / dx**2 - velocity / (2.0 * dx)
    times = np.unique(coordinates[:, 1])
    values_at_time: dict[float, np.ndarray] = {0.0: solution.copy()}
    current_time = 0.0

    for target_time in times:
        if target_time == 0.0:
            continue
        if target_time < current_time:
            raise ValueError("ADE evaluation times must be nondecreasing")
        steps = max(1, int(math.ceil((target_time - current_time) / max_dt)))
        dt = (target_time - current_time) / steps
        banded = np.empty((3, solution.size), dtype=np.float64)
        banded[0, 0] = 0.0
        banded[0, 1:] = -0.5 * dt * upper_l
        banded[1, :] = 1.0 - 0.5 * dt * diagonal_l
        banded[2, :-1] = -0.5 * dt * lower_l
        banded[2, -1] = 0.0
        for _ in range(steps):
            rhs = (1.0 + 0.5 * dt * diagonal_l) * solution
            rhs[1:] += 0.5 * dt * lower_l * solution[:-1]
            rhs[:-1] += 0.5 * dt * upper_l * solution[1:]
            solution = solve_banded((1, 1), banded, rhs, overwrite_ab=False, overwrite_b=True)
        current_time = float(target_time)
        values_at_time[current_time] = solution.copy()

    output = np.empty((coordinates.shape[0], 1), dtype=np.float64)
    for time_value in times:
        mask = coordinates[:, 1] == time_value
        full_solution = np.empty(n_cells + 1, dtype=np.float64)
        full_solution[0] = 0.0
        full_solution[-1] = 0.0
        full_solution[1:-1] = values_at_time[float(time_value)]
        output[mask, 0] = np.interp(coordinates[mask, 0], grid, full_solution)
    return output


def ade_exact_numpy(
    coordinates: np.ndarray, diffusivity: float = 0.01, truncation: int = 1000
) -> np.ndarray:
    if diffusivity <= 0.015:
        return ade_reference_fdm(coordinates, diffusivity)
    output = np.empty((coordinates.shape[0], 1), dtype=np.float64)
    velocity = 1.0
    modes = np.arange(truncation + 1, dtype=np.float64)[None, :]
    for start in range(0, coordinates.shape[0], 1024):
        block = coordinates[start:start + 1024]
        x = block[:, 0:1]
        t = block[:, 1:2]
        c0 = 16 * math.pi**2 * diffusivity**3 * velocity * np.exp(
            velocity / diffusivity / 2 * (x - velocity * t / 2)
        )
        c1_num = (-1.0) ** modes * 2 * modes * np.sin(modes * math.pi * x) * np.exp(
            -diffusivity * modes**2 * math.pi**2 * t
        )
        c1_den = velocity**4 + 8 * (velocity * math.pi * diffusivity) ** 2 * (modes**2 + 1)
        c1_den += 16 * (math.pi * diffusivity) ** 4 * (modes**2 - 1) ** 2
        c1 = np.sinh(velocity / diffusivity / 2) * np.sum(c1_num / c1_den, axis=1, keepdims=True)
        odd = 2 * modes + 1
        c2_num = (-1.0) ** modes * odd * np.cos((modes + 0.5) * math.pi * x) * np.exp(
            -diffusivity * odd**2 * math.pi**2 * t / 4
        )
        c2_den = velocity**4 + (velocity * math.pi * diffusivity) ** 2 * (8 * modes**2 + 8 * modes + 10)
        c2_den += (math.pi * diffusivity) ** 4 * (4 * modes**2 + 4 * modes - 3) ** 2
        values = c0 * (c1 + np.cosh(velocity / diffusivity / 2) * np.sum(c2_num / c2_den, axis=1, keepdims=True))
        at_initial = np.isclose(t[:, 0], 0.0)
        values[at_initial] = -np.sin(math.pi * x[at_initial])
        output[start:start + block.shape[0]] = values
    return output


def evaluation_data(name: str, ade_kappa: float | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    metadata: dict[str, object] = {}
    if name in {"poisson_1d_boundary", "poisson_1d_steep"}:
        coordinates = np.linspace(-1.0, 1.0, 2001)[:, None]
        exact = exact_torch(name, torch.as_tensor(coordinates, dtype=torch.float64)).numpy()
    elif name == "helmholtz_1d":
        coordinates = np.linspace(0.0, 1.0, 2001)[:, None]
        exact = np.sin(20.0 * math.pi * coordinates)
    elif name in {"poisson_2d_peak", "poisson_2d_steep", "helmholtz_2d"}:
        (xl, xr), (yl, yr) = SPECS[name].domain
        x = np.linspace(xl, xr, 201)
        y = np.linspace(yl, yr, 201)
        xx, yy = np.meshgrid(x, y)
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))
        exact = exact_torch(name, torch.as_tensor(coordinates, dtype=torch.float64)).numpy()
        metadata["grid_shape"] = [201, 201]
    elif name == "cavity_re100":
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 1.0, 100)
        xx, yy = np.meshgrid(x, y)
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))
        path = GRADIENT_DATA / "Lid-driven Cavity"
        u_ref = np.loadtxt(path / "reference_u.csv", delimiter=",")
        v_ref = np.loadtxt(path / "reference_v.csv", delimiter=",")
        exact = np.sqrt(u_ref**2 + v_ref**2).T.reshape(-1, 1)
        metadata.update(grid_shape=[100, 100], target="velocity_magnitude", reference=str(path))
    elif name == "burgers_1d":
        path = PINNS_DATA / "appendix" / "Data" / "burgers_shock.mat"
        data = loadmat(path)
        x = data["x"].reshape(-1)
        t = data["t"].reshape(-1)
        xx, tt = np.meshgrid(x, t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        exact = np.real(data["usol"]).T.reshape(-1, 1)
        metadata.update(grid_shape=list(xx.shape), reference=str(path))
    elif name == "allen_cahn_1d":
        path = PINNS_DATA / "main" / "Data" / "AC.mat"
        data = loadmat(path)
        x = data["x"].reshape(-1)
        t_key = "tt" if "tt" in data else "t"
        t = data[t_key].reshape(-1)
        xx, tt = np.meshgrid(x, t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        solution_key = "uu" if "uu" in data else "usol"
        exact = np.real(data[solution_key]).T.reshape(-1, 1)
        metadata.update(grid_shape=list(xx.shape), reference=str(path))
    elif name == "diffusion_reaction_1d":
        x = np.linspace(-math.pi, math.pi, 256)
        t = np.linspace(0.0, 1.0, 101)
        xx, tt = np.meshgrid(x, t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        exact = exact_torch(name, torch.as_tensor(coordinates, dtype=torch.float64)).numpy()
        metadata["grid_shape"] = list(xx.shape)
    elif name == "advection_diffusion_1d":
        x = np.linspace(-1.0, 1.0, 256)
        t = np.linspace(0.0, 1.0, 101)
        xx, tt = np.meshgrid(x, t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        if ade_kappa is None:
            ade_kappa = 0.01
        exact = ade_exact_numpy(coordinates, diffusivity=ade_kappa, truncation=1000)
        reference_method = "crank_nicolson_fdm_16384_cells" if ade_kappa <= 0.015 else "analytic_series"
        metadata.update(
            grid_shape=list(xx.shape), series_terms=1000, kappa=ade_kappa, reference_method=reference_method
        )
    elif name == "poisson_lshape":
        path = HP_DATA / "main" / "hp-VPINN-Poisson2D-LShaped" / "Data"
        coordinates_all = np.load(path / "X_test.npy")
        exact_all = np.load(path / "y_ref.npy")
        mask = np.isfinite(exact_all[:, 0])
        coordinates, exact = coordinates_all[mask], exact_all[mask]
        metadata.update(reference=str(path), excluded_nan_points=int(np.sum(~mask)))
    elif name == "advection_discontinuous_1d":
        # Cell centers avoid assigning positive measure to the two jump traces
        # when approximating the continuous L2 norm.
        x_edges = np.linspace(-1.5, 1.5, 2401)
        t_edges = np.linspace(0.0, 1.0, 801)
        x = (x_edges[:-1] + x_edges[1:]) / 2.0
        t = (t_edges[:-1] + t_edges[1:]) / 2.0
        xx, tt = np.meshgrid(x, t)
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        exact = (((xx - tt) >= -0.2) & ((xx - tt) <= 0.2)).astype(np.float64).reshape(-1, 1)
        metadata.update(grid_shape=list(xx.shape), quadrature="composite_midpoint")
    else:
        raise KeyError(name)
    return coordinates.astype(np.float64), exact.astype(np.float64), metadata


def predict(
    name: str, model: nn.Module, coordinates: np.ndarray, device: torch.device, dtype: torch.dtype
) -> np.ndarray:
    blocks = []
    for start in range(0, coordinates.shape[0], 32768):
        points = tensor(coordinates[start:start + 32768], device, dtype)
        if name == "cavity_re100":
            points.requires_grad_(False)
        with torch.no_grad():
            output = model(points)
        if name == "cavity_re100":
            output = torch.sqrt(output[:, 0:1] ** 2 + output[:, 1:2] ** 2)
        blocks.append(output.detach().cpu().numpy())
    return np.vstack(blocks).astype(np.float64)


def exact_residual_audit(
    spec: ProblemSpec, points: torch.Tensor
) -> float | None:
    supported = {
        "poisson_1d_boundary", "poisson_1d_steep", "helmholtz_1d",
        "poisson_2d_peak", "poisson_2d_steep", "helmholtz_2d", "diffusion_reaction_1d",
    }
    if spec.name not in supported:
        return None
    audit_points = points.detach().clone().requires_grad_(True)
    residual = strong_residual(spec.name, lambda z: exact_torch(spec.name, z), audit_points)
    return float(torch.max(torch.abs(residual)).detach().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=PROBLEM_ORDER, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--variational-form", choices=("strong", "r1", "p1"), default="strong")
    parser.add_argument(
        "--mesh-variant",
        choices=(
            "baseline",
            "official",
            "refined",
            "paper",
            "paper_figure_fine",
            "causal_refined",
        ),
        default="baseline",
    )
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--n-quad", type=int)
    parser.add_argument(
        "--test-basis",
        choices=("legendre_bubble", "legendre_full", "chebyshev_full"),
        default="legendre_bubble",
    )
    parser.add_argument(
        "--quadrature-rule",
        choices=("gauss_lobatto_legendre", "clenshaw_curtis"),
        default="gauss_lobatto_legendre",
    )
    parser.add_argument("--fourier-modes", type=int)
    parser.add_argument(
        "--coordinate-transform",
        choices=("standard", "characteristic"),
        default="standard",
    )
    parser.add_argument(
        "--initial-sampling",
        choices=("uniform", "include_interfaces", "interface_refined"),
        default="uniform",
    )
    parser.add_argument("--elements-file", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-step", type=int)
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--lbfgs-max-iter", type=int, default=0)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float)
    parser.add_argument("--initial-weight", type=float)
    parser.add_argument("--ade-kappa", type=float)
    parser.add_argument("--constraint-aggregation", choices=("separate", "combined"), default="separate")
    parser.add_argument("--temporal-test-basis", choices=("bubble", "full"), default="bubble")
    parser.add_argument("--integrate-time-by-parts", action="store_true")
    parser.add_argument("--hard-constraints", action="store_true")
    parser.add_argument("--allen-soft-periodic", action="store_true")
    parser.add_argument(
        "--projection-normalization",
        choices=("integral", "element_average"),
        default="integral",
    )
    parser.add_argument("--temporal-curriculum-stages", type=int, default=1)
    parser.add_argument("--curriculum-final-fraction", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SPECS[args.problem]
    overrides = {}
    if args.n_test is not None:
        overrides["n_test"] = args.n_test
    if args.n_quad is not None:
        overrides["n_quad"] = args.n_quad
    if args.fourier_modes is not None:
        if spec.fourier_dim is None or args.fourier_modes < 1:
            raise ValueError("--fourier-modes requires a periodic problem and a positive value")
        overrides["fourier_modes"] = args.fourier_modes
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.lr_step is not None:
        overrides["lr_step"] = args.lr_step
    if args.lr_gamma is not None:
        overrides["lr_gamma"] = args.lr_gamma
    if args.boundary_weight is not None:
        overrides["boundary_weight"] = args.boundary_weight
    if args.initial_weight is not None:
        overrides["initial_weight"] = args.initial_weight
    if args.ade_kappa is not None:
        if args.problem != "advection_diffusion_1d" or args.ade_kappa <= 0.0:
            raise ValueError("--ade-kappa must be positive and is only valid for advection_diffusion_1d")
        overrides["ade_kappa"] = args.ade_kappa
    if overrides:
        spec = replace(spec, **overrides)
    if args.allen_soft_periodic:
        if args.problem != "allen_cahn_1d":
            raise ValueError("--allen-soft-periodic is valid only for Allen-Cahn")
        if args.constraint_aggregation != "separate":
            raise ValueError("Soft periodic constraints require separate constraint aggregation")
        spec = replace(spec, fourier_dim=None, fourier_modes=0)
    if args.init_checkpoint is not None and args.resume:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.lbfgs_max_iter < 0 or args.lbfgs_learning_rate <= 0.0:
        raise ValueError("L-BFGS iterations must be nonnegative and its learning rate positive")
    if args.temporal_curriculum_stages < 1:
        raise ValueError("--temporal-curriculum-stages must be at least 1")
    if not 0.0 < args.curriculum_final_fraction < 1.0:
        raise ValueError("--curriculum-final-fraction must be between 0 and 1")
    if args.temporal_curriculum_stages > 1 and len(spec.domain) != 2:
        raise ValueError("Temporal curriculum is only valid for space-time problems")
    if args.integrate_time_by_parts and (
        args.variational_form != "p1"
        or args.problem not in {"burgers_1d", "allen_cahn_1d", "diffusion_reaction_1d"}
        or args.temporal_test_basis != "full"
    ):
        raise ValueError(
            "--integrate-time-by-parts requires a supported space-time P1 problem "
            "with a full temporal basis"
        )
    if args.hard_constraints and args.problem != "burgers_1d":
        raise ValueError("--hard-constraints is currently valid only for Burgers")
    if args.coordinate_transform == "characteristic" and args.problem != "advection_discontinuous_1d":
        raise ValueError(
            "The characteristic coordinate transform is restricted to discontinuous advection"
        )
    if args.initial_sampling != "uniform" and args.problem != "advection_discontinuous_1d":
        raise ValueError(
            "Interface-refined initial sampling is restricted to discontinuous advection"
        )
    if args.variational_form == "r1" and args.problem not in {
        "burgers_1d", "allen_cahn_1d", "diffusion_reaction_1d"
    }:
        raise ValueError(
            "The paper R1 mode is restricted to Burgers, Allen-Cahn, and diffusion-reaction"
        )
    if args.test_basis == "chebyshev_full" and args.variational_form != "r1":
        raise ValueError("Full Chebyshev tests are currently restricted to the paper R1 mode")
    if args.test_basis == "legendre_full" and args.variational_form != "strong":
        raise ValueError("Full Legendre tests are restricted to the strong residual form")
    if args.quadrature_rule == "clenshaw_curtis" and args.test_basis != "chebyshev_full":
        raise ValueError("Clenshaw-Curtis quadrature requires the full Chebyshev test basis")
    if args.variational_form == "r1" and (
        args.test_basis != "chebyshev_full" or args.quadrature_rule != "clenshaw_curtis"
    ):
        raise ValueError("Paper R1 mode requires full Chebyshev tests and Clenshaw-Curtis quadrature")
    if args.variational_form == "p1" and args.problem not in {
        "poisson_2d_peak", "poisson_2d_steep", "poisson_lshape", "helmholtz_1d",
        "helmholtz_2d", "burgers_1d", "allen_cahn_1d", "diffusion_reaction_1d",
        "advection_diffusion_1d"
    }:
        raise ValueError(f"p1 is not implemented for {args.problem}")
    iterations = args.iterations if args.iterations is not None else spec.iterations
    if args.smoke:
        iterations = min(iterations, 3)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    set_seed(args.seed)

    output_dir = args.output_dir or ROOT / "results" / "table1_hpvpinn" / spec.name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.elements_file is None:
        elements = build_elements(spec.name, args.mesh_variant)
    else:
        elements = np.load(args.elements_file, allow_pickle=False)
        expected_shape = (2, len(spec.domain))
        if elements.ndim != 3 or tuple(elements.shape[1:]) != expected_shape:
            raise ValueError(
                f"Element file must have shape (n, {expected_shape[0]}, {expected_shape[1]})"
            )
        if elements.shape[0] < 1 or not np.all(np.isfinite(elements)):
            raise ValueError("Element file must contain at least one finite element")
        if np.any(elements[:, 1, :] <= elements[:, 0, :]):
            raise ValueError("Every element must have strictly increasing bounds")
        for dimension, (low, high) in enumerate(spec.domain):
            if np.any(elements[:, 0, dimension] < low) or np.any(elements[:, 1, dimension] > high):
                raise ValueError("Element file contains bounds outside the problem domain")
    local_points, weighted_tests, weighted_test_x, weighted_test_y = quadrature_and_tests(
        spec.n_quad, spec.n_test, len(spec.domain), args.temporal_test_basis == "full",
        args.test_basis, args.quadrature_rule,
    )
    quadrature_points, jacobians = map_quadrature(elements, local_points)
    points = tensor(quadrature_points.reshape(-1, len(spec.domain)), device, dtype)
    weighted_tests_t = tensor(weighted_tests, device, dtype)
    weighted_test_x_t = None if weighted_test_x is None else tensor(weighted_test_x, device, dtype)
    weighted_test_y_t = None if weighted_test_y is None else tensor(weighted_test_y, device, dtype)
    half_sizes_t = tensor((elements[:, 1, :] - elements[:, 0, :]) / 2.0, device, dtype)
    jacobians_t = tensor(jacobians, device, dtype)
    constraints = build_constraints(
        spec,
        device,
        dtype,
        args.constraint_aggregation,
        args.allen_soft_periodic,
        args.initial_sampling,
    )

    time_boundary_points_t: tuple[torch.Tensor, torch.Tensor] | None = None
    weighted_space_tests_t: torch.Tensor | None = None
    time_test_boundary_values_t: tuple[torch.Tensor, torch.Tensor] | None = None
    if args.integrate_time_by_parts:
        one_dimensional_points, one_dimensional_weights = GaussLobattoJacobiWeights(
            spec.n_quad, 0, 0
        )
        one_dimensional_points = np.asarray(one_dimensional_points, dtype=np.float64).reshape(-1)
        one_dimensional_weights = np.asarray(one_dimensional_weights, dtype=np.float64).reshape(-1)
        space_tests = np.asarray(
            [
                legendre_value(n + 1, one_dimensional_points)
                - legendre_value(n - 1, one_dimensional_points)
                for n in range(1, spec.n_test + 1)
            ]
        )
        time_degrees = range(spec.n_test)
        time_test_bottom = np.asarray([legendre_value(n, np.asarray([-1.0]))[0] for n in time_degrees])
        time_test_top = np.asarray([legendre_value(n, np.asarray([1.0]))[0] for n in time_degrees])
        local_bottom = np.column_stack(
            (one_dimensional_points, -np.ones_like(one_dimensional_points))
        )
        local_top = np.column_stack(
            (one_dimensional_points, np.ones_like(one_dimensional_points))
        )
        bottom_points, _ = map_quadrature(elements, local_bottom)
        top_points, _ = map_quadrature(elements, local_top)
        time_boundary_points_t = (
            tensor(bottom_points.reshape(-1, 2), device, dtype),
            tensor(top_points.reshape(-1, 2), device, dtype),
        )
        weighted_space_tests_t = tensor(
            space_tests * one_dimensional_weights[None, :], device, dtype
        )
        time_test_boundary_values_t = (
            tensor(time_test_bottom, device, dtype),
            tensor(time_test_top, device, dtype),
        )

    audit = exact_residual_audit(spec, points[: min(points.shape[0], 4096)])
    if audit is not None:
        tolerance = 2e-5 if dtype == torch.float32 else 1e-8
        if not np.isfinite(audit) or audit > tolerance:
            raise RuntimeError(f"Analytic residual audit failed for {spec.name}: {audit:.3e}")

    model: nn.Module = ModifiedMLP(spec, dtype, args.coordinate_transform)
    if args.hard_constraints:
        model = BurgersConstrainedTrial(model)
    model = model.to(device)
    initial_checkpoint_iteration = -1
    prior_total_iterations = 0
    if args.init_checkpoint is not None:
        source_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        source_model = source_checkpoint.get("model", source_checkpoint.get("model_state_dict"))
        if source_model is None:
            raise ValueError(f"No model state found in {args.init_checkpoint}")
        model.load_state_dict(source_model)
        initial_checkpoint_iteration = int(source_checkpoint.get("iteration", -1))
        prior_total_iterations = int(
            source_checkpoint.get("total_iterations", initial_checkpoint_iteration + 1)
        )
        print(f"Initialized model from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=spec.lr_step, gamma=spec.lr_gamma)
    start_iteration = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_iteration = int(checkpoint["iteration"]) + 1
        prior_total_iterations = int(checkpoint.get("prior_total_iterations", 0))
        history = checkpoint.get("history", [])
        print(f"Resuming at iteration {start_iteration}", flush=True)

    run_config = {
        **asdict(spec), "seed": args.seed, "iterations": iterations, "dtype": args.dtype,
        "device": str(device), "network": "modified_mlp", "hidden_layers": 6, "width": 128,
        "elements": elements.tolist(), "n_elements": int(elements.shape[0]),
        "mesh_variant": args.mesh_variant,
        "coordinate_transform": args.coordinate_transform,
        "initial_sampling": args.initial_sampling,
        "elements_file": str(args.elements_file) if args.elements_file is not None else None,
        "variational_form": args.variational_form,
        "test_function_derivatives": args.variational_form == "p1",
        "test_basis": args.test_basis,
        "quadrature_rule": args.quadrature_rule,
        "constraint_aggregation": args.constraint_aggregation,
        "temporal_test_basis": args.temporal_test_basis,
        "integrate_time_by_parts": args.integrate_time_by_parts,
        "hard_constraints": args.hard_constraints,
        "allen_soft_periodic": args.allen_soft_periodic,
        "projection_normalization": args.projection_normalization,
        "temporal_curriculum_stages": args.temporal_curriculum_stages,
        "curriculum_final_fraction": args.curriculum_final_fraction,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "lbfgs_learning_rate": args.lbfgs_learning_rate,
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "initial_checkpoint_iteration": initial_checkpoint_iteration,
        "prior_total_iterations": prior_total_iterations,
        "fixed_endpoint_metrics": True,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True)

    element_count = elements.shape[0]
    points_per_element = local_points.shape[0]
    temporal_slab_ranks: torch.Tensor | None = None
    temporal_slab_count = 0
    if args.temporal_curriculum_stages > 1:
        temporal_centers = np.mean(elements[:, :, 1], axis=1)
        unique_temporal_centers = np.unique(temporal_centers)
        temporal_slab_count = int(unique_temporal_centers.size)
        if args.temporal_curriculum_stages > temporal_slab_count:
            raise ValueError(
                "--temporal-curriculum-stages cannot exceed the number of temporal elements "
                f"({temporal_slab_count})"
            )
        slab_ranks = np.searchsorted(unique_temporal_centers, temporal_centers)
        temporal_slab_ranks = torch.as_tensor(slab_ranks, device=device, dtype=torch.long)

    def active_temporal_slab_count(iteration: int) -> int:
        if temporal_slab_ranks is None:
            return temporal_slab_count
        denominator = max(iterations - 1, 1)
        progress = iteration / denominator
        ramp_fraction = 1.0 - args.curriculum_final_fraction
        if progress >= ramp_fraction:
            return temporal_slab_count
        stage = min(
            args.temporal_curriculum_stages,
            int(progress / ramp_fraction * args.temporal_curriculum_stages) + 1,
        )
        return max(1, math.ceil(stage * temporal_slab_count / args.temporal_curriculum_stages))

    start_time = time.perf_counter()
    for iteration in range(start_iteration, iterations):
        optimizer.zero_grad(set_to_none=True)
        active_points = points.detach().clone().requires_grad_(True)
        projection = variational_projection(
            spec.name, args.variational_form, model, active_points, weighted_tests_t,
            weighted_test_x_t, weighted_test_y_t, half_sizes_t, element_count, points_per_element, spec.ade_kappa,
            args.integrate_time_by_parts, time_boundary_points_t, weighted_space_tests_t,
            time_test_boundary_values_t,
        )
        if args.projection_normalization == "element_average":
            projection = projection / jacobians_t[:, None, None]
        element_variational_loss = torch.mean(projection**2, dim=(1, 2))
        active_slab_count = active_temporal_slab_count(iteration)
        if temporal_slab_ranks is None:
            variational_loss = torch.sum(element_variational_loss)
        else:
            active_elements = temporal_slab_ranks < active_slab_count
            variational_loss = torch.sum(element_variational_loss[active_elements])
        conditions_loss, condition_parts = constraint_loss(spec.name, model, constraints)
        loss = variational_loss + conditions_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0 or iteration == iterations - 1:
            record: dict[str, float | int] = {
                "iteration": iteration,
                "loss": float(loss.detach().cpu()),
                "variational_loss": float(variational_loss.detach().cpu()),
                "constraint_loss": float(conditions_loss.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "active_temporal_slabs": active_slab_count,
                **condition_parts,
            }
            history.append(record)
            print(
                f"{spec.name} it={iteration:6d} loss={record['loss']:.4e} "
                f"var={record['variational_loss']:.4e} con={record['constraint_loss']:.4e} "
                f"lr={record['learning_rate']:.2e}", flush=True,
            )

        if args.checkpoint_every > 0 and iteration > 0 and iteration % args.checkpoint_every == 0:
            torch.save(
                {"iteration": iteration, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history,
                 "prior_total_iterations": prior_total_iterations,
                 "total_iterations": prior_total_iterations + iteration + 1}, checkpoint_path,
            )

    lbfgs_function_evaluations = 0
    if args.lbfgs_max_iter > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_learning_rate,
            max_iter=args.lbfgs_max_iter,
            max_eval=max(args.lbfgs_max_iter * 5 // 4, args.lbfgs_max_iter + 1),
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def lbfgs_closure() -> torch.Tensor:
            nonlocal lbfgs_function_evaluations
            lbfgs.zero_grad(set_to_none=True)
            active_points = points.detach().clone().requires_grad_(True)
            projection = variational_projection(
                spec.name, args.variational_form, model, active_points, weighted_tests_t,
                weighted_test_x_t, weighted_test_y_t, half_sizes_t, element_count,
                points_per_element, spec.ade_kappa, args.integrate_time_by_parts,
                time_boundary_points_t, weighted_space_tests_t, time_test_boundary_values_t,
            )
            if args.projection_normalization == "element_average":
                projection = projection / jacobians_t[:, None, None]
            variational_loss = torch.sum(torch.mean(projection**2, dim=(1, 2)))
            conditions_loss, _ = constraint_loss(spec.name, model, constraints)
            loss = variational_loss + conditions_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite L-BFGS loss")
            loss.backward()
            lbfgs_function_evaluations += 1
            if lbfgs_function_evaluations % args.log_every == 0:
                print(
                    f"{spec.name} lbfgs_eval={lbfgs_function_evaluations:6d} "
                    f"loss={float(loss.detach().cpu()):.4e} "
                    f"var={float(variational_loss.detach().cpu()):.4e} "
                    f"con={float(conditions_loss.detach().cpu()):.4e}",
                    flush=True,
                )
            return loss

        lbfgs.step(lbfgs_closure)
        active_points = points.detach().clone().requires_grad_(True)
        projection = variational_projection(
            spec.name, args.variational_form, model, active_points, weighted_tests_t,
            weighted_test_x_t, weighted_test_y_t, half_sizes_t, element_count,
            points_per_element, spec.ade_kappa, args.integrate_time_by_parts,
            time_boundary_points_t, weighted_space_tests_t, time_test_boundary_values_t,
        )
        if args.projection_normalization == "element_average":
            projection = projection / jacobians_t[:, None, None]
        final_variational_loss = torch.sum(torch.mean(projection**2, dim=(1, 2)))
        final_conditions_loss, final_condition_parts = constraint_loss(spec.name, model, constraints)
        final_loss = final_variational_loss + final_conditions_loss
        record = {
            "iteration": iterations,
            "loss": float(final_loss.detach().cpu()),
            "variational_loss": float(final_variational_loss.detach().cpu()),
            "constraint_loss": float(final_conditions_loss.detach().cpu()),
            "learning_rate": args.lbfgs_learning_rate,
            "active_temporal_slabs": temporal_slab_count,
            "lbfgs_function_evaluations": lbfgs_function_evaluations,
            **final_condition_parts,
        }
        history.append(record)
        print(
            f"{spec.name} lbfgs_done evals={lbfgs_function_evaluations} "
            f"loss={record['loss']:.4e} var={record['variational_loss']:.4e} "
            f"con={record['constraint_loss']:.4e}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    coordinates, y_true, evaluation_metadata = evaluation_data(spec.name, spec.ade_kappa)
    y_pred = predict(spec.name, model, coordinates, device, dtype)
    errors = np.abs(y_true - y_pred)
    metrics = {
        "problem": spec.name,
        "title": spec.title,
        "seed": args.seed,
        "iterations": iterations,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "lbfgs_function_evaluations": lbfgs_function_evaluations,
        "total_iterations_including_initial_checkpoint": prior_total_iterations + iterations,
        "relative_l2_error": relative_l2(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
        "evaluation_points": int(y_true.shape[0]),
        "elapsed_seconds": elapsed,
        "final_loss": history[-1]["loss"],
        "final_variational_loss": history[-1]["variational_loss"],
        "final_constraint_loss": history[-1]["constraint_loss"],
        "exact_strong_residual_max": audit,
        "evaluation": evaluation_metadata,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output_dir / "predictions.npz", coordinates=coordinates, y_true=y_true, y_pred=y_pred,
        absolute_error=errors,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(
        {"iteration": iterations - 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(), "history": history,
         "prior_total_iterations": prior_total_iterations,
         "total_iterations": prior_total_iterations + iterations}, checkpoint_path,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
