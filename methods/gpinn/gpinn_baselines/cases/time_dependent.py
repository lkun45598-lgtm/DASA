"""Time-dependent equations (19)-(24) and (26)-(28)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..autodiff import partial, residual_gradients, second
from ..sampling import lhs, time_interval_boundary
from .base import PDECase


class TimeDirichletCase(PDECase):
    """Common sampler for (x,t) problems with two spatial Dirichlet boundaries."""

    coordinate_names = ("x", "t")

    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def boundary_numpy(self, points: np.ndarray) -> np.ndarray:
        return np.zeros((len(points), 1), dtype=np.float64)

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        sampling = self.config["sampling"]
        boundary = time_interval_boundary(
            sampling["n_boundary"],
            self.lower[0],
            self.upper[0],
            self.lower[1],
            self.upper[1],
            rng,
        )
        initial_x = lhs(
            sampling["n_initial"], (self.lower[0],), (self.upper[0],), rng
        )
        initial = np.column_stack((initial_x[:, 0], np.full(len(initial_x), self.lower[1])))
        values.update(
            {
                "boundary_points": boundary,
                "boundary_values": self.boundary_numpy(boundary),
                "initial_points": initial,
                "initial_values": self.initial_numpy(initial_x),
            }
        )
        return values


class Burgers1D(TimeDirichletCase):
    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        # Yu et al.'s original Burgers code uses only 1500 interior residual
        # points.  Initial and boundary conditions are imposed exactly by the
        # output transform below rather than by sampled soft losses.
        return PDECase.sample_training(self, rng)

    def output_transform(
        self, points: torch.Tensor, raw_output: torch.Tensor
    ) -> torch.Tensor:
        x = points[:, 0:1]
        t = points[:, 1:2]
        return (
            (1.0 - x)
            * (1.0 + x)
            * (1.0 - torch.exp(-t))
            * raw_output
            - torch.sin(torch.pi * x)
        )

    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        return -np.sin(np.pi * x)

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        ux = partial(u, points, 0)
        ut = partial(u, points, 1)
        uxx = second(u, points, 0)
        nu = float(self.parameters["viscosity"])
        return {"burgers": ut + u * ux - nu * uxx}

    def physics_loss_terms(
        self,
        model: Callable,
        residual_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Preserve the repository objective L_f + w L_fx + w L_ft."""

        residual = self.residuals(model, residual_points)
        enhanced = residual_gradients(
            residual, residual_points, self.residual_gradient_dims()
        )
        return {
            "pde": torch.stack(
                [torch.mean(value**2) for value in residual.values()]
            ).sum(),
            "gradient": torch.stack(
                [torch.mean(value**2) for value in enhanced.values()]
            ).sum(),
        }


class AllenCahn1D(PDECase):
    coordinate_names = ("x", "t")

    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        return x**2 * np.cos(np.pi * x)

    def sample_training(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        values = super().sample_training(rng)
        sampling = self.config["sampling"]
        initial_x = lhs(
            sampling["n_initial"], (self.lower[0],), (self.upper[0],), rng
        )
        initial = np.column_stack((initial_x[:, 0], np.zeros(len(initial_x))))
        times = lhs(
            sampling["n_boundary"], (self.lower[1],), (self.upper[1],), rng
        )[:, 0]
        left = np.column_stack((np.full(len(times), self.lower[0]), times))
        right = np.column_stack((np.full(len(times), self.upper[0]), times))
        values.update(
            {
                "initial_points": initial,
                "initial_values": self.initial_numpy(initial_x),
                "periodic_left": left,
                "periodic_right": right,
            }
        )
        return values

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        diffusion = float(self.parameters["diffusion"])
        reaction = float(self.parameters["reaction"])
        return {
            "allen_cahn": partial(u, points, 1)
            - diffusion * second(u, points, 0)
            + reaction * u**3
            - reaction * u
        }

    def constraint_terms(self, model: Callable, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        terms = super().constraint_terms(model, batch)
        left, right = batch["periodic_left"], batch["periodic_right"]
        u_left = model(left)[:, 0:1]
        u_right = model(right)[:, 0:1]
        terms["periodic_value"] = torch.mean((u_left - u_right) ** 2)
        ux_left = partial(u_left, left, 0)
        ux_right = partial(u_right, right, 0)
        terms["periodic_derivative"] = torch.mean((ux_left - ux_right) ** 2)
        return terms


class DiffusionReaction1D(TimeDirichletCase):
    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        value = np.sin(8.0 * x) / 8.0
        for k in range(1, 5):
            value = value + np.sin(k * x) / k
        return value

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x, t = points[:, 0:1], points[:, 1:2]
        return np.exp(-t) * self.initial_numpy(x)

    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, t = points[:, 0:1], points[:, 1:2]
        value = torch.sin(8.0 * x) / 8.0
        for k in range(1, 5):
            value = value + torch.sin(k * x) / k
        return torch.exp(-t) * value

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        x, t = points[:, 0:1], points[:, 1:2]
        reaction = torch.exp(-t) * (
            1.5 * torch.sin(2.0 * x)
            + (8.0 / 3.0) * torch.sin(3.0 * x)
            + (15.0 / 4.0) * torch.sin(4.0 * x)
            + (63.0 / 8.0) * torch.sin(8.0 * x)
        )
        diffusion = float(self.parameters["diffusion"])
        return {
            "diffusion_reaction": partial(u, points, 1)
            - diffusion * second(u, points, 0)
            - reaction
        }


class AdvectionDiffusion1D(TimeDirichletCase):
    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        return -np.sin(np.pi * x)

    @staticmethod
    def _safe_erfc(z: np.ndarray) -> np.ndarray:
        from scipy.special import erfc

        values = np.zeros_like(z, dtype=np.complex128)
        active = np.real(z) < 26.0
        values[active] = erfc(z[active])
        return values

    def _gaussian_sine_integral(self, center: np.ndarray, time: np.ndarray) -> np.ndarray:
        kappa = float(self.parameters["diffusivity"])
        root = np.sqrt(kappa * time)
        z0 = -center / (2.0 * root) - 1j * np.pi * root
        z1 = (2.0 - center) / (2.0 * root) - 1j * np.pi * root
        delta = np.empty_like(z0, dtype=np.complex128)
        left, right = center < 0.0, center > 2.0
        middle = ~(left | right)
        delta[left] = self._safe_erfc(z0[left]) - self._safe_erfc(z1[left])
        delta[right] = self._safe_erfc(-z1[right]) - self._safe_erfc(-z0[right])
        delta[middle] = 2.0 - self._safe_erfc(z1[middle]) - self._safe_erfc(-z0[middle])
        phase = np.exp(1j * np.pi * center - np.pi**2 * kappa * time)
        return 0.5 * np.imag(phase * delta)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        """Stable image-series equivalent of the paper's 1000-term exact series."""
        x, time = points[:, 0], points[:, 1]
        values = -np.sin(np.pi * x)
        positive = time > 0.0
        if not np.any(positive):
            return values[:, None]
        velocity = float(self.parameters["velocity"])
        kappa = float(self.parameters["diffusivity"])
        terms = int(self.parameters["series_terms"])
        y, t = x[positive] + 1.0, time[positive]
        result = np.zeros_like(y)
        denominator = 4.0 * kappa * t
        half = terms // 2
        cutoff = -50.0
        for m in range(-half, terms - half):
            log_scale_1 = -velocity * m * 2.0 / kappa
            center_1 = y + 4.0 * m - velocity * t
            distance_1 = np.maximum(np.maximum(-center_1, center_1 - 2.0), 0.0)
            active_1 = log_scale_1 - distance_1**2 / denominator > cutoff
            if np.any(active_1):
                result[active_1] += np.exp(log_scale_1) * self._gaussian_sine_integral(
                    center_1[active_1], t[active_1]
                )

            log_scale_2 = velocity * (y + 2.0 * m) / kappa
            center_2 = -y - 4.0 * m - velocity * t
            distance_2 = np.maximum(np.maximum(-center_2, center_2 - 2.0), 0.0)
            active_2 = log_scale_2 - distance_2**2 / denominator > cutoff
            if np.any(active_2):
                result[active_2] -= np.exp(log_scale_2[active_2]) * self._gaussian_sine_integral(
                    center_2[active_2], t[active_2]
                )
        values[positive] = result
        return values[:, None]

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        velocity = float(self.parameters["velocity"])
        kappa = float(self.parameters["diffusivity"])
        return {
            "advection_diffusion": partial(u, points, 1)
            + velocity * partial(u, points, 0)
            - kappa * second(u, points, 0)
        }


class AdvectionDiscontinuous1D(TimeDirichletCase):
    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        return (np.abs(x) <= 0.2).astype(np.float64)

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        x, t = points[:, 0:1], points[:, 1:2]
        velocity = float(self.parameters["velocity"])
        return (np.abs(x - velocity * t) <= 0.2).astype(np.float64)

    def _exact_torch(self, points: torch.Tensor) -> torch.Tensor:
        x, t = points[:, 0:1], points[:, 1:2]
        velocity = float(self.parameters["velocity"])
        indicator = (torch.abs(x - velocity * t) <= 0.2).to(points.dtype)
        return indicator + 0.0 * x + 0.0 * t

    def residuals(self, model: Callable, points: torch.Tensor) -> dict[str, torch.Tensor]:
        u = model(points)[:, 0:1]
        velocity = float(self.parameters["velocity"])
        kappa = float(self.parameters["diffusivity"])
        residual = partial(u, points, 1) + velocity * partial(u, points, 0)
        if kappa != 0.0:
            residual = residual - kappa * second(u, points, 0)
        return {
            "advection": residual
        }
