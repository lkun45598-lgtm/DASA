"""One-space-dimensional time-dependent benchmark definitions."""

from __future__ import annotations

import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import tensorflow as tf

from ..config import SUITE_ROOT
from ..grids import time_1d_grid
from .base import PDEProblem, TrainingData, latin_hypercube


PROJECT_ROOT = SUITE_ROOT.parents[1]
ORIGINAL_DATASETS = {
    "burgers": (PROJECT_ROOT / "Burgers" / "burgers_shock.mat", "t", "usol"),
    "allen_cahn": (PROJECT_ROOT / "Allen-Cahn" / "AC.mat", "tt", "uu"),
}


class Time1DProblem(PDEProblem):
    def boundary_prediction(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        if self.config.case_id != "allen_cahn":
            return super().boundary_prediction(model, points)
        with tf.GradientTape() as tape:
            tape.watch(points)
            values = model(points)
        spatial_derivative = tape.gradient(
            values, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 1:2]
        per_side = tf.shape(points)[0] // 2
        value_error = values[:per_side] - values[per_side:]
        derivative_error = spatial_derivative[:per_side] - spatial_derivative[per_side:]
        return tf.concat((value_error, derivative_error), axis=0)

    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        case_id = self.config.case_id
        if case_id in {"burgers", "advection_diffusion"}:
            return -np.sin(np.pi * x)
        if case_id == "allen_cahn":
            return x**2 * np.cos(np.pi * x)
        if case_id == "diffusion_reaction":
            result = np.zeros_like(x)
            for frequency in range(1, 5):
                result += np.sin(frequency * x) / frequency
            return result + np.sin(8.0 * x) / 8.0
        if case_id == "discontinuous_advection":
            half_width = self.config.parameters["pulse_half_width"]
            return (np.abs(x) <= half_width).astype(np.float64)
        raise ValueError(f"unsupported time case: {case_id}")

    def initial_tf(self, x: tf.Tensor) -> tf.Tensor:
        dtype = x.dtype
        case_id = self.config.case_id
        pi = tf.cast(math.pi, dtype)
        if case_id in {"burgers", "advection_diffusion"}:
            return -tf.sin(pi * x)
        if case_id == "allen_cahn":
            return tf.square(x) * tf.cos(pi * x)
        if case_id == "diffusion_reaction":
            result = tf.zeros_like(x)
            for frequency in range(1, 5):
                value = tf.cast(frequency, dtype)
                result += tf.sin(value * x) / value
            return result + tf.sin(tf.cast(8.0, dtype) * x) / tf.cast(8.0, dtype)
        if case_id == "discontinuous_advection":
            half_width = tf.cast(self.config.parameters["pulse_half_width"], dtype)
            return tf.cast(tf.abs(x) <= half_width, dtype)
        raise ValueError(f"unsupported time case: {case_id}")

    @staticmethod
    def _advection_coefficients(
        terms: int, velocity: float, kappa: float
    ) -> tuple[np.ndarray, np.ndarray]:
        dtype = np.longdouble
        length = dtype(2.0)
        a = dtype(velocity) / (dtype(2.0) * dtype(kappa))
        n = np.arange(1, terms + 1, dtype=dtype)
        beta = n * np.longdouble(np.pi) / length
        initial_frequency = np.longdouble(np.pi)

        def damped_cosine_integral(frequency: np.ndarray) -> np.ndarray:
            numerator = a + np.exp(-a * length) * (
                -a * np.cos(frequency * length)
                + frequency * np.sin(frequency * length)
            )
            return numerator / (a**2 + frequency**2)

        coefficients = dtype(0.5) * (
            damped_cosine_integral(initial_frequency - beta)
            - damped_cosine_integral(initial_frequency + beta)
        )
        return np.asarray(beta, dtype=np.float64), np.asarray(coefficients, dtype=np.float64)

    def _advection_exact_tf(self, points: tf.Tensor) -> tf.Tensor:
        dtype = points.dtype
        velocity = self.config.parameters["velocity"]
        kappa = self.config.parameters["kappa"]
        terms = int(self.config.parameters["series_terms"])
        beta_np, coefficients_np = self._advection_coefficients(terms, velocity, kappa)
        beta = tf.cast(beta_np[None, :], dtype)
        coefficients = tf.cast(coefficients_np[None, :], dtype)
        t = points[:, 0:1]
        x = points[:, 1:2]
        z = x + tf.cast(1.0, dtype)
        a = tf.cast(velocity / (2.0 * kappa), dtype)
        temporal_shift = tf.cast(velocity**2 / (4.0 * kappa), dtype)
        exponent = a * z - temporal_shift * t - tf.cast(kappa, dtype) * t * beta**2
        series = tf.reduce_sum(
            coefficients * tf.sin(z * beta) * tf.exp(exponent), axis=1, keepdims=True
        )
        series = tf.where(t == 0.0, self.initial_tf(x), series)
        boundary = (x == tf.cast(-1.0, dtype)) | (x == tf.cast(1.0, dtype))
        return tf.where(boundary, tf.zeros_like(series), series)

    def _advection_exact_numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        velocity = self.config.parameters["velocity"]
        kappa = self.config.parameters["kappa"]
        spatial_points = 2049
        x_grid = np.linspace(-1.0, 1.0, spatial_points, dtype=np.float64)
        spacing = x_grid[1] - x_grid[0]
        interior_size = spatial_points - 2

        rows: list[int] = []
        columns: list[int] = []
        entries: list[float] = []
        weight_cache: dict[tuple[int, ...], np.ndarray] = {}
        for grid_index in range(1, spatial_points - 1):
            if grid_index == 1:
                stencil = np.arange(0, 6)
            elif grid_index == spatial_points - 2:
                stencil = np.arange(spatial_points - 6, spatial_points)
            else:
                stencil = np.arange(grid_index - 2, grid_index + 3)
            offsets = tuple(int(value) for value in stencil - grid_index)
            if offsets not in weight_cache:
                vandermonde = np.asarray(offsets, dtype=np.float64)[None, :] ** np.arange(
                    len(offsets), dtype=np.float64
                )[:, None]
                first_target = np.zeros(len(offsets), dtype=np.float64)
                first_target[1] = 1.0
                second_target = np.zeros(len(offsets), dtype=np.float64)
                second_target[2] = 2.0
                first = np.linalg.solve(vandermonde, first_target) / spacing
                second = np.linalg.solve(vandermonde, second_target) / spacing**2
                weight_cache[offsets] = kappa * second - velocity * first
            for column, weight in zip(stencil, weight_cache[offsets]):
                if 0 < column < spatial_points - 1:
                    rows.append(grid_index - 1)
                    columns.append(int(column - 1))
                    entries.append(float(weight))
        operator = csr_matrix(
            (entries, (rows, columns)), shape=(interior_size, interior_size)
        )

        initial = -np.sin(np.pi * x_grid[1:-1])
        times = np.unique(points[:, 0])
        if times.size > 1 and times[0] == 0.0 and np.allclose(
            np.diff(times), times[1] - times[0], rtol=0.0, atol=1e-14
        ):
            interior_solutions = expm_multiply(
                operator,
                initial,
                start=0.0,
                stop=float(times[-1]),
                num=times.size,
                endpoint=True,
            )
        else:
            snapshots = []
            current = initial
            previous_time = 0.0
            for time in times:
                if time < previous_time:
                    raise ValueError("advection-diffusion times must be nondecreasing")
                if time > previous_time:
                    current = expm_multiply(operator * (time - previous_time), current)
                snapshots.append(current.copy())
                previous_time = float(time)
            interior_solutions = np.asarray(snapshots)

        full_solutions = np.zeros((times.size, spatial_points), dtype=np.float64)
        full_solutions[:, 1:-1] = interior_solutions
        result = np.empty((points.shape[0], 1), dtype=np.float64)
        for time_index, time in enumerate(times):
            selected = points[:, 0] == time
            interpolator = CubicSpline(x_grid, full_solutions[time_index])
            result[selected, 0] = interpolator(points[selected, 1])
        at_initial = points[:, 0] == 0.0
        result[at_initial] = self.initial_numpy(points[at_initial, 1:2])
        at_boundary = np.isin(points[:, 1], (-1.0, 1.0))
        result[at_boundary] = 0.0
        return result

    def exact_tf(self, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points)
        dtype = points.dtype
        t = points[:, 0:1]
        x = points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "diffusion_reaction":
            return tf.exp(-t) * self.initial_tf(x)
        if case_id == "advection_diffusion":
            return self._advection_exact_tf(points)
        if case_id == "discontinuous_advection":
            velocity = tf.cast(self.config.parameters["velocity"], dtype)
            return self.initial_tf(x - velocity * t)
        raise ValueError(f"{case_id} uses numerical reference truth")

    def exact_numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        t = points[:, 0:1]
        x = points[:, 1:2]
        case_id = self.config.case_id
        if case_id == "diffusion_reaction":
            return np.exp(-t) * self.initial_numpy(x)
        if case_id == "advection_diffusion":
            return self._advection_exact_numpy(points)
        if case_id == "discontinuous_advection":
            return self.initial_numpy(x - self.config.parameters["velocity"] * t)
        raise ValueError(f"{case_id} uses numerical reference truth")

    def _diffusion_reaction_forcing(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        dtype = x.dtype
        return tf.exp(-t) * (
            tf.cast(3.0 / 2.0, dtype) * tf.sin(tf.cast(2.0, dtype) * x)
            + tf.cast(8.0 / 3.0, dtype) * tf.sin(tf.cast(3.0, dtype) * x)
            + tf.cast(15.0 / 4.0, dtype) * tf.sin(tf.cast(4.0, dtype) * x)
            + tf.cast(63.0 / 8.0, dtype) * tf.sin(tf.cast(8.0, dtype) * x)
        )

    def residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(points)
        with tf.GradientTape() as outer:
            outer.watch(points)
            with tf.GradientTape() as inner:
                inner.watch(points)
                values = model(points)
            first = inner.gradient(
                values, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            u_t = first[:, 0:1]
            u_x = first[:, 1:2]
        u_xx = outer.gradient(
            u_x, points, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )[:, 1:2]
        case_id = self.config.case_id
        if case_id == "burgers":
            return u_t + values * u_x - self.config.parameters["viscosity"] * u_xx
        if case_id == "allen_cahn":
            diffusion = self.config.parameters["diffusion"]
            reaction = self.config.parameters["reaction"]
            return u_t - diffusion * u_xx + reaction * values**3 - reaction * values
        if case_id == "diffusion_reaction":
            return (
                u_t
                - self.config.parameters["diffusion"] * u_xx
                - self._diffusion_reaction_forcing(points[:, 0:1], points[:, 1:2])
            )
        if case_id in {"advection_diffusion", "discontinuous_advection"}:
            return (
                u_t
                + self.config.parameters["velocity"] * u_x
                - self.config.parameters["kappa"] * u_xx
            )
        raise ValueError(f"unsupported time case: {case_id}")

    def sample_training(self, seed: int) -> TrainingData:
        residual = latin_hypercube(
            self.config.domain, self.config.sampling.residual, seed
        )
        x_domain = (self.config.domain[1],)
        initial_x = latin_hypercube(x_domain, self.config.sampling.initial, seed + 1)
        initial_points = np.column_stack((np.zeros(initial_x.shape[0]), initial_x[:, 0]))

        boundary_points = None
        boundary_values = None
        if self.config.sampling.boundary:
            per_side = self.config.sampling.boundary
            time_domain = (self.config.domain[0],)
            boundary_t = latin_hypercube(time_domain, per_side, seed + 2)[:, 0]
            boundary_points = np.vstack(
                (
                    np.column_stack((boundary_t, np.full(per_side, self.config.domain[1][0]))),
                    np.column_stack((boundary_t, np.full(per_side, self.config.domain[1][1]))),
                )
            )
            boundary_values = np.zeros((2 * per_side, 1), dtype=np.float64)

        return TrainingData(
            residual=residual,
            boundary_points=boundary_points,
            boundary_values=boundary_values,
            initial_points=initial_points,
            initial_values=self.initial_numpy(initial_x),
        )

    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.case_id in {"burgers", "allen_cahn"}:
            path, time_key, value_key = ORIGINAL_DATASETS[self.config.case_id]
            data = loadmat(path)
            x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
            t = np.asarray(data[time_key], dtype=np.float64).reshape(-1)
            field = np.asarray(data[value_key], dtype=np.float64)
            if field.shape != (x.size, t.size):
                raise ValueError(f"invalid original dataset shape for {self.config.case_id}")
            tt, xx = np.meshgrid(t, x, indexing="ij")
            points = np.column_stack((tt.ravel(), xx.ravel()))
            values = field.T.reshape(-1, 1)
            return points, values
        points = time_1d_grid(self.config.domain)
        return points, self.exact_numpy(points)
