"""Independent high-resolution numerical references and convergence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


NUMERICAL_CASES = (
    "burgers_1d",
    "allen_cahn_1d",
    "poisson_lshape",
    "lid_driven_cavity",
)


def _save(path: Path, metadata: dict, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return path


class _Tridiagonal:
    def __init__(self, lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray):
        self.lower = lower.astype(np.float64)
        self.diagonal = diagonal.astype(np.float64)
        self.upper = upper.astype(np.float64)
        n = len(diagonal)
        self.cprime = np.empty(n - 1, dtype=np.float64)
        self.denominator = np.empty(n, dtype=np.float64)
        self.denominator[0] = self.diagonal[0]
        self.cprime[0] = self.upper[0] / self.denominator[0]
        for i in range(1, n - 1):
            self.denominator[i] = self.diagonal[i] - self.lower[i - 1] * self.cprime[i - 1]
            self.cprime[i] = self.upper[i] / self.denominator[i]
        self.denominator[-1] = self.diagonal[-1] - self.lower[-1] * self.cprime[-1]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        n = len(rhs)
        dprime = np.empty(n, dtype=np.float64)
        dprime[0] = rhs[0] / self.denominator[0]
        for i in range(1, n):
            dprime[i] = (rhs[i] - self.lower[i - 1] * dprime[i - 1]) / self.denominator[i]
        solution = np.empty(n, dtype=np.float64)
        solution[-1] = dprime[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = dprime[i] - self.cprime[i] * solution[i + 1]
        return solution


def _burgers_grid(nx: int, dt: float, save_dt: float = 0.0025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order IMEX BDF2 finite differences for equation (19)."""
    viscosity = 0.01 / np.pi
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    h = x[1] - x[0]
    steps = int(round(1.0 / dt))
    stride = int(round(save_dt / dt))
    if not np.isclose(stride * dt, save_dt):
        raise ValueError("save_dt must be an integer multiple of dt")
    times = np.arange(0, steps + 1, stride, dtype=np.float64) * dt
    history = np.empty((len(times), nx), dtype=np.float64)
    current = -np.sin(np.pi * x)
    current[[0, -1]] = 0.0
    history[0] = current

    def nonlinear(value: np.ndarray) -> np.ndarray:
        derivative = (value[2:] - value[:-2]) / (2.0 * h)
        return -value[1:-1] * derivative

    m = nx - 2
    alpha1 = viscosity / h**2
    first_solver = _Tridiagonal(
        -alpha1 * np.ones(m - 1),
        (1.0 / dt + 2.0 * alpha1) * np.ones(m),
        -alpha1 * np.ones(m - 1),
    )
    previous = current.copy()
    first = current.copy()
    first[1:-1] = first_solver.solve(current[1:-1] / dt + nonlinear(current))
    current = first
    save_index = 1
    if stride == 1:
        history[save_index] = current
        save_index += 1

    alpha2 = viscosity / h**2
    bdf_solver = _Tridiagonal(
        -alpha2 * np.ones(m - 1),
        (3.0 / (2.0 * dt) + 2.0 * alpha2) * np.ones(m),
        -alpha2 * np.ones(m - 1),
    )
    nonlinear_previous = nonlinear(previous)
    nonlinear_current = nonlinear(current)
    for step in range(2, steps + 1):
        rhs = (
            4.0 * current[1:-1] / (2.0 * dt)
            - previous[1:-1] / (2.0 * dt)
            + 2.0 * nonlinear_current
            - nonlinear_previous
        )
        following = np.zeros_like(current)
        following[1:-1] = bdf_solver.solve(rhs)
        previous, current = current, following
        nonlinear_previous, nonlinear_current = nonlinear_current, nonlinear(following)
        if step % stride == 0:
            history[save_index] = current
            save_index += 1
    return x, times, history


def generate_burgers(path: Path) -> Path:
    coarse_x, coarse_t, coarse_u = _burgers_grid(1025, 5.0e-4)
    fine_x, fine_t, fine_u = _burgers_grid(2049, 2.5e-4)
    difference = fine_u[:, ::2] - coarse_u
    convergence = {
        "coarse_nx": 1025,
        "coarse_dt": 5.0e-4,
        "fine_nx": 2049,
        "fine_dt": 2.5e-4,
        "relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(fine_u[:, ::2])),
        "max_difference": float(np.max(np.abs(difference))),
    }
    metadata = {
        "case": "burgers_1d",
        "method": "centered finite differences plus second-order IMEX-BDF2",
        "equation": "u_t + u*u_x - (0.01/pi)*u_xx = 0",
        "boundary": "u(-1,t)=u(1,t)=0",
        "time_interval": [0.0, 1.0],
        "saved_time_spacing": 0.0025,
        "convergence": convergence,
    }
    return _save(path, metadata, x=fine_x, t=fine_t, u=fine_u)


def _allen_cahn_grid(n: int, dt: float, save_dt: float = 0.005) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fourier pseudospectral ETDRK4 integration with diffusion 1e-4."""
    x = np.linspace(-1.0, 1.0, n, endpoint=False, dtype=np.float64)
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=2.0 / n)
    linear = -1.0e-4 * wave**2
    steps = int(round(1.0 / dt))
    stride = int(round(save_dt / dt))
    times = np.arange(0, steps + 1, stride, dtype=np.float64) * dt
    history = np.empty((len(times), n), dtype=np.float64)
    value = x**2 * np.cos(np.pi * x)
    spectral = np.fft.fft(value)
    history[0] = value

    E = np.exp(dt * linear)
    E2 = np.exp(dt * linear / 2.0)
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32.0)
    lr = dt * linear[:, None] + roots[None, :]
    Q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
    f1 = dt * np.real(
        np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
    )
    f2 = dt * np.real(
        np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
    )

    def nonlinear(vhat: np.ndarray) -> np.ndarray:
        physical = np.fft.ifft(vhat).real
        return np.fft.fft(5.0 * physical - 5.0 * physical**3)

    save_index = 1
    for step in range(1, steps + 1):
        Nv = nonlinear(spectral)
        a = E2 * spectral + Q * Nv
        Na = nonlinear(a)
        b = E2 * spectral + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        spectral = E * spectral + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        if step % stride == 0:
            history[save_index] = np.fft.ifft(spectral).real
            save_index += 1
    return x, times, history


def generate_allen_cahn(path: Path) -> Path:
    coarse_x, coarse_t, coarse_u = _allen_cahn_grid(512, 5.0e-4)
    fine_x, fine_t, fine_u = _allen_cahn_grid(1024, 2.5e-4)
    difference = fine_u[:, ::2] - coarse_u
    metadata = {
        "case": "allen_cahn_1d",
        "method": "periodic Fourier pseudospectral ETDRK4",
        "equation": "u_t - 1e-4*u_xx + 5*u^3 - 5*u = 0",
        "diffusion": 1.0e-4,
        "fine_grid": {"nx": 1024, "dt": 2.5e-4, "saved_dt": 0.005},
        "convergence": {
            "coarse_nx": 512,
            "coarse_dt": 5.0e-4,
            "relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(fine_u[:, ::2])),
            "max_difference": float(np.max(np.abs(difference))),
        },
    }
    return _save(path, metadata, x=fine_x, t=fine_t, u=fine_u)


def _lshape_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg

    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    domain = ~((X > 0.0) & (Y > 0.0))
    boundary = (
        np.isclose(np.abs(X), 1.0)
        | np.isclose(np.abs(Y), 1.0)
        | (np.isclose(X, 0.0) & (Y >= 0.0))
        | (np.isclose(Y, 0.0) & (X >= 0.0))
    )
    unknown = domain & ~boundary
    index = -np.ones((n, n), dtype=np.int64)
    index[unknown] = np.arange(np.count_nonzero(unknown))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for iy, ix in np.argwhere(unknown):
        row = int(index[iy, ix])
        rows.append(row)
        cols.append(row)
        data.append(4.0 / h**2)
        for jy, jx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if unknown[jy, jx]:
                rows.append(row)
                cols.append(int(index[jy, jx]))
                data.append(-1.0 / h**2)
    matrix = csr_matrix((data, (rows, cols)), shape=(np.count_nonzero(unknown),) * 2)
    rhs = np.ones(np.count_nonzero(unknown), dtype=np.float64)
    solution, info = cg(matrix, rhs, rtol=1.0e-11, atol=0.0, maxiter=20000)
    if info != 0:
        raise RuntimeError(f"L-shaped Poisson CG did not converge, info={info}")
    # Values in the removed quadrant are stored as finite zeros together with a mask.
    # This keeps every result NPZ NaN/Inf-free; they are never evaluation points.
    values = np.zeros((n, n), dtype=np.float64)
    values[unknown] = solution
    residual = matrix @ solution - rhs
    details = {
        "n": n,
        "h": h,
        "unknowns": int(len(solution)),
        "linear_solver": "conjugate gradient",
        "rtol": 1.0e-11,
        "max_algebraic_residual": float(np.max(np.abs(residual))),
    }
    return x, y, values, domain, details


def generate_lshape(path: Path) -> Path:
    coarse_x, coarse_y, coarse_u, coarse_mask, coarse_details = _lshape_grid(201)
    fine_x, fine_y, fine_u, fine_mask, fine_details = _lshape_grid(401)
    fine_common = fine_u[::2, ::2]
    difference = fine_common[coarse_mask] - coarse_u[coarse_mask]
    metadata = {
        "case": "poisson_lshape",
        "method": "five-point finite differences on the exact L-shaped mask",
        "equation": "-Delta u = 1, u|boundary=0",
        "fine_grid": fine_details,
        "convergence": {
            "coarse_grid": coarse_details,
            "relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(fine_common[coarse_mask])),
            "max_difference": float(np.max(np.abs(difference))),
        },
    }
    return _save(path, metadata, x=fine_x, y=fine_y, u=fine_u, domain_mask=fine_mask)


def _cavity_grid(n: int, dt: float = 0.01, tolerance: float = 1.0e-9, max_steps: int = 12000):
    """Semi-Lagrangian implicit-diffusion vorticity-streamfunction solver."""
    from scipy.fft import dstn, idstn
    from scipy.ndimage import map_coordinates

    reynolds = 100.0
    viscosity = 1.0 / reynolds
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    m = n - 2
    indices = np.arange(1, m + 1)
    eigen_1d = 2.0 * (np.cos(np.pi * indices / (n - 1)) - 1.0) / h**2
    laplace_eigen = eigen_1d[:, None] + eigen_1d[None, :]
    helmholtz = 1.0 - dt * viscosity * laplace_eigen
    alpha = dt * viscosity / h**2
    omega = np.zeros((n, n), dtype=np.float64)
    psi = np.zeros_like(omega)
    previous_u = np.zeros_like(omega)

    def poisson(vorticity: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vorticity)
        transformed = dstn(-vorticity[1:-1, 1:-1], type=1, norm="ortho")
        result[1:-1, 1:-1] = idstn(transformed / laplace_eigen, type=1, norm="ortho")
        return result

    def velocity(stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.gradient(stream, h, axis=0, edge_order=2)
        v = -np.gradient(stream, h, axis=1, edge_order=2)
        u[0, :] = 0.0
        u[-1, :] = 1.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[[0, -1], :] = 0.0
        v[:, [0, -1]] = 0.0
        return u, v

    def set_boundary(vorticity: np.ndarray, stream: np.ndarray) -> None:
        vorticity[0, 1:-1] = -2.0 * stream[1, 1:-1] / h**2
        vorticity[-1, 1:-1] = -2.0 * stream[-2, 1:-1] / h**2 - 2.0 / h
        vorticity[1:-1, 0] = -2.0 * stream[1:-1, 1] / h**2
        vorticity[1:-1, -1] = -2.0 * stream[1:-1, -2] / h**2
        vorticity[0, 0] = 0.5 * (vorticity[0, 1] + vorticity[1, 0])
        vorticity[0, -1] = 0.5 * (vorticity[0, -2] + vorticity[1, -1])
        vorticity[-1, 0] = 0.5 * (vorticity[-1, 1] + vorticity[-2, 0])
        vorticity[-1, -1] = 0.5 * (vorticity[-1, -2] + vorticity[-2, -1])

    converged = False
    change = np.inf
    for step in range(1, max_steps + 1):
        psi = poisson(omega)
        u, v = velocity(psi)
        departure_x = np.clip(X - dt * u, 0.0, 1.0) / h
        departure_y = np.clip(Y - dt * v, 0.0, 1.0) / h
        advected = map_coordinates(
            omega, [departure_y, departure_x], order=1, mode="nearest", prefilter=False
        )
        set_boundary(omega, psi)
        rhs = advected[1:-1, 1:-1].copy()
        rhs[0, :] += alpha * omega[0, 1:-1]
        rhs[-1, :] += alpha * omega[-1, 1:-1]
        rhs[:, 0] += alpha * omega[1:-1, 0]
        rhs[:, -1] += alpha * omega[1:-1, -1]
        transformed = dstn(rhs, type=1, norm="ortho")
        following = np.zeros_like(omega)
        following[1:-1, 1:-1] = idstn(
            transformed / helmholtz, type=1, norm="ortho"
        )
        set_boundary(following, psi)
        omega = following
        if step % 20 == 0:
            change = float(np.max(np.abs(u - previous_u)))
            previous_u = u.copy()
            if change < tolerance:
                converged = True
                break
    psi = poisson(omega)
    u, v = velocity(psi)
    details = {
        "n": n,
        "h": h,
        "dt": dt,
        "iterations": step,
        "tolerance": tolerance,
        "final_velocity_change": change,
        "converged": converged,
    }
    if not converged:
        raise RuntimeError(f"Cavity reference did not converge: {details}")
    return x, y, u, v, details


def generate_cavity(path: Path) -> Path:
    # Scaling dt with h suppresses the corner-vorticity instability while the
    # semi-Lagrangian update remains efficient enough for a steady solve.
    coarse_x, coarse_y, coarse_u, coarse_v, coarse_details = _cavity_grid(
        129, dt=1.0e-3, tolerance=1.0e-8, max_steps=40000
    )
    fine_x, fine_y, fine_u, fine_v, fine_details = _cavity_grid(
        257, dt=5.0e-4, tolerance=1.0e-8, max_steps=80000
    )
    du = fine_u[::2, ::2] - coarse_u
    dv = fine_v[::2, ::2] - coarse_v
    numerator = np.sqrt(np.sum(du**2 + dv**2))
    denominator = np.sqrt(np.sum(fine_u[::2, ::2] ** 2 + fine_v[::2, ::2] ** 2))
    metadata = {
        "case": "lid_driven_cavity",
        "method": "steady vorticity-streamfunction, semi-Lagrangian advection, implicit diffusion, DST Poisson solves",
        "reynolds": 100.0,
        "fine_grid": fine_details,
        "convergence": {
            "coarse_grid": coarse_details,
            "joint_velocity_relative_l2_difference": float(numerator / denominator),
            "max_pointwise_velocity_difference": float(np.max(np.sqrt(du**2 + dv**2))),
        },
        "pressure": "not reconstructed; gPINN pressure is gauge-fixed but excluded from velocity metrics",
    }
    return _save(path, metadata, x=fine_x, y=fine_y, u=fine_u, v=fine_v)


GENERATORS: dict[str, Callable[[Path], Path]] = {
    "burgers_1d": generate_burgers,
    "allen_cahn_1d": generate_allen_cahn,
    "poisson_lshape": generate_lshape,
    "lid_driven_cavity": generate_cavity,
}


def reference_path(root: Path, case: str) -> Path:
    return root / case / "reference.npz"


def ensure_reference(root: Path, case: str, force: bool = False) -> Path:
    path = reference_path(root, case)
    if force or not path.exists():
        GENERATORS[case](path)
    return path


def interpolate_reference(path: Path, case: str, points: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.interpolate import RegularGridInterpolator

    with np.load(path) as data:
        metadata = json.loads(str(data["metadata"]))
        x = data["x"]
        if case == "burgers_1d":
            t, u = data["t"], data["u"]
            interpolator = RegularGridInterpolator((t, x), u, bounds_error=True)
            values = interpolator(np.column_stack((points[:, 1], points[:, 0])))[:, None]
        elif case == "allen_cahn_1d":
            t, u = data["t"], data["u"]
            # The Fourier grid is periodic and deliberately excludes x=1.
            # Add the equivalent endpoint so the PINN test grid may include
            # both -1 and 1 without relaxing bounds checks.
            periodic_x = np.concatenate((x, np.asarray([1.0], dtype=np.float64)))
            periodic_u = np.concatenate((u, u[:, 0:1]), axis=1)
            interpolator = RegularGridInterpolator(
                (t, periodic_x), periodic_u, bounds_error=True
            )
            values = interpolator(
                np.column_stack((points[:, 1], points[:, 0]))
            )[:, None]
        elif case == "poisson_lshape":
            y, u = data["y"], data["u"]
            interpolator = RegularGridInterpolator((y, x), u, bounds_error=True)
            values = interpolator(np.column_stack((points[:, 1], points[:, 0])))[:, None]
        elif case == "lid_driven_cavity":
            y, u, v = data["y"], data["u"], data["v"]
            query = np.column_stack((points[:, 1], points[:, 0]))
            u_values = RegularGridInterpolator((y, x), u, bounds_error=True)(query)
            v_values = RegularGridInterpolator((y, x), v, bounds_error=True)(query)
            values = np.column_stack((u_values, v_values))
        else:
            raise KeyError(case)
    return values.astype(np.float64), metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=NUMERICAL_CASES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("data/reference_data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.all and args.case is None:
        parser.error("use --all or --case")
    cases = NUMERICAL_CASES if args.all else (args.case,)
    for case in cases:
        path = ensure_reference(args.root, case, args.force)
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata"]))
        print(json.dumps({"case": case, "path": str(path), "convergence": metadata["convergence"]}, indent=2))


if __name__ == "__main__":
    main()
