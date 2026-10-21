import numpy as np
import pytest
import torch

from methods.sa_pinn_aligned.dasa_sa.config import load_case_config
from methods.deepxde_pinn.problems import build_problem


TIME_CASES = (
    "burgers",
    "allen_cahn",
    "diffusion_reaction",
    "advection_diffusion",
    "discontinuous_advection",
)


@pytest.mark.parametrize("case_id", TIME_CASES)
def test_time_residual_matches_configured_equation(case_id):
    problem = build_problem(
        load_case_config(case_id),
        seed=2,
        sampling_overrides={"residual": 3, "boundary": 2, "initial": 3},
    )
    points = torch.tensor(
        [[0.17, -0.41], [0.53, 0.22], [0.81, 0.67]],
        dtype=torch.float64,
        requires_grad=True,
    )
    t, x = points[:, 0:1], points[:, 1:2]
    values = t.square() + x**3
    u_t, u_x, u_xx = 2.0 * t, 3.0 * x.square(), 6.0 * x
    parameters = problem.config.parameters

    if case_id == "burgers":
        expected = u_t + values * u_x - parameters["viscosity"] * u_xx
    elif case_id == "allen_cahn":
        expected = (
            u_t
            - parameters["diffusion"] * u_xx
            + parameters["reaction"] * values**3
            - parameters["reaction"] * values
        )
    elif case_id == "diffusion_reaction":
        forcing = torch.exp(-t) * (
            1.5 * torch.sin(2.0 * x)
            + (8.0 / 3.0) * torch.sin(3.0 * x)
            + (15.0 / 4.0) * torch.sin(4.0 * x)
            + (63.0 / 8.0) * torch.sin(8.0 * x)
        )
        expected = u_t - parameters["diffusion"] * u_xx - forcing
    else:
        expected = (
            u_t
            + parameters["velocity"] * u_x
            - parameters["kappa"] * u_xx
        )

    torch.testing.assert_close(problem.pde(points, values), expected)


@pytest.mark.parametrize("case_id", TIME_CASES)
def test_time_initial_values(case_id):
    problem = build_problem(
        load_case_config(case_id),
        seed=2,
        sampling_overrides={"residual": 2, "boundary": 2, "initial": 3},
    )
    x = np.array([[-0.5], [0.0], [0.5]])
    values = problem.initial_numpy(x)

    if case_id in {"burgers", "advection_diffusion"}:
        expected = -np.sin(np.pi * x)
    elif case_id == "allen_cahn":
        expected = x**2 * np.cos(np.pi * x)
    elif case_id == "diffusion_reaction":
        expected = sum(np.sin(n * x) / n for n in range(1, 5)) + np.sin(8 * x) / 8
    else:
        expected = (np.abs(x) <= 0.2).astype(np.float64)

    np.testing.assert_allclose(values, expected)


def test_time_boundary_count_is_per_side_and_allen_cahn_is_periodic():
    burgers = build_problem(load_case_config("burgers"), seed=4)
    allen_cahn = build_problem(load_case_config("allen_cahn"), seed=4)

    assert burgers.training_counts["boundary"] == 256
    assert burgers.data.bcs[0].points.shape == (512, 2)
    assert allen_cahn.training_counts["boundary"] == 256
    assert [bc.points.shape for bc in allen_cahn.data.bcs[:2]] == [(512, 2), (512, 2)]
    assert allen_cahn.loss_names[1:3] == (
        "boundary_periodic_value",
        "boundary_periodic_derivative",
    )


def test_original_time_references_keep_sa_pinn_points_and_values():
    burgers = build_problem(
        load_case_config("burgers"),
        seed=0,
        sampling_overrides={"residual": 2, "boundary": 2, "initial": 2},
    )
    allen_cahn = build_problem(
        load_case_config("allen_cahn"),
        seed=0,
        sampling_overrides={"residual": 2, "boundary": 2, "initial": 2},
    )

    burgers_points, burgers_values = burgers.evaluation_data()
    ac_points, ac_values = allen_cahn.evaluation_data()

    assert burgers_points.shape == (256 * 100, 2)
    assert burgers_values.shape == (256 * 100, 1)
    assert ac_points.shape == (512 * 201, 2)
    assert ac_values.shape == (512 * 201, 1)
    assert np.all(np.isfinite(burgers_values))
    assert np.all(np.isfinite(ac_values))
