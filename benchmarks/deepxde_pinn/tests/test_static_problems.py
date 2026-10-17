import numpy as np
import pytest
import torch

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.deepxde_pinn.problems import build_problem


STATIC_CASES = (
    "poisson_1d_boundary",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
)


@pytest.mark.parametrize("case_id", STATIC_CASES)
def test_exact_solution_satisfies_static_residual(case_id):
    problem = build_problem(load_case_config(case_id), seed=5)
    if len(problem.config.domain) == 1:
        values = [[0.13], [0.37], [0.71]]
    else:
        values = [[0.21, 0.32], [0.43, 0.57], [0.72, 0.81]]
    points = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    exact = problem.exact_torch(points)
    residual = problem.pde(points, exact)

    assert torch.max(torch.abs(residual)).item() < 1e-4


def test_static_training_counts_follow_dasa_definitions():
    poisson = build_problem(load_case_config("poisson_2d_peak"), seed=3)
    helmholtz = build_problem(load_case_config("helmholtz_2d"), seed=3)

    assert poisson.training_counts == {"residual": 2500, "boundary": 50, "initial": 0}
    assert helmholtz.training_counts == {
        "residual": 10201,
        "boundary": 200,
        "initial": 0,
    }


def test_static_evaluation_data_matches_shared_reference_grids():
    one_dimensional = build_problem(load_case_config("poisson_1d_steep"), seed=0)
    two_dimensional = build_problem(load_case_config("helmholtz_2d"), seed=0)

    points_1d, truth_1d = one_dimensional.evaluation_data()
    points_2d, truth_2d = two_dimensional.evaluation_data()

    assert points_1d.shape == truth_1d.shape == (10001, 1)
    assert points_2d.shape == (401 * 401, 2)
    assert truth_2d.shape == (401 * 401, 1)
    assert np.all(np.isfinite(truth_1d))
    assert np.all(np.isfinite(truth_2d))


def test_homogeneous_helmholtz_is_flagged_as_amplitude_underdetermined():
    problem = build_problem(load_case_config("helmholtz_1d"), seed=0)
    assert problem.metadata["scientific_status"] == "amplitude_underdetermined"
