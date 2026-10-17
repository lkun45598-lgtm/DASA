import numpy as np
import torch

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.deepxde_pinn.problems import build_problem


def test_cavity_residual_has_two_momenta_and_continuity():
    problem = build_problem(
        load_case_config("cavity"),
        seed=0,
        sampling_overrides={"residual": 3, "boundary": 8},
    )
    points = torch.tensor(
        [[0.21, 0.34], [0.45, 0.61], [0.72, 0.83]],
        dtype=torch.float64,
        requires_grad=True,
    )
    x, y = points[:, 0:1], points[:, 1:2]
    u, v, pressure = x.square() + y, x - y.square(), x * y
    residuals = problem.pde(points, torch.cat((u, v, pressure), dim=1))

    assert len(residuals) == 3
    torch.testing.assert_close(residuals[0], u * (2.0 * x) + v + y - 0.02)
    torch.testing.assert_close(residuals[1], u - 2.0 * y * v + x + 0.02)
    torch.testing.assert_close(residuals[2], 2.0 * x - 2.0 * y)


def test_cavity_uses_fixed_grid_lid_boundary_gauge_and_reference():
    problem = build_problem(load_case_config("cavity"), seed=7)

    assert problem.data.anchors.shape == (2500, 2)
    assert problem.data.bcs[0].points.shape == (512, 2)
    assert problem.data.bcs[1].points.shape == (1, 2)
    assert problem.loss_weights == (1.0, 1.0, 1.0, 100.0, 1.0)
    points, values = problem.evaluation_data()
    assert points.shape == (10000, 2)
    assert values.shape == (10000, 1)


def test_lshape_residual_sampling_boundary_and_reference():
    problem = build_problem(load_case_config("lshape_poisson"), seed=3)
    points = torch.tensor(
        [[-0.7, -0.2], [-0.4, 0.6], [0.5, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = points[:, 0:1].square() + points[:, 1:2].square()

    torch.testing.assert_close(
        problem.pde(points, values), torch.full((3, 1), -5.0, dtype=torch.float64)
    )
    residual = problem.data.anchors
    assert residual.shape == (10000, 2)
    assert not np.any((residual[:, 0] >= 0.0) & (residual[:, 1] >= 0.0))
    assert problem.data.bcs[0].points.shape == (1024, 2)
    evaluation_points, truth = problem.evaluation_data()
    assert evaluation_points.shape == (120801, 2)
    assert truth.shape == (120801, 1)
