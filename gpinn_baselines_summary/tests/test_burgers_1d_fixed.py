from __future__ import annotations

import numpy as np
import torch

from experiments.burgers_1d_fixed.run import (
    BurgersMLP,
    burgers_reference_grid,
    select_rar_anchors,
)


def test_fixed_burgers_hard_initial_and_boundary_conditions():
    model = BurgersMLP(hidden_layers=2, width=16)
    x = torch.linspace(-1.0, 1.0, 31, dtype=torch.float64)
    initial = torch.column_stack((x, torch.zeros_like(x)))
    np.testing.assert_allclose(
        model(initial).detach().numpy()[:, 0],
        -np.sin(np.pi * x.numpy()),
        atol=1.0e-13,
    )
    t = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
    left = torch.column_stack((-torch.ones_like(t), t))
    right = torch.column_stack((torch.ones_like(t), t))
    np.testing.assert_allclose(model(left).detach().numpy(), 0.0, atol=1.0e-13)
    np.testing.assert_allclose(model(right).detach().numpy(), 0.0, atol=1.0e-13)

    interior_x = torch.linspace(0.0, 1.0, 23, dtype=torch.float64)
    interior_t = torch.linspace(0.0, 1.0, 23, dtype=torch.float64)
    positive = torch.column_stack((interior_x, interior_t))
    negative = torch.column_stack((-interior_x, interior_t))
    np.testing.assert_allclose(
        model(negative).detach().numpy(),
        -model(positive).detach().numpy(),
        atol=1.0e-13,
    )


def test_rar_selects_high_residual_points():
    candidates = np.column_stack((np.linspace(-1.0, 1.0, 100), np.linspace(0.0, 1.0, 100)))
    scores = np.linspace(0.0, 1.0, 100)
    selected, details = select_rar_anchors(
        candidates,
        scores,
        np.empty((0, 2)),
        count=5,
        min_distance=0.0,
    )
    assert len(selected) == 5
    np.testing.assert_allclose(selected, candidates[-5:][::-1])
    assert details["selected_min_abs_residual"] >= scores[-5]


def test_reference_satisfies_conditions_and_is_finite():
    x, t, u = burgers_reference_grid(
        nx=257, dt=1.0e-3, viscosity=0.01 / np.pi, save_dt=1.0e-2
    )
    assert u.shape == (len(t), len(x))
    assert np.isfinite(u).all()
    np.testing.assert_allclose(u[0], -np.sin(np.pi * x), atol=1.0e-13)
    np.testing.assert_allclose(u[:, [0, -1]], 0.0, atol=1.0e-13)
