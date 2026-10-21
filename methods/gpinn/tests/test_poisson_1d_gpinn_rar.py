from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from experiments.poisson_1d_gpinn_rar.cases import CASES, build_case
from experiments.poisson_1d_gpinn_rar.model import MLP, derivative, objective_terms
from experiments.poisson_1d_gpinn_rar.run import (
    METHOD_NAMES,
    RunConfig,
    run,
    select_top_residual_points,
    uniform_interior,
)


def test_default_is_explicit_upstream_burgers_rar_transfer():
    config = RunConfig()
    config.apply_protocol_defaults()
    assert config.protocol == "burgers_rar_transfer"
    assert config.initial_points == 500
    assert config.sampling_distribution == "latin_hypercube"
    assert config.initial_adam_steps == 20_000
    assert config.initial_lbfgs_iterations == 15_000
    assert config.rar_cycles == 40
    assert config.rar_candidate_points == 100_000
    assert config.rar_points_per_cycle == 10
    assert config.rar_adam_steps == 10_000
    assert config.rar_lbfgs_iterations == 15_000
    assert config.rar_early_stopping
    assert config.initial_points + config.rar_cycles * config.rar_points_per_cycle == 900


def test_upstream_poisson_protocol_is_15_uniform_points_without_rar():
    config = RunConfig(protocol="upstream_poisson")
    config.apply_protocol_defaults()
    config.validate()
    assert config.initial_points == 15
    assert config.sampling_distribution == "uniform"
    assert config.initial_adam_steps == 20_000
    assert config.gradient_weight == 0.01
    assert config.initial_lbfgs_iterations == 0
    assert config.rar_cycles == 0
    np.testing.assert_allclose(
        uniform_interior(3, -1.0, 1.0)[:, 0], [-0.5, 0.0, 0.5]
    )


def test_paper_gpinn_protocol_is_500_lhs_points_without_rar():
    config = RunConfig(protocol="paper_gpinn")
    config.apply_protocol_defaults()
    config.validate()
    assert config.initial_points == 500
    assert config.sampling_distribution == "latin_hypercube"
    assert config.initial_adam_steps == 20_000
    assert config.initial_lbfgs_iterations == 0
    assert config.rar_cycles == 0


@pytest.mark.parametrize("key", sorted(CASES))
def test_equation_and_forcing_gradient(key):
    case = build_case(key)
    x = torch.linspace(case.lower, case.upper, 4001, dtype=torch.float64)[:, None]
    x.requires_grad_(True)
    exact = case.exact(x)
    exact_xx = derivative(derivative(exact, x), x)
    assert float((-exact_xx - case.forcing(x)).detach().abs().max()) < 1.0e-9
    autodiff = derivative(case.forcing(x), x)
    scale = max(1.0, float(case.forcing_x(x).detach().abs().max()))
    assert float((autodiff - case.forcing_x(x)).detach().abs().max()) < 1.0e-9 * scale


@pytest.mark.parametrize("key", sorted(CASES))
def test_network_is_6x128_and_satisfies_dirichlet_exactly(key):
    case = build_case(key)
    model = MLP(case)
    assert len(model.hidden) == 6
    assert all(layer.out_features == 128 for layer in model.hidden)
    assert model.output.out_features == 1
    assert next(model.parameters()).dtype == torch.float64
    endpoints = torch.tensor([[case.lower], [case.upper]], dtype=torch.float64)
    np.testing.assert_allclose(
        model(endpoints).detach().numpy(),
        case.exact(endpoints).detach().numpy(),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_objective_is_raw_gpinn_without_extra_methods():
    case = build_case("steep")
    model = MLP(case)
    x = torch.linspace(-0.9, 0.9, 31, dtype=torch.float64)[:, None]
    x.requires_grad_(True)
    terms = objective_terms(model, x, gradient_weight=0.01)
    assert set(terms) == {"total", "pde", "gradient"}
    torch.testing.assert_close(
        terms["total"], terms["pde"] + 0.01 * terms["gradient"]
    )


def test_rar_selects_top_raw_residuals():
    candidates = np.linspace(-1.0, 1.0, 20, dtype=np.float64)[:, None]
    scores = np.arange(20, dtype=np.float64) ** 2
    selected, indices = select_top_residual_points(candidates, scores, 4)
    np.testing.assert_array_equal(indices, [19, 18, 17, 16])
    np.testing.assert_allclose(selected, candidates[[19, 18, 17, 16]])


@pytest.mark.parametrize("key", sorted(CASES))
def test_smoke_run_refines_saves_and_recomputes_metrics(tmp_path, key):
    config = RunConfig(case=key, seed=0, mode="smoke", device="cpu")
    config.apply_protocol_defaults()
    config.apply_smoke_defaults()
    metrics = run(config, tmp_path)
    directory = tmp_path / "smoke" / key / "seed_0"
    assert metrics["method"] == METHOD_NAMES["burgers_rar_transfer"]
    assert metrics["protocol"] == "burgers_rar_transfer"
    assert metrics["rar_score"] == "absolute_raw_pde_residual"
    assert metrics["rar_cycles_completed"] == config.rar_cycles
    assert config.initial_adam_steps <= metrics["adam_steps_completed"] <= (
        config.initial_adam_steps + config.rar_cycles * config.rar_adam_steps
    )
    assert metrics["lbfgs_closure_evaluations"] == 0
    assert metrics["collocation_points"] == (
        config.initial_points + config.rar_cycles * config.rar_points_per_cycle
    )
    assert metrics["boundary_max_abs"] < 1.0e-12
    required = (
        "config.json",
        "last.pt",
        "metrics.json",
        "prediction.npz",
        "reference.npz",
        "rar_points.npz",
        "rar_history.jsonl",
        "training_log.csv",
        "solution_error_rar.png",
        "absolute_error.png",
        "loss_curve.png",
    )
    assert all((directory / name).exists() for name in required)
    with np.load(directory / "prediction.npz") as data:
        prediction = data["prediction"]
    with np.load(directory / "reference.npz") as data:
        reference = data["reference"]
        error = prediction - reference
        relative_l2 = np.linalg.norm(error) / np.linalg.norm(reference)
        max_abs = np.max(np.abs(error))
    assert np.isclose(relative_l2, metrics["relative_l2"], rtol=1e-12, atol=1e-14)
    assert np.isclose(max_abs, metrics["max_abs"], rtol=1e-12, atol=1e-14)
    history = [
        json.loads(line)
        for line in (directory / "rar_history.jsonl").read_text().splitlines()
    ]
    assert len(history) == config.rar_cycles
    assert [row["cycle"] for row in history] == list(range(1, config.rar_cycles + 1))
    assert all(
        row["added_points"] == config.rar_points_per_cycle for row in history
    )
    phases = {
        row.split(",", 1)[0]
        for row in (directory / "training_log.csv").read_text().splitlines()[1:]
    }
    assert phases == {"initial_adam", "rar_adam"}
