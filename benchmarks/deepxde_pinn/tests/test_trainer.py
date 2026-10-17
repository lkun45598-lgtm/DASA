import json
import math

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.deepxde_pinn.problems import build_problem
from benchmarks.deepxde_pinn.trainer import Trainer


def test_one_step_training_writes_only_requested_metric_fields(tmp_path):
    config = load_case_config("poisson_1d_steep")
    problem = build_problem(
        config,
        seed=9,
        sampling_overrides={"residual": 8, "boundary": 2},
    )
    trainer = Trainer(config, problem, seed=9, output_dir=tmp_path)

    result = trainer.run(adam_steps=1, lbfgs_steps=0)

    assert (tmp_path / "config_resolved.json").is_file()
    assert (tmp_path / "progress.jsonl").is_file()
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {
        "case",
        "seed",
        "l2_relative_error",
        "maximum_pointwise_error",
    }
    assert result == metrics
    assert math.isfinite(metrics["l2_relative_error"])
    assert math.isfinite(metrics["maximum_pointwise_error"])

    resolved = json.loads(
        (tmp_path / "config_resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["baseline_network"] == {
        "architecture": "classical_fnn",
        "hidden_layers": 6,
        "width": 128,
        "activation": "tanh",
        "initializer": "glorot_normal",
        "input_mapping": "normalized_raw_coordinates",
    }


def test_raissi_profile_accepts_lbfgs_budget_override(tmp_path):
    config = load_case_config("poisson_1d_steep")
    problem = build_problem(
        config,
        seed=3,
        sampling_overrides={"residual": 8, "boundary": 2},
    )
    trainer = Trainer(config, problem, seed=3, output_dir=tmp_path)

    trainer.run(lbfgs_steps=2, training_profile="raissi_lbfgs")

    resolved = json.loads(
        (tmp_path / "config_resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["run_budget"] == {
        "adam_steps": 0,
        "lbfgs_steps": 2,
        "lbfgs_calls": 1,
        "full_batch": True,
    }
    assert resolved["loss_components"]["effective_weights"] == [1.0, 1.0]
