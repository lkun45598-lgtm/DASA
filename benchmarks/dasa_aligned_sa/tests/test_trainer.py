from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.dasa_aligned_sa.dasa_sa.lbfgs import minimize_lbfgs
from benchmarks.dasa_aligned_sa.dasa_sa.problems import build_problem
from benchmarks.dasa_aligned_sa.dasa_sa.trainer import (
    Trainer,
    learning_rate_for_step,
)


class TrainerTests(unittest.TestCase):
    def test_learning_rate_starts_decay_after_mapped_stage_two(self):
        optimizer = load_case_config("helmholtz_2d").optimizer
        warmup = optimizer.stages[0] + optimizer.stages[1]
        self.assertEqual(learning_rate_for_step(optimizer, warmup - 1), 0.001)
        self.assertEqual(learning_rate_for_step(optimizer, warmup), 0.001)
        self.assertEqual(learning_rate_for_step(optimizer, warmup + 1999), 0.001)
        self.assertAlmostEqual(
            learning_rate_for_step(optimizer, warmup + 2000), 0.0009
        )

    def test_model_and_mask_optimizers_share_the_same_learning_rate(self):
        config = load_case_config("poisson_1d_boundary")
        problem = build_problem(config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, problem, seed=0, output_dir=Path(directory))
            self.assertEqual(
                float(trainer.model_optimizer.learning_rate.numpy()),
                float(trainer.mask_optimizer.learning_rate.numpy()),
            )
            self.assertEqual(float(trainer.model_optimizer.beta_1), 0.99)
            self.assertEqual(float(trainer.mask_optimizer.beta_1), 0.99)

    def test_lbfgs_converges_and_does_not_modify_unlisted_mask(self):
        parameter = tf.Variable([0.0], dtype=tf.float32)
        mask = tf.Variable([0.75], dtype=tf.float32)

        def loss():
            return tf.reduce_sum(tf.square(parameter - 3.0)) * mask

        result = minimize_lbfgs(loss, [parameter], max_iterations=30)
        self.assertTrue(result.success)
        np.testing.assert_allclose(parameter.numpy(), [3.0], atol=1e-5)
        np.testing.assert_array_equal(mask.numpy(), [0.75])

    def test_tiny_run_writes_only_the_approved_result_fields(self):
        config = load_case_config("helmholtz_1d")
        config = replace(
            config,
            sampling=replace(config.sampling, residual=16),
            optimizer=replace(config.optimizer, stages=(1, 0, 0, 0), adam_steps=1),
        )
        problem = build_problem(config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, problem, seed=0, output_dir=Path(directory))
            result = trainer.run(adam_steps=1, lbfgs_steps=0)
            self.assertEqual(
                set(result),
                {"case", "seed", "l2_relative_error", "maximum_pointwise_error"},
            )
            self.assertTrue(np.isfinite(result["l2_relative_error"]))
            self.assertTrue(np.isfinite(result["maximum_pointwise_error"]))
            stored = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(stored, result)
            resolved = json.loads((Path(directory) / "config_resolved.json").read_text())
            self.assertEqual(resolved["run_budget"], {"adam_steps": 1, "lbfgs_steps": 0})


if __name__ == "__main__":
    unittest.main()
