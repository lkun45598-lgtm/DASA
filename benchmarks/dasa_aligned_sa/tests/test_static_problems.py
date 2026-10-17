import unittest

import numpy as np
import tensorflow as tf

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.dasa_aligned_sa.dasa_sa.problems import build_problem


STATIC_CASES = (
    "poisson_1d_boundary",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
)


class _ExactModel(tf.keras.Model):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def call(self, points):
        return self.problem.exact_tf(points)


class StaticProblemTests(unittest.TestCase):
    def test_manufactured_solutions_satisfy_the_equations(self):
        samples = {
            "poisson_1d_boundary": [[-0.8], [0.0], [0.7]],
            "poisson_1d_steep": [[-0.4], [-0.1], [0.3]],
            "helmholtz_1d": [[0.13], [0.41], [0.82]],
            "poisson_2d_peak": [[0.2, 0.3], [0.5, 0.5], [0.7, 0.6]],
            "poisson_2d_steep": [[0.2, 0.3], [0.5, 0.5], [0.7, 0.6]],
            "helmholtz_2d": [[-0.7, -0.3], [0.2, 0.4], [0.8, 0.6]],
        }
        for case_id in STATIC_CASES:
            with self.subTest(case_id=case_id):
                problem = build_problem(load_case_config(case_id))
                points = tf.constant(samples[case_id], dtype=tf.float32)
                residual = problem.residual(_ExactModel(problem), points).numpy()
                np.testing.assert_allclose(residual, 0.0, atol=2e-2, rtol=2e-5)

    def test_sampling_counts_and_boundary_labels_are_exact(self):
        for case_id in STATIC_CASES:
            with self.subTest(case_id=case_id):
                config = load_case_config(case_id)
                problem = build_problem(config)
                data = problem.sample_training(seed=3)
                self.assertEqual(data.residual.shape, (config.sampling.residual, len(config.domain)))
                self.assertEqual(data.boundary_points.shape[0], config.sampling.boundary)
                expected = problem.exact_numpy(data.boundary_points)
                np.testing.assert_allclose(data.boundary_values, expected, atol=1e-10)

    def test_1d_poisson_boundary_losses_sum_endpoint_errors(self):
        prediction = tf.constant([[1.0], [1.0]], dtype=tf.float32)
        target = tf.zeros_like(prediction)
        for case_id in ("poisson_1d_boundary", "poisson_1d_steep"):
            with self.subTest(case_id=case_id):
                problem = build_problem(load_case_config(case_id))
                self.assertEqual(float(problem.boundary_loss(prediction, target)), 2.0)

    def test_1d_helmholtz_forcing_is_nonzero(self):
        problem = build_problem(load_case_config("helmholtz_1d"))
        x = tf.constant([[0.0125]], dtype=tf.float32)
        expected = 800.0 * np.pi**2 * np.sin(20.0 * np.pi * 0.0125)
        np.testing.assert_allclose(problem.forcing_tf(x).numpy(), expected, rtol=1e-6)

    def test_evaluation_data_uses_the_fixed_grid(self):
        for case_id in STATIC_CASES:
            with self.subTest(case_id=case_id):
                problem = build_problem(load_case_config(case_id))
                points, truth = problem.evaluation_data()
                expected_count = 10001 if points.shape[1] == 1 else 401 * 401
                self.assertEqual(points.shape[0], expected_count)
                self.assertEqual(truth.shape, (expected_count, 1))
                self.assertTrue(np.all(np.isfinite(truth)))


if __name__ == "__main__":
    unittest.main()
