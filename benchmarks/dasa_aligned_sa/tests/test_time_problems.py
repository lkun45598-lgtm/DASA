import unittest

import numpy as np
import tensorflow as tf

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.dasa_aligned_sa.dasa_sa.problems import build_problem


TIME_CASES = (
    "burgers",
    "allen_cahn",
    "diffusion_reaction",
    "advection_diffusion",
    "discontinuous_advection",
)


class _ExactModel(tf.keras.Model):
    def __init__(self, problem):
        super().__init__(dtype=tf.float64)
        self.problem = problem

    def call(self, points):
        return self.problem.exact_tf(points)


class _PolynomialModel(tf.keras.Model):
    def call(self, points):
        t = points[:, 0:1]
        x = points[:, 1:2]
        return t * x + x**2


class TimeProblemTests(unittest.TestCase):
    def test_burgers_evaluation_uses_original_sa_pinn_dataset(self):
        problem = build_problem(load_case_config("burgers"))
        points, values = problem.evaluation_data()

        self.assertEqual(points.shape, (100 * 256, 2))
        self.assertEqual(values.shape, (100 * 256, 1))
        np.testing.assert_array_equal(points[0], [0.0, -1.0])
        np.testing.assert_array_equal(points[-1], [0.99, 1.0])
        np.testing.assert_allclose(
            values[:256], -np.sin(np.pi * points[:256, 1:2]), atol=1e-12
        )

    def test_allen_cahn_evaluation_uses_original_sa_pinn_dataset(self):
        problem = build_problem(load_case_config("allen_cahn"))
        points, values = problem.evaluation_data()

        self.assertEqual(points.shape, (201 * 512, 2))
        self.assertEqual(values.shape, (201 * 512, 1))
        np.testing.assert_array_equal(points[0], [0.0, -1.0])
        np.testing.assert_array_equal(points[-1], [1.0, 0.99609375])
        np.testing.assert_allclose(
            values[:512],
            points[:512, 1:2] ** 2 * np.cos(np.pi * points[:512, 1:2]),
            atol=1e-12,
        )

    def test_sampling_counts_and_condition_values(self):
        for case_id in TIME_CASES:
            with self.subTest(case_id=case_id):
                config = load_case_config(case_id)
                problem = build_problem(config)
                data = problem.sample_training(seed=9)
                self.assertEqual(data.residual.shape, (config.sampling.residual, 2))
                self.assertEqual(data.initial_points.shape, (config.sampling.initial, 2))
                np.testing.assert_allclose(
                    data.initial_values,
                    problem.initial_numpy(data.initial_points[:, 1:2]),
                    atol=1e-12,
                )
                if config.sampling.boundary:
                    self.assertEqual(
                        data.boundary_points.shape,
                        (2 * config.sampling.boundary, 2),
                    )
                    np.testing.assert_array_equal(
                        data.boundary_values,
                        np.zeros((2 * config.sampling.boundary, 1)),
                    )
                else:
                    self.assertIsNone(data.boundary_points)

    def test_boundary_sampling_uses_256_paired_times_per_side(self):
        for case_id in TIME_CASES:
            with self.subTest(case_id=case_id):
                problem = build_problem(load_case_config(case_id))
                points = problem.sample_training(seed=9).boundary_points
                self.assertIsNotNone(points)
                self.assertEqual(points.shape, (512, 2))
                np.testing.assert_array_equal(points[:256, 0], points[256:, 0])
                np.testing.assert_array_equal(
                    points[:256, 1], np.full(256, problem.config.domain[1][0])
                )
                np.testing.assert_array_equal(
                    points[256:, 1], np.full(256, problem.config.domain[1][1])
                )

    def test_allen_cahn_boundary_prediction_is_periodic_value_and_derivative_error(self):
        problem = build_problem(load_case_config("allen_cahn"))
        times = tf.constant([[0.25], [0.75]], dtype=tf.float32)
        points = tf.concat(
            (
                tf.concat((times, -tf.ones_like(times)), axis=1),
                tf.concat((times, tf.ones_like(times)), axis=1),
            ),
            axis=0,
        )
        prediction = problem.boundary_prediction(_PolynomialModel(), points)
        expected = tf.constant([[-0.5], [-1.5], [-4.0], [-4.0]], dtype=tf.float32)
        np.testing.assert_allclose(prediction.numpy(), expected.numpy(), atol=1e-7)

    def test_diffusion_reaction_exact_solution_satisfies_residual(self):
        problem = build_problem(load_case_config("diffusion_reaction"))
        points = tf.constant(
            [[0.1, -2.1], [0.4, -0.2], [0.8, 1.7]], dtype=tf.float64
        )
        residual = problem.residual(_ExactModel(problem), points).numpy()
        np.testing.assert_allclose(residual, 0.0, atol=1e-7)

    def test_advection_diffusion_series_is_finite_and_satisfies_conditions(self):
        problem = build_problem(load_case_config("advection_diffusion"))
        points = np.array(
            [[0.0, -0.7], [0.0, 0.3], [0.4, -1.0], [0.4, 1.0], [0.3, -0.4]],
            dtype=np.float64,
        )
        values = problem.exact_numpy(points)
        self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(values[:2], -np.sin(np.pi * points[:2, 1:2]), atol=1e-12)
        np.testing.assert_array_equal(values[2:4], np.zeros((2, 1)))

        residual_points = tf.constant([[0.25, -0.8], [0.5, -0.4]], dtype=tf.float64)
        residual = problem.residual(_ExactModel(problem), residual_points).numpy()
        np.testing.assert_allclose(residual, 0.0, atol=2e-7, rtol=2e-7)

    def test_advection_diffusion_evaluation_respects_maximum_principle(self):
        problem = build_problem(load_case_config("advection_diffusion"))
        points, values = problem.evaluation_data()

        self.assertEqual(points.shape, (201 * 512, 2))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertLessEqual(float(np.max(np.abs(values))), 1.0 + 1e-12)
        boundary = np.isin(points[:, 1], (-1.0, 1.0))
        np.testing.assert_array_equal(values[boundary], np.zeros((np.sum(boundary), 1)))

    def test_discontinuous_exact_solution_translates_the_closed_pulse(self):
        problem = build_problem(load_case_config("discontinuous_advection"))
        points = np.array(
            [[0.0, -0.2], [0.5, 0.3], [0.5, 0.71], [1.0, 1.2]], dtype=np.float64
        )
        np.testing.assert_array_equal(
            problem.exact_numpy(points), np.array([[1.0], [1.0], [0.0], [1.0]])
        )
        smooth_points = tf.constant([[0.2, -1.0], [0.7, 1.3]], dtype=tf.float64)
        residual = problem.residual(_ExactModel(problem), smooth_points).numpy()
        np.testing.assert_array_equal(residual, np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
