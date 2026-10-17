import unittest

import numpy as np
import tensorflow as tf

from benchmarks.dasa_aligned_sa.dasa_sa.config import load_case_config
from benchmarks.dasa_aligned_sa.dasa_sa.problems import build_problem


class _ZeroCavityModel(tf.keras.Model):
    def call(self, points):
        zero = points[:, 0:1] * 0.0
        return tf.concat((zero, zero, zero), axis=1)


class _ZeroScalarModel(tf.keras.Model):
    def call(self, points):
        return points[:, 0:1] * 0.0


class IrregularProblemTests(unittest.TestCase):
    def test_cavity_evaluation_uses_paper_reference_dataset(self):
        problem = build_problem(load_case_config("cavity"))
        points, values = problem.evaluation_data()

        self.assertEqual(points.shape, (100 * 100, 2))
        self.assertEqual(values.shape, (100 * 100, 1))
        field = values.reshape(100, 100)
        np.testing.assert_array_equal(field[:, 0], np.zeros(100))
        np.testing.assert_array_equal(field[:, -1], np.ones(100))
        np.testing.assert_array_equal(field[0, :-1], np.zeros(99))
        np.testing.assert_array_equal(field[-1, :-1], np.zeros(99))

    def test_cavity_sampling_and_boundary_policy(self):
        problem = build_problem(load_case_config("cavity"))
        data = problem.sample_training(seed=4)
        self.assertEqual(data.residual.shape, (2500, 2))
        self.assertEqual(data.boundary_points.shape, (512, 2))
        self.assertEqual(data.boundary_values.shape, (512, 2))
        self.assertEqual(data.gauge_points.shape, (1, 2))
        np.testing.assert_array_equal(data.gauge_points[0], [0.5, 0.5])

        corners = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
        self.assertTrue(corners.isdisjoint(map(tuple, data.boundary_points)))
        top = data.boundary_points[:, 1] == 1.0
        np.testing.assert_array_equal(data.boundary_values[top, 0], np.ones(128))
        np.testing.assert_array_equal(data.boundary_values[~top], np.zeros((384, 2)))

    def test_cavity_has_three_zero_residuals_for_zero_flow(self):
        problem = build_problem(load_case_config("cavity"))
        points = tf.constant([[0.2, 0.3], [0.7, 0.8]], dtype=tf.float32)
        residual = problem.residual(_ZeroCavityModel(), points).numpy()
        self.assertEqual(residual.shape, (2, 3))
        np.testing.assert_array_equal(residual, np.zeros((2, 3)))

    def test_lshape_sampling_respects_geometry_and_eight_segments(self):
        problem = build_problem(load_case_config("lshape_poisson"))
        data = problem.sample_training(seed=4)
        self.assertEqual(data.residual.shape, (10000, 2))
        self.assertFalse(
            np.any((data.residual[:, 0] >= 0.0) & (data.residual[:, 1] >= 0.0))
        )
        self.assertEqual(data.boundary_points.shape, (8 * 128, 2))
        segments = data.boundary_points.reshape(8, 128, 2)
        self.assertTrue(all(np.unique(segment, axis=0).shape[0] == 128 for segment in segments))
        np.testing.assert_array_equal(data.boundary_values, np.zeros((1024, 1)))

    def test_lshape_zero_model_has_minus_one_residual(self):
        problem = build_problem(load_case_config("lshape_poisson"))
        points = tf.constant([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]])
        residual = problem.residual(_ZeroScalarModel(), points).numpy()
        np.testing.assert_array_equal(residual, -np.ones((3, 1)))


if __name__ == "__main__":
    unittest.main()
