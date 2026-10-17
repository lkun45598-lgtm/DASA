import unittest

import numpy as np

from benchmarks.dasa_aligned_sa.dasa_sa.metrics import (
    compute_metrics,
    l2_relative_error,
    maximum_pointwise_error,
)


class MetricTests(unittest.TestCase):
    def test_uses_the_requested_relative_l2_formula(self):
        y_true = np.array([3.0, 4.0], dtype=np.float32)
        y_pred = np.array([0.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(l2_relative_error(y_true, y_pred), 3.0 / 5.0)

    def test_uses_the_requested_maximum_pointwise_formula(self):
        y_true = np.array([[1.0], [-2.0], [4.0]])
        y_pred = np.array([[0.5], [1.0], [3.0]])
        self.assertEqual(maximum_pointwise_error(y_true, y_pred), 3.0)

    def test_compute_metrics_returns_only_the_two_approved_metrics(self):
        result = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        self.assertEqual(
            set(result), {"l2_relative_error", "maximum_pointwise_error"}
        )
        self.assertTrue(all(isinstance(value, float) for value in result.values()))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_metrics(np.ones((2, 1)), np.ones((2,)))


if __name__ == "__main__":
    unittest.main()
