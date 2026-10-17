import unittest

import numpy as np

from benchmarks.dasa_aligned_sa.dasa_sa.grids import (
    lshape_grid,
    regular_2d_grid,
    static_1d_grid,
    time_1d_grid,
)


class EvaluationGridTests(unittest.TestCase):
    def test_static_1d_grid_has_10001_points_and_endpoints(self):
        points = static_1d_grid((-1.0, 1.0))
        self.assertEqual(points.shape, (10001, 1))
        np.testing.assert_array_equal(points[[0, -1], 0], [-1.0, 1.0])

    def test_regular_2d_grid_has_401_by_401_points(self):
        points = regular_2d_grid(((0.0, 1.0), (0.0, 1.0)))
        self.assertEqual(points.shape, (401 * 401, 2))
        self.assertTrue(np.any(np.all(points == [0.0, 0.0], axis=1)))
        self.assertTrue(np.any(np.all(points == [1.0, 1.0], axis=1)))

    def test_time_grid_has_512_space_by_201_time_points(self):
        points = time_1d_grid(((0.0, 1.0), (-1.0, 1.0)))
        self.assertEqual(points.shape, (512 * 201, 2))
        np.testing.assert_array_equal(points[0], [0.0, -1.0])
        np.testing.assert_array_equal(points[-1], [1.0, 1.0])

    def test_lshape_grid_keeps_axes_and_removes_strict_quadrant(self):
        points = lshape_grid()
        self.assertEqual(points.shape, (401 * 401 - 200 * 200, 2))
        self.assertFalse(np.any((points[:, 0] > 0.0) & (points[:, 1] > 0.0)))
        self.assertTrue(np.any(np.all(points == [0.5, 0.0], axis=1)))
        self.assertTrue(np.any(np.all(points == [0.0, 0.5], axis=1)))


if __name__ == "__main__":
    unittest.main()
