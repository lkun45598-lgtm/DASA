import unittest

import numpy as np
import tensorflow as tf

from benchmarks.dasa_aligned_sa.dasa_sa.sa_weights import (
    create_point_masks,
    minimax_gradients,
)


class SAWeightTests(unittest.TestCase):
    def test_seeded_masks_are_positive_and_reproducible(self):
        first = create_point_masks({"residual": 4, "initial": 2}, seed=13)
        second = create_point_masks({"residual": 4, "initial": 2}, seed=13)
        self.assertEqual(set(first), {"residual", "initial"})
        for key in first:
            self.assertTrue(np.all(first[key].numpy() >= 0.0))
            self.assertTrue(np.all(first[key].numpy() < 1.0))
            np.testing.assert_array_equal(first[key].numpy(), second[key].numpy())

    def test_minimax_gradients_descend_model_and_ascend_masks(self):
        parameter = tf.Variable([[2.0]], dtype=tf.float32)
        mask = tf.Variable([[0.5]], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            loss = tf.reduce_mean(tf.square(mask * parameter))
        model_gradients, mask_gradients = minimax_gradients(
            tape, loss, [parameter], [mask]
        )
        self.assertGreater(float(model_gradients[0].numpy().item()), 0.0)
        self.assertLess(float(mask_gradients[0].numpy().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
