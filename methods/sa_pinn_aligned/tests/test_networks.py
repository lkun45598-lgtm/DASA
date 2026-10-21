import unittest

import numpy as np
import tensorflow as tf

from methods.sa_pinn_aligned.dasa_sa.networks import ModifiedMLP


class ModifiedMLPTests(unittest.TestCase):
    def test_scalar_and_cavity_output_shapes(self):
        scalar = ModifiedMLP(((-1.0, 1.0),), output_dim=1, seed=7)
        cavity = ModifiedMLP(((0.0, 1.0), (0.0, 1.0)), output_dim=3, seed=7)
        self.assertEqual(scalar(tf.zeros((4, 1))).shape, (4, 1))
        self.assertEqual(cavity(tf.zeros((4, 2))).shape, (4, 3))

    def test_has_six_backbone_hidden_transformations_of_width_128(self):
        model = ModifiedMLP(((-1.0, 1.0),), output_dim=1, seed=7)
        model(tf.zeros((1, 1)))
        self.assertEqual(model.hidden_depth, 6)
        self.assertEqual(model.initial_hidden.units, 128)
        self.assertEqual(len(model.mixing_layers), 5)
        self.assertTrue(all(layer.units == 128 for layer in model.mixing_layers))

    def test_seeded_initialization_is_deterministic(self):
        first = ModifiedMLP(((-1.0, 1.0),), output_dim=1, seed=17)
        second = ModifiedMLP(((-1.0, 1.0),), output_dim=1, seed=17)
        sample = tf.constant([[0.25]], dtype=tf.float32)
        np.testing.assert_allclose(first(sample).numpy(), second(sample).numpy())

    def test_supports_finite_second_derivatives(self):
        model = ModifiedMLP(((-1.0, 1.0),), output_dim=1, seed=11)
        x = tf.constant([[0.2], [0.4]], dtype=tf.float32)
        with tf.GradientTape() as outer:
            outer.watch(x)
            with tf.GradientTape() as inner:
                inner.watch(x)
                values = model(x)
            first = inner.gradient(values, x)
        second = outer.gradient(first, x)
        self.assertTrue(np.all(np.isfinite(second.numpy())))

    def test_fourier_embeddings_identify_periodic_endpoints(self):
        cases = (
            ("periodic_unit", -1.0, 1.0),
            ("periodic_pi", -np.pi, np.pi),
        )
        for fourier, left, right in cases:
            with self.subTest(fourier=fourier):
                model = ModifiedMLP(
                    ((0.0, 1.0), (left, right)),
                    output_dim=1,
                    fourier=fourier,
                    seed=5,
                )
                inputs = tf.constant([[0.4, left], [0.4, right]], dtype=tf.float32)
                predictions = model(inputs).numpy().ravel()
                np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
