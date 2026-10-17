import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
CASES = ("Allen-Cahn", "Burgers", "Helmholtz")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlottingCompatibilityTests(unittest.TestCase):
    def test_plotting_helpers_import_without_latex(self):
        import matplotlib.pyplot as plt

        for case in CASES:
            with self.subTest(case=case):
                with mock.patch("shutil.which", return_value=None):
                    module = load_module(ROOT / case / "plotting.py", f"plotting_{case}")
                self.assertFalse(module.mpl.rcParams["text.usetex"])
                fig, _ = module.newfig(0.5)
                plt.close(fig)


class LbfgsCompatibilityTests(unittest.TestCase):
    def test_stops_when_loss_change_is_below_tolerance(self):
        for case in CASES:
            with self.subTest(case=case):
                module = load_module(ROOT / case / "eager_lbfgs.py", f"lbfgs_tol_{case}")
                evaluated = []

                def numerically_unchanged_loss(x):
                    x = tf.convert_to_tensor(x)
                    evaluated.append(x.numpy().copy())
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        value = tf.constant(1e20, dtype=x.dtype) + tf.reduce_sum(x)
                    return value, tape.gradient(value, x)

                _, _, function_evaluations = module.lbfgs(
                    numerically_unchanged_loss,
                    tf.constant([0.0]),
                    module.Struct(),
                    maxIter=5,
                    do_verbose=False,
                )

                self.assertEqual(function_evaluations, 2)
                self.assertEqual(len(evaluated), 2)

    def test_final_parameter_is_evaluated(self):
        for case in CASES:
            with self.subTest(case=case):
                module = load_module(ROOT / case / "eager_lbfgs.py", f"lbfgs_{case}")
                evaluated = []

                def quadratic(x):
                    x = tf.convert_to_tensor(x)
                    evaluated.append(x.numpy().copy())
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        value = tf.reduce_sum(tf.square(x - 3.0))
                    return value, tape.gradient(value, x)

                result, history, _ = module.lbfgs(
                    quadratic,
                    tf.constant([0.0]),
                    module.Struct(),
                    maxIter=2,
                    learningRate=0.1,
                    do_verbose=False,
                )

                np.testing.assert_allclose(evaluated[-1], result.numpy())
                self.assertLess(float(history[-1]), float(history[0]))


if __name__ == "__main__":
    unittest.main()
