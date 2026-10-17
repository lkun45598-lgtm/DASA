import numpy as np

from benchmarks.deepxde_pinn.metrics import maximum_pointwise_error, relative_l2_error


def test_requested_metric_formulas():
    truth = np.array([[1.0], [-2.0], [4.0]])
    prediction = np.array([[0.5], [-1.0], [3.0]])

    expected_l2 = np.linalg.norm(truth - prediction) / np.linalg.norm(truth)
    expected_max = np.max(np.abs(truth - prediction))

    assert relative_l2_error(truth, prediction) == expected_l2
    assert maximum_pointwise_error(truth, prediction) == expected_max


def test_metrics_reject_shape_mismatch():
    truth = np.ones((3, 1))
    prediction = np.ones((3,))

    for metric in (relative_l2_error, maximum_pointwise_error):
        try:
            metric(truth, prediction)
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("shape mismatch must be rejected")

