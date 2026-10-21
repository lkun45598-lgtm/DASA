import unittest

import torch

from plain_pinn_baselines.allen_cahn_1d import periodic_fourier_features
from plain_pinn_baselines.common import PlainFNN, dde


class PlainFNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dde.config.set_default_float("float64")

    def test_six_hidden_layer_plain_fnn(self):
        net = PlainFNN(input_dim=2, output_dim=1, depth=6, width=128)

        self.assertEqual(len(net.linears), 7)
        self.assertEqual(tuple(net.linears[0].weight.shape), (128, 2))
        self.assertEqual(tuple(net.linears[-1].weight.shape), (1, 128))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 83073)
        self.assertFalse(any("encoder" in name for name, _ in net.named_modules()))

        x = torch.zeros((4, 2), dtype=torch.float64, requires_grad=True)
        y = net(x)
        gradient = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        self.assertEqual(tuple(y.shape), (4, 1))
        self.assertEqual(tuple(gradient.shape), (4, 2))

    def test_fourier_feature_input(self):
        def features(points):
            x = points[:, 0:1]
            t = points[:, 1:2]
            return torch.cat((t, torch.cos(torch.pi * x), torch.sin(torch.pi * x)), 1)

        net = PlainFNN(
            input_dim=2,
            output_dim=1,
            depth=6,
            width=128,
            fourier_features=features,
        )

        self.assertEqual(net.feature_dim, 3)
        self.assertEqual(tuple(net.linears[0].weight.shape), (128, 3))
        self.assertEqual(
            tuple(net(torch.zeros((2, 2), dtype=torch.float64)).shape), (2, 1)
        )

    def test_allen_cahn_order_eight_periodicity(self):
        net = PlainFNN(
            input_dim=2,
            output_dim=1,
            depth=6,
            width=128,
            fourier_features=periodic_fourier_features,
        )

        self.assertEqual(net.feature_dim, 17)
        self.assertEqual(sum(p.numel() for p in net.parameters()), 84993)

        left = torch.tensor([[-1.0, 0.37]], dtype=torch.float64, requires_grad=True)
        right = torch.tensor([[1.0, 0.37]], dtype=torch.float64, requires_grad=True)
        left_value = net(left)
        right_value = net(right)
        left_dx = torch.autograd.grad(left_value.sum(), left)[0][:, :1]
        right_dx = torch.autograd.grad(right_value.sum(), right)[0][:, :1]

        self.assertTrue(torch.allclose(left_value, right_value, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(left_dx, right_dx, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
