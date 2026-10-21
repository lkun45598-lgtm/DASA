import torch

from methods.deepxde_pinn.network import ClassicalMLP


def test_classical_mlp_has_approved_depth_width_and_outputs():
    scalar = ClassicalMLP(((-1.0, 1.0),), output_dim=1, seed=7)
    cavity = ClassicalMLP(((0.0, 1.0), (0.0, 1.0)), output_dim=3, seed=7)

    assert scalar.hidden_depth == 6
    assert scalar.width == 128
    assert len(scalar.hidden_layers) == 6
    assert sum(parameter.numel() for parameter in scalar.parameters()) == 82945
    assert not hasattr(scalar, "u_encoder")
    assert not hasattr(scalar, "v_encoder")
    assert not hasattr(scalar, "mixing_layers")
    assert scalar(torch.zeros((4, 1))).shape == (4, 1)
    assert cavity(torch.zeros((4, 2))).shape == (4, 3)


def test_classical_mlp_normalizes_raw_coordinates():
    model = ClassicalMLP(((-2.0, 2.0), (0.0, 4.0)), output_dim=1, seed=11)
    points = torch.tensor([[-2.0, 0.0], [0.0, 2.0], [2.0, 4.0]])

    torch.testing.assert_close(
        model.normalize(points),
        torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
    )


def test_seeded_initialization_is_deterministic():
    first = ClassicalMLP(((-1.0, 1.0),), output_dim=1, seed=19)
    second = ClassicalMLP(((-1.0, 1.0),), output_dim=1, seed=19)
    points = torch.tensor([[-0.2], [0.7]], dtype=torch.float32)

    torch.testing.assert_close(first(points), second(points))
