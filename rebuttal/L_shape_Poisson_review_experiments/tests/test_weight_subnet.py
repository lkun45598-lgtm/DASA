import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lshape_experiment import ModifiedMLP, VanillaMLP, WeightSubnet, build_backbone


def test_vanilla_mlp_has_six_standard_hidden_layers():
    model = VanillaMLP(dim_in=2, dim_out=1, width=128, depth=6)

    assert len(model.hidden_layers) == 6
    assert sum(parameter.numel() for parameter in model.parameters()) == 83073
    assert not hasattr(model, "encoder_u")
    assert not hasattr(model, "encoder_v")
    assert model(torch.zeros((4, 2))).shape == (4, 1)


def test_build_backbone_selects_configured_architecture():
    common = {"depth": 6, "width": 128}

    assert isinstance(build_backbone({"type": "vanilla_mlp", **common}), VanillaMLP)
    assert isinstance(build_backbone({"type": "modified_mlp", **common}), ModifiedMLP)


def test_scaled_pointwise_weights_have_unit_mean():
    subnet = WeightSubnet(width=3, depth=1)
    points = torch.tensor(
        [[-1.0, -1.0], [-0.5, 0.25], [0.25, -0.5], [1.0, 1.0]]
    )

    weights = subnet(points)

    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
