import deepxde as dde


def test_deepxde_uses_pytorch_backend():
    assert dde.__version__ == "1.15.0"
    assert dde.backend.backend_name == "pytorch"
