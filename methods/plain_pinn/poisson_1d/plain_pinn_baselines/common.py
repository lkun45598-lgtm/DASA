import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

WORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORK_ROOT.parent
DEEPXDE_ROOT = PROJECT_ROOT / "deepxde"
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", str(WORK_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(DEEPXDE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPXDE_ROOT))

import deepxde as dde
import torch

from deepxde.nn.pytorch.fnn import FNN


ArrayFn = Callable[[np.ndarray], np.ndarray]
BACKBONE_VERSION = "plain_fnn_original_pinn"


@dataclass(frozen=True)
class TrainConfig:
    adam_iterations: int
    lr: float = 1e-3
    decay_steps: int = 2000
    decay_rate: float = 0.9
    lbfgs_iterations: int = 0
    display_every: int = 1000


class PlainFNN(FNN):
    """Ordinary fully connected PINN backbone without modified-MLP gates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 6,
        width: int = 128,
        activation: str = "tanh",
        initializer: str = "Glorot normal",
        fourier_features: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if depth < 1:
            raise ValueError("depth must be at least 1")

        feature_dim = input_dim
        if fourier_features is not None:
            with torch.no_grad():
                probe = torch.zeros((1, input_dim), dtype=dde.config.real(torch))
                feature_dim = int(fourier_features(probe).shape[1])

        super().__init__(
            [feature_dim] + [width] * depth + [output_dim],
            activation,
            initializer,
        )
        if fourier_features is not None:
            self.apply_feature_transform(fourier_features)

        self.backbone_version = BACKBONE_VERSION
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.depth = depth
        self.width = width
        self.activation_name = activation
        self.initializer_name = initializer
        self.fourier_features = fourier_features

    def architecture_metadata(self) -> dict:
        return {
            "type": "plain_fnn",
            "version": self.backbone_version,
            "source": "Original PINNs ordinary fully connected network family",
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "hidden_layer_count": self.depth,
            "width": self.width,
            "activation": self.activation_name,
            "initializer": self.initializer_name,
            "gated_modified_mlp": False,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def set_seed(seed: int):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compile_and_train(model: dde.Model, config: TrainConfig, loss_weights, loss="MSE"):
    model.compile(
        "adam",
        lr=config.lr,
        decay=("step", config.decay_steps, config.decay_rate),
        loss=loss,
        loss_weights=loss_weights,
    )
    losshistory, train_state = model.train(
        iterations=config.adam_iterations,
        display_every=config.display_every,
    )
    if config.lbfgs_iterations > 0:
        dde.optimizers.set_LBFGS_options(maxiter=config.lbfgs_iterations)
        model.compile("L-BFGS", loss=loss, loss_weights=loss_weights)
        losshistory, train_state = model.train(display_every=config.display_every)
    return losshistory, train_state


def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def max_pointwise_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def evaluate_errors(model: dde.Model, points: np.ndarray, exact: ArrayFn) -> dict[str, float]:
    y_true = exact(points)
    y_pred = model.predict(points)
    return {
        "l2_relative_error": l2_relative_error(y_true, y_pred),
        "max_pointwise_error": max_pointwise_error(y_true, y_pred),
    }


def save_predictions(path: Path, points: np.ndarray, exact: ArrayFn, model: dde.Model):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=points, y_true=exact(points), y_pred=model.predict(points))


def experiment_dir(output_root: Path, equation: str, case: str) -> Path:
    path = output_root / equation / case
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(path: Path, metrics: dict, metadata: Optional[dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def positive_int(value: str) -> int:
    value_int = int(value)
    if value_int < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value_int
