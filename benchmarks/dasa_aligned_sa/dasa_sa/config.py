"""Strict loading for resolved benchmark case configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SUITE_ROOT / "configs" / "cases"
ALLOWED_PROVENANCE = frozenset({"paper", "user", "sa_reference", "inferred"})
NETWORK_OVERRIDES = {
    "diffusion_reaction": (4, 64),
    "discontinuous_advection": (4, 64),
}


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: int
    width: int
    activation: str
    initializer: str
    output_dim: int
    fourier: str


@dataclass(frozen=True)
class OptimizerConfig:
    stages: tuple[int, int, int, int]
    adam_steps: int
    lbfgs_steps: int
    learning_rate: float
    adam_beta_1: float
    decay_rate: float
    decay_steps: int
    loss_weights: dict[str, float]


@dataclass(frozen=True)
class SamplingConfig:
    residual: int
    boundary: int
    initial: int
    method: str
    residual_grid: tuple[int, ...]


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    family: str
    domain: tuple[tuple[float, float], ...]
    parameters: dict[str, float]
    sampling: SamplingConfig
    network: NetworkConfig
    optimizer: OptimizerConfig
    provenance: dict[str, str]


_TOP_LEVEL_KEYS = {
    "case_id",
    "family",
    "domain",
    "parameters",
    "sampling",
    "network",
    "optimizer",
    "provenance",
}


def _require_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {context} keys: missing={missing}, unknown={unknown}")


def _positive(value: int, name: str, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def list_case_ids() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in CONFIG_ROOT.glob("*.yaml")))


def load_case_config(case_id: str) -> CaseConfig:
    path = CONFIG_ROOT / f"{case_id}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown case: {case_id}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    _require_keys(data, _TOP_LEVEL_KEYS, "case")
    if data["case_id"] != case_id:
        raise ValueError(f"case_id mismatch in {path}")

    network_data = data["network"]
    _require_keys(
        network_data,
        {"hidden_layers", "width", "activation", "initializer", "output_dim", "fourier"},
        "network",
    )
    network = NetworkConfig(**network_data)
    expected_depth, expected_width = NETWORK_OVERRIDES.get(case_id, (6, 128))
    if network.hidden_layers != expected_depth or network.width != expected_width:
        raise ValueError(
            f"{case_id} must use a {expected_depth}x{expected_width} network"
        )
    if network.activation != "tanh" or network.initializer != "glorot_normal":
        raise ValueError(f"{case_id} has an unsupported network definition")

    sampling_data = data["sampling"]
    _require_keys(
        sampling_data,
        {"residual", "boundary", "initial", "method", "residual_grid"},
        "sampling",
    )
    sampling = SamplingConfig(
        residual=_positive(sampling_data["residual"], "sampling.residual"),
        boundary=_positive(sampling_data["boundary"], "sampling.boundary", allow_zero=True),
        initial=_positive(sampling_data["initial"], "sampling.initial", allow_zero=True),
        method=str(sampling_data["method"]),
        residual_grid=tuple(int(v) for v in sampling_data["residual_grid"]),
    )

    optimizer_data = dict(data["optimizer"])
    optimizer_data.setdefault("adam_beta_1", 0.9)
    _require_keys(
        optimizer_data,
        {
            "stages",
            "adam_steps",
            "lbfgs_steps",
            "learning_rate",
            "adam_beta_1",
            "decay_rate",
            "decay_steps",
            "loss_weights",
        },
        "optimizer",
    )
    stages = tuple(int(v) for v in optimizer_data["stages"])
    if len(stages) != 4 or any(v < 0 for v in stages):
        raise ValueError("optimizer.stages must contain four nonnegative integers")
    optimizer = OptimizerConfig(
        stages=stages,
        adam_steps=_positive(optimizer_data["adam_steps"], "optimizer.adam_steps"),
        lbfgs_steps=_positive(
            optimizer_data["lbfgs_steps"], "optimizer.lbfgs_steps", allow_zero=True
        ),
        learning_rate=float(optimizer_data["learning_rate"]),
        adam_beta_1=float(optimizer_data["adam_beta_1"]),
        decay_rate=float(optimizer_data["decay_rate"]),
        decay_steps=_positive(optimizer_data["decay_steps"], "optimizer.decay_steps"),
        loss_weights={key: float(value) for key, value in optimizer_data["loss_weights"].items()},
    )
    if optimizer.adam_steps != sum(optimizer.stages):
        raise ValueError(f"{case_id} Adam budget does not equal the four-stage sum")
    if optimizer.learning_rate <= 0.0:
        raise ValueError(f"{case_id} optimizer learning rate must be positive")
    if not 0.0 <= optimizer.adam_beta_1 < 1.0:
        raise ValueError(f"{case_id} optimizer adam_beta_1 must be in [0, 1)")

    provenance = {str(key): str(value) for key, value in data["provenance"].items()}
    invalid = set(provenance.values()) - ALLOWED_PROVENANCE
    if invalid:
        raise ValueError(f"invalid provenance labels: {sorted(invalid)}")

    domain = tuple(tuple(float(bound) for bound in axis) for axis in data["domain"])
    if not domain or any(len(axis) != 2 or axis[0] >= axis[1] for axis in domain):
        raise ValueError(f"invalid domain for {case_id}")

    return CaseConfig(
        case_id=case_id,
        family=str(data["family"]),
        domain=domain,
        parameters={key: float(value) for key, value in data["parameters"].items()},
        sampling=sampling,
        network=network,
        optimizer=optimizer,
        provenance=provenance,
    )
