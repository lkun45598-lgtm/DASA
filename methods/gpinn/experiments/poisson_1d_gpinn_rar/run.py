"""Run one audited 1D Poisson gPINN protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .cases import CASES, build_case
from .model import MLP, objective_terms, pde_residual


LOG_FIELDS = (
    "phase",
    "rar_cycle",
    "step",
    "lbfgs_closures",
    "total",
    "pde",
    "gradient",
    "collocation_points",
    "learning_rate",
    "wall_seconds",
)


PROTOCOLS = ("upstream_poisson", "paper_gpinn", "burgers_rar_transfer")
METHOD_NAMES = {
    "upstream_poisson": "gPINN (upstream Poisson protocol; target equation)",
    "paper_gpinn": "gPINN (paper sampling + upstream Poisson optimizer)",
    "burgers_rar_transfer": "gPINN + RAR (explicit upstream Burgers-RAR transfer)",
}


@dataclass
class RunConfig:
    case: str = "boundary_layer"
    seed: int = 0
    mode: str = "full"
    device: str = "auto"
    protocol: str = "burgers_rar_transfer"
    # Fixed benchmark network.
    hidden_layers: int = 6
    width: int = 128
    activation: str = "tanh"
    dtype: str = "float64"
    initializer: str = "xavier_uniform"
    # Target-paper sampling plus the upstream Poisson gPINN optimizer/loss.
    initial_points: int = 500
    sampling_distribution: str = "latin_hypercube"
    gradient_weight: float = 0.01
    learning_rate: float = 1.0e-3
    initial_adam_steps: int = 20_000
    initial_lbfgs_iterations: int = 15_000
    # RAR is absent from upstream poisson_1d.py.  Nonzero values are only set
    # by the explicitly named burgers_rar_transfer protocol.
    rar_cycles: int = 40
    rar_candidate_points: int = 100_000
    rar_points_per_cycle: int = 10
    rar_adam_steps: int = 10_000
    rar_lbfgs_iterations: int = 15_000
    rar_early_stopping: bool = True
    early_stopping_min_delta: float = 1.0e-4
    early_stopping_patience: int = 2_000
    early_stopping_check_every: int = 1_000
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1.0e-8
    lbfgs_tolerance_change: float = 0.0
    log_every: int = 100
    checkpoint_every: int = 1_000
    test_points: int = 2_001

    def apply_protocol_defaults(self) -> None:
        """Apply a fully named protocol; never silently synthesize Poisson RAR."""

        if self.protocol == "upstream_poisson":
            # Exact upstream poisson_1d.py numerical protocol, apart from the
            # user-required equation and 6x128 architecture: 15 fixed uniform
            # interior points, Adam(1e-3) for 20k epochs, w_grad=0.01, no RAR
            # and no L-BFGS.
            self.initial_points = 15
            self.sampling_distribution = "uniform"
            self.gradient_weight = 0.01
            self.initial_adam_steps = 20_000
            self.initial_lbfgs_iterations = 0
            self.rar_cycles = 0
            self.rar_candidate_points = 100_000
            self.rar_points_per_cycle = 10
            self.rar_adam_steps = 10_000
            self.rar_lbfgs_iterations = 15_000
            self.rar_early_stopping = True
        elif self.protocol == "paper_gpinn":
            self.initial_points = 500
            self.sampling_distribution = "latin_hypercube"
            self.gradient_weight = 0.01
            self.initial_adam_steps = 20_000
            self.initial_lbfgs_iterations = 0
            self.rar_cycles = 0
            self.rar_candidate_points = 100_000
            self.rar_points_per_cycle = 10
            self.rar_adam_steps = 10_000
            self.rar_lbfgs_iterations = 15_000
            self.rar_early_stopping = True
        elif self.protocol == "burgers_rar_transfer":
            # Equation/domain/500 LHS points and w=0.01 remain the target-paper
            # Poisson setup.  Only the RAR schedule comes from upstream
            # burgers.py: initial Adam+L-BFGS, then 40 rounds of a 100k random
            # pool, Top-10 |r| anchors, Adam<=10k with early stopping, L-BFGS.
            self.initial_points = 500
            self.sampling_distribution = "latin_hypercube"
            self.gradient_weight = 0.01
            self.initial_adam_steps = 20_000
            self.initial_lbfgs_iterations = 15_000
            self.rar_cycles = 40
            self.rar_candidate_points = 100_000
            self.rar_points_per_cycle = 10
            self.rar_adam_steps = 10_000
            self.rar_lbfgs_iterations = 15_000
            self.rar_early_stopping = True
        else:
            raise ValueError(f"unknown protocol {self.protocol!r}")

    def apply_smoke_defaults(self) -> None:
        # Network and equation stay identical; only work counts are reduced.
        self.initial_points = 64
        self.initial_adam_steps = 2
        self.initial_lbfgs_iterations = 0
        self.rar_cycles = 0 if self.protocol != "burgers_rar_transfer" else 2
        self.rar_candidate_points = 501
        self.rar_points_per_cycle = 8
        self.rar_adam_steps = 2
        self.rar_lbfgs_iterations = 0
        self.early_stopping_patience = 2
        self.log_every = 1
        self.checkpoint_every = 2
        self.test_points = 201

    def validate(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if self.case not in CASES:
            raise ValueError(f"unknown case {self.case!r}")
        if self.hidden_layers != 6 or self.width != 128:
            raise ValueError("the benchmark contract requires a 6x128 network")
        if self.activation != "tanh" or self.dtype != "float64":
            raise ValueError("the benchmark contract requires tanh and float64")
        if self.initializer != "xavier_uniform":
            raise ValueError("upstream poisson_1d.py requires Xavier/Glorot uniform")
        if self.sampling_distribution not in ("uniform", "latin_hypercube"):
            raise ValueError("sampling distribution must be uniform or latin_hypercube")
        expected_sampling = (
            "uniform" if self.protocol == "upstream_poisson" else "latin_hypercube"
        )
        if self.sampling_distribution != expected_sampling:
            raise ValueError(
                f"{self.protocol} requires {expected_sampling} sampling; "
                "use a different named protocol instead of silently overriding it"
            )
        positive = (
            "initial_points",
            "rar_candidate_points",
            "rar_points_per_cycle",
            "lbfgs_history_size",
            "log_every",
            "checkpoint_every",
            "test_points",
            "early_stopping_patience",
            "early_stopping_check_every",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = (
            "initial_adam_steps",
            "initial_lbfgs_iterations",
            "rar_cycles",
            "rar_adam_steps",
            "rar_lbfgs_iterations",
        )
        for name in nonnegative:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.rar_points_per_cycle > self.rar_candidate_points:
            raise ValueError("RAR cannot select more points than its candidate pool")
        if self.gradient_weight < 0.0 or self.learning_rate <= 0.0:
            raise ValueError("gradient weight must be non-negative and LR positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early stopping min_delta must be non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "mps":
        device = torch.device("cpu")
    probe = torch.tensor([[0.3]], dtype=torch.float64, device=device, requires_grad=True)
    value = torch.tanh(probe) ** 4
    for _ in range(3):
        value = torch.autograd.grad(
            value, probe, torch.ones_like(value), create_graph=True
        )[0]
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{device} failed the float64 third-derivative probe")
    return device


def latin_hypercube(
    count: int,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    return (lower + (upper - lower) * strata[rng.permutation(count)])[:, None]


def uniform_interior(count: int, lower: float, upper: float) -> np.ndarray:
    """DeepXDE Interval.uniform_points(count, boundary=False) equivalent."""

    return np.linspace(lower, upper, count + 2, dtype=np.float64)[1:-1, None]


def input_tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.as_tensor(points, dtype=torch.float64, device=device)
        .detach()
        .requires_grad_(True)
    )


def _cpu_cuda_states(states) -> list[torch.Tensor]:
    if states is None:
        return []
    return [state.detach().cpu().to(dtype=torch.uint8).contiguous() for state in states]


def select_top_residual_points(
    candidates: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidates with the largest raw absolute PDE residuals."""

    candidates = np.asarray(candidates, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(candidates) != len(scores):
        raise ValueError("candidate and score lengths differ")
    if count <= 0 or count > len(candidates):
        raise ValueError("selection count is outside the candidate pool")
    if not np.isfinite(scores).all():
        raise FloatingPointError("RAR scores contain NaN or Inf")
    indices = np.argpartition(scores, -count)[-count:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return candidates[indices], indices


class RARTrainer:
    """Initial gPINN fit followed by residual-greedy anchor refinement."""

    def __init__(
        self,
        model: MLP,
        config: RunConfig,
        result_dir: Path,
        device: torch.device,
        resume: bool,
    ) -> None:
        self.model = model
        self.config = config
        self.result_dir = result_dir
        self.device = device
        self.rng = np.random.default_rng(config.seed)
        if config.sampling_distribution == "uniform":
            self.points = uniform_interior(config.initial_points, model.lower, model.upper)
        else:
            self.points = latin_hypercube(
                config.initial_points, model.lower, model.upper, self.rng
            )
        self.stage = "initial_adam"
        self.rar_cycle = 0
        self.stage_step = 0
        self.global_step = 0
        self.lbfgs_closures = 0
        self.adam_best_loss = float("inf")
        self.adam_stale_steps = 0
        self.started = time.perf_counter()
        self.optimizer = self._new_adam()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.resumed = False
        if resume and (self.result_dir / "last.pt").exists():
            self._load()
            self.resumed = True

    def _new_adam(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "points": self.points,
            "stage": self.stage,
            "rar_cycle": self.rar_cycle,
            "stage_step": self.stage_step,
            "global_step": self.global_step,
            "lbfgs_closures": self.lbfgs_closures,
            "adam_best_loss": self.adam_best_loss,
            "adam_stale_steps": self.adam_stale_steps,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "config": asdict(self.config),
        }

    def _save(self) -> None:
        temporary = self.result_dir / "last.pt.tmp"
        torch.save(self._payload(), temporary)
        temporary.replace(self.result_dir / "last.pt")

    def _load(self) -> None:
        state = torch.load(
            self.result_dir / "last.pt", map_location=self.device, weights_only=False
        )
        saved = state.get("config", {})
        mismatched = {
            key: (saved.get(key), value)
            for key, value in asdict(self.config).items()
            if key in saved and saved[key] != value and key != "device"
        }
        if mismatched:
            raise ValueError(f"cannot resume after configuration changes: {mismatched}")
        self.model.load_state_dict(state["model"])
        self.optimizer = self._new_adam()
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.points = np.asarray(state["points"], dtype=np.float64)
        self.stage = str(state["stage"])
        self.rar_cycle = int(state["rar_cycle"])
        self.stage_step = int(state["stage_step"])
        self.global_step = int(state["global_step"])
        self.lbfgs_closures = int(state.get("lbfgs_closures", 0))
        self.adam_best_loss = float(state.get("adam_best_loss", float("inf")))
        self.adam_stale_steps = int(state.get("adam_stale_steps", 0))
        self.rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_states = _cpu_cuda_states(state.get("cuda_rng_state_all"))
        if torch.cuda.is_available() and cuda_states:
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                torch.cuda.set_rng_state(cuda_states[0])

    def _write_row(
        self,
        writer: csv.DictWriter,
        handle,
        phase: str,
        terms: dict[str, float],
    ) -> None:
        writer.writerow(
            {
                "phase": phase,
                "rar_cycle": self.rar_cycle,
                "step": self.global_step,
                "lbfgs_closures": self.lbfgs_closures,
                "total": terms["total"],
                "pde": terms["pde"],
                "gradient": terms["gradient"],
                "collocation_points": len(self.points),
                "learning_rate": (
                    1.0 if "lbfgs" in phase else self.config.learning_rate
                ),
                "wall_seconds": time.perf_counter() - self.started,
            }
        )
        handle.flush()

    def _adam(
        self,
        writer,
        handle,
        phase: str,
        target: int,
        early_stopping: bool = False,
    ) -> None:
        if early_stopping and not np.isfinite(self.adam_best_loss):
            x_initial = input_tensor(self.points, self.device)
            self.adam_best_loss = float(
                objective_terms(
                    self.model, x_initial, self.config.gradient_weight
                )["total"].detach()
            )
            self.adam_stale_steps = 0
        while self.stage_step < target:
            x = input_tensor(self.points, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            tensors = objective_terms(self.model, x, self.config.gradient_weight)
            if not torch.isfinite(tensors["total"]):
                raise FloatingPointError(f"non-finite {phase} objective")
            tensors["total"].backward()
            self.optimizer.step()
            self.stage_step += 1
            self.global_step += 1
            check_early_stopping = early_stopping and (
                self.stage_step % self.config.early_stopping_check_every == 0
                or self.stage_step == target
            )
            if check_early_stopping:
                x_check = input_tensor(self.points, self.device)
                current = float(
                    objective_terms(
                        self.model, x_check, self.config.gradient_weight
                    )["total"].detach()
                )
                if current < self.adam_best_loss - self.config.early_stopping_min_delta:
                    self.adam_best_loss = current
                    self.adam_stale_steps = 0
                else:
                    self.adam_stale_steps += self.config.early_stopping_check_every
            if (
                self.global_step == 1
                or self.stage_step % self.config.log_every == 0
                or self.stage_step == target
            ):
                terms = {name: float(value.detach()) for name, value in tensors.items()}
                self._write_row(writer, handle, phase, terms)
            if (
                self.stage_step % self.config.checkpoint_every == 0
                or self.stage_step == target
            ):
                self._save()
            if early_stopping and self.adam_stale_steps >= self.config.early_stopping_patience:
                break

    def _lbfgs(self, writer, handle, phase: str, iterations: int) -> None:
        if iterations <= 0:
            return
        x = input_tensor(self.points, self.device)
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=iterations,
            max_eval=int(iterations * 1.25),
            history_size=self.config.lbfgs_history_size,
            tolerance_grad=self.config.lbfgs_tolerance_grad,
            tolerance_change=self.config.lbfgs_tolerance_change,
            line_search_fn=None,
        )
        latest = {"total": float("nan"), "pde": float("nan"), "gradient": float("nan")}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            tensors = objective_terms(self.model, x, self.config.gradient_weight)
            if not torch.isfinite(tensors["total"]):
                raise FloatingPointError(f"non-finite {phase} objective")
            tensors["total"].backward()
            latest.update({name: float(value.detach()) for name, value in tensors.items()})
            self.lbfgs_closures += 1
            if self.lbfgs_closures % self.config.log_every == 0:
                self._write_row(writer, handle, phase, latest)
            return tensors["total"]

        optimizer.step(closure)
        self._write_row(writer, handle, phase, latest)

    def _refine(self) -> None:
        """Classic RAR: random pool, rank by raw |PDE residual|, add Top-K."""

        config = self.config
        candidates = self.rng.uniform(
            self.model.lower,
            self.model.upper,
            size=(config.rar_candidate_points, 1),
        ).astype(np.float64)
        score_parts: list[np.ndarray] = []
        for start in range(0, len(candidates), 4096):
            x = input_tensor(candidates[start : start + 4096], self.device)
            residual = pde_residual(self.model, x)
            score_parts.append(residual.detach().abs().cpu().numpy()[:, 0])
        scores = np.concatenate(score_parts)
        selected, indices = select_top_residual_points(
            candidates, scores, config.rar_points_per_cycle
        )
        self.points = np.concatenate((self.points, selected), axis=0)
        record = {
            "cycle": self.rar_cycle + 1,
            "candidate_points": config.rar_candidate_points,
            "added_points": config.rar_points_per_cycle,
            "total_points": len(self.points),
            "pool_mean_abs_residual": float(np.mean(scores)),
            "pool_max_abs_residual": float(np.max(scores)),
            "selected_min_abs_residual": float(np.min(scores[indices])),
            "selected_x": selected[:, 0].tolist(),
        }
        # Make this write idempotent.  If the process was interrupted after the
        # history write but before the following checkpoint, resume repeats the
        # same refinement cycle from the saved RNG state.  Discarding a stale
        # record for this cycle prevents duplicate history rows.
        history_path = self.result_dir / "rar_history.jsonl"
        retained: list[dict] = []
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                previous = json.loads(line)
                if int(previous["cycle"]) < int(record["cycle"]):
                    retained.append(previous)
        with history_path.open("w", encoding="utf-8") as handle:
            for previous in retained:
                handle.write(json.dumps(previous, sort_keys=True) + "\n")
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        np.savez_compressed(
            self.result_dir / "rar_points.npz",
            points=self.points,
            initial_points=np.asarray(config.initial_points),
        )

    def train(self) -> dict[str, int | float]:
        config = self.config
        log_path = self.result_dir / "training_log.csv"
        append = self.resumed and log_path.exists()
        if not append:
            stale = self.result_dir / "rar_history.jsonl"
            if stale.exists():
                stale.unlink()
            stale.touch()
            np.savez_compressed(
                self.result_dir / "rar_points.npz",
                points=self.points,
                initial_points=np.asarray(config.initial_points),
            )
        with log_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not append:
                writer.writeheader()

            if self.stage == "initial_adam":
                self._adam(writer, handle, "initial_adam", config.initial_adam_steps)
                self.stage = "initial_lbfgs"
                self.stage_step = 0
                self._save()

            if self.stage == "initial_lbfgs":
                self._lbfgs(
                    writer,
                    handle,
                    "initial_lbfgs",
                    config.initial_lbfgs_iterations,
                )
                self.stage = "rar_refine"
                self.stage_step = 0
                self._save()

            while self.rar_cycle < config.rar_cycles:
                if self.stage == "rar_refine":
                    self._refine()
                    self.stage = "rar_adam"
                    self.stage_step = 0
                    self.adam_best_loss = float("inf")
                    self.adam_stale_steps = 0
                    self.optimizer = self._new_adam()
                    self._save()
                if self.stage == "rar_adam":
                    self._adam(
                        writer,
                        handle,
                        "rar_adam",
                        config.rar_adam_steps,
                        early_stopping=config.rar_early_stopping,
                    )
                    self.stage = "rar_lbfgs"
                    self.stage_step = 0
                    self._save()
                if self.stage == "rar_lbfgs":
                    self._lbfgs(
                        writer,
                        handle,
                        "rar_lbfgs",
                        config.rar_lbfgs_iterations,
                    )
                    self.rar_cycle += 1
                    self.stage = "rar_refine"
                    self.stage_step = 0
                    self._save()

            if self.stage == "rar_refine":
                self.stage = "complete"
            self.stage = "complete"
            self._save()
        return {
            "adam_steps_completed": self.global_step,
            "lbfgs_closure_evaluations": self.lbfgs_closures,
            "rar_cycles_completed": self.rar_cycle,
            "collocation_points": int(len(self.points)),
        }


def evaluate(
    model: MLP,
    config: RunConfig,
    result_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    case = model.case
    x = np.linspace(case.lower, case.upper, config.test_points, dtype=np.float64)[:, None]
    tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(tensor).cpu().numpy()
    reference = case.exact_numpy(x)
    error = prediction - reference
    relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(reference))
    max_abs = float(np.max(np.abs(error)))
    if not np.isfinite([relative_l2, max_abs]).all():
        raise FloatingPointError("evaluation contains NaN or Inf")
    np.savez_compressed(
        result_dir / "prediction.npz",
        points=x,
        prediction=prediction,
    )
    np.savez_compressed(
        result_dir / "reference.npz",
        points=x,
        reference=reference,
        reference_type=np.asarray("analytic"),
    )
    return {
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "test_points": len(x),
        "boundary_max_abs": float(np.max(np.abs(error[[0, -1]]))),
        "reference_type": "analytic",
    }


def plot_outputs(result_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(result_dir / "prediction.npz") as data:
        x = data["points"][:, 0]
        prediction = data["prediction"][:, 0]
    with np.load(result_dir / "reference.npz") as data:
        reference = data["reference"][:, 0]
    with np.load(result_dir / "rar_points.npz") as data:
        points = data["points"][:, 0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    axes[0].plot(x, reference, label="Exact")
    axes[0].plot(x, prediction, "--", label="gPINN prediction")
    axes[0].legend()
    axes[0].set_ylabel("u")
    axes[1].semilogy(x, np.maximum(np.abs(prediction - reference), 1.0e-18))
    axes[1].set_ylabel("absolute error")
    axes[2].hist(points, bins=100, range=(x[0], x[-1]))
    axes[2].set_ylabel("collocation points")
    axes[2].set_xlabel("x")
    fig.savefig(result_dir / "solution_error_rar.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(x, np.abs(prediction - reference))
    axis.set_xlabel("x")
    axis.set_ylabel("absolute error")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "absolute_error.png", dpi=180)
    plt.close(fig)

    steps: list[int] = []
    totals: list[float] = []
    with (result_dir / "training_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            totals.append(float(row["total"]))
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.semilogy(steps, np.maximum(totals, 1.0e-30))
    axis.set_xlabel("Adam step")
    axis.set_ylabel("gPINN loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def run(config: RunConfig, output_root: Path, resume: bool = False) -> dict:
    config.validate()
    set_seed(config.seed)
    torch.set_default_dtype(torch.float64)
    device = select_device(config.device)
    case = build_case(config.case)
    result_dir = (output_root / config.mode / config.case / f"seed_{config.seed}").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
    model = MLP(case).to(device=device, dtype=torch.float64)
    trainer = RARTrainer(model, config, result_dir, device, resume)
    training = trainer.train()
    metrics = {
        "case": config.case,
        "seed": config.seed,
        "mode": config.mode,
        "protocol": config.protocol,
        "method": METHOD_NAMES[config.protocol],
        "rar_enabled": config.rar_cycles > 0,
        "rar_score": (
            "absolute_raw_pde_residual" if config.rar_cycles > 0 else "not_applicable"
        ),
        "architecture": "1-[128x6]-1, tanh, Xavier-uniform, float64",
        "device": str(device),
        "gradient_weight": config.gradient_weight,
        "initial_adam_steps_configured": config.initial_adam_steps,
        "initial_lbfgs_iterations_configured": config.initial_lbfgs_iterations,
        "rar_adam_steps_per_cycle_configured": config.rar_adam_steps,
        "rar_cycles_configured": config.rar_cycles,
        "rar_candidate_points_configured": config.rar_candidate_points,
        "rar_points_per_cycle_configured": config.rar_points_per_cycle,
        "rar_lbfgs_iterations_per_cycle_configured": config.rar_lbfgs_iterations,
        "protocol_provenance": {
            "target_equation_and_domain": "Zhang et al. Eq. (10)-(11), local PDF",
            "network": "user-required 6 hidden layers x 128",
            "poisson_optimizer_and_gpinn_weight": (
                "lu-group/gpinn commit 09dd590, src/poisson_1d.py"
            ),
            "rar_schedule": (
                "lu-group/gpinn commit 09dd590, src/burgers.py; transferred"
                if config.protocol == "burgers_rar_transfer"
                else "none"
            ),
        },
        "full_training_completed": config.mode == "full" and trainer.stage == "complete",
        **training,
        **evaluate(model, config, result_dir, device),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    plot_outputs(result_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


INT_FIELDS = (
    "initial_points",
    "initial_adam_steps",
    "initial_lbfgs_iterations",
    "rar_cycles",
    "rar_candidate_points",
    "rar_points_per_cycle",
    "rar_adam_steps",
    "rar_lbfgs_iterations",
    "early_stopping_patience",
    "early_stopping_check_every",
    "lbfgs_history_size",
    "log_every",
    "checkpoint_every",
    "test_points",
)
FLOAT_FIELDS = (
    "gradient_weight",
    "learning_rate",
    "lbfgs_tolerance_grad",
    "lbfgs_tolerance_change",
    "early_stopping_min_delta",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="burgers_rar_transfer")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rar-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    for name in INT_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    for name in FLOAT_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float)
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        case=args.case,
        seed=args.seed,
        mode=args.mode,
        device=args.device,
        protocol=args.protocol,
    )
    config.apply_protocol_defaults()
    if args.mode == "smoke":
        config.apply_smoke_defaults()
    for name in (*INT_FIELDS, *FLOAT_FIELDS):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    if args.rar_early_stopping is not None:
        config.rar_early_stopping = args.rar_early_stopping
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.output_root is None:
        args.output_root = (
            Path(__file__).resolve().parents[2]
            / "results"
            / f"poisson_1d_{args.protocol}_10seeds"
        )
    return run(config_from_args(args), args.output_root, resume=args.resume)


if __name__ == "__main__":
    main()
