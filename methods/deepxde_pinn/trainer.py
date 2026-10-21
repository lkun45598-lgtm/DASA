"""Full-batch classical PINN training and two-metric evaluation."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import deepxde as dde
import numpy as np
import torch

from methods.sa_pinn_aligned.dasa_sa.config import CaseConfig, OptimizerConfig

from .artifacts import append_jsonl, atomic_write_checkpoint, atomic_write_json
from .metrics import maximum_pointwise_error, relative_l2_error
from .network import ClassicalMLP
from .problems.base import BaseProblem


DEFAULT_TRAINING_PROFILE = "case_config"
RAISSI_LBFGS_PROFILE = "raissi_lbfgs"
TRAINING_PROFILES = (DEFAULT_TRAINING_PROFILE, RAISSI_LBFGS_PROFILE)


def learning_rate_for_step(config: OptimizerConfig, step: int) -> float:
    if step < 0:
        raise ValueError("step must be nonnegative")
    warmup = config.stages[0] + config.stages[1]
    decay_count = max(0, step - warmup) // config.decay_steps
    return config.learning_rate * config.decay_rate**decay_count


class _AdamLearningRate(dde.callbacks.Callback):
    def __init__(self, config: OptimizerConfig) -> None:
        super().__init__()
        self.config = config

    def on_epoch_begin(self):
        rate = learning_rate_for_step(self.config, int(self.model.train_state.step))
        for group in self.model.opt.param_groups:
            group["lr"] = rate


class _ProgressLogger(dde.callbacks.Callback):
    def __init__(self, path: Path, *, final_step: int, started_at: float) -> None:
        super().__init__()
        self.path = path
        self.final_step = final_step
        self.started_at = started_at

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        if step != 1 and step % 100 != 0 and step != self.final_step:
            return
        losses = np.asarray(self.model.train_state.loss_train, dtype=np.float64)
        if not np.all(np.isfinite(losses)):
            raise FloatingPointError(f"non-finite loss at Adam step {step}")
        append_jsonl(
            self.path,
            {
                "phase": "adam",
                "step": step,
                "learning_rate": float(self.model.opt.param_groups[0]["lr"]),
                "loss": float(np.sum(losses)),
                "loss_components": [float(value) for value in losses],
                "elapsed_seconds": time.time() - self.started_at,
            },
        )


class Trainer:
    def __init__(
        self,
        config: CaseConfig,
        problem: BaseProblem,
        *,
        seed: int,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.problem = problem
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dde.config.set_random_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.network = ClassicalMLP(
            config.domain,
            config.network.output_dim,
            width=config.network.width,
            hidden_layers=config.network.hidden_layers,
            seed=self.seed,
        )
        self.model = dde.Model(problem.data, self.network)

    def _predict(self, points: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        device = next(self.network.parameters()).device
        predictions: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, points.shape[0], batch_size):
                batch = torch.as_tensor(
                    points[start : start + batch_size], dtype=torch.float32, device=device
                )
                values = self.network(batch).detach().cpu().numpy()
                predictions.append(self.problem.prediction_values(values))
        return np.vstack(predictions).astype(np.float64)

    def run(
        self,
        *,
        adam_steps: int | None = None,
        lbfgs_steps: int | None = None,
        training_profile: str = DEFAULT_TRAINING_PROFILE,
    ) -> dict[str, int | float | str]:
        if training_profile not in TRAINING_PROFILES:
            raise ValueError(f"unsupported training profile: {training_profile}")
        if training_profile == RAISSI_LBFGS_PROFILE:
            if adam_steps is not None:
                raise ValueError(
                    "Adam step overrides cannot be combined with raissi_lbfgs"
                )
            adam_budget = 0
            lbfgs_budget = 1000 if lbfgs_steps is None else int(lbfgs_steps)
            effective_loss_weights = tuple(1.0 for _ in self.problem.loss_weights)
        else:
            adam_budget = (
                self.config.optimizer.adam_steps
                if adam_steps is None
                else int(adam_steps)
            )
            lbfgs_budget = (
                self.config.optimizer.lbfgs_steps
                if lbfgs_steps is None
                else int(lbfgs_steps)
            )
            effective_loss_weights = tuple(self.problem.loss_weights)
        if adam_budget < 0 or lbfgs_budget < 0:
            raise ValueError("optimizer budgets must be nonnegative")

        atomic_write_json(
            self.output_dir / "config_resolved.json",
            {
                "case": asdict(self.config),
                "baseline_network": {
                    "architecture": "classical_fnn",
                    "hidden_layers": self.network.hidden_depth,
                    "width": self.network.width,
                    "activation": "tanh",
                    "initializer": "glorot_normal",
                    "input_mapping": "normalized_raw_coordinates",
                },
                "seed": self.seed,
                "training_profile": training_profile,
                "problem_metadata": self.problem.metadata,
                "training_counts": self.problem.training_counts,
                "loss_components": {
                    "names": list(self.problem.loss_names),
                    "configured_weights": list(self.problem.loss_weights),
                    "effective_weights": list(effective_loss_weights),
                },
                "run_budget": {
                    "adam_steps": adam_budget,
                    "lbfgs_steps": lbfgs_budget,
                    "lbfgs_calls": int(lbfgs_budget > 0),
                    "full_batch": True,
                },
                "execution": {
                    "hostname": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
        )
        started_at = time.time()
        if adam_budget:
            self.model.compile(
                "adam",
                lr=self.config.optimizer.learning_rate,
                loss="MSE",
                loss_weights=list(effective_loss_weights),
                verbose=0,
            )
            self.model.train(
                iterations=adam_budget,
                display_every=min(100, adam_budget),
                callbacks=[
                    _AdamLearningRate(self.config.optimizer),
                    _ProgressLogger(
                        self.output_dir / "progress.jsonl",
                        final_step=adam_budget,
                        started_at=started_at,
                    ),
                ],
                verbose=0,
            )

        if lbfgs_budget:
            dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_budget)
            self.model.compile(
                "L-BFGS",
                loss="MSE",
                loss_weights=list(effective_loss_weights),
                verbose=0,
            )
            before = int(self.model.train_state.step)
            self.model.train(verbose=0)
            append_jsonl(
                self.output_dir / "progress.jsonl",
                {
                    "phase": "lbfgs",
                    "iterations": int(self.model.train_state.step) - before,
                    "loss": float(np.sum(self.model.train_state.loss_train)),
                    "elapsed_seconds": time.time() - started_at,
                },
            )

        evaluation_points, truth = self.problem.evaluation_data()
        prediction = self._predict(evaluation_points)
        result: dict[str, int | float | str] = {
            "case": self.config.case_id,
            "seed": self.seed,
            "l2_relative_error": relative_l2_error(truth, prediction),
            "maximum_pointwise_error": maximum_pointwise_error(truth, prediction),
        }
        if not all(
            math.isfinite(float(result[key]))
            for key in ("l2_relative_error", "maximum_pointwise_error")
        ):
            raise FloatingPointError("non-finite evaluation metric")
        atomic_write_checkpoint(
            self.output_dir / "checkpoint.pt",
            {
                "case": self.config.case_id,
                "seed": self.seed,
                "model_state_dict": self.network.state_dict(),
            },
        )
        atomic_write_json(self.output_dir / "metrics.json", result)
        return result
