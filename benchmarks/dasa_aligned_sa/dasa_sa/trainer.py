"""Shared full-batch SA-PINNs trainer and two-metric evaluation."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from .artifacts import append_jsonl, atomic_write_json, atomic_write_weights
from .config import CaseConfig, OptimizerConfig
from .lbfgs import minimize_lbfgs
from .metrics import compute_metrics
from .networks import ModifiedMLP
from .problems.base import PDEProblem, TrainingData
from .sa_weights import create_point_masks, minimax_gradients


def learning_rate_for_step(config: OptimizerConfig, step: int) -> float:
    if step < 0:
        raise ValueError("step must be nonnegative")
    warmup = config.stages[0] + config.stages[1]
    decay_count = max(0, step - warmup) // config.decay_steps
    return config.learning_rate * config.decay_rate**decay_count


class Trainer:
    def __init__(
        self,
        config: CaseConfig,
        problem: PDEProblem,
        *,
        seed: int,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.problem = problem
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tf.keras.utils.set_random_seed(self.seed)

        self.model = ModifiedMLP(
            config.domain,
            config.network.output_dim,
            width=config.network.width,
            hidden_layers=config.network.hidden_layers,
            fourier=config.network.fourier,
            seed=self.seed,
        )
        self.model(tf.zeros((1, len(config.domain)), dtype=tf.float32))
        self.data = problem.sample_training(self.seed)
        mask_counts = {"residual": self.data.residual.shape[0]}
        if self.data.initial_points is not None:
            mask_counts["initial"] = self.data.initial_points.shape[0]
        self.masks = create_point_masks(mask_counts, self.seed + 1000)

        self.residual_points = tf.convert_to_tensor(self.data.residual, dtype=tf.float32)
        self.boundary_points = self._tensor(self.data.boundary_points)
        self.boundary_values = self._tensor(self.data.boundary_values)
        self.initial_points = self._tensor(self.data.initial_points)
        self.initial_values = self._tensor(self.data.initial_values)

        initial_rate = config.optimizer.learning_rate
        beta_1 = config.optimizer.adam_beta_1
        self.model_optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_rate, beta_1=beta_1
        )
        self.mask_optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_rate, beta_1=beta_1
        )

    @staticmethod
    def _tensor(value: np.ndarray | None) -> tf.Tensor | None:
        if value is None:
            return None
        return tf.convert_to_tensor(value, dtype=tf.float32)

    def loss(self) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        weights = self.config.optimizer.loss_weights
        residual = self.problem.residual(self.model, self.residual_points)
        weighted_residual = self.masks["residual"] * residual
        residual_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(weighted_residual), axis=1)
        )
        total = tf.cast(weights["residual"], residual_loss.dtype) * residual_loss
        components = {"residual_loss": residual_loss}

        if self.boundary_points is not None and self.boundary_values is not None:
            prediction = self.problem.boundary_prediction(self.model, self.boundary_points)
            boundary_loss = self.problem.boundary_loss(
                prediction, self.boundary_values
            )
            total += tf.cast(weights["boundary"], boundary_loss.dtype) * boundary_loss
            components["boundary_loss"] = boundary_loss

        if self.initial_points is not None and self.initial_values is not None:
            prediction = self.problem.initial_prediction(self.model, self.initial_points)
            weighted_error = self.masks["initial"] * (prediction - self.initial_values)
            initial_loss = tf.reduce_mean(tf.square(weighted_error))
            total += tf.cast(weights["initial"], initial_loss.dtype) * initial_loss
            components["initial_loss"] = initial_loss

        extra_loss = self.problem.extra_loss(self.model, self.data)
        total += extra_loss
        components["extra_loss"] = extra_loss
        components["total_loss"] = total
        return total, components

    @tf.function(reduce_retracing=True)
    def _adam_step(self) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        with tf.GradientTape(persistent=True) as tape:
            total, components = self.loss()
        model_gradients, mask_gradients = minimax_gradients(
            tape,
            total,
            self.model.trainable_variables,
            list(self.masks.values()),
        )
        del tape
        self.model_optimizer.apply_gradients(
            zip(model_gradients, self.model.trainable_variables)
        )
        self.mask_optimizer.apply_gradients(zip(mask_gradients, self.masks.values()))
        return total, components

    def _predict(self, points: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        predictions = []
        for start in range(0, points.shape[0], batch_size):
            batch = tf.convert_to_tensor(points[start : start + batch_size], dtype=tf.float32)
            prediction = self.model(batch, training=False).numpy()
            if self.config.case_id == "cavity":
                prediction = prediction[:, 0:1]
            predictions.append(prediction)
        return np.vstack(predictions).astype(np.float64)

    def run(
        self, *, adam_steps: int | None = None, lbfgs_steps: int | None = None
    ) -> dict[str, int | float | str]:
        adam_budget = self.config.optimizer.adam_steps if adam_steps is None else adam_steps
        lbfgs_budget = (
            self.config.optimizer.lbfgs_steps if lbfgs_steps is None else lbfgs_steps
        )
        if adam_budget < 0 or lbfgs_budget < 0:
            raise ValueError("optimizer budgets must be nonnegative")
        atomic_write_json(
            self.output_dir / "config_resolved.json",
            {
                "case": asdict(self.config),
                "seed": self.seed,
                "run_budget": {
                    "adam_steps": int(adam_budget),
                    "lbfgs_steps": int(lbfgs_budget),
                },
            },
        )
        start_time = time.time()
        for step in range(adam_budget):
            rate = learning_rate_for_step(self.config.optimizer, step)
            self.model_optimizer.learning_rate.assign(rate)
            self.mask_optimizer.learning_rate.assign(rate)
            total, components = self._adam_step()
            if not bool(tf.reduce_all(tf.math.is_finite(total))):
                raise FloatingPointError(f"non-finite loss at Adam step {step}")
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == adam_budget:
                append_jsonl(
                    self.output_dir / "progress.jsonl",
                    {
                        "phase": "adam",
                        "step": step + 1,
                        "learning_rate": rate,
                        "elapsed_seconds": time.time() - start_time,
                        **{key: float(value.numpy()) for key, value in components.items()},
                    },
                )

        if lbfgs_budget:
            result = minimize_lbfgs(
                lambda: self.loss()[0],
                self.model.trainable_variables,
                max_iterations=lbfgs_budget,
            )
            append_jsonl(
                self.output_dir / "progress.jsonl",
                {
                    "phase": "lbfgs",
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "success": bool(result.success),
                    "final_loss": float(result.fun),
                },
            )

        points, truth = self.problem.evaluation_data()
        prediction = self._predict(points)
        metric_values = compute_metrics(truth, prediction)
        result = {
            "case": self.config.case_id,
            "seed": self.seed,
            **metric_values,
        }
        if not all(
            math.isfinite(float(result[key]))
            for key in ("l2_relative_error", "maximum_pointwise_error")
        ):
            raise FloatingPointError("non-finite evaluation metric")
        atomic_write_weights(
            self.output_dir / "model_weights.npz", self.model.get_weights()
        )
        atomic_write_json(self.output_dir / "metrics.json", result)
        return result
