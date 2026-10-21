"""Adam/L-BFGS trainer for the pure, non-adaptive gPINN objective."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .evaluation import evaluate
from .plotting import plot_loss


LOG_FIELDS = (
    "phase",
    "step",
    "total",
    "pde",
    "gradient",
    "boundary",
    "initial",
    "periodic_value",
    "periodic_derivative",
    "pressure_gauge",
    "lr",
    "wall_seconds",
)


def _cpu_byte_rng_states(states) -> list[torch.Tensor]:
    """Normalize checkpointed CUDA RNG states for PyTorch's restore API.

    ``torch.load(..., map_location=cuda_device)`` also maps CUDA RNG-state
    tensors to CUDA.  PyTorch deliberately requires CPU ByteTensors in
    ``set_rng_state[_all]``, so resuming a GPU checkpoint must move these
    small tensors back to CPU explicitly.
    """

    if states is None:
        return []
    if isinstance(states, torch.Tensor):
        states = [states]
    return [
        torch.as_tensor(state).detach().to(device="cpu", dtype=torch.uint8).contiguous()
        for state in states
    ]


def _restore_cuda_rng_states(states) -> None:
    if not torch.cuda.is_available() or states is None:
        return
    normalized = _cpu_byte_rng_states(states)
    if not normalized:
        return
    visible_devices = torch.cuda.device_count()
    if len(normalized) == visible_devices:
        torch.cuda.set_rng_state_all(normalized)
    else:
        # Checkpoints can be moved between a multi-GPU process and the
        # one-visible-GPU workers used by run_server_pipeline.sh.
        torch.cuda.set_rng_state(normalized[0], device=torch.cuda.current_device())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _constraint_tensors(
    arrays: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in arrays.items():
        if name == "residual":
            continue
        requires_grad = name.endswith("points") or name in {
            "periodic_left",
            "periodic_right",
            "gauge_points",
        }
        tensors[name] = torch.as_tensor(
            value, dtype=torch.float64, device=device
        ).detach().requires_grad_(requires_grad)
    return tensors


def _weighted_total(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    missing = set(terms) - set(weights)
    if missing:
        raise KeyError(f"Loss weights are not centrally configured for {sorted(missing)}")
    return sum((weights[name] * value for name, value in terms.items()), torch.zeros((), dtype=torch.float64, device=next(iter(terms.values())).device))


class Trainer:
    def __init__(
        self,
        model,
        case,
        config: dict,
        result_dir: Path,
        reference_root: Path,
        device: torch.device,
        resume_path: Path | None = None,
    ) -> None:
        self.model = model
        self.case = case
        self.config = config
        self.result_dir = result_dir
        self.reference_root = reference_root
        self.device = device
        self.resume_path = resume_path
        self.weights = config["loss_weights"]
        self.optimizer_config = config["optimizer"]
        self.result_dir.mkdir(parents=True, exist_ok=True)
        with (self.result_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    def _payload(
        self,
        adam,
        scheduler,
        adam_step: int,
        phase: str,
        best_loss: float,
        lbfgs_closures: int = 0,
    ) -> dict:
        rng = getattr(self, "_rng", None)
        payload = {
            "model": self.model.state_dict(),
            "adam": adam.state_dict() if adam is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "adam_step": adam_step,
            "phase": phase,
            "best_loss": best_loss,
            "lbfgs_closures": lbfgs_closures,
            "config": self.config,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_generator_state": rng.bit_generator.state if rng is not None else None,
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def _write_log(self, writer, phase: str, step: int, terms: dict, total: float, lr: float, started: float) -> None:
        row = {name: 0.0 for name in LOG_FIELDS}
        row.update(
            {
                "phase": phase,
                "step": step,
                "total": total,
                "lr": lr,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        row.update({name: float(value) for name, value in terms.items()})
        writer.writerow(row)

    def train(self) -> dict:
        set_seed(int(self.config["seed"]))
        rng = np.random.default_rng(int(self.config["seed"]))
        self._rng = rng
        arrays = self.case.sample_training(rng)
        residual_array = arrays["residual"]
        adam = torch.optim.Adam(
            self.model.parameters(), lr=float(self.optimizer_config["learning_rate"])
        )
        decay_rate = float(self.optimizer_config["lr_decay_rate"])
        decay_steps = int(self.optimizer_config["lr_decay_steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            adam, lr_lambda=lambda step: decay_rate ** (step // decay_steps)
        )
        start_step, best_loss, phase = 0, float("inf"), "adam"
        resumed_lbfgs_closures = 0
        if self.resume_path is not None:
            state = load_checkpoint(self.resume_path, self.device)
            self.model.load_state_dict(state["model"])
            if state.get("adam") is not None:
                adam.load_state_dict(state["adam"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            start_step = int(state.get("adam_step", 0))
            best_loss = float(state.get("best_loss", best_loss))
            phase = str(state.get("phase", "adam"))
            resumed_lbfgs_closures = int(state.get("lbfgs_closures", 0))
            if state.get("numpy_generator_state") is not None:
                rng.bit_generator.state = state["numpy_generator_state"]
            if state.get("torch_rng_state") is not None:
                torch.set_rng_state(state["torch_rng_state"].cpu())
            _restore_cuda_rng_states(state.get("cuda_rng_state_all"))

        log_path = self.result_dir / "training_log.csv"
        append = start_step > 0 and log_path.exists()
        started = time.perf_counter()
        with log_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if not append:
                writer.writeheader()
            target_steps = int(self.optimizer_config["adam_iterations"])
            batch_size = int(self.optimizer_config["adam_batch_size"])
            if phase == "adam":
                for step in range(start_step + 1, target_steps + 1):
                    if batch_size < len(residual_array):
                        indices = rng.choice(len(residual_array), batch_size, replace=False)
                        selected = residual_array[indices]
                    else:
                        selected = residual_array
                    residual = torch.as_tensor(
                        selected, dtype=torch.float64, device=self.device
                    ).detach().requires_grad_(True)
                    constraints = _constraint_tensors(arrays, self.device)
                    adam.zero_grad(set_to_none=True)
                    terms = self.case.raw_loss_terms(self.model, residual, constraints)
                    total = _weighted_total(terms, self.weights)
                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            f"Non-finite loss at Adam iteration {step}: "
                            + ", ".join(f"{k}={float(v.detach())}" for k, v in terms.items())
                        )
                    total.backward()
                    adam.step()
                    scheduler.step()
                    total_value = float(total.detach())
                    term_values = {name: float(value.detach()) for name, value in terms.items()}
                    if total_value < best_loss:
                        best_loss = total_value
                        save_checkpoint(
                            self.result_dir / "best.pt",
                            self._payload(adam, scheduler, step, "adam", best_loss),
                        )
                    if step == 1 or step % int(self.optimizer_config["log_every"]) == 0 or step == target_steps:
                        self._write_log(
                            writer,
                            "adam",
                            step,
                            term_values,
                            total_value,
                            adam.param_groups[0]["lr"],
                            started,
                        )
                        handle.flush()
                    if step % int(self.optimizer_config["checkpoint_every"]) == 0 or step == target_steps:
                        save_checkpoint(
                            self.result_dir / "last.pt",
                            self._payload(adam, scheduler, step, "lbfgs", best_loss),
                        )
                start_step = target_steps
                phase = "lbfgs"

            lbfgs_iterations = int(self.optimizer_config["lbfgs_iterations"])
            lbfgs_closures = resumed_lbfgs_closures
            if phase == "lbfgs" and lbfgs_iterations > 0:
                lbfgs = torch.optim.LBFGS(
                    self.model.parameters(),
                    max_iter=lbfgs_iterations,
                    history_size=int(self.optimizer_config["lbfgs_history_size"]),
                    tolerance_grad=float(self.optimizer_config["lbfgs_tolerance_grad"]),
                    tolerance_change=float(self.optimizer_config["lbfgs_tolerance_change"]),
                    line_search_fn="strong_wolfe",
                )
                chunk_size = int(self.optimizer_config["adam_batch_size"])
                last_terms: dict[str, float] = {}
                last_total = float("nan")

                def closure():
                    nonlocal lbfgs_closures, last_terms, last_total
                    lbfgs.zero_grad(set_to_none=True)
                    aggregate = {"pde": 0.0, "gradient": 0.0}
                    objective_value = 0.0
                    for start in range(0, len(residual_array), chunk_size):
                        selected = residual_array[start : start + chunk_size]
                        residual = torch.as_tensor(
                            selected, dtype=torch.float64, device=self.device
                        ).detach().requires_grad_(True)
                        physics = self.case.physics_loss_terms(self.model, residual)
                        fraction = len(selected) / len(residual_array)
                        piece = _weighted_total(physics, self.weights) * fraction
                        piece.backward()
                        objective_value += float(piece.detach())
                        for name, value in physics.items():
                            aggregate[name] += fraction * float(value.detach())
                    constraints = self.case.constraint_terms(
                        self.model, _constraint_tensors(arrays, self.device)
                    )
                    if constraints:
                        constraint_loss = _weighted_total(constraints, self.weights)
                        constraint_loss.backward()
                        objective_value += float(constraint_loss.detach())
                        aggregate.update(
                            {name: float(value.detach()) for name, value in constraints.items()}
                        )
                    if not np.isfinite(objective_value):
                        raise FloatingPointError("Non-finite L-BFGS objective")
                    lbfgs_closures += 1
                    last_terms = aggregate
                    last_total = objective_value
                    return torch.as_tensor(objective_value, dtype=torch.float64, device=self.device)

                lbfgs.step(closure)
                self._write_log(
                    writer,
                    "lbfgs",
                    start_step + lbfgs_closures,
                    last_terms,
                    last_total,
                    0.0,
                    started,
                )
                handle.flush()
                phase = "complete"
            elif phase == "lbfgs":
                phase = "complete"
            save_checkpoint(
                self.result_dir / "last.pt",
                self._payload(
                    adam, scheduler, start_step, phase, best_loss, lbfgs_closures
                ),
            )

        metrics = evaluate(
            self.model,
            self.case,
            self.config,
            self.result_dir,
            self.reference_root,
            self.device,
        )
        metrics.update(
            {
                "adam_iterations_completed": start_step,
                "adam_iterations_configured": int(self.optimizer_config["adam_iterations"]),
                "lbfgs_iterations_configured": int(self.optimizer_config["lbfgs_iterations"]),
                "lbfgs_closure_evaluations": lbfgs_closures,
                "full_training_completed": bool(
                    self.config["mode"] == "full" and phase == "complete"
                ),
                "device": str(self.device),
            }
        )
        with (self.result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        plot_loss(log_path, self.result_dir / "loss_curve.png")
        return metrics


def evaluate_checkpoint(
    model,
    case,
    config: dict,
    result_dir: Path,
    reference_root: Path,
    device: torch.device,
    checkpoint_path: Path,
) -> dict:
    state = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state["model"])
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return evaluate(model, case, config, result_dir, reference_root, device)
