#!/usr/bin/env python3
"""Controlled PINN/DASA experiments for the one-dimensional wave equation."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SENSITIVITY_VARIANTS = (
    "baseline",
    "arch_3x5",
    "arch_7x20",
    "lr_both_5e-4",
    "lr_subnet_5e-4",
    "lr_subnet_2e-3",
    "lr_both_2e-3",
    "ratio_1to5",
    "ratio_5to1",
)

STAGE_VARIANTS = (
    "baseline",
    "stage1_2500",
    "stage1_10000",
    "stage2_1000",
    "stage2_4000",
    "stage3_25000",
    "stage3_100000",
    "stage4_25000",
    "stage4_100000",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_seed(base_seed: int, stream: str, block: int) -> int:
    payload = f"{base_seed}:{stream}:{block}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def latin_hypercube(n: int, dim: int, rng: np.random.RandomState) -> np.ndarray:
    if n <= 0 or dim <= 0:
        raise ValueError("Latin-hypercube dimensions must be positive")
    result = np.empty((n, dim), dtype=np.float64)
    for axis in range(dim):
        result[:, axis] = (rng.permutation(n) + rng.random_sample(n)) / float(n)
    return result


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(module: Optional[nn.Module], trainable_only: bool = True) -> int:
    if module is None:
        return 0
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad or not trainable_only
        )
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return torch.device(requested)


def base_config() -> Dict[str, Any]:
    return {
        "equation": "u_tt - 4*u_xx = 0",
        "domain": {"t": [0.0, 1.0], "x": [0.0, 1.0]},
        "method": "dasa",
        "backbone": {"type": "st_mff", "depth": 6, "width": 128},
        "subnet": {"depth": 5, "width": 10},
        "data": {
            "n_interior": 256,
            "n_initial": 128,
            "n_boundary_per_side": 64,
            "resample_every": 100,
            "n_interior_lbfgs": 25600,
            "n_initial_lbfgs": 512,
            "n_boundary_per_side_lbfgs": 256,
            "test_nt": 501,
            "test_nx": 501,
        },
        "loss": {"initial_velocity_weight": 10000.0, "condition_weight": 10000.0},
        "optimization": {
            "lr_backbone": 1.0e-3,
            "lr_subnet": 1.0e-3,
            "warmup_steps": 5000,
            "subnet_pretrain_steps": 2000,
            "main_steps": 50000,
            "finetune_steps": 50000,
            "scheduler_step": 1000,
            "scheduler_gamma": 0.9,
            "subnet_update_every": 1,
            "subnet_updates_per_backbone": 1,
            "controlled_finetune_start_lr": None,
            "lbfgs_outer_steps": 1000,
            "lbfgs_lr": 0.8,
            "lbfgs_max_iter": 100,
            "lbfgs_tolerance_grad": 1.0e-5,
            "lbfgs_tolerance_change": 1.0e-9,
            "lbfgs_history_size": 100,
        },
        "logging": {"log_every": 1000, "lbfgs_log_every": 10},
    }


def make_config(group: str, variant: str) -> Dict[str, Any]:
    config = base_config()
    config.update({"category": group, "variant": variant, "experiment_name": f"{group}_{variant}"})
    opt = config["optimization"]

    if group == "cost":
        if variant not in {"pinn", "dasa"}:
            raise ValueError(f"Unknown cost variant: {variant}")
        config["method"] = variant
        if variant == "pinn":
            config["subnet"] = None
            opt["subnet_pretrain_steps"] = 0
        return config

    if group == "sensitivity":
        if variant not in SENSITIVITY_VARIANTS:
            raise ValueError(f"Unknown sensitivity variant: {variant}")
        opt["finetune_steps"] = 0
        opt["lbfgs_outer_steps"] = 0
        if variant == "arch_3x5":
            config["subnet"] = {"depth": 3, "width": 5}
        elif variant == "arch_7x20":
            config["subnet"] = {"depth": 7, "width": 20}
        elif variant == "lr_both_5e-4":
            opt["lr_backbone"] = 5.0e-4
            opt["lr_subnet"] = 5.0e-4
        elif variant == "lr_subnet_5e-4":
            opt["lr_subnet"] = 5.0e-4
        elif variant == "lr_subnet_2e-3":
            opt["lr_subnet"] = 2.0e-3
        elif variant == "lr_both_2e-3":
            opt["lr_backbone"] = 2.0e-3
            opt["lr_subnet"] = 2.0e-3
        elif variant == "ratio_1to5":
            opt["subnet_update_every"] = 5
        elif variant == "ratio_5to1":
            opt["subnet_updates_per_backbone"] = 5
        return config

    if group == "stage_length":
        if variant not in STAGE_VARIANTS:
            raise ValueError(f"Unknown stage-length variant: {variant}")
        changes = {
            "stage1_2500": ("warmup_steps", 2500),
            "stage1_10000": ("warmup_steps", 10000),
            "stage2_1000": ("subnet_pretrain_steps", 1000),
            "stage2_4000": ("subnet_pretrain_steps", 4000),
            "stage3_25000": ("main_steps", 25000),
            "stage3_100000": ("main_steps", 100000),
            "stage4_25000": ("finetune_steps", 25000),
            "stage4_100000": ("finetune_steps", 100000),
        }
        if variant in changes:
            key, value = changes[variant]
            opt[key] = value
        # The Stage-4 start rate is fixed to the rate reached by the default
        # 50,000-step Stage 3, so varying Stage 3 does not alter Stage 4's LR budget.
        default_decays = 50000 // int(opt["scheduler_step"])
        opt["controlled_finetune_start_lr"] = float(
            opt["lr_backbone"] * opt["scheduler_gamma"] ** default_decays
        )
        return config

    raise ValueError(f"Unknown experiment group: {group}")


def apply_smoke_overrides(config: Dict[str, Any]) -> None:
    data = config["data"]
    opt = config["optimization"]
    data.update(
        {
            "n_interior": 8,
            "n_initial": 8,
            "n_boundary_per_side": 4,
            "resample_every": 1,
            "n_interior_lbfgs": 8,
            "n_initial_lbfgs": 8,
            "n_boundary_per_side_lbfgs": 4,
            "test_nt": 11,
            "test_nx": 11,
        }
    )
    opt.update(
        {
            "warmup_steps": 1,
            "subnet_pretrain_steps": 1 if config["method"] == "dasa" else 0,
            "main_steps": 1,
            "finetune_steps": 1,
            "scheduler_step": 1,
            "controlled_finetune_start_lr": None,
            "lbfgs_outer_steps": 0,
        }
    )
    config["logging"]["log_every"] = 1


class STMFF(nn.Module):
    """Spatio-temporal multi-scale Fourier-feature network from the wave source."""

    def __init__(self, width: int = 128, depth: int = 6) -> None:
        super().__init__()
        if width % 2 != 0:
            raise ValueError("ST-MFF width must be even")
        layers = []
        for _ in range(depth):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        self.shared_mlp = nn.Sequential(*layers)
        self.output = nn.Linear(2 * width, 1)
        self.register_buffer("w_t_low", torch.randn(1, width // 2) * 1.0)
        self.register_buffer("w_t_high", torch.randn(1, width // 2) * 10.0)
        self.register_buffer("w_x", torch.randn(1, width // 2) * 1.0)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)

    @staticmethod
    def fourier(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        projection = values @ frequencies
        return torch.cat((torch.sin(projection), torch.cos(projection)), dim=1)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        low_t = self.shared_mlp(self.fourier(t, self.w_t_low))
        high_t = self.shared_mlp(self.fourier(t, self.w_t_high))
        space = self.shared_mlp(self.fourier(x, self.w_x))
        return self.output(torch.cat((low_t * space, high_t * space), dim=1))


class WeightSubnet(nn.Module):
    def __init__(self, width: int = 10, depth: int = 5) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Subnet depth must be positive")
        layers = []
        in_features = 2
        for _ in range(depth):
            layers.extend((nn.Linear(in_features, width), nn.Sigmoid()))
            in_features = width
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        self.output.bias.requires_grad_(False)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.features(tx))
        low = raw.amin(dim=0, keepdim=True)
        high = raw.amax(dim=0, keepdim=True)
        scaled = (raw - low) / (high - low).clamp_min(1.0e-12)
        return scaled / scaled.sum(dim=0, keepdim=True).clamp_min(1.0e-12) * raw.shape[0]


@dataclass
class WaveBatch:
    interior: torch.Tensor
    initial: torch.Tensor
    initial_u: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor


def make_batch(
    *,
    seed: int,
    stream: str,
    block: int,
    n_interior: int,
    n_initial: int,
    n_boundary_per_side: int,
    device: torch.device,
) -> WaveBatch:
    rng = np.random.RandomState(stable_seed(seed, stream, block))
    interior = latin_hypercube(n_interior, 2, rng)

    initial_x = latin_hypercube(n_initial, 1, rng)
    initial = np.column_stack((np.zeros(n_initial), initial_x[:, 0]))
    initial_u = np.sin(np.pi * initial_x) + 0.5 * np.sin(4.0 * np.pi * initial_x)

    boundary_t = latin_hypercube(n_boundary_per_side, 1, rng)
    left = np.column_stack((boundary_t[:, 0], np.zeros(n_boundary_per_side)))
    right = np.column_stack((boundary_t[:, 0], np.ones(n_boundary_per_side)))

    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    return WaveBatch(
        interior=tensor(interior),
        initial=tensor(initial),
        initial_u=tensor(initial_u),
        left=tensor(left),
        right=tensor(right),
    )


def wave_residual(model: nn.Module, tx: torch.Tensor) -> torch.Tensor:
    points = tx.detach().requires_grad_(True)
    u = model(points)
    grad_u = torch.autograd.grad(
        u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]
    u_tt = torch.autograd.grad(
        u_t, points, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 1:2]
    return u_tt - 4.0 * u_xx


def compute_loss(
    model: nn.Module,
    batch: WaveBatch,
    velocity_weight: float,
    condition_weight: float,
    subnet: Optional[nn.Module] = None,
    detach_residual: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    residual = wave_residual(model, batch.interior)
    if detach_residual:
        residual = residual.detach()
    weighted_residual = residual if subnet is None else subnet(batch.interior) * residual
    loss_r = weighted_residual.square().mean()

    initial = batch.initial.detach().requires_grad_(True)
    initial_pred = model(initial)
    initial_grad = torch.autograd.grad(
        initial_pred,
        initial,
        grad_outputs=torch.ones_like(initial_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    loss_ut = initial_grad[:, 0:1].square().mean()
    loss_conditions = (
        model(batch.left).square().mean()
        + model(batch.right).square().mean()
        + (initial_pred - batch.initial_u).square().mean()
    )
    total = loss_r + velocity_weight * loss_ut + condition_weight * loss_conditions
    return total, {"loss": total, "loss_r": loss_r, "loss_ut": loss_ut, "loss_conditions": loss_conditions}


@contextlib.contextmanager
def temporarily_freeze(module: nn.Module) -> Iterator[None]:
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class Experiment:
    def __init__(self, config: Dict[str, Any], seed: int, device: torch.device) -> None:
        self.config = config
        self.seed = seed
        self.device = device
        self.method = str(config["method"])
        backbone = config["backbone"]
        self.model = STMFF(width=int(backbone["width"]), depth=int(backbone["depth"])).to(device)
        subnet = config.get("subnet")
        self.subnet = (
            WeightSubnet(width=int(subnet["width"]), depth=int(subnet["depth"])).to(device)
            if subnet is not None
            else None
        )

        opt = config["optimization"]
        self.backbone_optimizer = optim.Adam(
            self.model.parameters(), lr=float(opt["lr_backbone"]), betas=(0.9, 0.999)
        )
        self.subnet_optimizer = (
            optim.Adam(
                (parameter for parameter in self.subnet.parameters() if parameter.requires_grad),
                lr=float(opt["lr_subnet"]),
                betas=(0.9, 0.999),
            )
            if self.subnet is not None
            else None
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.backbone_optimizer, gamma=float(opt["scheduler_gamma"])
        )
        self.stage_timings: Dict[str, Dict[str, Any]] = {}
        self.evaluations: Dict[str, Dict[str, float]] = {}

    @property
    def data(self) -> Dict[str, Any]:
        return self.config["data"]

    @property
    def opt(self) -> Dict[str, Any]:
        return self.config["optimization"]

    def batch_for(self, stream: str, step: int, full: bool = False) -> WaveBatch:
        interval = int(self.data["resample_every"])
        block = 0 if full or interval <= 0 else step // interval
        suffix = "_lbfgs" if full else ""
        return make_batch(
            seed=self.seed,
            stream=stream,
            block=block,
            n_interior=int(self.data[f"n_interior{suffix}"]),
            n_initial=int(self.data[f"n_initial{suffix}"]),
            n_boundary_per_side=int(self.data[f"n_boundary_per_side{suffix}"]),
            device=self.device,
        )

    def loss(self, batch: WaveBatch, subnet: Optional[nn.Module], detach: bool = False):
        loss_cfg = self.config["loss"]
        return compute_loss(
            self.model,
            batch,
            float(loss_cfg["initial_velocity_weight"]),
            float(loss_cfg["condition_weight"]),
            subnet=subnet,
            detach_residual=detach,
        )

    @staticmethod
    def detached_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in stats.items()}

    def backbone_step(self, batch: WaveBatch) -> Dict[str, float]:
        self.backbone_optimizer.zero_grad(set_to_none=True)
        loss, stats = self.loss(batch, None)
        loss.backward()
        self.backbone_optimizer.step()
        return self.detached_stats(stats)

    def subnet_step(self, batch: WaveBatch) -> Dict[str, float]:
        if self.subnet is None or self.subnet_optimizer is None:
            raise RuntimeError("Subnet step requested for PINN")
        self.subnet_optimizer.zero_grad(set_to_none=True)
        with temporarily_freeze(self.model):
            loss, stats = self.loss(batch, self.subnet, detach=True)
            (-loss).backward()
        self.subnet_optimizer.step()
        return self.detached_stats(stats)

    def adversarial_step(self, batch: WaveBatch, update_subnet: bool) -> Tuple[Dict[str, float], int]:
        if self.subnet is None or self.subnet_optimizer is None:
            raise RuntimeError("Adversarial step requested for PINN")
        self.backbone_optimizer.zero_grad(set_to_none=True)
        self.subnet_optimizer.zero_grad(set_to_none=True)
        loss, stats = self.loss(batch, self.subnet)
        loss.backward()
        self.backbone_optimizer.step()
        if not update_subnet:
            return self.detached_stats(stats), 0
        for parameter in self.subnet.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.neg_()
        self.subnet_optimizer.step()
        return self.detached_stats(stats), 1

    def run_stage(self, name: str, steps: int, stream: str, mode: str, scheduler: bool = False) -> None:
        if steps <= 0:
            self.stage_timings[name] = {
                "elapsed_s": 0.0,
                "steps": 0,
                "backbone_updates": 0,
                "subnet_updates": 0,
                "seconds_per_step": None,
            }
            return

        interval = int(self.data["resample_every"])
        schedule_every = int(self.opt["scheduler_step"])
        subnet_every = int(self.opt["subnet_update_every"])
        subnet_per_backbone = int(self.opt["subnet_updates_per_backbone"])
        log_every = int(self.config["logging"]["log_every"])
        if subnet_every < 1 or subnet_per_backbone < 1:
            raise ValueError("Update ratios must be positive")

        batch = None
        backbone_updates = 0
        subnet_updates = 0
        latest: Dict[str, float] = {}
        sync(self.device)
        started = time.perf_counter()
        for step in range(steps):
            if batch is None or (interval > 0 and step % interval == 0):
                batch = self.batch_for(stream, step)
            if mode == "backbone":
                latest = self.backbone_step(batch)
                backbone_updates += 1
            elif mode == "subnet":
                latest = self.subnet_step(batch)
                subnet_updates += 1
            elif mode == "adversarial":
                due = (step + 1) % subnet_every == 0
                latest, updates = self.adversarial_step(batch, due)
                backbone_updates += 1
                subnet_updates += updates
                if due:
                    for _ in range(subnet_per_backbone - 1):
                        latest = self.subnet_step(batch)
                        subnet_updates += 1
            else:
                raise ValueError(f"Unknown stage mode: {mode}")

            if scheduler and schedule_every > 0 and (step + 1) % schedule_every == 0:
                self.scheduler.step()
            if not math.isfinite(latest.get("loss", math.nan)):
                raise FloatingPointError(f"Non-finite loss in {name} at step {step + 1}")
            if log_every > 0 and (step == 0 or (step + 1) % log_every == 0 or step + 1 == steps):
                print(
                    f"[{name}] {step + 1}/{steps} loss={latest['loss']:.6e} "
                    f"lr_b={self.backbone_optimizer.param_groups[0]['lr']:.6e}",
                    flush=True,
                )
        sync(self.device)
        elapsed = time.perf_counter() - started
        self.stage_timings[name] = {
            "elapsed_s": elapsed,
            "steps": steps,
            "backbone_updates": backbone_updates,
            "subnet_updates": subnet_updates,
            "seconds_per_step": elapsed / steps,
            "last_loss": latest,
            "backbone_lr_end": float(self.backbone_optimizer.param_groups[0]["lr"]),
        }

    def run_lbfgs(self) -> None:
        steps = int(self.opt["lbfgs_outer_steps"])
        if steps <= 0:
            self.stage_timings["lbfgs"] = {
                "elapsed_s": 0.0,
                "steps": 0,
                "backbone_updates": 0,
                "subnet_updates": 0,
                "seconds_per_step": None,
            }
            return

        batch = self.batch_for("lbfgs", 0, full=True)
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=float(self.opt["lbfgs_lr"]),
            max_iter=int(self.opt["lbfgs_max_iter"]),
            tolerance_grad=float(self.opt["lbfgs_tolerance_grad"]),
            tolerance_change=float(self.opt["lbfgs_tolerance_change"]),
            history_size=int(self.opt["lbfgs_history_size"]),
        )
        log_every = int(self.config["logging"]["lbfgs_log_every"])
        last_loss = math.nan
        sync(self.device)
        started = time.perf_counter()
        for step in range(steps):
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.loss(batch, None)
                loss.backward()
                return loss

            result = optimizer.step(closure)
            last_loss = float(result.detach().cpu()) if torch.is_tensor(result) else float(result)
            if not math.isfinite(last_loss):
                raise FloatingPointError(f"Non-finite L-BFGS loss at outer step {step + 1}")
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
                print(f"[lbfgs] {step + 1}/{steps} returned_loss={last_loss:.6e}", flush=True)
        sync(self.device)
        elapsed = time.perf_counter() - started
        self.stage_timings["lbfgs"] = {
            "elapsed_s": elapsed,
            "steps": steps,
            "backbone_updates": steps,
            "subnet_updates": 0,
            "seconds_per_step": elapsed / steps,
            "last_loss": last_loss,
            "max_iter_per_outer_step": int(self.opt["lbfgs_max_iter"]),
        }

    def evaluate(self, label: str) -> None:
        nt = int(self.data["test_nt"])
        nx = int(self.data["test_nx"])
        t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
        x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
        tt, xx = np.meshgrid(t, x, indexing="xy")
        points = np.column_stack((tt.reshape(-1), xx.reshape(-1)))
        truth = (
            np.sin(np.pi * points[:, 1]) * np.cos(2.0 * np.pi * points[:, 0])
            + 0.5 * np.sin(4.0 * np.pi * points[:, 1]) * np.cos(8.0 * np.pi * points[:, 0])
        ).astype(np.float64)

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for begin in range(0, points.shape[0], 65536):
                inputs = torch.as_tensor(points[begin : begin + 65536], device=self.device)
                predictions.append(self.model(inputs).detach().cpu().numpy().reshape(-1))
        self.model.train()
        prediction = np.concatenate(predictions).astype(np.float64)
        error = truth - prediction
        self.evaluations[label] = {
            "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
            "max_pointwise_error": float(np.max(np.abs(error))),
            "n_test": int(truth.size),
        }
        print(
            f"[eval:{label}] relative_l2={self.evaluations[label]['relative_l2']:.6e} "
            f"max_pointwise_error={self.evaluations[label]['max_pointwise_error']:.6e}",
            flush=True,
        )

    def run(self) -> None:
        self.run_stage("stage1", int(self.opt["warmup_steps"]), "stage1", "backbone")
        if self.method == "dasa":
            self.run_stage(
                "stage2", int(self.opt["subnet_pretrain_steps"]), "stage2", "subnet"
            )
            main_mode = "adversarial"
        else:
            self.run_stage("stage2", 0, "stage2", "subnet")
            main_mode = "backbone"
        self.run_stage(
            "stage3", int(self.opt["main_steps"]), "stage3", main_mode, scheduler=True
        )
        self.evaluate("after_stage3")

        controlled_lr = self.opt.get("controlled_finetune_start_lr")
        if controlled_lr is not None:
            for group in self.backbone_optimizer.param_groups:
                group["lr"] = float(controlled_lr)
        self.run_stage("stage4", int(self.opt["finetune_steps"]), "stage4", "backbone")
        self.run_lbfgs()
        self.evaluate("final")

    def metrics(self) -> Dict[str, Any]:
        adam_time = sum(
            self.stage_timings[name]["elapsed_s"]
            for name in ("stage1", "stage2", "stage3", "stage4")
        )
        total_time = adam_time + self.stage_timings["lbfgs"]["elapsed_s"]
        backbone_trainable = count_parameters(self.model)
        subnet_trainable = count_parameters(self.subnet)
        return {
            "schema_version": 1,
            "equation": "wave",
            "experiment_name": self.config["experiment_name"],
            "category": self.config["category"],
            "variant": self.config["variant"],
            "method": self.method,
            "seed": self.seed,
            "config": self.config,
            "parameter_counts": {
                "backbone_trainable": backbone_trainable,
                "subnet_trainable": subnet_trainable,
                "total_trainable": backbone_trainable + subnet_trainable,
                "trainable_overhead_vs_backbone_pct": 100.0 * subnet_trainable / backbone_trainable,
            },
            "stage_timings": self.stage_timings,
            "timing_summary": {
                "adam_total_s": adam_time,
                "training_total_s": total_time,
            },
            "evaluations": self.evaluations,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(self.device),
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("cost", "sensitivity", "stage_length"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = make_config(args.group, args.variant)
    if args.smoke:
        apply_smoke_overrides(config)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = args.output_root / args.group / config["experiment_name"] / f"seed_{args.seed}"
    metrics_path = output / "metrics.json"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)

    experiment = Experiment(copy.deepcopy(config), args.seed, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    experiment.run()
    payload = experiment.metrics()
    if device.type == "cuda":
        payload["timing_summary"]["peak_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
    save_json(metrics_path, payload)
    temporary_model = output / "model.pt.tmp"
    torch.save(experiment.model.state_dict(), temporary_model)
    temporary_model.replace(output / "model.pt")
    print(f"[complete] {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
