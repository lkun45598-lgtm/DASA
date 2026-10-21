#!/usr/bin/env python3
"""Controlled L-shaped Poisson experiments for reviewer responses.

The PDE, L-shaped domain, Modified MLP backbone, DASA weighting subnet,
collocation size, boundary penalty, and four-stage Adam schedule follow the
provided notebook. The runner adds deterministic sampling, explicit timing,
parameter counting, structured outputs, and configurable DASA update ratios.
"""
from __future__ import annotations

import argparse
import contextlib
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
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------- utilities ---------------------------------

def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_seed(base_seed: int, stream: str, block: int) -> int:
    payload = f"{base_seed}:{stream}:{block}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2**32 - 1)


def latin_hypercube(n: int, dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Random Latin hypercube in [0, 1]^dim using an explicit RNG."""
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive")
    out = np.empty((n, dim), dtype=np.float64)
    for j in range(dim):
        out[:, j] = (rng.permutation(n) + rng.random_sample(n)) / float(n)
    return out


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return int(
        sum(
            p.numel()
            for p in module.parameters()
            if (p.requires_grad or not trainable_only)
        )
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# --------------------------------- models -----------------------------------

def activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU(inplace=False)
    if key in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(inplace=False)
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unknown activation: {name}")


class ModifiedMLP(nn.Module):
    """Gated Modified MLP shared by the aligned PINN and DASA runners.

    The initial hidden state is computed directly from the input features.
    Each subsequent hidden layer predicts a gate and blends the two input
    encoders, matching ``common_modified_mlp_corrected.ModifiedMLP`` used by
    the standard PINN baselines.
    """

    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        width: int = 128,
        depth: int = 6,
        act_name: str = "tanh",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Backbone depth must be at least 1")
        self.depth = depth
        self.width = width
        self.activation = activation(act_name)
        self.encoder_u = nn.Sequential(nn.Linear(dim_in, width), activation(act_name))
        self.encoder_v = nn.Sequential(nn.Linear(dim_in, width), activation(act_name))
        self.input_layer = nn.Linear(dim_in, width)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth - 1)
        )
        self.output = nn.Linear(width, dim_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.encoder_u[0],
            self.encoder_v[0],
            self.input_layer,
            *self.hidden_layers,
            self.output,
        ):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.encoder_u(x)
        v = self.encoder_v(x)
        h = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            z = self.activation(layer(h))
            h = (1.0 - z) * u + z * v
        return self.output(h)


class OriginalModifiedMLP(nn.Module):
    """Modified MLP implementation used by the supplied DASA notebook."""

    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        width: int = 128,
        depth: int = 6,
        act_name: str = "tanh",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Backbone depth must be at least 1")
        self.encoder_u = nn.Sequential(nn.Linear(dim_in, width), activation(act_name))
        self.encoder_v = nn.Sequential(nn.Linear(dim_in, width), activation(act_name))
        layers = [nn.Linear(dim_in, width)]
        layers.extend(nn.Linear(width, width) for _ in range(depth - 1))
        self.hidden_layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activation(act_name) for _ in range(depth))
        self.output = nn.Linear(width, dim_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.encoder_u(x)
        v = self.encoder_v(x)
        h = x
        for layer, act in zip(self.hidden_layers, self.activations):
            h = act(layer(h))
            h = (1.0 - h) * u + h * v
        return self.output(h)


def build_backbone(config: Dict[str, Any]) -> nn.Module:
    architecture = str(config.get("type", "modified_mlp")).lower()
    common = {
        "width": int(config["width"]),
        "depth": int(config["depth"]),
    }
    if architecture == "modified_mlp":
        return ModifiedMLP(**common)
    if architecture == "original_modified_mlp":
        return OriginalModifiedMLP(**common)
    raise ValueError(f"Unsupported backbone type: {architecture}")


class WeightSubnet(nn.Module):
    """DASA subnetwork predicting positive pointwise residual weights."""

    def __init__(self, width: int = 10, depth: int = 5) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Subnet depth must be at least 1")
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Sigmoid())
            in_dim = width
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        # The supplied notebook freezes the output bias (fc5.bias for depth=5).
        self.output.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.features(x))
        low = raw.amin(dim=0, keepdim=True)
        high = raw.amax(dim=0, keepdim=True)
        normalized = (raw - low) / (high - low).clamp_min(1.0e-12)
        normalized_sum = normalized.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        return normalized / normalized_sum * float(raw.shape[0])


# ---------------------------------- data ------------------------------------
@dataclass
class CollocationBatch:
    interior: torch.Tensor
    boundary: torch.Tensor
    candidate_count: int
    interior_count: int
    boundary_count: int


def make_collocation_batch(
    *,
    base_seed: int,
    stream: str,
    block: int,
    n_interior_candidates: int,
    n_boundary_per_segment: int,
    device: torch.device,
) -> CollocationBatch:
    rng = np.random.RandomState(stable_seed(base_seed, stream, block))

    xy = 2.0 * latin_hypercube(n_interior_candidates, 2, rng) - 1.0
    keep = ~((xy[:, 0] >= 0.0) & (xy[:, 1] >= 0.0))
    xy = xy[keep]

    neg = latin_hypercube(n_boundary_per_segment, 1, rng) - 1.0
    pos = latin_hypercube(n_boundary_per_segment, 1, rng)
    zeros_neg = np.zeros_like(neg)
    zeros_pos = np.zeros_like(pos)

    boundary = np.vstack(
        [
            np.hstack((neg, -np.ones_like(neg))),
            np.hstack((pos, -np.ones_like(pos))),
            np.hstack((-np.ones_like(neg), neg)),
            np.hstack((-np.ones_like(pos), pos)),
            np.hstack((neg, np.ones_like(neg))),
            np.hstack((pos, zeros_pos)),
            np.hstack((np.ones_like(neg), neg)),
            np.hstack((zeros_pos, pos)),
        ]
    )

    interior_t = torch.as_tensor(xy, dtype=torch.float32, device=device)
    boundary_t = torch.as_tensor(boundary, dtype=torch.float32, device=device)
    return CollocationBatch(
        interior=interior_t,
        boundary=boundary_t,
        candidate_count=n_interior_candidates,
        interior_count=int(interior_t.shape[0]),
        boundary_count=int(boundary_t.shape[0]),
    )


# ------------------------------- PDE / losses -------------------------------
def poisson_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy_req = xy.detach().requires_grad_(True)
    u = model(xy_req)
    grad_u = torch.autograd.grad(
        u,
        xy_req,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x,
        xy_req,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y,
        xy_req,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    return -(u_xx + u_yy) - 1.0


def compute_loss(
    model: nn.Module,
    batch: CollocationBatch,
    boundary_weight: float,
    subnet: Optional[nn.Module] = None,
    detach_residual_for_subnet_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = poisson_residual(model, batch.interior)
    if subnet is None:
        residual_term = residual
    else:
        weights = subnet(batch.interior)
        if detach_residual_for_subnet_only:
            residual = residual.detach()
        residual_term = weights * residual
    loss_r = torch.mean(residual_term.square())
    boundary_pred = model(batch.boundary)
    loss_b = torch.mean(boundary_pred.square())
    loss = loss_r + boundary_weight * loss_b
    return loss, loss_r, loss_b


@contextlib.contextmanager
def temporarily_freeze(module: nn.Module) -> Iterator[None]:
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


# -------------------------------- training ----------------------------------
class Experiment:
    def __init__(self, cfg: Dict[str, Any], seed: int, device: torch.device) -> None:
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.method = str(cfg["method"]).lower()
        if self.method not in {"pinn", "dasa"}:
            raise ValueError("method must be 'pinn' or 'dasa'")

        backbone_cfg = cfg["backbone"]
        self.model = build_backbone(backbone_cfg).to(device)

        self.subnet: Optional[WeightSubnet]
        if self.method == "dasa":
            subnet_cfg = cfg["subnet"]
            self.subnet = WeightSubnet(
                width=int(subnet_cfg["width"]),
                depth=int(subnet_cfg["depth"]),
            ).to(device)
        else:
            self.subnet = None

        opt_cfg = cfg["optimization"]
        self.backbone_optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(opt_cfg["lr_backbone"]),
            betas=(0.9, 0.999),
        )
        self.subnet_optimizer: Optional[optim.Optimizer]
        if self.subnet is not None:
            self.subnet_optimizer = optim.Adam(
                (p for p in self.subnet.parameters() if p.requires_grad),
                lr=float(opt_cfg["lr_subnet"]),
                betas=(0.9, 0.999),
            )
        else:
            self.subnet_optimizer = None

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.backbone_optimizer,
            gamma=float(opt_cfg["scheduler_gamma"]),
        )
        self.stage_records: Dict[str, Dict[str, Any]] = {}
        self.evaluations: Dict[str, Dict[str, float]] = {}
        self.first_batch_counts: Optional[Dict[str, int]] = None

    @property
    def data_cfg(self) -> Dict[str, Any]:
        return self.cfg["data"]

    @property
    def opt_cfg(self) -> Dict[str, Any]:
        return self.cfg["optimization"]

    @property
    def boundary_weight(self) -> float:
        return float(self.cfg["loss"]["boundary_weight"])

    def batch_for(self, stream: str, step: int) -> CollocationBatch:
        interval = int(self.data_cfg["resample_every"])
        block = 0 if interval <= 0 else step // interval
        batch = make_collocation_batch(
            base_seed=self.seed,
            stream=stream,
            block=block,
            n_interior_candidates=int(self.data_cfg["n_interior_candidates"]),
            n_boundary_per_segment=int(self.data_cfg["n_boundary_per_segment"]),
            device=self.device,
        )
        if self.first_batch_counts is None:
            self.first_batch_counts = {
                "interior_candidates": batch.candidate_count,
                "retained_interior": batch.interior_count,
                "boundary": batch.boundary_count,
            }
        return batch

    def unweighted_backbone_step(self, batch: CollocationBatch) -> Dict[str, float]:
        self.backbone_optimizer.zero_grad(set_to_none=True)
        loss, loss_r, loss_b = compute_loss(
            self.model, batch, self.boundary_weight, subnet=None
        )
        loss.backward()
        self.backbone_optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_r": float(loss_r.detach().cpu()),
            "loss_b": float(loss_b.detach().cpu()),
        }

    def subnet_only_step(self, batch: CollocationBatch) -> Dict[str, float]:
        if self.subnet is None or self.subnet_optimizer is None:
            raise RuntimeError("Subnet-only step requested for PINN")
        self.subnet_optimizer.zero_grad(set_to_none=True)
        # Freezing the backbone removes gradients that the supplied notebook
        # computes but never uses. The objective and subnet gradient are unchanged.
        with temporarily_freeze(self.model):
            loss, loss_r, loss_b = compute_loss(
                self.model,
                batch,
                self.boundary_weight,
                subnet=self.subnet,
                detach_residual_for_subnet_only=True,
            )
            (-loss).backward()
        self.subnet_optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_r": float(loss_r.detach().cpu()),
            "loss_b": float(loss_b.detach().cpu()),
        }

    def adversarial_backbone_step(
        self, batch: CollocationBatch, update_subnet: bool
    ) -> Tuple[Dict[str, float], int]:
        if self.subnet is None or self.subnet_optimizer is None:
            raise RuntimeError("Adversarial step requested for PINN")
        self.backbone_optimizer.zero_grad(set_to_none=True)
        self.subnet_optimizer.zero_grad(set_to_none=True)
        loss, loss_r, loss_b = compute_loss(
            self.model, batch, self.boundary_weight, subnet=self.subnet
        )
        loss.backward()
        self.backbone_optimizer.step()
        subnet_updates = 0
        if update_subnet:
            for p in self.subnet.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.neg_()
            self.subnet_optimizer.step()
            subnet_updates = 1
        return (
            {
                "loss": float(loss.detach().cpu()),
                "loss_r": float(loss_r.detach().cpu()),
                "loss_b": float(loss_b.detach().cpu()),
            },
            subnet_updates,
        )

    def run_repeated_stage(
        self,
        *,
        name: str,
        steps: int,
        stream: str,
        mode: str,
        apply_scheduler: bool = False,
    ) -> None:
        backbone_lr_start = float(self.backbone_optimizer.param_groups[0]["lr"])
        subnet_lr_start = (
            float(self.subnet_optimizer.param_groups[0]["lr"])
            if self.subnet_optimizer is not None
            else None
        )
        if steps <= 0:
            self.stage_records[name] = {
                "elapsed_s": 0.0,
                "primary_steps": 0,
                "backbone_updates": 0,
                "subnet_updates": 0,
                "seconds_per_primary_step": None,
                "last_loss": None,
                "backbone_lr_start": backbone_lr_start,
                "subnet_lr_start": subnet_lr_start,
            }
            return

        log_every = int(self.cfg.get("logging", {}).get("log_every", 1000))
        scheduler_every = int(self.opt_cfg["scheduler_step"])
        subnet_every = int(self.opt_cfg.get("subnet_update_every", 1))
        subnet_per_backbone = int(
            self.opt_cfg.get("subnet_updates_per_backbone", 1)
        )
        if subnet_every < 1 or subnet_per_backbone < 1:
            raise ValueError("DASA update-frequency values must be positive")

        batch: Optional[CollocationBatch] = None
        interval = int(self.data_cfg["resample_every"])
        backbone_updates = 0
        subnet_updates = 0
        last_stats: Dict[str, float] = {}

        sync(self.device)
        start = time.perf_counter()
        for step in range(steps):
            if batch is None or (interval > 0 and step % interval == 0):
                batch = self.batch_for(stream, step)

            if mode == "unweighted":
                last_stats = self.unweighted_backbone_step(batch)
                backbone_updates += 1
            elif mode == "subnet_only":
                last_stats = self.subnet_only_step(batch)
                subnet_updates += 1
            elif mode == "adversarial":
                due = ((step + 1) % subnet_every) == 0
                last_stats, used = self.adversarial_backbone_step(batch, due)
                backbone_updates += 1
                subnet_updates += used
                if due and subnet_per_backbone > 1:
                    for _ in range(subnet_per_backbone - 1):
                        last_stats = self.subnet_only_step(batch)
                        subnet_updates += 1
            else:
                raise ValueError(f"Unknown stage mode: {mode}")

            if apply_scheduler and scheduler_every > 0 and (step + 1) % scheduler_every == 0:
                gamma = float(self.opt_cfg["scheduler_gamma"])
                for group in self.backbone_optimizer.param_groups:
                    group["lr"] *= gamma

            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
                print(
                    f"[{name}] {step + 1}/{steps} "
                    f"loss={last_stats.get('loss', math.nan):.3e} "
                    f"lr_b={self.backbone_optimizer.param_groups[0]['lr']:.3e}"
                )

        sync(self.device)
        elapsed = time.perf_counter() - start
        self.stage_records[name] = {
            "elapsed_s": elapsed,
            "primary_steps": steps,
            "backbone_updates": backbone_updates,
            "subnet_updates": subnet_updates,
            "seconds_per_primary_step": elapsed / steps,
            "last_loss": last_stats,
            "backbone_lr_start": backbone_lr_start,
            "subnet_lr_start": subnet_lr_start,
            "backbone_lr_end": float(self.backbone_optimizer.param_groups[0]["lr"]),
            "subnet_lr_end": (
                float(self.subnet_optimizer.param_groups[0]["lr"])
                if self.subnet_optimizer is not None
                else None
            ),
        }

    def run_lbfgs(self) -> None:
        name = "lbfgs"
        outer_steps = int(self.opt_cfg["lbfgs_outer_steps"])
        if outer_steps <= 0:
            self.stage_records[name] = {
                "elapsed_s": 0.0,
                "primary_steps": 0,
                "backbone_updates": 0,
                "subnet_updates": 0,
                "seconds_per_primary_step": None,
                "last_loss": None,
            }
            return

        lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=float(self.opt_cfg["lbfgs_lr"]),
            max_iter=int(self.opt_cfg["lbfgs_max_iter"]),
            tolerance_grad=float(self.opt_cfg["lbfgs_tolerance_grad"]),
            tolerance_change=float(self.opt_cfg["lbfgs_tolerance_change"]),
            history_size=int(self.opt_cfg.get("lbfgs_history_size", 100)),
            line_search_fn=self.opt_cfg.get("lbfgs_line_search_fn", None),
        )
        interval = int(self.data_cfg["resample_every"])
        batch: Optional[CollocationBatch] = None
        log_every = max(1, int(self.cfg.get("logging", {}).get("lbfgs_log_every", 10)))
        last_loss = math.nan

        sync(self.device)
        start = time.perf_counter()
        for outer in range(outer_steps):
            if batch is None or (interval > 0 and outer % interval == 0):
                batch = self.batch_for("lbfgs", outer)

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                loss, _, _ = compute_loss(
                    self.model, batch, self.boundary_weight, subnet=None
                )
                loss.backward()
                return loss

            returned = lbfgs.step(closure)
            last_loss = float(returned.detach().cpu()) if torch.is_tensor(returned) else float(returned)
            if (outer + 1) % log_every == 0 or outer == 0 or outer + 1 == outer_steps:
                print(f"[lbfgs] {outer + 1}/{outer_steps} returned_loss={last_loss:.3e}")

        sync(self.device)
        elapsed = time.perf_counter() - start
        self.stage_records[name] = {
            "elapsed_s": elapsed,
            "primary_steps": outer_steps,
            "backbone_updates": outer_steps,
            "subnet_updates": 0,
            "seconds_per_primary_step": elapsed / outer_steps,
            "last_loss": {"returned_loss": last_loss},
            "max_iter_per_outer_step": int(self.opt_cfg["lbfgs_max_iter"]),
        }

    def evaluate(self, label: str, mat_path: Path, chunk_size: int = 65536) -> None:
        raw = scipy.io.loadmat(mat_path)
        required = {"x", "y", "z"}
        missing = required.difference(raw)
        if missing:
            raise KeyError(f"Missing MAT keys: {sorted(missing)}")
        x = np.asarray(raw["x"]).reshape(-1)
        y = np.asarray(raw["y"]).reshape(-1)
        truth = np.asarray(raw["z"]).reshape(-1).astype(np.float64)
        coords = np.column_stack((x, y)).astype(np.float32)

        was_training = self.model.training
        self.model.eval()
        pred_parts = []
        sync(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            for begin in range(0, coords.shape[0], chunk_size):
                end = min(coords.shape[0], begin + chunk_size)
                inp = torch.as_tensor(coords[begin:end], device=self.device)
                pred_parts.append(self.model(inp).detach().cpu().numpy().reshape(-1))
        sync(self.device)
        elapsed = time.perf_counter() - start
        if was_training:
            self.model.train()

        pred = np.concatenate(pred_parts).astype(np.float64)
        error = pred - truth
        denom = np.linalg.norm(truth)
        self.evaluations[label] = {
            "relative_l2": float(np.linalg.norm(error) / denom),
            "max_abs_error": float(np.max(np.abs(error))),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "evaluation_s": elapsed,
            "n_test": int(truth.size),
        }
        print(
            f"[eval:{label}] rel_l2={self.evaluations[label]['relative_l2']:.6e} "
            f"max_abs={self.evaluations[label]['max_abs_error']:.6e}"
        )

    def run(self, mat_path: Path) -> None:
        self.model.train()
        if self.subnet is not None:
            self.subnet.train()

        self.run_repeated_stage(
            name="warmup",
            steps=int(self.opt_cfg["warmup_steps"]),
            stream="warmup",
            mode="unweighted",
        )
        self.evaluate("after_warmup", mat_path)

        if self.method == "dasa":
            self.run_repeated_stage(
                name="subnet_pretrain",
                steps=int(self.opt_cfg["subnet_pretrain_steps"]),
                stream="subnet_pretrain",
                mode="subnet_only",
            )
            main_mode = "adversarial"
        else:
            self.stage_records["subnet_pretrain"] = {
                "elapsed_s": 0.0,
                "primary_steps": 0,
                "backbone_updates": 0,
                "subnet_updates": 0,
                "seconds_per_primary_step": None,
                "last_loss": None,
            }
            main_mode = "unweighted"

        self.run_repeated_stage(
            name="main",
            steps=int(self.opt_cfg["main_steps"]),
            stream="main",
            mode=main_mode,
            apply_scheduler=True,
        )
        self.evaluate("after_main", mat_path)

        finetune_start_lr = self.opt_cfg.get("finetune_start_lr")
        if finetune_start_lr is not None:
            finetune_start_lr = float(finetune_start_lr)
            if finetune_start_lr <= 0.0:
                raise ValueError("finetune_start_lr must be positive")
            for group in self.backbone_optimizer.param_groups:
                group["lr"] = finetune_start_lr
            print(f"[finetune] reset backbone lr to {finetune_start_lr:.6e}")

        self.run_repeated_stage(
            name="finetune",
            steps=int(self.opt_cfg["finetune_steps"]),
            stream="finetune",
            mode="unweighted",
            apply_scheduler=bool(self.opt_cfg.get("schedule_finetune", False)),
        )
        self.evaluate("after_finetune", mat_path)

        self.run_lbfgs()
        self.evaluate("final", mat_path)

    def build_metrics(self) -> Dict[str, Any]:
        backbone_trainable = count_parameters(self.model, trainable_only=True)
        backbone_all = count_parameters(self.model, trainable_only=False)
        subnet_trainable = (
            count_parameters(self.subnet, trainable_only=True)
            if self.subnet is not None
            else 0
        )
        subnet_all = (
            count_parameters(self.subnet, trainable_only=False)
            if self.subnet is not None
            else 0
        )
        total_elapsed = float(sum(r["elapsed_s"] for r in self.stage_records.values()))
        adam_elapsed = float(
            sum(
                self.stage_records[name]["elapsed_s"]
                for name in ("warmup", "subnet_pretrain", "main", "finetune")
            )
        )
        peak_memory = None
        if self.device.type == "cuda":
            peak_memory = float(torch.cuda.max_memory_allocated(self.device) / 1024**2)

        return {
            "schema_version": 1,
            "experiment_name": self.cfg["experiment_name"],
            "category": self.cfg.get("category", "unspecified"),
            "variant": self.cfg.get("variant", self.cfg["experiment_name"]),
            "sensitivity_axis": self.cfg.get("sensitivity_axis"),
            "method": self.method,
            "seed": self.seed,
            "config": self.cfg,
            "parameter_counts": {
                "backbone_trainable": backbone_trainable,
                "backbone_all": backbone_all,
                "subnet_trainable": subnet_trainable,
                "subnet_all": subnet_all,
                "total_trainable": backbone_trainable + subnet_trainable,
                "total_all": backbone_all + subnet_all,
                "trainable_overhead_vs_backbone_pct": (
                    100.0 * subnet_trainable / backbone_trainable
                    if backbone_trainable > 0
                    else None
                ),
            },
            "data_counts_first_batch": self.first_batch_counts,
            "stage_timings": self.stage_records,
            "evaluations": self.evaluations,
            "timing_summary": {
                "adam_total_s": adam_elapsed,
                "training_total_s": total_elapsed,
                "peak_gpu_memory_mb": peak_memory,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
            },
        }


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    opt = cfg["optimization"]
    mapping = {
        "warmup_steps": "warmup_steps",
        "subnet_pretrain_steps": "subnet_pretrain_steps",
        "main_steps": "main_steps",
        "finetune_steps": "finetune_steps",
        "lbfgs_outer_steps": "lbfgs_outer_steps",
        "lbfgs_max_iter": "lbfgs_max_iter",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            opt[cfg_name] = int(value)
    if args.log_every is not None:
        cfg.setdefault("logging", {})["log_every"] = int(args.log_every)
    if args.n_interior_candidates is not None:
        cfg["data"]["n_interior_candidates"] = int(args.n_interior_candidates)
    if args.n_boundary_per_segment is not None:
        cfg["data"]["n_boundary_per_segment"] = int(args.n_boundary_per_segment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--subnet-pretrain-steps", type=int)
    parser.add_argument("--main-steps", type=int)
    parser.add_argument("--finetune-steps", type=int)
    parser.add_argument("--lbfgs-outer-steps", type=int)
    parser.add_argument("--lbfgs-max-iter", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--n-interior-candidates", type=int)
    parser.add_argument("--n-boundary-per-segment", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-metrics", type=Path)
    parser.add_argument("--lbfgs-only", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    apply_overrides(cfg, args)
    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=True)

    project_root = Path(__file__).resolve().parents[1]
    mat_path = args.data or project_root / "data" / "Lpoission_more.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Reference MAT file not found: {mat_path}")

    run_dir = args.output_root / cfg["experiment_name"] / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"[skip] Existing result: {metrics_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    experiment = Experiment(cfg, args.seed, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print(json.dumps({"config": cfg, "seed": args.seed, "device": str(device)}, indent=2))
    if args.lbfgs_only:
        if args.checkpoint is None or args.base_metrics is None:
            parser.error("--lbfgs-only requires --checkpoint and --base-metrics")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        experiment.model.load_state_dict(checkpoint["backbone"])
        if experiment.subnet is not None and checkpoint.get("subnet") is not None:
            experiment.subnet.load_state_dict(checkpoint["subnet"])
        base_metrics = load_json(args.base_metrics)
        if int(base_metrics["seed"]) != args.seed:
            raise ValueError("Checkpoint/base-metrics seed does not match --seed")
        experiment.stage_records = dict(base_metrics["stage_timings"])
        experiment.evaluations = dict(base_metrics["evaluations"])
        experiment.first_batch_counts = base_metrics.get("data_counts_first_batch")
        experiment.run_lbfgs()
        experiment.evaluate("final", mat_path)
    else:
        experiment.run(mat_path)
    metrics = experiment.build_metrics()
    save_json(metrics_path, metrics)
    torch.save(
        {
            "config": cfg,
            "seed": args.seed,
            "backbone": experiment.model.state_dict(),
            "subnet": experiment.subnet.state_dict() if experiment.subnet is not None else None,
        },
        run_dir / "final_model.pt",
    )
    print(f"[done] {metrics_path}")


if __name__ == "__main__":
    main()
