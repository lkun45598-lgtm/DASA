"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)
