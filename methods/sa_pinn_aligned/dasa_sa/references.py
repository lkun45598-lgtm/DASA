"""Immutable numerical-reference artifacts with SHA-256 verification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .metrics import compute_metrics


REQUIRED_MANIFEST_KEYS = {
    "case_id",
    "equation_parameters",
    "solver",
    "grid_levels",
    "tolerances",
    "convergence",
    "generated_at",
    "generator_sha256",
    "reference_sha256",
}


def compare_refinements(coarse_values: np.ndarray, fine_values: np.ndarray) -> dict[str, float]:
    return compute_metrics(fine_values, coarse_values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reference(
    case_id: str,
    points: np.ndarray,
    values: np.ndarray,
    *,
    root: Path,
    generator_path: Path,
    equation_parameters: dict[str, Any],
    solver: dict[str, Any],
    grid_levels: list[int],
    tolerances: dict[str, float],
    convergence: dict[str, float],
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or values.shape != (points.shape[0], 1):
        raise ValueError("reference points must be (N,D) and values must be (N,1)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("reference arrays must be finite")
    if set(tolerances) != set(convergence):
        raise ValueError("convergence and tolerance metric keys must match")
    exceeded = {
        key: (convergence[key], tolerances[key])
        for key in convergence
        if convergence[key] > tolerances[key]
    }
    if exceeded:
        raise ValueError(f"reference convergence limits exceeded: {exceeded}")

    root.mkdir(parents=True, exist_ok=True)
    reference_path = root / f"{case_id}.npz"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".npz", delete=False) as handle:
        temporary_reference = Path(handle.name)
        np.savez_compressed(handle, points=points, values=values)
    os.replace(temporary_reference, reference_path)

    manifest = {
        "case_id": case_id,
        "equation_parameters": equation_parameters,
        "solver": solver,
        "grid_levels": list(grid_levels),
        "tolerances": {key: float(value) for key, value in tolerances.items()},
        "convergence": {key: float(value) for key, value in convergence.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": sha256_file(generator_path),
        "reference_sha256": sha256_file(reference_path),
    }
    manifest_path = root / f"{case_id}.json"
    with tempfile.NamedTemporaryFile(
        dir=root, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary_manifest = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_reference(
    case_id: str, *, root: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_path = root / f"{case_id}.npz"
    manifest_path = root / f"{case_id}.json"
    if not reference_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen reference or manifest for {case_id}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if set(manifest) != REQUIRED_MANIFEST_KEYS:
        raise ValueError(f"invalid reference manifest keys for {case_id}")
    actual_digest = sha256_file(reference_path)
    if actual_digest != manifest["reference_sha256"]:
        raise ValueError(f"reference SHA-256 mismatch for {case_id}")
    with np.load(reference_path) as reference:
        points = np.asarray(reference["points"], dtype=np.float64)
        values = np.asarray(reference["values"], dtype=np.float64)
    return points, values, manifest
