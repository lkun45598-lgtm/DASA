"""Reference interpolation, prediction, metrics and output artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .metrics import (
    assert_finite_arrays,
    cavity_metrics,
    scalar_metrics,
)
from .plotting import plot_absolute_error
from .references import ensure_reference, interpolate_reference


def predict(model, points: np.ndarray, device: torch.device, chunk_size: int = 8192) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            tensor = torch.as_tensor(
                points[start : start + chunk_size], dtype=torch.float64, device=device
            )
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(
    model,
    case,
    config: dict,
    result_dir: Path,
    reference_root: Path,
    device: torch.device,
) -> dict:
    points = case.make_test_points()
    prediction = predict(model, points, device)
    if case.reference_type == "analytic":
        reference = case.reference_values(points)
        reference_metadata = {
            "type": "paper analytical/manufactured solution",
            "case": config["case"],
        }
    else:
        master_path = ensure_reference(reference_root, config["case"])
        reference, reference_metadata = interpolate_reference(
            master_path, config["case"], points
        )
        shutil.copy2(master_path, result_dir / "reference_master.npz")
    assert_finite_arrays(points=points, prediction=prediction, reference=reference)
    if config["case"] == "lid_driven_cavity":
        values = cavity_metrics(prediction, reference)
    else:
        values = scalar_metrics(prediction[:, :1], reference[:, :1])
    values.update(
        {
            "case": config["case"],
            "seed": config["seed"],
            "mode": config["mode"],
            "reference_type": config["reference_type"],
            "full_training_completed": bool(config["mode"] == "full"),
            "test_points": int(len(points)),
        }
    )
    np.savez_compressed(
        result_dir / "prediction.npz",
        points=points,
        prediction=prediction,
        output_names=np.asarray(case.output_names),
    )
    np.savez_compressed(
        result_dir / "reference.npz",
        points=points,
        reference=reference,
        metadata=np.asarray(json.dumps(reference_metadata)),
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
    plot_absolute_error(
        points,
        prediction,
        reference,
        case.coordinate_names,
        result_dir / "absolute_error.png",
    )
    return values
