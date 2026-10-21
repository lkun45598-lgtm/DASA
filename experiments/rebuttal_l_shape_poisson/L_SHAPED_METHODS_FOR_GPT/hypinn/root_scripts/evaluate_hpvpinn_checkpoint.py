"""Evaluate a saved hp-VPINN checkpoint without changing a training run."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

import hpvpinn_experiments as hpv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--residual-points", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.run_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    base_spec = hpv.SPECS[config["name"]]
    spec_keys = {field.name for field in fields(hpv.ProblemSpec)}
    overrides = {key: value for key, value in config.items() if key in spec_keys}
    spec = replace(base_spec, **overrides)
    dtype = torch.float64 if config.get("dtype", "float64") == "float64" else torch.float32
    device = torch.device(args.device)

    model: torch.nn.Module = hpv.ModifiedMLP(spec, dtype)
    if config.get("hard_constraints", False):
        model = hpv.BurgersConstrainedTrial(model)
    model = model.to(device)
    checkpoint = torch.load(args.run_dir / "checkpoint_latest.pt", map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()

    coordinates, y_true, evaluation = hpv.evaluation_data(spec.name, spec.ade_kappa)
    y_pred = hpv.predict(spec.name, model, coordinates, device, dtype)
    result: dict[str, object] = {
        "problem": spec.name,
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "relative_l2_error": hpv.relative_l2(y_true, y_pred),
        "max_pointwise_error": hpv.max_pointwise_error(y_true, y_pred),
        "evaluation": evaluation,
    }

    if coordinates.shape[1] == 2 and spec.name not in {
        "poisson_2d_peak", "poisson_2d_steep", "helmholtz_2d", "cavity_re100", "poisson_lshape"
    }:
        times = np.unique(coordinates[:, 1])
        slice_errors = []
        for time_value in times:
            mask = np.isclose(coordinates[:, 1], time_value, rtol=0.0, atol=1.0e-12)
            slice_errors.append(hpv.relative_l2(y_true[mask], y_pred[mask]))
        slice_errors_array = np.asarray(slice_errors)
        requested_times = np.linspace(times[0], times[-1], 17)
        selected = {}
        for requested in requested_times:
            index = int(np.argmin(np.abs(times - requested)))
            selected[f"{times[index]:.8g}"] = float(slice_errors_array[index])
        result.update(
            initial_relative_l2_error=float(slice_errors_array[0]),
            final_time_relative_l2_error=float(slice_errors_array[-1]),
            worst_time_slice_relative_l2_error=float(np.max(slice_errors_array)),
            worst_time=float(times[int(np.argmax(slice_errors_array))]),
            selected_time_slice_relative_l2=selected,
        )

    if args.residual_points > 0 and spec.name in {"burgers_1d", "allen_cahn_1d"}:
        sample_count = min(args.residual_points, coordinates.shape[0])
        indices = np.linspace(0, coordinates.shape[0] - 1, sample_count, dtype=np.int64)
        residual_blocks = []
        for start in range(0, sample_count, 2048):
            block_indices = indices[start:start + 2048]
            points = hpv.tensor(coordinates[block_indices], device, dtype).requires_grad_(True)
            residual = hpv.strong_residual(spec.name, model, points, spec.ade_kappa)
            residual_blocks.append(residual.detach().cpu().numpy())
        residual_values = np.vstack(residual_blocks)
        result.update(
            strong_residual_rms=float(np.sqrt(np.mean(residual_values**2))),
            strong_residual_max=float(np.max(np.abs(residual_values))),
            strong_residual_points=sample_count,
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
