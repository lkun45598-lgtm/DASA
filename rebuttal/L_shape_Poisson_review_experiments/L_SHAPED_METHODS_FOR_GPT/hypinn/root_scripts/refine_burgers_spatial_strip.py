"""Create a Burgers spatial strip mesh from an independent strong-residual indicator."""

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
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--x-order", type=int, default=201)
    parser.add_argument("--t-order", type=int, default=101)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--search-half-width", type=float, default=0.5)
    parser.add_argument("--output-mesh", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.theta <= 1.0:
        raise ValueError("--theta must be in (0, 1]")
    if min(args.x_order, args.t_order) < 3 or args.batch_size < 1:
        raise ValueError("Indicator orders must be at least 3 and batch size must be positive")
    if not 0.0 < args.search_half_width < 1.0:
        raise ValueError("--search-half-width must be in (0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    with (args.run_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config["name"] != "burgers_1d":
        raise ValueError("Spatial strip refinement is only valid for Burgers")
    spec_names = {field.name for field in fields(hpv.ProblemSpec)}
    overrides = {key: value for key, value in config.items() if key in spec_names and key != "name"}
    overrides["domain"] = tuple(tuple(bounds) for bounds in overrides["domain"])
    spec = replace(hpv.SPECS["burgers_1d"], **overrides)
    dtype = torch.float64 if config["dtype"] == "float64" else torch.float32
    device = torch.device(args.device)

    model: torch.nn.Module = hpv.ModifiedMLP(spec, dtype)
    checkpoint_path = args.run_dir / "checkpoint_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
    model = model.to(device)
    model.eval()

    x_local, x_weights = np.polynomial.legendre.leggauss(args.x_order)
    t_local, t_weights = np.polynomial.legendre.leggauss(args.t_order)
    (x_low, x_high), (t_low, t_high) = spec.domain
    x = (x_high + x_low) / 2.0 + (x_high - x_low) / 2.0 * x_local
    t = (t_high + t_low) / 2.0 + (t_high - t_low) / 2.0 * t_local
    xx, tt = np.meshgrid(x, t, indexing="ij")
    coordinates = np.column_stack((xx.reshape(-1), tt.reshape(-1)))
    residual_sq = np.empty(coordinates.shape[0], dtype=np.float64)
    for start in range(0, coordinates.shape[0], args.batch_size):
        block = hpv.tensor(coordinates[start:start + args.batch_size], device, dtype).requires_grad_(True)
        residual = hpv.strong_residual("burgers_1d", model, block)
        residual_sq[start:start + block.shape[0]] = (
            torch.sum(residual**2, dim=1).detach().cpu().numpy()
        )
    residual_sq = residual_sq.reshape(args.x_order, args.t_order)

    time_integrated = residual_sq @ (t_weights * (t_high - t_low) / 2.0)
    spatial_contributions = time_integrated * x_weights * (x_high - x_low) / 2.0
    total = float(np.sum(spatial_contributions))
    search_mask = np.abs(x) <= args.search_half_width
    search_indices = np.flatnonzero(search_mask)
    search_total = float(np.sum(spatial_contributions[search_indices]))
    peak = int(search_indices[np.argmax(spatial_contributions[search_indices])])
    first = last = peak
    marked_energy = float(spatial_contributions[peak])
    target = args.theta * search_total
    search_first, search_last = int(search_indices[0]), int(search_indices[-1])
    while marked_energy < target and (first > search_first or last < search_last):
        left_value = spatial_contributions[first - 1] if first > search_first else -1.0
        right_value = spatial_contributions[last + 1] if last < search_last else -1.0
        if left_value >= right_value:
            first -= 1
            marked_energy += float(spatial_contributions[first])
        else:
            last += 1
            marked_energy += float(spatial_contributions[last])
    marked = np.arange(first, last + 1)
    left_edge = (x[first - 1] + x[first]) / 2.0 if first > 0 else x_low
    right_edge = (x[last] + x[last + 1]) / 2.0 if last + 1 < x.size else x_high
    if not x_low < left_edge < right_edge < x_high:
        raise RuntimeError("Residual marking did not produce an interior spatial strip")
    elements = hpv.regular_elements(
        np.asarray([x_low, left_edge, right_edge, x_high]), np.asarray([t_low, t_high])
    )

    output_mesh = args.output_mesh or args.run_dir / "spatial_refined_elements.npy"
    output_report = args.output_report or args.run_dir / "spatial_refinement_report.json"
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mesh, elements, allow_pickle=False)
    report = {
        "problem": "burgers_1d",
        "source_run_dir": str(args.run_dir.resolve()),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "selection_uses_reference_solution": False,
        "indicator": "time-integrated independent strong-residual energy by spatial node",
        "quadrature": "tensor-product Gauss-Legendre",
        "x_order": args.x_order,
        "t_order": args.t_order,
        "dorfler_theta": args.theta,
        "dorfler_achieved_fraction_in_search_window": float(marked_energy / search_total),
        "search_window": [-args.search_half_width, args.search_half_width],
        "search_window_fraction_of_total_residual_energy": float(search_total / total),
        "marked_spatial_indices": marked.tolist(),
        "marked_x_min": float(x[first]),
        "marked_x_max": float(x[last]),
        "spatial_strip": [float(left_edge), float(right_edge)],
        "elements": elements.tolist(),
        "new_element_count": int(elements.shape[0]),
        "integrated_residual_squared": total,
        "strong_residual_domain_rms": float(np.sqrt(total / ((x_high - x_low) * (t_high - t_low)))),
        "output_mesh": str(output_mesh.resolve()),
    }
    with output_report.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
