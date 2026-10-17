"""Create two Allen-Cahn spatial refinement bands from independent strong residuals."""

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
    parser.add_argument("--time-elements", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-mesh", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def contiguous_dorfler_band(
    contributions: np.ndarray, candidate_indices: np.ndarray, theta: float
) -> tuple[int, int, float]:
    candidate_total = float(np.sum(contributions[candidate_indices]))
    peak = int(candidate_indices[np.argmax(contributions[candidate_indices])])
    first = last = peak
    marked_energy = float(contributions[peak])
    target = theta * candidate_total
    candidate_first, candidate_last = int(candidate_indices[0]), int(candidate_indices[-1])
    while marked_energy < target and (first > candidate_first or last < candidate_last):
        left_value = contributions[first - 1] if first > candidate_first else -1.0
        right_value = contributions[last + 1] if last < candidate_last else -1.0
        if left_value >= right_value:
            first -= 1
            marked_energy += float(contributions[first])
        else:
            last += 1
            marked_energy += float(contributions[last])
    return first, last, marked_energy / candidate_total


def main() -> None:
    args = parse_args()
    if not 0.0 < args.theta <= 1.0:
        raise ValueError("--theta must be in (0, 1]")
    if min(args.x_order, args.t_order) < 3 or args.batch_size < 1:
        raise ValueError("Indicator orders must be at least 3 and batch size must be positive")
    if args.time_elements < 1:
        raise ValueError("--time-elements must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    with (args.run_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config["name"] != "allen_cahn_1d":
        raise ValueError("Spatial band refinement is only valid for Allen-Cahn")
    spec_names = {field.name for field in fields(hpv.ProblemSpec)}
    overrides = {key: value for key, value in config.items() if key in spec_names and key != "name"}
    overrides["domain"] = tuple(tuple(bounds) for bounds in overrides["domain"])
    spec = replace(hpv.SPECS["allen_cahn_1d"], **overrides)
    dtype = torch.float64 if config["dtype"] == "float64" else torch.float32
    device = torch.device(args.device)

    model = hpv.ModifiedMLP(spec, dtype).to(device)
    checkpoint_path = args.run_dir / "checkpoint_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
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
        residual = hpv.strong_residual("allen_cahn_1d", model, block)
        residual_sq[start:start + block.shape[0]] = (
            torch.sum(residual**2, dim=1).detach().cpu().numpy()
        )
    residual_sq = residual_sq.reshape(args.x_order, args.t_order)

    time_integrated = residual_sq @ (t_weights * (t_high - t_low) / 2.0)
    contributions = time_integrated * x_weights * (x_high - x_low) / 2.0
    negative = np.flatnonzero(x < 0.0)
    positive = np.flatnonzero(x > 0.0)
    neg_first, neg_last, neg_fraction = contiguous_dorfler_band(contributions, negative, args.theta)
    pos_first, pos_last, pos_fraction = contiguous_dorfler_band(contributions, positive, args.theta)

    def left_boundary(index: int) -> float:
        return float((x[index - 1] + x[index]) / 2.0) if index > 0 else x_low

    def right_boundary(index: int) -> float:
        return float((x[index] + x[index + 1]) / 2.0) if index + 1 < x.size else x_high

    edges = np.asarray(
        [
            x_low,
            left_boundary(neg_first),
            right_boundary(neg_last),
            left_boundary(pos_first),
            right_boundary(pos_last),
            x_high,
        ]
    )
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError(f"Residual bands overlap or touch the domain boundary: {edges.tolist()}")
    time_edges = np.linspace(t_low, t_high, args.time_elements + 1)
    elements = hpv.regular_elements(edges, time_edges)

    output_mesh = args.output_mesh or args.run_dir / "spatial_band_elements.npy"
    output_report = args.output_report or args.run_dir / "spatial_band_report.json"
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mesh, elements, allow_pickle=False)
    total = float(np.sum(contributions))
    report = {
        "problem": "allen_cahn_1d",
        "source_run_dir": str(args.run_dir.resolve()),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "selection_uses_reference_solution": False,
        "indicator": "time-integrated independent strong-residual energy by spatial node",
        "quadrature": "tensor-product Gauss-Legendre",
        "x_order": args.x_order,
        "t_order": args.t_order,
        "dorfler_theta_per_half_domain": args.theta,
        "negative_band_achieved_fraction": neg_fraction,
        "positive_band_achieved_fraction": pos_fraction,
        "negative_band": [float(edges[1]), float(edges[2])],
        "positive_band": [float(edges[3]), float(edges[4])],
        "spatial_edges": edges.tolist(),
        "time_edges": time_edges.tolist(),
        "time_elements": args.time_elements,
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
