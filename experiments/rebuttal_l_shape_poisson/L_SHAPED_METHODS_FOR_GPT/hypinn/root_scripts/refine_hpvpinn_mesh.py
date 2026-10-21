"""Build one residual-driven h-refinement for a completed hp-VPINN stage."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

import hpvpinn_experiments as hpv


SUPPORTED_PROBLEMS = ("burgers_1d", "allen_cahn_1d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-mesh", type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def tensor_product_rule(order: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    points_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    point_grids = np.meshgrid(*([points_1d] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights_1d] * dimension), indexing="ij")
    points = np.stack([grid.reshape(-1) for grid in point_grids], axis=1)
    weights = np.prod(np.stack(weight_grids, axis=-1), axis=-1).reshape(-1)
    return points, weights


def split_marked_elements(elements: np.ndarray, marked: set[int]) -> np.ndarray:
    children: list[np.ndarray] = []
    for index, element in enumerate(elements):
        if index not in marked:
            children.append(element.copy())
            continue
        left, right = element
        midpoint = (left + right) / 2.0
        for x_side in (0, 1):
            for t_side in (0, 1):
                child_left = np.asarray(
                    [left[0] if x_side == 0 else midpoint[0], left[1] if t_side == 0 else midpoint[1]]
                )
                child_right = np.asarray(
                    [midpoint[0] if x_side == 0 else right[0], midpoint[1] if t_side == 0 else right[1]]
                )
                children.append(np.stack((child_left, child_right)))
    return np.asarray(children, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.theta <= 1.0:
        raise ValueError("--theta must be in (0, 1]")
    if args.quadrature_order < 2:
        raise ValueError("--quadrature-order must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_path = args.run_dir / "config.json"
    checkpoint_path = args.run_dir / "checkpoint_latest.pt"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    problem = config["name"]
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f"Residual refinement only supports {SUPPORTED_PROBLEMS}, got {problem}")

    spec_field_names = {field.name for field in fields(hpv.ProblemSpec)}
    spec_overrides = {key: value for key, value in config.items() if key in spec_field_names and key != "name"}
    if "domain" in spec_overrides:
        spec_overrides["domain"] = tuple(tuple(bounds) for bounds in spec_overrides["domain"])
    spec = replace(hpv.SPECS[problem], **spec_overrides)
    dtype = torch.float64 if config["dtype"] == "float64" else torch.float32
    device = torch.device(args.device)

    model: torch.nn.Module = hpv.ModifiedMLP(spec, dtype)
    if config.get("hard_constraints", False):
        model = hpv.BurgersConstrainedTrial(model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
    model = model.to(device)
    model.eval()

    elements = np.asarray(config["elements"], dtype=np.float64)
    local_points, local_weights = tensor_product_rule(args.quadrature_order, elements.shape[2])
    physical_points, jacobians = hpv.map_quadrature(elements, local_points)
    active_points = hpv.tensor(
        physical_points.reshape(-1, elements.shape[2]), device, dtype
    ).requires_grad_(True)
    residual = hpv.strong_residual(problem, model, active_points, spec.ade_kappa)
    residual_sq = torch.sum(residual**2, dim=1).reshape(elements.shape[0], -1)
    weights = hpv.tensor(local_weights, device, dtype)
    indicators_sq = (
        torch.sum(residual_sq * weights[None, :], dim=1)
        * hpv.tensor(jacobians, device, dtype)
    ).detach().cpu().numpy()
    residual_abs_max = float(torch.max(torch.abs(residual)).detach().cpu())

    total_indicator_sq = float(np.sum(indicators_sq))
    descending = np.argsort(-indicators_sq, kind="stable")
    if total_indicator_sq > 0.0:
        marked_count = int(
            np.searchsorted(
                np.cumsum(indicators_sq[descending]), args.theta * total_indicator_sq, side="left"
            )
        ) + 1
    else:
        marked_count = 1
    marked_indices = np.sort(descending[:marked_count])
    refined_elements = split_marked_elements(elements, set(int(index) for index in marked_indices))

    output_mesh = args.output_mesh or args.run_dir / "refined_elements.npy"
    output_report = args.output_report or args.run_dir / "refinement_report.json"
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mesh, refined_elements, allow_pickle=False)
    marked_fraction = (
        float(np.sum(indicators_sq[marked_indices]) / total_indicator_sq)
        if total_indicator_sq > 0.0 else 0.0
    )
    domain_measure = float(np.sum(np.prod(elements[:, 1, :] - elements[:, 0, :], axis=1)))
    report = {
        "problem": problem,
        "source_run_dir": str(args.run_dir.resolve()),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "selection_uses_reference_solution": False,
        "indicator": "eta_e^2 = integral_element |strong_residual|^2",
        "quadrature": "independent tensor-product Gauss-Legendre",
        "quadrature_order_per_dimension": args.quadrature_order,
        "dorfler_theta": args.theta,
        "dorfler_achieved_fraction": marked_fraction,
        "old_element_count": int(elements.shape[0]),
        "new_element_count": int(refined_elements.shape[0]),
        "marked_indices": marked_indices.tolist(),
        "element_indicators_squared": indicators_sq.tolist(),
        "integrated_residual_squared": total_indicator_sq,
        "strong_residual_domain_rms": float(np.sqrt(total_indicator_sq / domain_measure)),
        "strong_residual_pointwise_max_on_indicator_nodes": residual_abs_max,
        "output_mesh": str(output_mesh.resolve()),
    }
    with output_report.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
