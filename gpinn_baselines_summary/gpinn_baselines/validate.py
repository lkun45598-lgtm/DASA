"""Static, reference and completed-result validation without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cases import create_case
from .config import CASE_NAMES, build_config
from .metrics import assert_finite_arrays, recompute_saved_metrics
from .network import MLP
from .references import NUMERICAL_CASES, reference_path
from .seeds import parse_seed_spec


AUTODIFF_EXACT_CASES = (
    "poisson_1d_boundary_layer",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
    "diffusion_reaction_1d",
    "advection_discontinuous_1d",
)


def validate_static() -> dict:
    torch.set_default_dtype(torch.float64)
    package_root = Path(__file__).resolve().parent
    forbidden_hits: list[str] = []
    for source in package_root.rglob("*.py"):
        if source.resolve() == Path(__file__).resolve():
            continue
        text = source.read_text(encoding="utf-8").lower()
        if "import tensorflow" in text or "deepxde" in text:
            forbidden_hits.append(str(source))
        if "usol_d_0.001_k_5" in text:
            forbidden_hits.append(str(source))
    if forbidden_hits:
        raise AssertionError(f"Forbidden backend/reference in package: {forbidden_hits}")

    architecture_checked = []
    boundary_checked = []
    for name in CASE_NAMES:
        config = build_config(name, "smoke", 0)
        case = create_case(config)
        model = MLP(
            config["input_dim"],
            config["output_dim"],
            config["lower"],
            config["upper"],
        )
        if len(model.hidden) != 6:
            raise AssertionError(f"{name}: hidden layer count is not 6")
        if any(layer.out_features != 128 for layer in model.hidden):
            raise AssertionError(f"{name}: a hidden layer is not width 128")
        if model.output.out_features != config["output_dim"]:
            raise AssertionError(f"{name}: output dimension mismatch")
        architecture_checked.append(name)
        batch = case.sample_training(np.random.default_rng(7))
        if "boundary_values" in batch:
            assert_finite_arrays(
                boundary_points=batch["boundary_points"],
                boundary_values=batch["boundary_values"],
            )
        if "initial_values" in batch:
            assert_finite_arrays(initial_values=batch["initial_values"])
        boundary_checked.append(name)

    # The Allen-Cahn equation and all of its residual derivatives inherit this exact value.
    allen = build_config("allen_cahn_1d", "full", 0)
    if allen["parameters"]["diffusion"] != 1.0e-4:
        raise AssertionError("Allen-Cahn diffusion is not exactly 1e-4")
    allen_case = create_case(allen)
    allen_batch = allen_case.sample_training(np.random.default_rng(8))
    if not np.array_equal(allen_batch["periodic_left"][:, 1], allen_batch["periodic_right"][:, 1]):
        raise AssertionError("Allen-Cahn periodic boundary pairs use different times")
    if not (
        np.all(allen_batch["periodic_left"][:, 0] == -1.0)
        and np.all(allen_batch["periodic_right"][:, 0] == 1.0)
    ):
        raise AssertionError("Allen-Cahn periodic boundary coordinates are wrong")

    cavity = create_case(build_config("lid_driven_cavity", "smoke", 0))
    cavity_batch = cavity.sample_training(np.random.default_rng(9))
    boundary, target = cavity_batch["boundary_points"], cavity_batch["boundary_values"]
    top = np.isclose(boundary[:, 1], 1.0)
    if not (np.all(target[top, 0] == 1.0) and np.all(target[top, 1] == 0.0)):
        raise AssertionError("moving-lid velocity is wrong")
    if not np.all(target[~top] == 0.0):
        raise AssertionError("stationary cavity wall velocity is wrong")

    residual_maxima = {}
    for name in AUTODIFF_EXACT_CASES:
        config = build_config(name, "smoke", 0)
        case = create_case(config)
        points_np = case.sample_residual(9, np.random.default_rng(11))
        if name == "advection_discontinuous_1d":
            # Keep points away from the two moving discontinuities.
            points_np[:, 0] = np.linspace(-1.4, -0.7, len(points_np))
            points_np[:, 1] = np.linspace(0.1, 0.9, len(points_np))
        points = torch.as_tensor(points_np, dtype=torch.float64).requires_grad_(True)
        exact = getattr(case, "_exact_torch")
        residuals = case.residuals(lambda value: exact(value), points)
        maximum = max(float(torch.max(torch.abs(value)).detach()) for value in residuals.values())
        tolerance = 1.0e-7 if name == "poisson_1d_boundary_layer" else 1.0e-9
        if maximum > tolerance:
            raise AssertionError(f"{name}: exact PDE residual {maximum} > {tolerance}")
        residual_maxima[name] = maximum

    for name in ("poisson_1d_boundary_layer", "poisson_1d_steep"):
        config = build_config(name, "full", 0)
        case = create_case(config)
        if config["sampling"]["n_boundary"] != 0:
            raise AssertionError(f"{name}: hard Dirichlet protocol sampled a boundary loss")
        if config["optimizer"]["lr_decay_rate"] != 1.0:
            raise AssertionError(f"{name}: original gPINN constant learning rate is disabled")
        batch = case.sample_training(np.random.default_rng(12))
        if tuple(batch) != ("residual",):
            raise AssertionError(f"{name}: unexpected soft-constraint batch keys")
        model = MLP(1, 1, config["lower"], config["upper"])
        model.set_output_transform(case.output_transform)
        endpoints = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
        predicted = model(endpoints).detach().numpy()
        exact_boundary = case.exact_numpy(endpoints.numpy())
        if not np.allclose(predicted, exact_boundary, rtol=0.0, atol=1.0e-13):
            raise AssertionError(f"{name}: hard Dirichlet transform is not exact")

    # Equation (24)'s stable image series is checked by centered finite differences.
    config = build_config("advection_diffusion_1d", "smoke", 0)
    case = create_case(config)
    x, t = 0.1, 0.4
    hx, ht = 2.0e-4, 2.0e-5

    def exact(x_value, t_value):
        return float(case.exact_numpy(np.asarray([[x_value, t_value]], dtype=np.float64))[0, 0])

    u = exact(x, t)
    ux = (exact(x + hx, t) - exact(x - hx, t)) / (2.0 * hx)
    uxx = (exact(x + hx, t) - 2.0 * u + exact(x - hx, t)) / hx**2
    ut = (exact(x, t + ht) - exact(x, t - ht)) / (2.0 * ht)
    ade_residual = ut + config["parameters"]["velocity"] * ux - config["parameters"]["diffusivity"] * uxx
    if abs(ade_residual) > 2.0e-4:
        raise AssertionError(f"advection_diffusion_1d exact-series residual={ade_residual}")
    residual_maxima["advection_diffusion_1d"] = abs(ade_residual)
    return {
        "cases_imported": len(CASE_NAMES),
        "architectures_checked": architecture_checked,
        "conditions_checked": boundary_checked,
        "analytic_residual_maxima": residual_maxima,
        "allen_cahn_diffusion": allen["parameters"]["diffusion"],
    }


def validate_references(root: Path) -> dict:
    thresholds = {
        "burgers_1d": 2.0e-2,
        "allen_cahn_1d": 2.0e-2,
        "poisson_lshape": 8.0e-2,
        "lid_driven_cavity": 8.0e-2,
    }
    report = {}
    for case in NUMERICAL_CASES:
        path = reference_path(root, case)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            for key in data.files:
                if key == "metadata":
                    continue
                assert_finite_arrays(**{f"{case}:{key}": data[key]})
            metadata = json.loads(str(data["metadata"]))
        convergence = metadata["convergence"]
        candidates = [
            value
            for key, value in convergence.items()
            if "relative_l2" in key and isinstance(value, (int, float))
        ]
        if not candidates:
            raise AssertionError(f"{case}: no relative grid-convergence measure")
        relative = max(candidates)
        if relative > thresholds[case]:
            raise AssertionError(
                f"{case}: grid difference {relative} exceeds {thresholds[case]}"
            )
        fine = metadata.get("fine_grid", {})
        if fine and fine.get("converged") is False:
            raise AssertionError(f"{case}: fine reference did not reach steady tolerance")
        report[case] = convergence
    return report


def validate_results(
    output_root: Path,
    mode: str,
    expected_seeds: tuple[int, ...] | None = None,
    case_names: tuple[str, ...] = CASE_NAMES,
) -> dict:
    report = {}
    unknown = sorted(set(case_names) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"Unknown case names: {unknown}")
    for case in case_names:
        if expected_seeds is None:
            candidates = sorted((output_root / mode / case).glob("seed_*/metrics.json"))
        else:
            candidates = [
                output_root / mode / case / f"seed_{seed}" / "metrics.json"
                for seed in expected_seeds
            ]
        if not candidates:
            raise FileNotFoundError(f"No {mode} result for {case}")
        case_report = {}
        for metrics_path in candidates:
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            result_dir = metrics_path.parent
            with metrics_path.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            if mode == "full" and not bool(saved.get("full_training_completed")):
                raise AssertionError(f"{case}/{result_dir.name}: full training incomplete")
            recomputed = recompute_saved_metrics(result_dir)
            for key in ("relative_l2", "max_abs"):
                if not np.isclose(
                    saved[key], recomputed[key], rtol=1.0e-12, atol=1.0e-14
                ):
                    raise AssertionError(
                        f"{case}/{result_dir.name}: saved {key}={saved[key]} "
                        f"!= recomputed {recomputed[key]}"
                    )
            for npz_path in result_dir.glob("*.npz"):
                with np.load(npz_path) as data:
                    for key in data.files:
                        if key == "metadata" or data[key].dtype.kind in {"U", "S", "O"}:
                            continue
                        assert_finite_arrays(**{f"{npz_path.name}:{key}": data[key]})
            case_report[result_dir.name] = recomputed
        report[case] = case_report
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("static", "references", "results"), required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("data/reference_data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gpinn_baselines"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--seeds",
        help="Require exactly these seeds, e.g. 0-9 or 0,1,2",
    )
    parser.add_argument(
        "--cases",
        help="Comma-separated result subset; the default is all 13 equations.",
    )
    args = parser.parse_args(argv)
    if args.level == "static":
        report = validate_static()
    elif args.level == "references":
        report = validate_references(args.reference_root)
    else:
        expected_seeds = parse_seed_spec(args.seeds) if args.seeds else None
        case_names = (
            tuple(item.strip() for item in args.cases.split(",") if item.strip())
            if args.cases
            else CASE_NAMES
        )
        if not case_names:
            parser.error("--cases must contain at least one case")
        report = validate_results(
            args.output_root, args.mode, expected_seeds, case_names=case_names
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
