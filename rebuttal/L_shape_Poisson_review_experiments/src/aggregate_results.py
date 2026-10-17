#!/usr/bin/env python3
"""Aggregate L-shaped Poisson experiment metrics into CSV and Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_metrics(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        item["_path"] = str(path)
        records.append(item)
    return records


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else math.nan


def std(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}g}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_record(m: Dict[str, Any]) -> Dict[str, Any]:
    cfg = m["config"]
    opt = cfg["optimization"]
    subnet = cfg.get("subnet") or {}
    stages = m["stage_timings"]
    final_eval = m["evaluations"]["final"]
    after_main = m["evaluations"]["after_main"]
    return {
        "category": m.get("category"),
        "variant": m.get("variant"),
        "sensitivity_axis": m.get("sensitivity_axis"),
        "method": m["method"],
        "seed": m["seed"],
        "backbone_depth": cfg["backbone"]["depth"],
        "backbone_width": cfg["backbone"]["width"],
        "subnet_depth": subnet.get("depth"),
        "subnet_width": subnet.get("width"),
        "lr_backbone": opt["lr_backbone"],
        "lr_subnet": opt.get("lr_subnet"),
        "subnet_update_every": opt.get("subnet_update_every"),
        "subnet_updates_per_backbone": opt.get("subnet_updates_per_backbone"),
        "total_trainable_params": m["parameter_counts"]["total_trainable"],
        "subnet_trainable_params": m["parameter_counts"]["subnet_trainable"],
        "param_overhead_pct": m["parameter_counts"]["trainable_overhead_vs_backbone_pct"],
        "warmup_s_per_step": stages["warmup"]["seconds_per_primary_step"],
        "subnet_pretrain_s_per_step": stages["subnet_pretrain"]["seconds_per_primary_step"],
        "main_s_per_step": stages["main"]["seconds_per_primary_step"],
        "finetune_s_per_step": stages["finetune"]["seconds_per_primary_step"],
        "lbfgs_s_per_outer_step": stages["lbfgs"]["seconds_per_primary_step"],
        "adam_total_s": m["timing_summary"]["adam_total_s"],
        "training_total_s": m["timing_summary"]["training_total_s"],
        "peak_gpu_memory_mb": m["timing_summary"]["peak_gpu_memory_mb"],
        "after_main_relative_l2": after_main["relative_l2"],
        "after_main_max_abs_error": after_main["max_abs_error"],
        "final_relative_l2": final_eval["relative_l2"],
        "final_max_abs_error": final_eval["max_abs_error"],
        "metrics_path": m["_path"],
    }


def aggregate_cost(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["category"] == "cost":
            grouped[str(row["variant"])].append(row)
    out = []
    for variant, items in sorted(grouped.items()):
        out.append(
            {
                "variant": variant,
                "n_seeds": len(items),
                "trainable_params": items[0]["total_trainable_params"],
                "subnet_params": items[0]["subnet_trainable_params"],
                "param_overhead_pct": items[0]["param_overhead_pct"],
                "main_ms_per_step_mean": 1000.0 * mean(i["main_s_per_step"] for i in items),
                "main_ms_per_step_std": 1000.0 * std(i["main_s_per_step"] for i in items),
                "adam_total_min_mean": mean(i["adam_total_s"] for i in items) / 60.0,
                "adam_total_min_std": std(i["adam_total_s"] for i in items) / 60.0,
                "training_total_min_mean": mean(i["training_total_s"] for i in items) / 60.0,
                "training_total_min_std": std(i["training_total_s"] for i in items) / 60.0,
                "peak_gpu_memory_mb_mean": mean(
                    i["peak_gpu_memory_mb"]
                    for i in items
                    if i["peak_gpu_memory_mb"] is not None
                ),
                "final_relative_l2_mean": mean(i["final_relative_l2"] for i in items),
                "final_relative_l2_std": std(i["final_relative_l2"] for i in items),
                "final_max_abs_error_mean": mean(i["final_max_abs_error"] for i in items),
                "final_max_abs_error_std": std(i["final_max_abs_error"] for i in items),
            }
        )
    return out


def aggregate_sensitivity(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["category"] == "sensitivity":
            grouped[str(row["variant"])].append(row)
    out = []
    for variant, items in sorted(grouped.items()):
        first = items[0]
        if int(first["subnet_update_every"] or 1) > 1:
            ratio = f"1:{int(first['subnet_update_every'])}"
        else:
            ratio = f"{int(first['subnet_updates_per_backbone'] or 1)}:1"
        out.append(
            {
                "axis": first["sensitivity_axis"],
                "variant": variant,
                "n_seeds": len(items),
                "subnet_architecture": f"{first['subnet_depth']}x{first['subnet_width']}",
                "lr_backbone": first["lr_backbone"],
                "lr_subnet": first["lr_subnet"],
                "subnet_to_backbone_updates": ratio,
                "trainable_params": first["total_trainable_params"],
                "after_main_relative_l2_mean": mean(i["after_main_relative_l2"] for i in items),
                "after_main_relative_l2_std": std(i["after_main_relative_l2"] for i in items),
                "after_main_max_abs_error_mean": mean(i["after_main_max_abs_error"] for i in items),
                "after_main_max_abs_error_std": std(i["after_main_max_abs_error"] for i in items),
                "training_total_min_mean": mean(i["training_total_s"] for i in items) / 60.0,
            }
        )
    baseline = next((r for r in out if str(r["variant"]).startswith("baseline_")), None)
    if baseline is not None and baseline["after_main_relative_l2_mean"] != 0:
        base = baseline["after_main_relative_l2_mean"]
        for row in out:
            row["relative_l2_change_vs_baseline_pct"] = 100.0 * (row["after_main_relative_l2_mean"] / base - 1.0)
    else:
        for row in out:
            row["relative_l2_change_vs_baseline_pct"] = math.nan
    return out


def cost_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {str(r["variant"]).upper(): r for r in rows}
    if "PINN" not in by_name or "DASA" not in by_name:
        return []
    p = by_name["PINN"]
    d = by_name["DASA"]
    def pct(new: float, old: float) -> float:
        return 100.0 * (new / old - 1.0) if old != 0 else math.nan
    return [{
        "comparison": "DASA_vs_PINN",
        "parameter_overhead_pct": pct(d["trainable_params"], p["trainable_params"]),
        "main_time_overhead_pct": pct(d["main_ms_per_step_mean"], p["main_ms_per_step_mean"]),
        "adam_total_time_overhead_pct": pct(d["adam_total_min_mean"], p["adam_total_min_mean"]),
        "full_training_time_overhead_pct": pct(d["training_total_min_mean"], p["training_total_min_mean"]),
        "relative_l2_change_pct": pct(d["final_relative_l2_mean"], p["final_relative_l2_mean"]),
        "relative_l2_reduction_pct": -pct(d["final_relative_l2_mean"], p["final_relative_l2_mean"]),
        "max_error_change_pct": pct(d["final_max_abs_error_mean"], p["final_max_abs_error_mean"]),
    }]


def cost_delta_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# DASA vs PINN deltas\n\nInsufficient completed runs.\n"
    r = rows[0]
    return (
        "# DASA vs PINN deltas\n\n"
        f"- Parameter overhead: {r['parameter_overhead_pct']:.4f}%\n"
        f"- Main-stage time overhead: {r['main_time_overhead_pct']:.4f}%\n"
        f"- Adam total-time overhead: {r['adam_total_time_overhead_pct']:.4f}%\n"
        f"- Full training-time overhead: {r['full_training_time_overhead_pct']:.4f}%\n"
        f"- Relative L2 reduction: {r['relative_l2_reduction_pct']:.4f}%\n"
        f"- Maximum-error change: {r['max_error_change_pct']:.4f}%\n"
    )


def sensitivity_overview_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# Sensitivity overview\n\nNo completed sensitivity runs.\n"
    baseline = next((r for r in rows if str(r["variant"]).startswith("baseline_")), None)
    if baseline is None:
        return "# Sensitivity overview\n\nBaseline run is missing.\n"
    best = min(rows, key=lambda r: r["after_main_relative_l2_mean"])
    worst = max(rows, key=lambda r: r["after_main_relative_l2_mean"])
    return (
        "# Sensitivity overview\n\n"
        f"- Baseline relative L2: {baseline['after_main_relative_l2_mean']:.6e}\n"
        f"- Best variant: {best['variant']} ({best['after_main_relative_l2_mean']:.6e}, "
        f"{best['relative_l2_change_vs_baseline_pct']:+.2f}% vs baseline)\n"
        f"- Worst variant: {worst['variant']} ({worst['after_main_relative_l2_mean']:.6e}, "
        f"{worst['relative_l2_change_vs_baseline_pct']:+.2f}% vs baseline)\n"
        f"- Tested-range max/min ratio: {worst['after_main_relative_l2_mean'] / best['after_main_relative_l2_mean']:.4f}\n"
    )


def cost_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Cost comparison",
        "",
        "| Method | Seeds | Trainable params | Added subnet params | Param overhead | Main Adam time/step (ms) | Adam total (min) | Full training total (min) | Final relative L2 | Max error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {variant} | {n_seeds} | {params} | {subnet} | {over}% | {step} ± {step_std} | "
            "{adam} ± {adam_std} | {total} ± {total_std} | {l2} ± {l2_std} | {mx} ± {mx_std} |".format(
                variant=r["variant"],
                n_seeds=r["n_seeds"],
                params=r["trainable_params"],
                subnet=r["subnet_params"],
                over=fmt(r["param_overhead_pct"], 3),
                step=fmt(r["main_ms_per_step_mean"], 5),
                step_std=fmt(r["main_ms_per_step_std"], 3),
                adam=fmt(r["adam_total_min_mean"], 5),
                adam_std=fmt(r["adam_total_min_std"], 3),
                total=fmt(r["training_total_min_mean"], 5),
                total_std=fmt(r["training_total_min_std"], 3),
                l2=fmt(r["final_relative_l2_mean"], 5),
                l2_std=fmt(r["final_relative_l2_std"], 3),
                mx=fmt(r["final_max_abs_error_mean"], 5),
                mx_std=fmt(r["final_max_abs_error_std"], 3),
            )
        )
    return "\n".join(lines) + "\n"


def sensitivity_markdown(rows: List[Dict[str, Any]]) -> str:
    order = {"architecture": 0, "learning_rate": 1, "update_frequency": 2, "baseline": 3}
    rows = sorted(rows, key=lambda r: (order.get(str(r["axis"]), 99), str(r["variant"])))
    lines = [
        "# DASA sensitivity on L-shaped Poisson",
        "",
        "All errors are measured immediately after the adversarial/main stage, before unweighted fine-tuning and L-BFGS.",
        "",
        "| Axis | Variant | Seeds | Subnet depth×width | Backbone LR | Subnet LR | Subnet:backbone updates | Params | Relative L2 | ΔL2 vs baseline | Max error | Total time (min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {axis} | {variant} | {n} | {arch} | {lrb:.1e} | {lrs:.1e} | {ratio} | {params} | {l2} ± {l2s} | {delta:+.2f}% | {mx} ± {mxs} | {time} |".format(
                axis=r["axis"],
                variant=r["variant"],
                n=r["n_seeds"],
                arch=r["subnet_architecture"],
                lrb=float(r["lr_backbone"]),
                lrs=float(r["lr_subnet"]),
                ratio=r["subnet_to_backbone_updates"],
                params=r["trainable_params"],
                l2=fmt(r["after_main_relative_l2_mean"], 5),
                l2s=fmt(r["after_main_relative_l2_std"], 3),
                delta=float(r["relative_l2_change_vs_baseline_pct"]),
                mx=fmt(r["after_main_max_abs_error_mean"], 5),
                mxs=fmt(r["after_main_max_abs_error_std"], 3),
                time=fmt(r["training_total_min_mean"], 5),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("summaries"))
    args = parser.parse_args()

    metrics = load_metrics(args.results)
    flat = [flatten_record(m) for m in metrics]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "all_runs.csv", flat)

    cost = aggregate_cost(flat)
    sens = aggregate_sensitivity(flat)
    deltas = cost_deltas(cost)
    write_csv(args.out / "cost_summary.csv", cost)
    write_csv(args.out / "cost_deltas.csv", deltas)
    write_csv(args.out / "sensitivity_summary.csv", sens)
    (args.out / "cost_summary.md").write_text(cost_markdown(cost), encoding="utf-8")
    (args.out / "cost_deltas.md").write_text(cost_delta_markdown(deltas), encoding="utf-8")
    (args.out / "sensitivity_summary.md").write_text(
        sensitivity_markdown(sens), encoding="utf-8"
    )
    (args.out / "sensitivity_overview.md").write_text(
        sensitivity_overview_markdown(sens), encoding="utf-8"
    )
    print(f"Aggregated {len(metrics)} runs into {args.out}")


if __name__ == "__main__":
    main()
