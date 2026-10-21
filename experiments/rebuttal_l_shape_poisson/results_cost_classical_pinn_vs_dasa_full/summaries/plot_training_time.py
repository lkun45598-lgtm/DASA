#!/usr/bin/env python3
"""Plot matched PINN/DASA training-stage time composition on two PDEs."""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sa_pinns_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHODS = ("PINN", "DASA")
METHOD_DIRS = {"PINN": "pinn", "DASA": "dasa"}
LABELS = {"PINN": "PINN", "DASA": "DASA-PINN"}
STAGES = (
    "Warm-up (S1)",
    "Subnet pre-training (S2)",
    "Main Adam (S3)",
    "Fine-tuning (S4)",
    "L-BFGS",
)
METHOD_COLORS = {"PINN": "#1c5cab", "DASA": "#b23a2e"}
METHOD_MARKERS = {"PINN": "o", "DASA": "D"}

# Chrome / ink tokens (validated data-viz palette)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
EXPERIMENTS = {
    "L-shaped Poisson": {
        "root": ROOT.parents[1] / "results_cost_unitmean_full_seed0",
        "stage_keys": {
            "Warm-up (S1)": "warmup",
            "Subnet pre-training (S2)": "subnet_pretrain",
            "Main Adam (S3)": "main",
            "Fine-tuning (S4)": "finetune",
            "L-BFGS": "lbfgs",
        },
    },
    "Wave equation": {
        "root": (
            ROOT.parents[3]
            / "wave_equation"
            / "rebuttal_experiments"
            / "preliminary_stage4_seed0_4_20260815"
            / "merged"
            / "cost"
        ),
        "stage_keys": {
            "Warm-up (S1)": "stage1",
            "Subnet pre-training (S2)": "stage2",
            "Main Adam (S3)": "stage3",
            "Fine-tuning (S4)": "stage4",
            "L-BFGS": "lbfgs",
        },
    },
}


def load_stage_summary(config: dict[str, object]) -> dict[str, dict[str, object]]:
    root = config["root"]
    stage_keys = config["stage_keys"]
    if not isinstance(root, Path) or not isinstance(stage_keys, dict):
        raise TypeError("Invalid experiment configuration")

    summary: dict[str, dict[str, object]] = {}
    for method in METHODS:
        stage_samples = {stage: [] for stage in STAGES}
        total_samples = []
        for seed in range(3):
            metrics_path = root / f"cost_{METHOD_DIRS[method]}" / f"seed_{seed}" / "metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)

            stage_minutes = {
                stage: float(metrics["stage_timings"][stage_keys[stage]]["elapsed_s"]) / 60.0
                for stage in STAGES
            }
            recorded_total = float(metrics["timing_summary"]["training_total_s"]) / 60.0
            if abs(sum(stage_minutes.values()) - recorded_total) > 1e-7:
                raise ValueError(
                    f"Stage sum does not match total training time: {metrics_path}"
                )
            for stage, value in stage_minutes.items():
                stage_samples[stage].append(value)
            total_samples.append(recorded_total)

        summary[method] = {
            "stages": {stage: statistics.mean(values) for stage, values in stage_samples.items()},
            "stage_stds": {stage: statistics.stdev(values) for stage, values in stage_samples.items()},
            "total_mean": statistics.mean(total_samples),
            "total_std": statistics.stdev(total_samples),
        }
    return summary


def main() -> None:
    summaries = {
        equation: load_stage_summary(config)
        for equation, config in EXPERIMENTS.items()
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.05))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.29, top=0.80, wspace=0.19)
    centers = tuple(range(len(STAGES)))
    stage_labels = (
        "Warm-up",
        "Pre-train*",
        "Main Adam",
        "Fine-tune",
        "L-BFGS",
    )

    for panel, (ax, (equation, summary)) in enumerate(zip(axes, summaries.items())):
        ymax = 0.0
        for method in METHODS:
            means = [summary[method]["stages"][stage] for stage in STAGES]
            # Per-stage line (unchanged from before).
            ax.plot(
                centers,
                means,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=7.0,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#ffffff",
                markeredgewidth=1.1,
                linewidth=2.2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=LABELS[method],
                clip_on=False,
                zorder=3,
            )
            # Separate cumulative ("added-up") line, ending at the total.
            cumulative = []
            running = 0.0
            for value in means:
                running += value
                cumulative.append(running)
            total = cumulative[-1]
            ax.plot(
                centers,
                cumulative,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5.0,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
                linewidth=1.8,
                linestyle=(0, (4, 3)),
                label="_nolegend_",
                clip_on=False,
                zorder=3,
            )
            ax.annotate(
                f"{total:.1f} min",
                (centers[-1], total),
                textcoords="offset points",
                xytext=(9, 0),
                ha="left",
                va="center",
                fontsize=8.6,
                fontweight="bold",
                color=METHOD_COLORS[method],
                clip_on=False,
                zorder=4,
            )
            ymax = max(ymax, total)

        ax.set_ylim(0.0, ymax * 1.12)
        ax.set_xlim(-0.35, len(STAGES) - 1 + 1.15)
        ax.set_xticks(centers)
        ax.set_xticklabels(stage_labels, fontsize=8.2, color=INK_PRIMARY)
        ax.set_ylabel("Time (min)", fontsize=10.0, color=INK_PRIMARY)


        # Footnote + panel caption below the axis.
        ax.text(
            0.5,
            -0.155,
            "* DASA subnet only",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            color=INK_MUTED,
        )
        ax.text(
            0.5,
            -0.325,
            f"({chr(ord('a') + panel)}) {equation}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.8,
            fontweight="bold",
            color=INK_PRIMARY,
        )

        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BASELINE_COLOR)
        ax.spines["bottom"].set_color(BASELINE_COLOR)
        ax.tick_params(axis="both", length=0, labelsize=8.9, colors=INK_PRIMARY)

    # One unified legend: colour = method, line style = per-stage vs cumulative.
    legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS["PINN"],
            marker=METHOD_MARKERS["PINN"],
            markerfacecolor=METHOD_COLORS["PINN"],
            markeredgecolor="#ffffff",
            markeredgewidth=1.1,
            markersize=7.0,
            linewidth=2.2,
        ),
        Line2D(
            [0], [0],
            color=METHOD_COLORS["DASA"],
            marker=METHOD_MARKERS["DASA"],
            markerfacecolor=METHOD_COLORS["DASA"],
            markeredgecolor="#ffffff",
            markeredgewidth=1.1,
            markersize=7.0,
            linewidth=2.2,
        ),
        Line2D([0], [0], color=INK_PRIMARY, linewidth=2.2, linestyle="-"),
        Line2D([0], [0], color=INK_PRIMARY, linewidth=1.8, linestyle=(0, (4, 3))),
    ]
    legend_labels = [
        LABELS["PINN"],
        LABELS["DASA"],
        "Per-stage",
        "Cumulative (total)",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=10.0,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.8,
        labelcolor=INK_PRIMARY,
    )
    fig.savefig(ROOT / "training_time_lshape_wave_pinn_vs_dasa.png", dpi=300, facecolor="white")
    fig.savefig(ROOT / "training_time_lshape_wave_pinn_vs_dasa.pdf", facecolor="white")


if __name__ == "__main__":
    main()
