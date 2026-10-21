"""Non-interactive loss and absolute-error plots."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def plot_loss(log_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, totals = [], []
    with log_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            totals.append(float(row["total"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(steps, totals, color="#2455a4", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total gPINN loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_absolute_error(
    points: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    coordinate_names: tuple[str, ...],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if prediction.shape[1] >= 2 and reference.shape[1] >= 2:
        error = np.linalg.norm(prediction[:, :2] - reference[:, :2], axis=1)
        label = "Velocity error norm"
    else:
        error = np.abs(prediction[:, 0] - reference[:, 0])
        label = "Absolute error"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if points.shape[1] == 1:
        order = np.argsort(points[:, 0])
        ax.plot(points[order, 0], error[order], color="#b7352d", linewidth=1.1)
        ax.set_xlabel(coordinate_names[0])
        ax.set_ylabel(label)
    else:
        plot = ax.scatter(
            points[:, 0], points[:, 1], c=error, s=6, cmap="magma", linewidths=0
        )
        ax.set_xlabel(coordinate_names[0])
        ax.set_ylabel(coordinate_names[1])
        fig.colorbar(plot, ax=ax, label=label)
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
