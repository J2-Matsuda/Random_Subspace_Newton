"""Plotting utilities for convergence curves."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_from_csv(
    csv_path: str | Path,
    metrics: Sequence[str],
    save_path: str | Path,
) -> Path:
    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)
    iterations: list[int] = []
    series = {metric: [] for metric in metrics}
    with csv_file.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            iterations.append(int(row["k"]))
            for metric in metrics:
                value = row.get(metric, "")
                series[metric].append(float(value) if value else np.nan)
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.semilogy(iterations, series[metric], label=metric)
    ax.set_xlabel("iteration")
    ax.set_ylabel("metric value")
    ax.set_title("Convergence")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    save_target = Path(save_path).expanduser().resolve()
    save_target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_target, bbox_inches="tight")
    plt.close(fig)
    return save_target
