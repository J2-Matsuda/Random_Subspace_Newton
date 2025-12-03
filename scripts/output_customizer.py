"""Plot selected columns from multiple experiment CSVs on shared axes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay metrics from multiple CSV logs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML file specifying series and plotting parameters.",
    )
    return parser.parse_args()


def load_series(csv_path: Path, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if x_key not in row or y_key not in row:
                raise KeyError(f"Missing column(s) {x_key},{y_key} in {csv_path}")
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]))
    return np.asarray(xs), np.asarray(ys)


def parse_series_args(series_args: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in series_args:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        mapping[label.strip()] = Path(path.strip())
    return mapping


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    metric = cfg.get("metric", "grad_norm")
    x_key = cfg.get("x", "k")
    logy = bool(cfg.get("logy", False))
    output = cfg.get("output", "comparison.png")
    title = cfg.get("title")

    raw_series = cfg.get("series", [])
    series_list: List[Tuple[str, Path, str | None]] = []
    if isinstance(raw_series, list):
        for item in raw_series:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            path = item.get("path")
            if not label or not path:
                continue
            style = item.get("style") or item.get("linestyle")
            series_list.append((label, Path(path), style))
    else:
        series_map = parse_series_args(raw_series or [])
        series_list.extend((label, path, None) for label, path in series_map.items())

    if not series_list:
        raise ValueError("No series specified in config.")

    plt.figure()
    for label, csv_path, style in series_list:
        xs, ys = load_series(csv_path, x_key, metric)
        plot_kwargs = {"label": label}
        if style:
            plot_kwargs["linestyle"] = style
        if logy:
            plt.semilogy(xs, ys, **plot_kwargs)
        else:
            plt.plot(xs, ys, **plot_kwargs)

    plt.xlabel(x_key)
    plt.ylabel(metric)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if title:
        plt.title(title)
    plt.legend()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
