"""Plot grad_norm vs cumulative CPU time from multiple CSV logs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grad_norm vs CPU time.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config describing the CSV series and output path.",
    )
    return parser.parse_args()


def cumulative_cpu_series(
    csv_path: Path, grad_key: str, cpu_key: str
) -> tuple[np.ndarray, np.ndarray]:
    cpu_times: List[float] = []
    grads: List[float] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        has_cpu = cpu_key in fieldnames
        if grad_key not in fieldnames:
            raise KeyError(f"Missing required column {grad_key} in {csv_path}")
        fallback_value = 1.0
        warned = False
        for row in reader:
            grads.append(float(row[grad_key]))
            if has_cpu:
                cpu_times.append(float(row.get(cpu_key, 0.0) or 0.0))
            else:
                cpu_times.append(fallback_value)
                if not warned:
                    print(
                        f"[WARN] {csv_path} lacks '{cpu_key}'; "
                        f"using unit step fallback.",
                        file=sys.stderr,
                    )
                    warned = True
    cpu_cum = np.cumsum(np.asarray(cpu_times))
    return cpu_cum, np.asarray(grads)


def parse_series(series_cfg: Iterable[dict]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in series_cfg:
        if not isinstance(item, dict) or "label" not in item or "path" not in item:
            raise ValueError("Each series entry must have 'label' and 'path'.")
        mapping[item["label"]] = Path(item["path"])
    return mapping


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    series_cfg = cfg.get("series")
    if not series_cfg:
        raise ValueError("Config must contain a non-empty 'series' list.")
    series_map = parse_series(series_cfg)
    grad_key = cfg.get("metric", "grad_norm")
    cpu_key = cfg.get("cpu_column", "cpu_time")
    output = cfg.get("output", "plots/cpu_vs_grad.png")
    title = cfg.get("title")

    plt.figure()
    for label, csv_path in series_map.items():
        cpu, grad = cumulative_cpu_series(csv_path, grad_key, cpu_key)
        plt.semilogy(cpu, grad, label=label)

    plt.xlabel("cumulative CPU time [s]")
    plt.ylabel(grad_key)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if title:
        plt.title(title)
    plt.legend()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved CPU vs {grad_key} plot to {out_path.resolve()}")


if __name__ == "__main__":
    # ensure project root is importable when running as script
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    main()
