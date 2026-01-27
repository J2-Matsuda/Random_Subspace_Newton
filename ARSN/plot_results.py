from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from arsn.utils.config import load_config

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


def load_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if pd is not None:
        df = pd.read_csv(path)
        return {col: df[col].to_numpy() for col in df.columns}

    data: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for name in reader.fieldnames:
            data[name] = []
        for row in reader:
            for name in reader.fieldnames:
                val = row.get(name, "")
                if val is None or val == "":
                    data[name].append(np.nan)
                else:
                    try:
                        data[name].append(float(val))
                    except ValueError:
                        data[name].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in data.items()}


def _derive_label(path: Path, label_from: str | None) -> str:
    if label_from == "basename_dir":
        return path.parent.name
    if label_from == "basename":
        return path.stem
    if label_from == "path":
        return str(path)
    return path.stem


def _resolve_pattern(pattern: str, base_dir: Path) -> str:
    raw = Path(pattern)
    if raw.is_absolute():
        return str(raw)
    cand1 = (base_dir / raw).resolve()
    cand2 = (base_dir.parent / raw).resolve()
    prefer_parent = base_dir.name == "configs"
    if raw.parts and (base_dir.parent / raw.parts[0]).exists():
        prefer_parent = True
    order = [cand2, cand1] if prefer_parent else [cand1, cand2]
    if glob.has_magic(pattern):
        for cand in order:
            if glob.glob(str(cand)):
                return str(cand)
        return str(order[0])
    for cand in order:
        if cand.exists():
            return str(cand)
    return str(order[0])


def _expand_series(series: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    csv_spec = series.get("csv", None)
    if not csv_spec:
        raise ValueError("Each series must define 'csv'.")

    label = series.get("label", None)
    label_from = series.get("label_from", None)

    csv_spec = _resolve_pattern(str(csv_spec), base_dir)

    if glob.has_magic(str(csv_spec)):
        matches = sorted(glob.glob(str(csv_spec)))
    else:
        matches = [str(csv_spec)]

    expanded: list[dict[str, Any]] = []
    for match in matches:
        path = Path(match)
        if not path.exists():
            print(f"[warn] CSV not found: {path}", file=sys.stderr)
            continue
        series_label = label if label is not None else _derive_label(path, label_from)
        expanded.append({"path": path, "label": series_label})

    if not expanded:
        print(f"[warn] No CSV matched for series: {csv_spec}", file=sys.stderr)
    return expanded


def plot_figure(fig: dict[str, Any], base_dir: Path, out_dir: Path) -> None:
    x_col = fig.get("x", None)
    y_cols = fig.get("y", None)
    if not x_col or not y_cols:
        raise ValueError("Figure spec must include 'x' and 'y'.")
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    logy = bool(fig.get("logy", False))
    logx = bool(fig.get("logx", False))
    grid = bool(fig.get("grid", False))
    legend = bool(fig.get("legend", True))
    every = int(fig.get("every", 1))
    if every <= 0:
        raise ValueError("every must be >= 1")

    xlim = fig.get("xlim", None)
    ylim = fig.get("ylim", None)
    title = fig.get("title", None)
    xlabel = fig.get("xlabel", None)
    if xlabel is None:
        xlabel = fig.get("x_label", None)
    ylabel = fig.get("ylabel", None)
    if ylabel is None:
        ylabel = fig.get("y_label", None)
    y_labels = fig.get("y_labels", None)

    name = fig.get("name", "figure")
    save = fig.get("save", f"{name}.png")
    out_path = Path(save)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_plotted = False
    series_specs = fig.get("series", [])
    if not series_specs:
        raise ValueError(f"Figure '{name}' has no series defined.")

    y_label_map: dict[str, str] = {}
    if isinstance(y_labels, dict):
        y_label_map = {str(k): str(v) for k, v in y_labels.items()}
    elif isinstance(y_labels, list):
        if len(y_labels) != len(y_cols):
            print(
                f"[warn] y_labels length {len(y_labels)} does not match y columns "
                f"{len(y_cols)}; extra labels are ignored",
                file=sys.stderr,
            )
        for idx, y in enumerate(y_cols):
            if idx >= len(y_labels):
                break
            y_label_map[str(y)] = str(y_labels[idx])
    elif isinstance(y_labels, str) and len(y_cols) == 1:
        y_label_map[str(y_cols[0])] = y_labels
    elif y_labels is not None:
        print(
            "[warn] y_labels must be a dict, list, or string (when y has one column)",
            file=sys.stderr,
        )

    for spec in series_specs:
        for entry in _expand_series(spec, base_dir):
            data = load_csv(entry["path"])
            if x_col not in data:
                print(f"[warn] Missing column '{x_col}' in {entry['path']}", file=sys.stderr)
                continue
            missing = [y for y in y_cols if y not in data]
            if missing:
                print(
                    f"[warn] Missing columns {missing} in {entry['path']} (skipped)",
                    file=sys.stderr,
                )
                continue

            x = data[x_col][::every]
            for y in y_cols:
                y_vals = data[y][::every]
                label = entry["label"]
                y_display = y_label_map.get(y, y)
                if len(y_cols) > 1:
                    label = f"{label}:{y_display}"
                plt.plot(x, y_vals, label=label)
                any_plotted = True

    if not any_plotted:
        print(f"[warn] No series plotted for figure '{name}'", file=sys.stderr)

    if logy:
        plt.yscale("log")
    if logx:
        plt.xscale("log")
    if grid:
        plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()

    plt.xlabel(xlabel if xlabel is not None else x_col)
    plt.ylabel(ylabel if ylabel is not None else ", ".join(y_cols))
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _run_from_config(config_path: Path) -> None:
    cfg = load_config(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    plot_cfg = cfg.get("plot", None)
    if plot_cfg is None:
        raise ValueError("Config must contain a top-level 'plot' section.")

    out_dir = plot_cfg.get("out_dir", "outputs/plots")
    out_dir_path = Path(out_dir)
    if not out_dir_path.is_absolute():
        base_dir = config_path.parent
        prefer_parent = base_dir.name == "configs"
        if out_dir_path.parts and (base_dir.parent / out_dir_path.parts[0]).exists():
            prefer_parent = True
        if prefer_parent:
            out_dir_path = (base_dir.parent / out_dir_path).resolve()
        else:
            out_dir_path = (base_dir / out_dir_path).resolve()

    figures = plot_cfg.get("figures", [])
    if not figures:
        raise ValueError("plot.figures must be a non-empty list.")

    for fig in figures:
        plot_figure(fig, base_dir=config_path.parent, out_dir=out_dir_path)
        save = fig.get("save", f"{fig.get('name', 'figure')}.png")
        save_path = Path(save)
        if not save_path.is_absolute():
            save_path = out_dir_path / save_path
        print(f"[ok] saved: {save_path}")


def _parse_cli_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    args, remaining = config_parser.parse_known_args()
    if args.config:
        if remaining:
            raise ValueError("When using --config, do not pass other CLI arguments.")
        return args

    parser = argparse.ArgumentParser(description="Plot columns from ARSN CSV results.")
    parser.add_argument("--csv", required=True, help="Path to result.csv")
    parser.add_argument("--x", required=True, help="Column for x-axis")
    parser.add_argument("--y", required=True, nargs="+", help="One or more columns for y-axis")
    parser.add_argument("--logy", action="store_true", help="Log-scale y-axis")
    parser.add_argument("--logx", action="store_true", help="Log-scale x-axis")
    parser.add_argument("--out", default=None, help="Output image path (png/pdf)")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--every", type=int, default=1, help="Take every N-th row")
    parser.add_argument("--grid", action="store_true", help="Show grid")
    parser.add_argument("--legend", action="store_true", help="Show legend")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if getattr(args, "config", None):
        _run_from_config(Path(args.config).resolve())
        return

    csv_path = Path(args.csv)
    data = load_csv(csv_path)
    x_col = args.x
    y_cols = list(args.y)

    if x_col not in data:
        raise ValueError(f"Missing column '{x_col}' in {csv_path}")
    missing = [y for y in y_cols if y not in data]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data[x_col][:: max(1, int(args.every))]
    for y in y_cols:
        y_vals = data[y][:: max(1, int(args.every))]
        plt.plot(x, y_vals, label=y)

    if args.logy:
        plt.yscale("log")
    if args.logx:
        plt.xscale("log")
    if args.grid:
        plt.grid(True, alpha=0.3)
    if args.legend:
        plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(", ".join(y_cols))
    if args.title:
        plt.title(args.title)

    if args.out:
        plt.tight_layout()
        plt.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
