from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arsn import run_arsn, run_rsrnm
from arsn.problems import LogisticRegressionProblem, QuadraticProblem
from arsn.utils import CSVLogger, load_config, resolve_path, save_config_copy


def build_problem(cfg: dict[str, Any], base_dir: Path):
    prob_cfg = cfg.get("problem", {})
    name = str(prob_cfg.get("name", "")).lower()
    data_path = resolve_path(prob_cfg.get("data_path", None), base_dir)
    if data_path is None:
        raise ValueError("problem.data_path is required.")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if name in ("quadratic", "quad"):
        return QuadraticProblem.from_npz(data_path)
    if name in ("logistic", "logistic_regression", "lr"):
        reg_lambda = prob_cfg.get("reg_lambda", None)
        return LogisticRegressionProblem.from_npz(data_path, reg_lambda=reg_lambda)

    raise ValueError(f"Unknown problem.name: {name}")


def _get_record_columns(cfg: dict[str, Any]):
    if "record_columns" in cfg:
        return cfg["record_columns"]
    return cfg.get("logging", {}).get("record_columns", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARSN experiment (RK + Diagonal Shift)")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    base_dir = config_path.parent
    problem = build_problem(cfg, base_dir)

    algo_cfg = cfg.get("algorithm", {})
    algo_name = str(algo_cfg.get("name", "arsn")).lower()

    output_cfg = cfg.get("output", {})
    out_dir = resolve_path(output_cfg.get("dir", "outputs"), base_dir) or Path("outputs")
    run_name = output_cfg.get("run_name", config_path.stem)
    csv_name = output_cfg.get("csv_name", "result.csv")

    out_dir = Path(out_dir) / str(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(out_dir / csv_name, record_columns=_get_record_columns(cfg))
    if algo_name == "arsn":
        res = run_arsn(problem, cfg, logger=logger)
    elif algo_name == "rsrnm":
        if "inner" in cfg:
            print("[warn] inner config is ignored for RSRNM.")
        res = run_rsrnm(problem, cfg, logger=logger)
    else:
        raise ValueError(f"Unknown algorithm.name: {algo_name}")
    logger.close()

    save_config_copy(cfg, out_dir / "config_copy.yml")

    summary = {
        "status": res.status,
        "iters": res.iters,
        "elapsed_sec": res.elapsed_sec,
        "x_final_norm": float((res.x_final @ res.x_final) ** 0.5),
        "output_csv": str(out_dir / csv_name),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved CSV to: {out_dir / csv_name}")
    print(f"Saved config copy to: {out_dir / 'config_copy.yml'}")
    print(f"Saved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
