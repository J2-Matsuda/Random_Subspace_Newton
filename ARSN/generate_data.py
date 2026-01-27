from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from arsn.utils.config import load_config


def _save_matrix_npz(name: str, mat) -> dict[str, Any]:
    if sparse.issparse(mat):
        mat = mat.tocsr()
        return {
            f"{name}_data": mat.data,
            f"{name}_indices": mat.indices,
            f"{name}_indptr": mat.indptr,
            f"{name}_shape": np.array(mat.shape, dtype=int),
        }
    return {name: np.asarray(mat)}


def _resolve_output(out: str, out_dir: str | None, base_dir: Path) -> Path:
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    if out_dir is None:
        return (base_dir.parent / out_path).resolve()
    out_dir_path = Path(out_dir)
    if out_dir_path.is_absolute():
        return out_dir_path / out_path
    cand1 = (base_dir / out_dir_path).resolve()
    cand2 = (base_dir.parent / out_dir_path).resolve()
    prefer_parent = base_dir.name == "configs"
    order = [cand2, cand1] if prefer_parent else [cand1, cand2]
    for cand in order:
        if cand.exists():
            return cand / out_path
    return order[0] / out_path


def generate_quadratic(
    out: Path,
    n: int,
    *,
    cond: float = 100.0,
    seed: int = 0,
    indefinite: bool = False,
    sparse_matrix: bool = False,
    dtype: str | np.dtype = "float64",
) -> None:
    rng = np.random.default_rng(seed)
    dtype_np = np.dtype(dtype)

    if sparse_matrix:
        diag = rng.uniform(1.0, cond, size=n).astype(dtype_np, copy=False)
        if indefinite:
            signs = rng.choice([-1.0, 1.0], size=n)
            diag = diag * signs
        Q = sparse.diags(diag, format="csr", dtype=dtype_np)
    else:
        Q = rng.standard_normal(size=(n, n)).astype(dtype_np, copy=False)
        Q, _ = np.linalg.qr(Q)
        eigvals = np.logspace(0.0, np.log10(cond), n).astype(dtype_np, copy=False)
        if indefinite:
            signs = np.ones(n, dtype=dtype_np)
            signs[: n // 2] = -1.0
            rng.shuffle(signs)
            eigvals = eigvals * signs
        Q = Q @ np.diag(eigvals) @ Q.T
        Q = Q.astype(dtype_np, copy=False)

    b = rng.standard_normal(size=n).astype(dtype_np, copy=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _save_matrix_npz("Q", Q)
    payload["b"] = b
    np.savez(out, **payload)


def generate_logistic(
    out: Path,
    m: int,
    n: int,
    *,
    reg_lambda: float = 1.0e-3,
    seed: int = 0,
    beta: np.ndarray | None = None,
    beta_seed: int | None = None,
    beta_scale: float = 1.0,
    x_scale: float = 1.0,
    sparse_matrix: bool = False,
    density: float = 0.05,
    labels01: bool = False,
    dtype: str | np.dtype = "float64",
) -> None:
    seed_seq = np.random.SeedSequence(seed)
    seeds = seed_seq.spawn(2)
    rng_x = np.random.default_rng(seeds[0])
    rng_y = np.random.default_rng(seeds[1])
    dtype_np = np.dtype(dtype)

    if sparse_matrix:
        rng_rs = np.random.RandomState(seed)
        A = sparse.random(
            m,
            n,
            density=float(density),
            random_state=rng_rs,
            format="csr",
            data_rvs=rng_rs.standard_normal,
        ).astype(dtype_np, copy=False)
    else:
        A = rng_x.standard_normal(size=(m, n)).astype(dtype_np, copy=False) * float(x_scale)

    if beta is None:
        if beta_seed is not None:
            rng_beta = np.random.default_rng(beta_seed)
        else:
            rng_beta = np.random.default_rng(seed_seq.spawn(1)[0])
        beta = rng_beta.standard_normal(size=n).astype(dtype_np, copy=False)
        beta = beta * float(beta_scale)
    else:
        beta = np.asarray(beta, dtype=dtype_np).reshape(-1)
        if beta.shape[0] != n:
            raise ValueError(f"beta has dimension {beta.shape[0]}, expected n={n}")

    logits = A.dot(beta) if sparse.issparse(A) else A @ beta
    logits = np.clip(logits, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    y01 = rng_y.binomial(1, p).astype(dtype_np, copy=False)

    if labels01:
        y = y01
    else:
        y = 2.0 * y01 - 1.0

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _save_matrix_npz("A", A)
    payload["y"] = y
    payload["reg_lambda"] = float(reg_lambda)
    payload["beta_true"] = beta
    np.savez(out, **payload)


def _run_from_config(config_path: Path) -> None:
    cfg = load_config(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    data_cfg = cfg.get("data", None)
    if data_cfg is None:
        raise ValueError("Config must contain a top-level 'data' section.")

    out_dir = data_cfg.get("out_dir", ".")
    overwrite = bool(data_cfg.get("overwrite", False))
    datasets = data_cfg.get("datasets", [])
    if not datasets:
        raise ValueError("data.datasets must be a non-empty list.")

    for ds in datasets:
        name = ds.get("name", "<unnamed>")
        ds_type = str(ds.get("type", "")).lower()
        out = ds.get("out", None)
        if out is None:
            raise ValueError(f"Dataset '{name}' is missing required field: out")

        out_path = _resolve_output(str(out), out_dir, config_path.parent)
        if out_path.exists() and not overwrite:
            print(f"[skip] {name}: {out_path} exists (overwrite=false)")
            continue

        dtype = ds.get("dtype", "float64")

        if ds_type == "quadratic":
            n = int(ds.get("n", 0))
            if n <= 0:
                raise ValueError(f"Dataset '{name}' requires n > 0")
            generate_quadratic(
                out_path,
                n,
                cond=float(ds.get("cond", 100.0)),
                seed=int(ds.get("seed", 0)),
                indefinite=bool(ds.get("indefinite", False)),
                sparse_matrix=bool(ds.get("sparse", False)),
                dtype=dtype,
            )
            print(f"[ok] {name}: quadratic n={n} -> {out_path}")
        elif ds_type == "logistic":
            m = int(ds.get("m", 0))
            n = int(ds.get("n", 0))
            if m <= 0 or n <= 0:
                raise ValueError(f"Dataset '{name}' requires m > 0 and n > 0")
            beta = ds.get("beta", None)
            generate_logistic(
                out_path,
                m,
                n,
                reg_lambda=float(ds.get("reg_lambda", 1.0e-3)),
                seed=int(ds.get("seed", 0)),
                beta=beta,
                beta_seed=ds.get("beta_seed", None),
                beta_scale=float(ds.get("beta_scale", 1.0)),
                x_scale=float(ds.get("x_scale", 1.0)),
                sparse_matrix=bool(ds.get("sparse", False)),
                density=float(ds.get("density", 0.05)),
                labels01=bool(ds.get("labels01", False)),
                dtype=dtype,
            )
            print(f"[ok] {name}: logistic m={m} n={n} -> {out_path}")
        else:
            raise ValueError(f"Unknown dataset type '{ds_type}' for '{name}'")


def _parse_cli_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    args, remaining = config_parser.parse_known_args()
    if args.config:
        if remaining:
            raise ValueError("When using --config, do not pass other CLI arguments.")
        return args

    parser = argparse.ArgumentParser(description="Generate synthetic data for ARSN experiments.")
    subparsers = parser.add_subparsers(dest="problem", required=True)

    quad = subparsers.add_parser("quadratic", help="Generate quadratic data")
    quad.add_argument("--out", required=True, help="Output .npz path")
    quad.add_argument("--n", type=int, required=True, help="Dimension n")
    quad.add_argument("--cond", type=float, default=100.0, help="Condition number (approx)")
    quad.add_argument("--indefinite", action="store_true", help="Make Q indefinite")
    quad.add_argument("--sparse", action="store_true", help="Use sparse diagonal Q")
    quad.add_argument("--seed", type=int, default=0, help="Random seed")
    quad.add_argument("--dtype", default="float64", help="Data type (e.g., float32, float64)")

    logi = subparsers.add_parser("logistic", help="Generate logistic regression data")
    logi.add_argument("--out", required=True, help="Output .npz path")
    logi.add_argument("--m", type=int, required=True, help="Number of samples")
    logi.add_argument("--n", type=int, required=True, help="Number of features")
    logi.add_argument("--reg_lambda", type=float, default=1e-3, help="L2 regularization")
    logi.add_argument("--beta_seed", type=int, default=None, help="Seed for beta generation")
    logi.add_argument("--beta_scale", type=float, default=1.0, help="Scale for beta magnitude")
    logi.add_argument("--x_scale", type=float, default=1.0, help="Scale for feature magnitude")
    logi.add_argument("--beta", default=None, help="Comma-separated beta values (overrides beta_seed)")
    logi.add_argument("--sparse", action="store_true", help="Use sparse design matrix")
    logi.add_argument("--density", type=float, default=0.05, help="Density for sparse A")
    logi.add_argument("--labels01", action="store_true", help="Use labels in {0,1}")
    logi.add_argument("--seed", type=int, default=0, help="Random seed")
    logi.add_argument("--dtype", default="float64", help="Data type (e.g., float32, float64)")

    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if getattr(args, "config", None):
        _run_from_config(Path(args.config).resolve())
        return

    if args.problem == "quadratic":
        generate_quadratic(
            Path(args.out),
            args.n,
            cond=float(args.cond),
            seed=int(args.seed),
            indefinite=bool(args.indefinite),
            sparse_matrix=bool(args.sparse),
            dtype=args.dtype,
        )
        print(f"Saved quadratic data to: {args.out}")
    elif args.problem == "logistic":
        beta = None
        if args.beta:
            beta = np.array([float(x) for x in str(args.beta).split(",")])
        generate_logistic(
            Path(args.out),
            args.m,
            args.n,
            reg_lambda=float(args.reg_lambda),
            seed=int(args.seed),
            beta=beta,
            beta_seed=args.beta_seed,
            beta_scale=float(args.beta_scale),
            x_scale=float(args.x_scale),
            sparse_matrix=bool(args.sparse),
            density=float(args.density),
            labels01=bool(args.labels01),
            dtype=args.dtype,
        )
        print(f"Saved logistic data to: {args.out}")
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")


if __name__ == "__main__":
    main()
