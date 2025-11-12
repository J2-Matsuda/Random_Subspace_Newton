"""Metric helpers for experiment logging."""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping

import numpy as np

BASE_COLUMNS = ("k", "f", "grad_norm", "x_norm")
RS_COLUMNS = (
    "alpha",
    "subspace_dim_s",
    "inner_dim_r",
    "L",
    "rs_reg",
    "rk_reg",
    "rk_iters",
    "rk_residual",
)


def build_row(
    *,
    iteration: int,
    value: float,
    grad: np.ndarray,
    x: np.ndarray,
    extra: Mapping[str, float | int | bool] | None = None,
) -> Dict[str, float | int | bool]:
    row: MutableMapping[str, float | int | bool] = {
        "k": iteration,
        "f": float(value),
        "grad_norm": float(np.linalg.norm(grad)),
        "x_norm": float(np.linalg.norm(x)),
    }
    if extra:
        row.update(extra)
    return dict(row)
