"""Mathematical helper utilities for the RS/RK algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from utils.dtypes import DTYPE, as_dtype, eye, random_normal

Array = np.ndarray
ValueFn = Callable[[Array], float]


@dataclass
class ArmijoParams:
    t0: float = 1.0
    alpha: float = 1e-4
    beta: float = 0.5
    max_backtracks: int = 50
    min_alpha: float = 1e-16


def norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def sample_gaussian_matrix(rows: int, cols: int, variance: float) -> Array:
    std = np.sqrt(as_dtype(variance))
    return random_normal(0.0, std, size=(rows, cols))


def project_rows_orthogonal(matrix: Array, unit_vector: Array) -> Array:
    """Project each row of matrix onto the orthogonal complement of unit_vector."""
    u = unit_vector / (norm(unit_vector) + as_dtype(1e-12))
    projection = matrix - (matrix @ u[:, None]) * u[None, :]
    return projection


def orthonormalize_rows(matrix: Array) -> Array:
    """Return an orthonormal basis spanning the rows."""
    mat = np.asarray(matrix, dtype=float)
    mode = "reduced" if mat.shape[0] <= mat.shape[1] else "complete"
    q, _ = np.linalg.qr(mat.T, mode=mode)
    ortho = q.T[: mat.shape[0], :]
    return ortho


def solve_regularized_system(matrix: Array, rhs: Array, reg: float) -> Array:
    dim = matrix.shape[0]
    regularized = matrix + reg * eye(dim)
    try:
        return np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(regularized, rhs, rcond=None)
        return solution


def armijo_backtracking(
    value_fn: ValueFn,
    x: Array,
    direction: Array,
    grad: Array,
    params: ArmijoParams,
    fx: float | None = None,
) -> Tuple[float, float]:
    """Perform Armijo backtracking; returns (alpha, new_value)."""
    t = float(params.t0 or 1.0)
    alpha = float(params.alpha)
    beta = float(params.beta)
    fx = float(value_fn(x)) if fx is None else float(fx)

    gTd = float(grad @ direction)
    if not np.isfinite(gTd) or gTd >= 0.0:
        return 0.0, fx

    max_iters = int(params.max_backtracks)
    min_alpha = float(params.min_alpha)

    for _ in range(max_iters):
        candidate = x + t * direction
        f_candidate = float(value_fn(candidate))
        if np.isfinite(f_candidate) and f_candidate <= fx + alpha * t * gTd:
            return t, f_candidate
        t *= beta
        if t < min_alpha:
            break
    return 0.0, fx


def rk_residual(hessian: Array, y_vec: Array, grad_normed: Array) -> float:
    return norm(hessian @ y_vec - grad_normed)
