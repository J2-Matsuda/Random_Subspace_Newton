"""Mathematical helper utilities for the RS/RK algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

Array = np.ndarray
ValueFn = Callable[[Array], float]


@dataclass
class ArmijoParams:
    alpha0: float = 1.0
    c1: float = 1e-4
    rho: float = 0.5
    max_backtracks: int = 50
    min_alpha: float = 1e-16


def norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def sample_gaussian_matrix(rows: int, cols: int, variance: float) -> Array:
    std = np.sqrt(variance)
    return np.random.normal(loc=0.0, scale=std, size=(rows, cols))


def project_rows_orthogonal(matrix: Array, unit_vector: Array) -> Array:
    """Project each row of matrix onto the orthogonal complement of unit_vector."""
    u = unit_vector / (norm(unit_vector) + 1e-12)
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
    regularized = matrix + reg * np.eye(dim)
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
    t = float(params.alpha0 or 1.0)
    c1 = float(params.c1)
    rho = float(params.rho)
    fx = float(value_fn(x)) if fx is None else float(fx)

    gTd = float(grad @ direction)
    if not np.isfinite(gTd) or gTd >= 0.0:
        return 0.0, fx

    max_iters = int(getattr(params, "max_backtracks", 50))
    min_alpha = float(getattr(params, "min_alpha", 1e-16))

    for _ in range(max_iters):
        candidate = x + t * direction
        f_candidate = float(value_fn(candidate))
        if np.isfinite(f_candidate) and f_candidate <= fx + c1 * t * gTd:
            return t, f_candidate
        t *= rho
        if t < min_alpha:
            break
    return 0.0, fx


def rk_residual(hessian: Array, y_vec: Array, grad_normed: Array) -> float:
    return norm(hessian @ y_vec - grad_normed)
