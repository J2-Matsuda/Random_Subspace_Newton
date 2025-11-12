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
    alpha = params.alpha0
    c1 = params.c1
    rho = params.rho
    if fx is None:
        fx = float(value_fn(x))
    directional = float(grad @ direction)
    if directional >= 0:
        # Guard against non-descent directions
        directional = directional - abs(directional) - 1e-12
    min_alpha = 1e-16
    while alpha > min_alpha:
        candidate = x + alpha * direction
        f_candidate = float(value_fn(candidate))
        if f_candidate <= fx + c1 * alpha * directional:
            return alpha, f_candidate
        alpha *= rho
    return 0.0, fx


def rk_residual(hessian: Array, y_vec: Array, grad_normed: Array) -> float:
    return norm(hessian @ y_vec - grad_normed)
