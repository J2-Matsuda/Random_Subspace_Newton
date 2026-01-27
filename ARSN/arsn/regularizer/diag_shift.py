from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def solve_diagonal_shift(
    A: NDArray[np.floating],
    g_sub: NDArray[np.floating],
    norm_g: float,
    *,
    c1: float,
    c2: float,
    gamma: float,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    A = 0.5 * (A + A.T)
    eigvals = np.linalg.eigvalsh(A)
    lambda_min = float(eigvals.min())
    Lambda = max(0.0, -lambda_min)
    eta = c1 * Lambda + c2 * (norm_g ** gamma)

    B = A + eta * np.eye(A.shape[0], dtype=float)
    rhs = -g_sub
    try:
        L = np.linalg.cholesky(B)
        y = np.linalg.solve(L, rhs)
        u = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        u = np.linalg.solve(B, rhs)

    info = {
        "lambda_min_PHPT": lambda_min,
        "Lambda_shift": Lambda,
        "eta": eta,
    }
    return u.astype(float, copy=False), info
