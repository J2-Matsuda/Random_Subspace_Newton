from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator


def safe_norm(x: NDArray[np.floating]) -> float:
    return float(np.linalg.norm(x))


def make_linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    rmatvec: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    *,
    dtype: np.dtype | type = float,
) -> LinearOperator:
    return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=dtype)
