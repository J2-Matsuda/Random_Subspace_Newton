from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, minres

from arsn.linalg.gaussian_sketch import GaussianSketchOperator
from arsn.linalg.linear_operator_utils import safe_norm
from arsn.problems.base import Problem


def warm_start_y(
    problem: Problem,
    xk: NDArray[np.floating],
    x_prev: NDArray[np.floating],
    g_prev: NDArray[np.floating],
    y_prev: NDArray[np.floating],
) -> NDArray[np.floating]:
    dx = xk - x_prev
    ng_prev = safe_norm(g_prev) + 1e-30
    Hdx = problem.hvp(x_prev, dx)
    coef = float(g_prev @ Hdx) / (ng_prev**2)
    y0 = (1.0 - coef) * y_prev + (dx / ng_prev)
    return y0.astype(float, copy=False)


def compute_yk_rk(
    problem: Problem,
    xk: NDArray[np.floating],
    gk: NDArray[np.floating],
    y_init: NDArray[np.floating],
    *,
    T: int,
    r: int,
    ridge: float,
    minres_tol: float,
    minres_maxit: int,
    seed_base: int,
    sketch_mode: str,
    sketch_block_size: int,
    sketch_dtype: np.dtype,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    ng = safe_norm(gk) + 1e-30
    b = gk / ng

    y = y_init.astype(float, copy=True)
    n = y.shape[0]

    total_minres_iters = 0
    minres_fail = 0

    seed_seq = np.random.SeedSequence(seed_base)
    step_seqs = seed_seq.spawn(int(T))

    for t in range(int(T)):
        seed_R = int(step_seqs[t].generate_state(1)[0])
        R = GaussianSketchOperator(
            shape=(int(r), n),
            scale=1.0 / math.sqrt(max(1, int(r))),
            seed=seed_R,
            mode=sketch_mode,
            block_size=sketch_block_size,
            dtype=sketch_dtype,
        )

        Hy = problem.hvp(xk, y)
        rhs = R.matvec(Hy - b)

        def matvec(u: NDArray[np.floating]) -> NDArray[np.floating]:
            v = R.rmatvec(u)
            Hv = problem.hvp(xk, v)
            return R.matvec(Hv) + ridge * u

        Aop = LinearOperator((int(r), int(r)), matvec=matvec, dtype=float)
        iters = 0

        def callback(_x: NDArray[np.floating]) -> None:
            nonlocal iters
            iters += 1

        u, info = minres(Aop, rhs, tol=minres_tol, maxiter=minres_maxit, callback=callback)
        total_minres_iters += iters
        if info != 0:
            minres_fail += 1

        y = y - R.rmatvec(u)

    info = {
        "inner_T": int(T),
        "inner_r": int(r),
        "inner_minres_total_iters": int(total_minres_iters),
        "minres_fail": int(minres_fail),
    }
    return y.astype(float, copy=False), info
