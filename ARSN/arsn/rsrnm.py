from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from arsn.linalg.gaussian_sketch import GaussianSketchOperator
from arsn.linalg.linear_operator_utils import safe_norm
from arsn.problems.base import Problem
from arsn.regularizer.diag_shift import solve_diagonal_shift


@dataclass
class RSRNMResult:
    x_final: NDArray[np.floating]
    history: list[dict[str, Any]]
    status: str
    iters: int
    elapsed_sec: float


class _CountingProblem(Problem):
    def __init__(self, base: Problem) -> None:
        self._base = base
        self.hvp_calls = 0

    @property
    def n(self) -> int:
        return self._base.n

    def f(self, x: NDArray[np.floating]) -> float:
        return self._base.f(x)

    def grad(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._base.grad(x)

    def hvp(self, x: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
        self.hvp_calls += 1
        return self._base.hvp(x, v)


def armijo_backtracking(
    problem: Problem,
    xk: NDArray[np.floating],
    fk: float,
    gTd: float,
    dk: NDArray[np.floating],
    *,
    beta: float,
    tau: float,
    alpha0: float,
    max_iters: int,
) -> tuple[float, int]:
    alpha = float(alpha0)
    for t in range(max_iters):
        x_new = xk + alpha * dk
        f_new = problem.f(x_new)
        if f_new <= fk + beta * alpha * gTd:
            return alpha, t
        alpha *= tau
    return alpha, max_iters


def run_rsrnm(
    problem: Problem,
    config: Dict[str, Any],
    *,
    logger: Optional[Any] = None,
) -> RSRNMResult:
    rsrnm_cfg = config.get("rsrnm", {})
    if not rsrnm_cfg:
        raise ValueError("RSRNM requires a 'rsrnm' config section.")
    max_iter = int(rsrnm_cfg.get("max_iter", 200))
    tol_grad = float(rsrnm_cfg.get("tol_grad", 1e-6))
    s = int(rsrnm_cfg.get("s", 20))
    beta = float(rsrnm_cfg.get("beta", 1e-4))
    tau = float(rsrnm_cfg.get("tau", 0.5))
    alpha0 = float(rsrnm_cfg.get("alpha0", 1.0))
    max_ls_iters = int(rsrnm_cfg.get("max_ls_iters", 50))
    seed_outer = int(rsrnm_cfg.get("seed", 0))
    verbose = bool(rsrnm_cfg.get("verbose", True))
    print_every = int(rsrnm_cfg.get("print_every", 10))
    if s <= 0:
        raise ValueError("rsrnm.s must be positive.")

    reg_root = config.get("regularizer", {})
    reg_method = str(reg_root.get("method", "diag_shift")).lower()
    if reg_method not in ("diag_shift", "diagonal_shift", "ds"):
        raise ValueError(f"Only regularizer.method='diag_shift' is supported, got {reg_method}")
    reg_cfg = reg_root.get("diag_shift", {})
    c1 = float(reg_cfg.get("c1", 2.0))
    c2 = float(reg_cfg.get("c2", 1.0))
    gamma = float(reg_cfg.get("gamma", 1.0))

    sketch_cfg = config.get("sketch", {})
    sketch_mode = str(sketch_cfg.get("mode", "operator")).lower()
    if sketch_mode not in ("operator", "explicit"):
        raise ValueError(f"Unknown sketch.mode: {sketch_mode}")
    sketch_block_size = int(sketch_cfg.get("block_size", 256))
    sketch_dtype = np.dtype(sketch_cfg.get("dtype", "float64"))

    init_cfg = config.get("init", {})
    x0 = np.array(init_cfg.get("x0", [0.0] * problem.n), dtype=float).reshape(-1)
    if x0.shape[0] != problem.n:
        raise ValueError(f"x0 has dimension {x0.shape[0]}, but problem.n={problem.n}")

    seed_seq = np.random.SeedSequence(seed_outer)
    iter_seqs = seed_seq.spawn(max_iter)

    prob = _CountingProblem(problem)
    xk = x0.copy()

    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0
    run_start = time.perf_counter()

    if verbose:
        print(f"[RSRNM] start: n={problem.n}, s={s}, reg=diag_shift, sketch_mode={sketch_mode}")

    status = "max_iter_reached"
    for k in range(max_iter):
        iter_start = time.perf_counter()
        prob.hvp_calls = 0

        fk = float(prob.f(xk))
        gk = prob.grad(xk)
        ng = safe_norm(gk)

        if ng <= tol_grad:
            status = "converged"
            row = {
                "iter": k,
                "f": fk,
                "grad_norm": ng,
                "step_norm": None,
                "gTd": None,
                "alpha": None,
                "armijo_iters": None,
                "time_iter_sec": time.perf_counter() - iter_start,
                "time_cum_sec": time.perf_counter() - run_start,
                "hvp_calls_iter": prob.hvp_calls,
                "hvp_calls_cum": hvp_calls_cum,
                "yk_resid_rel": None,
                "inner_T": None,
                "inner_r": None,
                "inner_minres_total_iters": None,
                "eta": None,
                "lambda_min_PHPT": None,
                "Lambda_shift": None,
                "seed_outer": seed_outer,
                "seed_Sk": None,
                "seed_Rk_base": None,
            }
            history.append(row)
            if logger is not None:
                logger.log(row)
            if verbose:
                print(f"[RSRNM] converged at k={k}: f={fk:.6e}, ||g||={ng:.3e}")
            break

        seed_Sk = int(iter_seqs[k].generate_state(1)[0])
        S = GaussianSketchOperator(
            shape=(s, problem.n),
            scale=1.0 / math.sqrt(max(1, s)),
            seed=seed_Sk,
            mode=sketch_mode,
            block_size=sketch_block_size,
            dtype=sketch_dtype,
        )

        A = np.empty((s, s), dtype=float)
        for j in range(s):
            e = np.zeros(s, dtype=float)
            e[j] = 1.0
            v = S.rmatvec(e)
            w = prob.hvp(xk, v)
            A[:, j] = S.matvec(w)
        A = 0.5 * (A + A.T)

        g_sub = S.matvec(gk)
        u, reg_info = solve_diagonal_shift(A, g_sub, ng, c1=c1, c2=c2, gamma=gamma)
        dk = S.rmatvec(u)
        gTd = float(g_sub @ u)

        alpha, armijo_iters = armijo_backtracking(
            prob,
            xk,
            fk,
            gTd,
            dk,
            beta=beta,
            tau=tau,
            alpha0=alpha0,
            max_iters=max_ls_iters,
        )

        x_next = xk + alpha * dk
        step_norm = safe_norm(alpha * dk)

        hvp_calls_cum += prob.hvp_calls
        row = {
            "iter": k,
            "f": fk,
            "grad_norm": ng,
            "step_norm": step_norm,
            "gTd": gTd,
            "alpha": alpha,
            "armijo_iters": armijo_iters,
            "time_iter_sec": time.perf_counter() - iter_start,
            "time_cum_sec": time.perf_counter() - run_start,
            "hvp_calls_iter": prob.hvp_calls,
            "hvp_calls_cum": hvp_calls_cum,
            "yk_resid_rel": None,
            "inner_T": None,
            "inner_r": None,
            "inner_minres_total_iters": None,
            "eta": reg_info.get("eta"),
            "lambda_min_PHPT": reg_info.get("lambda_min_PHPT"),
            "Lambda_shift": reg_info.get("Lambda_shift"),
            "seed_outer": seed_outer,
            "seed_Sk": seed_Sk,
            "seed_Rk_base": None,
        }
        history.append(row)
        if logger is not None:
            logger.log(row)

        if verbose and (k % max(1, print_every) == 0):
            print(f"[k={k:04d}] f={fk:.6e} ||g||={ng:.3e} alpha={alpha:.2e} g^T d={gTd:.3e}")

        xk = x_next

    elapsed_sec = time.perf_counter() - run_start
    return RSRNMResult(x_final=xk, history=history, status=status, iters=len(history), elapsed_sec=elapsed_sec)
