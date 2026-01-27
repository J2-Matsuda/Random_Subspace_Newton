from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from arsn.inner.rk import compute_yk_rk, warm_start_y
from arsn.linalg.gaussian_sketch import GaussianSketchOperator
from arsn.linalg.linear_operator_utils import safe_norm
from arsn.problems.base import Problem
from arsn.regularizer.diag_shift import solve_diagonal_shift


@dataclass
class ARSNResult:
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


def _build_phpt(
    problem: Problem,
    xk: NDArray[np.floating],
    y_hat: NDArray[np.floating],
    S: GaussianSketchOperator,
) -> NDArray[np.floating]:
    m = 1 + S.shape[0]
    A = np.empty((m, m), dtype=float)

    def p_rmatvec(u: NDArray[np.floating]) -> NDArray[np.floating]:
        return y_hat * float(u[0]) + S.rmatvec(u[1:])

    def p_matvec(v: NDArray[np.floating]) -> NDArray[np.floating]:
        out = np.empty(m, dtype=float)
        out[0] = float(y_hat @ v)
        out[1:] = S.matvec(v)
        return out

    for j in range(m):
        e = np.zeros(m, dtype=float)
        e[j] = 1.0
        v = p_rmatvec(e)
        w = problem.hvp(xk, v)
        A[:, j] = p_matvec(w)

    return 0.5 * (A + A.T)


def run_arsn(
    problem: Problem,
    config: Dict[str, Any],
    *,
    logger: Optional[Any] = None,
) -> ARSNResult:
    arsn_cfg = config.get("arsn", {})
    max_iter = int(arsn_cfg.get("max_iter", 200))
    tol_grad = float(arsn_cfg.get("tol_grad", 1e-6))
    s = int(arsn_cfg.get("s", 20))
    beta = float(arsn_cfg.get("beta", 1e-4))
    tau = float(arsn_cfg.get("tau", 0.5))
    alpha0 = float(arsn_cfg.get("alpha0", 1.0))
    max_ls_iters = int(arsn_cfg.get("max_ls_iters", 50))
    seed_outer = int(arsn_cfg.get("seed", 0))
    verbose = bool(arsn_cfg.get("verbose", True))
    print_every = int(arsn_cfg.get("print_every", 10))

    inner_root = config.get("inner", {})
    inner_method = str(inner_root.get("method", "rk")).lower()
    if inner_method != "rk":
        raise ValueError(f"Only inner.method='rk' is supported, got {inner_method}")
    inner_cfg = inner_root.get("rk", {})
    T = int(inner_cfg.get("T", 10))
    r = int(inner_cfg.get("r", 10))
    ridge = float(inner_cfg.get("ridge", 1e-12))
    minres_tol = float(inner_cfg.get("minres_tol", 1e-6))
    minres_maxit = int(inner_cfg.get("minres_maxit", 200))
    warm_start = bool(inner_cfg.get("warm_start", True))
    if s <= 0:
        raise ValueError("arsn.s must be positive.")
    if T <= 0 or r <= 0:
        raise ValueError("inner.rk.T and inner.rk.r must be positive.")

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

    y0 = init_cfg.get("y0", None)
    if y0 is None:
        seed_y, seed_iter = seed_seq.spawn(2)
        rng_init = np.random.default_rng(seed_y)
        y_init = rng_init.normal(size=problem.n).astype(float)
        iter_seqs = seed_iter.spawn(max_iter)
    else:
        y_init = np.array(y0, dtype=float).reshape(-1)
        if y_init.shape[0] != problem.n:
            raise ValueError(f"y0 has dimension {y_init.shape[0]}, but problem.n={problem.n}")
        iter_seqs = seed_seq.spawn(max_iter)

    prob = _CountingProblem(problem)
    xk = x0.copy()
    yk = y_init.copy()

    x_prev: NDArray[np.floating] | None = None
    g_prev: NDArray[np.floating] | None = None
    y_prev: NDArray[np.floating] | None = None

    history: list[dict[str, Any]] = []
    hvp_calls_cum = 0
    run_start = time.perf_counter()

    if verbose:
        print(
            f"[ARSN] start: n={problem.n}, s={s}, inner=rk, reg=diag_shift, "
            f"sketch_mode={sketch_mode}"
        )

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
                print(f"[ARSN] converged at k={k}: f={fk:.6e}, ||g||={ng:.3e}")
            break

        seq_iter = iter_seqs[k]
        seq_S, seq_R = seq_iter.spawn(2)
        seed_Sk = int(seq_S.generate_state(1)[0])
        seed_Rk_base = int(seq_R.generate_state(1)[0])

        if warm_start and x_prev is not None and g_prev is not None and y_prev is not None:
            y_start = warm_start_y(prob, xk, x_prev, g_prev, y_prev)
        else:
            y_start = yk.copy()

        yk, inner_info = compute_yk_rk(
            prob,
            xk,
            gk,
            y_start,
            T=T,
            r=r,
            ridge=ridge,
            minres_tol=minres_tol,
            minres_maxit=minres_maxit,
            seed_base=seed_Rk_base,
            sketch_mode=sketch_mode,
            sketch_block_size=sketch_block_size,
            sketch_dtype=sketch_dtype,
        )

        y_norm = safe_norm(yk) + 1e-30
        y_hat = yk / y_norm

        S = GaussianSketchOperator(
            shape=(s, problem.n),
            scale=1.0 / math.sqrt(max(1, s)),
            seed=seed_Sk,
            mode=sketch_mode,
            block_size=sketch_block_size,
            dtype=sketch_dtype,
        )

        A = _build_phpt(prob, xk, y_hat, S)
        g_sub = np.empty(s + 1, dtype=float)
        g_sub[0] = float(y_hat @ gk)
        g_sub[1:] = S.matvec(gk)

        u, reg_info = solve_diagonal_shift(A, g_sub, ng, c1=c1, c2=c2, gamma=gamma)
        dk = y_hat * float(u[0]) + S.rmatvec(u[1:])
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

        Hy = prob.hvp(xk, yk)
        b = gk / (ng + 1e-30)
        yk_resid_rel = safe_norm(Hy - b) / (safe_norm(b) + 1e-30)

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
            "yk_resid_rel": yk_resid_rel,
            "inner_T": inner_info.get("inner_T"),
            "inner_r": inner_info.get("inner_r"),
            "inner_minres_total_iters": inner_info.get("inner_minres_total_iters"),
            "eta": reg_info.get("eta"),
            "lambda_min_PHPT": reg_info.get("lambda_min_PHPT"),
            "Lambda_shift": reg_info.get("Lambda_shift"),
            "seed_outer": seed_outer,
            "seed_Sk": seed_Sk,
            "seed_Rk_base": seed_Rk_base,
        }
        history.append(row)
        if logger is not None:
            logger.log(row)

        if verbose and (k % max(1, print_every) == 0):
            print(
                f"[k={k:04d}] f={fk:.6e} ||g||={ng:.3e} alpha={alpha:.2e} g^T d={gTd:.3e}"
            )

        x_prev = xk
        g_prev = gk
        y_prev = yk
        xk = x_next

    elapsed_sec = time.perf_counter() - run_start
    return ARSNResult(x_final=xk, history=history, status=status, iters=len(history), elapsed_sec=elapsed_sec)
