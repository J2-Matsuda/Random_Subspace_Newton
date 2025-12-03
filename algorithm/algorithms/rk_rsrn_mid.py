"""RK+RSRNM mid version following the provided algorithm description."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.dtypes import as_dtype, eye
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    norm,
    rk_residual as rk_residual_fn,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


def _lambda_shift(matrix: np.ndarray) -> float:
    """Return max(0, -lambda_min(matrix)) for stability."""
    matrix = symmetrize(matrix)
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        return float(max(0.0, -float(np.min(eigvals))))
    except np.linalg.LinAlgError:
        return 0.0


class RKRsrnMid(AlgorithmBase):
    """Implementation of the RK+RSRNM variant described in the pseudo code."""

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)

        s_rows = max(0, params.get("s", min(8, problem.dim)))
        r_dim = max(1, params.get("r", min(8, problem.dim)))
        L = max(1, params.get("L", 10))

        c1 = params.get("c1", 0.0)
        c2 = params.get("c2", 0.0)
        gamma = params.get("gamma", 1.0)
        c1_prime = params.get("c1_prime", 0.0)
        c2_prime = params.get("c2_prime", 0.0)
        gamma_prime = params.get("gamma_prime", 1.0)

        warm_start = params.get("warm_start", True)
        y0_vec = as_dtype(params.get("y0", np.zeros(problem.dim)))

        armijo_params = ArmijoParams(
            t0=params.get("t0", 1.0),
            alpha=params.get("alpha", 1e-4),
            beta=params.get("beta", 0.5),
            max_backtracks=params.get("max_backtracks", 50),
            min_alpha=params.get("min_alpha", 1e-16),
        )

        x = as_dtype(problem.initial_point())
        fx = problem.value(x)
        history = []
        converged = False
        prev_state: Optional[Dict[str, np.ndarray | float]] = None
        self._init_progress_tracker(max_iters)
        eps = as_dtype(1e-12)

        for k in range(max_iters):
            grad = as_dtype(problem.gradient(x))
            grad_norm = float(np.linalg.norm(grad))
            hessian = self._hessian(problem, x)

            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra=self._extras(
                        t_k=0.0,
                        subspace_rows=1 + s_rows,
                        r_dim=r_dim,
                        L=0,
                        lambda_k=0.0,
                        rk_iters=0,
                        rk_residual=0.0,
                    ),
                )
                logger.log(row)
                history.append(row)
                self._report_progress(k)
                break

            grad_unit = grad / (as_dtype(grad_norm) + eps)

            y_init = self._initialise_y(
                current_x=x,
                default=y0_vec,
                warm_start=warm_start,
                prev_state=prev_state,
                eps=eps,
            )

            y_vec, inner_iters, inner_residual = self._rk_inner_loop(
                y_init=y_init,
                hessian=hessian,
                grad_unit=grad_unit,
                r_dim=r_dim,
                L=L,
                grad_norm=grad_norm,
                c1_prime=c1_prime,
                c2_prime=c2_prime,
                gamma_prime=gamma_prime,
            )

            direction, P_rows, lambda_k = self._construct_direction(
                y_vec=y_vec,
                hessian=hessian,
                grad=grad,
                s_rows=s_rows,
                c1=c1,
                c2=c2,
                gamma=gamma,
                grad_norm=grad_norm,
            )

            direction = self._ensure_descent(direction, grad)

            t_k, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )

            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._extras(
                    t_k=t_k,
                    subspace_rows=P_rows,
                    r_dim=r_dim,
                    L=L,
                    lambda_k=lambda_k,
                    rk_iters=inner_iters,
                    rk_residual=inner_residual,
                ),
            )
            logger.log(row)
            history.append(row)
            self._report_progress(k)

            if t_k == 0.0:
                break

            prev_state = {
                "x": x.copy(),
                "grad": grad.copy(),
                "hessian": hessian.copy(),
                "y": y_vec.copy(),
                "grad_norm": grad_norm,
            }

            x = x + t_k * direction
            fx = new_fx

        logger.flush()
        final_grad = problem.gradient(x)
        return AlgorithmResult(
            x=x,
            f=problem.value(x),
            grad_norm=float(np.linalg.norm(final_grad)),
            iterations=len(history),
            converged=converged,
            history=history,
        )

    @staticmethod
    def _hessian(problem, x: np.ndarray) -> np.ndarray:
        hessian = getattr(problem, "hessian", lambda _: None)(x)
        if hessian is None:
            hessian = eye(x.shape[0])
        return symmetrize(hessian)

    @staticmethod
    def _initialise_y(
        *,
        current_x: np.ndarray,
        default: np.ndarray,
        warm_start: bool,
        prev_state: Optional[Dict[str, np.ndarray | float]],
        eps: float,
    ) -> np.ndarray:
        if not warm_start or prev_state is None:
            return default.copy()

        g_prev = prev_state["grad"]
        H_prev = prev_state["hessian"]
        x_prev = prev_state["x"]
        y_prev = prev_state["y"]
        s_prev = float(prev_state["grad_norm"])
        if s_prev <= eps:
            return default.copy()

        x_diff = current_x - x_prev
        numerator = g_prev @ (H_prev @ x_diff)
        factor = 1.0 - numerator / (s_prev**2 + eps)
        return factor * y_prev + x_diff / (s_prev + eps)

    def _rk_inner_loop(
        self,
        *,
        y_init: np.ndarray,
        hessian: np.ndarray,
        grad_unit: np.ndarray,
        r_dim: int,
        L: int,
        grad_norm: float,
        c1_prime: float,
        c2_prime: float,
        gamma_prime: float,
    ) -> Tuple[np.ndarray, int, float]:
        y = y_init.copy()
        performed = 0
        for i in range(L):
            performed = i + 1
            residual_vec = hessian @ y - grad_unit
            Q = sample_gaussian_matrix(r_dim, hessian.shape[0], 1.0 / r_dim)
            qhq = symmetrize(Q @ hessian @ Q.T)
            lambda_i = _lambda_shift(qhq)
            rhs = Q @ residual_vec
            reg = c1_prime * lambda_i + c2_prime * (grad_norm**gamma_prime)
            delta = solve_regularized_system(qhq, rhs, reg)
            y = y - Q.T @ delta
        final_residual = rk_residual_fn(hessian, y, grad_unit)
        return y, performed, final_residual

    def _construct_direction(
        self,
        *,
        y_vec: np.ndarray,
        hessian: np.ndarray,
        grad: np.ndarray,
        s_rows: int,
        c1: float,
        c2: float,
        gamma: float,
        grad_norm: float,
    ) -> Tuple[np.ndarray, int, float]:
        y_norm = norm(y_vec)
        if y_norm < 1e-16:
            base_row = grad / (norm(grad) + 1e-16)
        else:
            base_row = y_vec / y_norm

        Hy = hessian @ y_vec
        Hy_norm_sq = float(Hy @ Hy)
        dim = hessian.shape[0]
        if s_rows > 0:
            if Hy_norm_sq > 0:
                projector = eye(dim) - np.outer(Hy, Hy) / (Hy_norm_sq + 1e-16)
            else:
                projector = eye(dim)
            G = sample_gaussian_matrix(s_rows, dim, 1.0 / s_rows)
            tilde_P = G @ projector
            P = np.vstack([base_row, tilde_P])
        else:
            P = base_row[None, :]

        phpt = symmetrize(P @ hessian @ P.T)
        lambda_k = _lambda_shift(phpt)
        reg = c1 * lambda_k + c2 * (grad_norm**gamma)
        rhs = P @ grad
        delta = solve_regularized_system(phpt, rhs, reg)
        direction = -P.T @ delta
        return direction, P.shape[0], lambda_k

    @staticmethod
    def _ensure_descent(direction: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if grad @ direction >= 0:
            return -grad
        return direction

    @staticmethod
    def _extras(
        *,
        t_k: float,
        subspace_rows: int,
        r_dim: int,
        L: int,
        lambda_k: float,
        rk_iters: int,
        rk_residual: float,
    ) -> Dict[str, float]:
        return {
            "t_k": float(t_k),
            "subspace_dim_s": subspace_rows,
            "inner_dim_r": r_dim,
            "L": L,
            "lambda_k": float(lambda_k),
            "rk_iters": rk_iters,
            "rk_residual": float(rk_residual),
        }
