"""RK+RSRNM algorithm as specified in the requirements."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    norm,
    orthonormalize_rows,
    project_rows_orthogonal,
    rk_residual as rk_residual_fn,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


class RKRsrnm(AlgorithmBase):
    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)
        rs_rows = params.get("s", min(8, problem.dim))
        rs_rows = max(0, rs_rows)
        r_dim = max(1, params.get("r", min(8, problem.dim)))
        L = max(1, params.get("L", 10))
        rs_reg = params.get("rs_reg", 1e-8)
        rk_reg = params.get("rk_reg", 1e-8)
        rk_tol = params.get("rk_tol", None)
        y_cap = params.get("y_cap", 1e6)
        warm_start = params.get("warm_start", True)
        y0_vec = np.asarray(params.get("y0", np.zeros(problem.dim)), dtype=float)

        armijo_params = ArmijoParams(
            alpha0=params.get("alpha0", 1.0),
            c1=params.get("c1", 1e-4),
            rho=params.get("rho", 0.5),
        )

        x = problem.initial_point().astype(float)
        fx = problem.value(x)
        history = []
        converged = False
        prev_state: Optional[Dict[str, np.ndarray | float]] = None

        for k in range(max_iters):
            grad = problem.gradient(x)
            grad_norm = float(np.linalg.norm(grad))
            hessian = self._hessian(problem, x)

            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra=self._rk_extras(
                        alpha=0.0,
                        subspace_rows=1 + rs_rows,
                        r_dim=r_dim,
                        L=0,
                        rs_reg=rs_reg,
                        rk_reg=rk_reg,
                        rk_iters=0,
                        rk_residual=0.0,
                    ),
                )
                logger.log(row)
                history.append(row)
                break

            grad_unit = grad / (grad_norm + 1e-16)
            y_init = self._initialise_y(
                current_x=x,
                default=y0_vec,
                warm_start=warm_start,
                prev_state=prev_state,
            )

            y_vec, inner_iters, inner_residual = self._rk_inner_loop(
                y_init=y_init,
                hessian=hessian,
                grad_unit=grad_unit,
                r_dim=r_dim,
                L=L,
                rk_reg=rk_reg,
                rk_tol=rk_tol,
                y_cap=y_cap,
            )

            direction, P_rows = self._construct_direction(
                y_vec=y_vec,
                hessian=hessian,
                grad=grad,
                rs_rows=rs_rows,
                rs_reg=rs_reg,
            )
            direction = self._ensure_descent(direction, grad)

            alpha, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )

            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._rk_extras(
                    alpha=alpha,
                    subspace_rows=P_rows,
                    r_dim=r_dim,
                    L=L,
                    rs_reg=rs_reg,
                    rk_reg=rk_reg,
                    rk_iters=inner_iters,
                    rk_residual=inner_residual,
                ),
            )
            logger.log(row)
            history.append(row)

            if alpha == 0.0:
                break

            prev_state = {
                "x": x.copy(),
                "grad": grad.copy(),
                "grad_norm": grad_norm,
                "hessian": hessian.copy(),
                "y": y_vec.copy(),
            }

            x = x + alpha * direction
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
            hessian = np.eye(x.shape[0])
        return symmetrize(hessian)

    @staticmethod
    def _initialise_y(
        *,
        current_x: np.ndarray,
        default: np.ndarray,
        warm_start: bool,
        prev_state: Optional[Dict[str, np.ndarray | float]],
    ) -> np.ndarray:
        if not warm_start or prev_state is None:
            return default.copy()
        prev_grad_norm = float(prev_state["grad_norm"])
        if prev_grad_norm <= 0:
            return default.copy()
        x_diff = current_x - prev_state["x"]
        g_prev = prev_state["grad"]
        H_prev = prev_state["hessian"]
        y_prev = prev_state["y"]
        numerator = g_prev @ (H_prev @ x_diff)
        coeff = 1.0 - numerator / (prev_grad_norm**2 + 1e-16)
        update = x_diff / (prev_grad_norm + 1e-16)
        return coeff * y_prev + update

    @staticmethod
    def _rk_inner_loop(
        *,
        y_init: np.ndarray,
        hessian: np.ndarray,
        grad_unit: np.ndarray,
        r_dim: int,
        L: int,
        rk_reg: float,
        rk_tol: Optional[float],
        y_cap: float,
    ) -> Tuple[np.ndarray, int, float]:
        y = y_init.copy()
        residual_value = float("inf")
        performed = 0
        for l in range(L):
            performed = l + 1
            residual_vec = hessian @ y - grad_unit
            residual_value = norm(residual_vec)
            if not np.isfinite(residual_value):
                y = grad_unit.copy()
                residual_value = norm(hessian @ y - grad_unit)
                break
            if rk_tol is not None and residual_value <= rk_tol:
                break
            Q = sample_gaussian_matrix(r_dim, hessian.shape[0], 1.0 / r_dim)
            qhq = symmetrize(Q @ hessian @ Q.T)
            rhs = Q @ residual_vec
            delta = solve_regularized_system(qhq, rhs, rk_reg)
            y = y - Q.T @ delta
            if not np.all(np.isfinite(y)) or norm(y) > y_cap:
                y = grad_unit.copy()
                break
        final_residual = rk_residual_fn(hessian, y, grad_unit)
        return y, performed, final_residual

    def _construct_direction(
        self,
        *,
        y_vec: np.ndarray,
        hessian: np.ndarray,
        grad: np.ndarray,
        rs_rows: int,
        rs_reg: float,
    ) -> Tuple[np.ndarray, int]:
        y_norm = norm(y_vec)
        if y_norm < 1e-12:
            y_vec = grad / (norm(grad) + 1e-16)
            y_norm = norm(y_vec)
        u = (y_vec / y_norm)[None, :]
        if rs_rows > 0:
            rows = self._orth_complement(rs_rows, hessian.shape[0], u[0])
            P = np.vstack([u, rows])
        else:
            P = u
        phpt = symmetrize(P @ hessian @ P.T)
        rhs = P @ grad
        delta = solve_regularized_system(phpt, rhs, rs_reg)
        direction = -P.T @ delta
        return direction, P.shape[0]

    @staticmethod
    def _orth_complement(rows: int, dim: int, unit_vec: np.ndarray) -> np.ndarray:
        attempts = 0
        variance = 1.0 / max(rows, 1)
        while attempts < 5:
            attempts += 1
            gaussian = sample_gaussian_matrix(rows, dim, variance)
            projected = project_rows_orthogonal(gaussian, unit_vec)
            row_norms = np.linalg.norm(projected, axis=1)
            if np.all(row_norms > 1e-10):
                return orthonormalize_rows(projected)
        identity = np.eye(dim)
        projected = project_rows_orthogonal(identity[:rows], unit_vec)
        return orthonormalize_rows(projected)

    @staticmethod
    def _rk_extras(
        *,
        alpha: float,
        subspace_rows: int,
        r_dim: int,
        L: int,
        rs_reg: float,
        rk_reg: float,
        rk_iters: int,
        rk_residual: float,
    ) -> Dict[str, float]:
        return {
            "alpha": float(alpha),
            "subspace_dim_s": subspace_rows,
            "inner_dim_r": r_dim,
            "L": L,
            "rs_reg": float(rs_reg),
            "rk_reg": float(rk_reg),
            "rk_iters": rk_iters,
            "rk_residual": float(rk_residual),
        }

    @staticmethod
    def _ensure_descent(direction: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if grad @ direction >= 0:
            return -grad
        return direction
