"""Random Subspace Regularised Newton Method (direct solve)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    orthonormalize_rows,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


class RSRNM(AlgorithmBase):
    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)
        s_dim = max(1, params.get("s", min(8, problem.dim)))
        rs_reg = params.get("rs_reg", 1e-8)
        armijo_params = ArmijoParams(
            alpha0=params.get("alpha0", 1.0),
            c1=params.get("c1", 1e-4),
            rho=params.get("rho", 0.5),
        )

        x = problem.initial_point().astype(float)
        fx = problem.value(x)
        history = []
        converged = False

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
                    extra=self._rs_extras(alpha=0.0, s_dim=s_dim, rs_reg=rs_reg),
                )
                logger.log(row)
                history.append(row)
                break

            P = self._sample_p(rows=s_dim, dim=problem.dim)
            phpt = symmetrize(P @ hessian @ P.T)
            rhs = P @ grad
            delta_subspace = solve_regularized_system(phpt, rhs, rs_reg)
            direction = -P.T @ delta_subspace
            direction = self._ensure_descent(direction, grad)

            alpha, new_fx = armijo_backtracking(
                problem.value, x, direction, grad, armijo_params, fx=fx
            )
            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra=self._rs_extras(alpha=alpha, s_dim=s_dim, rs_reg=rs_reg),
            )
            logger.log(row)
            history.append(row)

            if alpha == 0.0:
                break

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
    def _sample_p(rows: int, dim: int) -> np.ndarray:
        variance = 1.0 / rows
        gaussian = sample_gaussian_matrix(rows, dim, variance)
        return orthonormalize_rows(gaussian)

    @staticmethod
    def _rs_extras(alpha: float, s_dim: int, rs_reg: float) -> Dict[str, float]:
        return {
            "alpha": float(alpha),
            "subspace_dim_s": s_dim,
            "inner_dim_r": 0,
            "L": 0,
            "rs_reg": float(rs_reg),
            "rk_reg": 0.0,
            "rk_iters": 0,
            "rk_residual": 0.0,
        }

    @staticmethod
    def _ensure_descent(direction: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if grad @ direction >= 0:
            return -grad
        return direction
