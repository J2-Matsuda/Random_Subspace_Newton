"""Randomized Subspace Regularized Newton method (mid variant, Algorithm 3.1)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.dtypes import as_dtype, eye
from utils.math_tools import (
    ArmijoParams,
    armijo_backtracking,
    norm,
    sample_gaussian_matrix,
    solve_regularized_system,
    symmetrize,
)


def _lambda_shift(matrix: np.ndarray) -> float:
    """Return max(0, -lambda_min(matrix)) for stabilization."""
    matrix = symmetrize(matrix)
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        return float(max(0.0, -float(np.min(eigvals))))
    except np.linalg.LinAlgError:
        return 0.0


class RSRNMid(AlgorithmBase):
    """Implementation of Algorithm 3.1 (RS-RNM with randomized subspace)."""

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 200)
        tol = params.get("tol", 1e-6)

        s_rows = max(1, params.get("s", min(8, problem.dim)))
        c1 = params.get("c1", 1.0)
        c2 = params.get("c2", 0.0)
        gamma = params.get("gamma", 1.0)

        armijo_params = ArmijoParams(
            t0=params.get("t0", 1.0),
            alpha=params.get("alpha", 1e-4),
            beta=params.get("beta", 0.5),
            max_backtracks=params.get("max_backtracks", 50),
            min_alpha=params.get("min_alpha", 1e-16),
        )

        x = as_dtype(problem.initial_point())
        fx = problem.value(x)
        history: list[Dict[str, float]] = []
        converged = False
        self._init_progress_tracker(max_iters)

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
                        subspace_rows=s_rows,
                        lambda_k=0.0,
                    ),
                )
                logger.log(row)
                history.append(row)
                self._report_progress(k)
                break

            P = sample_gaussian_matrix(s_rows, hessian.shape[0], 1.0 / s_rows)
            phpt = symmetrize(P @ hessian @ P.T)
            lambda_k = _lambda_shift(phpt)
            reg = c1 * lambda_k + c2 * (grad_norm**gamma)
            M = phpt + reg * eye(phpt.shape[0])

            rhs = P @ grad
            delta = solve_regularized_system(M, rhs, 0.0)  # already regularized
            direction = -P.T @ delta

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
                    subspace_rows=s_rows,
                    lambda_k=lambda_k,
                ),
            )
            logger.log(row)
            history.append(row)
            self._report_progress(k)

            if t_k == 0.0:
                break

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
    def _extras(
        *,
        t_k: float,
        subspace_rows: int,
        lambda_k: float,
    ) -> Dict[str, float]:
        return {
            "t_k": float(t_k),
            "subspace_dim_s": subspace_rows,
            "lambda_k": float(lambda_k),
        }
