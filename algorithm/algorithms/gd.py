"""Plain gradient descent with Armijo backtracking."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.dtypes import as_dtype
from utils.math_tools import ArmijoParams, armijo_backtracking


class GradientDescent(AlgorithmBase):
    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 100)
        tol = params.get("tol", 1e-6)
        armijo = ArmijoParams(
            t0=params.get("t0", 1.0),
            alpha=params.get("alpha", 1e-4),
            beta=params.get("beta", 0.5),
            max_backtracks=params.get("max_backtracks", 50),
            min_alpha=params.get("min_alpha", 1e-16),
        )
        x = as_dtype(problem.initial_point())
        history = []
        converged = False
        fx = problem.value(x)
        self._init_progress_tracker(max_iters)
        for k in range(max_iters):
            grad = as_dtype(problem.gradient(x))
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra={"t_k": 0.0},
                )
                logger.log(row)
                history.append(row)
                self._report_progress(k)
                break
            direction = -grad
            alpha, new_value = armijo_backtracking(
                problem.value, x, direction, grad, armijo, fx=fx
            )
            row = metrics.build_row(
                iteration=k,
                value=fx,
                grad=grad,
                x=x,
                extra={"t_k": alpha},
            )
            logger.log(row)
            history.append(row)
            self._report_progress(k)
            if alpha == 0.0:
                break
            x = x + alpha * direction
            fx = new_value
        logger.flush()
        return AlgorithmResult(
            x=x,
            f=fx,
            grad_norm=float(np.linalg.norm(problem.gradient(x))),
            iterations=len(history),
            converged=converged,
            history=history,
        )
