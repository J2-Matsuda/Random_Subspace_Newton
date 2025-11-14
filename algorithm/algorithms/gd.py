"""Plain gradient descent with Armijo backtracking."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from analysis import metrics
from utils.math_tools import ArmijoParams, armijo_backtracking


class GradientDescent(AlgorithmBase):
    def run(self, problem, logger: Logger) -> AlgorithmResult:
        params = self.params
        max_iters = params.get("max_iters", 100)
        tol = params.get("tol", 1e-6)
        armijo = ArmijoParams(
            alpha0=params.get("alpha0", 1.0),
            c1=params.get("c1", 1e-4),
            rho=params.get("rho", 0.5),
        )
        x = problem.initial_point().astype(float)
        history = []
        converged = False
        fx = problem.value(x)
        self._init_progress_tracker(max_iters)
        for k in range(max_iters):
            grad = problem.gradient(x)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm <= tol:
                converged = True
                row = metrics.build_row(
                    iteration=k,
                    value=fx,
                    grad=grad,
                    x=x,
                    extra={"alpha": 0.0},
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
                extra={"alpha": alpha},
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
