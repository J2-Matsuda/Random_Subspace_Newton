"""Rosenbrock banana valley benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RosenbrockProblem:
    dim: int
    a: float = 1.0
    b: float = 100.0

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return float(
            np.sum(self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2)
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        grad = np.zeros_like(x)
        grad[:-1] = (
            -4 * self.b * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (self.a - x[:-1])
        )
        grad[1:] += 2 * self.b * (x[1:] - x[:-1] ** 2)
        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        hess = np.zeros((self.dim, self.dim))
        diag = np.zeros(self.dim)
        diag[:-1] = 12 * self.b * x[:-1] ** 2 - 4 * self.b * x[1:] + 2
        diag[-1] = 2 * self.b
        hess[np.diag_indices(self.dim)] = diag
        off_diag = -4 * self.b * x[:-1]
        idx = np.arange(self.dim - 1)
        hess[idx, idx + 1] = off_diag
        hess[idx + 1, idx] = off_diag
        return hess

    def initial_point(self) -> np.ndarray:
        x0 = np.empty(self.dim)
        x0[::2] = -1.2
        x0[1::2] = 1.0
        return x0
