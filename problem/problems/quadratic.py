"""Random symmetric positive definite quadratic problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadraticProblem:
    dim: int
    condition_number: float = 100.0
    seed: int | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        eigenvalues = np.linspace(1.0, self.condition_number, self.dim)
        q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
        self.A = q @ np.diag(eigenvalues) @ q.T
        self.b = rng.normal(size=self.dim)
        self._x_star = np.linalg.solve(self.A, self.b)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return float(0.5 * x @ (self.A @ x) - self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def hessian(self, _: np.ndarray) -> np.ndarray:
        return self.A

    def initial_point(self) -> np.ndarray:
        return np.zeros(self.dim)

    @property
    def x_star(self) -> np.ndarray:
        return self._x_star
