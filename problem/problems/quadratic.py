"""Random symmetric positive definite quadratic problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.dtypes import DTYPE, as_dtype


@dataclass
class QuadraticProblem:
    dim: int
    condition_number: float = 100.0
    seed: int | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        eigenvalues = np.linspace(1.0, self.condition_number, self.dim, dtype=DTYPE)
        q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
        q = as_dtype(q)
        base_A = as_dtype(q @ np.diag(eigenvalues) @ q.T)
        base_b = as_dtype(rng.normal(size=self.dim))
        scale = as_dtype(self.scale)
        self.A = scale * base_A
        self.b = scale * base_b
        self._x_star = np.linalg.solve(self.A, self.b).astype(DTYPE)

    def value(self, x: np.ndarray) -> float:
        x = as_dtype(x)
        return float(0.5 * x @ (self.A @ x) - self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = as_dtype(x)
        return as_dtype(self.A @ x - self.b)

    def hessian(self, _: np.ndarray) -> np.ndarray:
        return self.A

    def initial_point(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=DTYPE)

    @property
    def x_star(self) -> np.ndarray:
        return self._x_star
