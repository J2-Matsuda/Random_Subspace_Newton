from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit

from arsn.problems.base import Problem


def _load_npz_matrix(npz: dict, name: str):
    if name in npz:
        return npz[name]
    data_key = f"{name}_data"
    if data_key in npz:
        data = npz[data_key]
        indices = npz[f"{name}_indices"]
        indptr = npz[f"{name}_indptr"]
        shape = tuple(npz[f"{name}_shape"].tolist())
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    raise KeyError(f"Matrix {name} not found in npz file.")


def _to_pm1(y: NDArray[np.floating]) -> NDArray[np.floating]:
    y_vals = np.unique(y)
    if set(y_vals.tolist()) <= {0, 1}:
        return 2.0 * y - 1.0
    if set(y_vals.tolist()) <= {-1, 1}:
        return y.astype(float, copy=False)
    raise ValueError("Labels must be in {0,1} or {-1,1}.")


@dataclass
class LogisticRegressionProblem(Problem):
    A: NDArray[np.floating] | sparse.spmatrix
    y_pm1: NDArray[np.floating]
    reg_lambda: float

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def _matvec(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        if sparse.issparse(self.A):
            return self.A.dot(v)
        return self.A @ v

    def _rmatvec(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        if sparse.issparse(self.A):
            return self.A.T.dot(v)
        return self.A.T @ v

    def f(self, x: NDArray[np.floating]) -> float:
        z = self.y_pm1 * self._matvec(x)
        loss = np.logaddexp(0.0, -z).mean()
        return float(loss + 0.5 * self.reg_lambda * (x @ x))

    def grad(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self.y_pm1 * self._matvec(x)
        p = expit(-z)
        g = -(self._rmatvec(self.y_pm1 * p)) / float(self.y_pm1.shape[0])
        g = g + self.reg_lambda * x
        return np.asarray(g, dtype=float).reshape(-1)

    def hvp(self, x: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self.y_pm1 * self._matvec(x)
        p = expit(-z)
        d = p * (1.0 - p)
        Av = self._matvec(v)
        Hv = self._rmatvec(d * Av) / float(self.y_pm1.shape[0])
        Hv = Hv + self.reg_lambda * v
        return np.asarray(Hv, dtype=float).reshape(-1)

    @staticmethod
    def from_npz(path: str | Path, *, reg_lambda: float | None = None) -> "LogisticRegressionProblem":
        npz = np.load(path, allow_pickle=False)
        A = _load_npz_matrix(npz, "A")
        y = npz["y"].astype(float, copy=False)
        y = y.reshape(-1)
        if reg_lambda is None:
            reg_lambda = float(npz["reg_lambda"]) if "reg_lambda" in npz else 0.0
        y_pm1 = _to_pm1(y)
        return LogisticRegressionProblem(A=A, y_pm1=y_pm1, reg_lambda=float(reg_lambda))
