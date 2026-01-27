from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

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


@dataclass
class QuadraticProblem(Problem):
    Q: NDArray[np.floating] | sparse.spmatrix
    b: NDArray[np.floating]

    @property
    def n(self) -> int:
        return int(self.b.shape[0])

    def _matvec(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        if sparse.issparse(self.Q):
            return self.Q.dot(v)
        return self.Q @ v

    def f(self, x: NDArray[np.floating]) -> float:
        Qx = self._matvec(x)
        return 0.5 * float(x @ Qx) + float(self.b @ x)

    def grad(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._matvec(x) + self.b

    def hvp(self, x: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._matvec(v)

    @staticmethod
    def from_npz(path: str | Path) -> "QuadraticProblem":
        npz = np.load(path, allow_pickle=False)
        Q = _load_npz_matrix(npz, "Q")
        b = npz["b"].astype(float, copy=False)
        if b.ndim != 1:
            b = b.reshape(-1)
        return QuadraticProblem(Q=Q, b=b)
