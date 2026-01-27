from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GaussianSketchOperator:
    shape: tuple[int, int]
    scale: float
    seed: int
    mode: str = "operator"
    block_size: int = 256
    dtype: np.dtype | type = float
    _mat: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        m, n = self.shape
        if m <= 0 or n <= 0:
            raise ValueError("GaussianSketchOperator shape must be positive.")
        if self.mode not in ("operator", "explicit"):
            raise ValueError(f"Unknown sketch mode: {self.mode}")
        if self.mode == "explicit":
            rng = np.random.default_rng(self.seed)
            mat = rng.standard_normal(size=self.shape).astype(self.dtype, copy=False)
            self._mat = (self.scale * mat).astype(self.dtype, copy=False)

    def matvec(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.mode == "explicit":
            if self._mat is None:
                raise ValueError("Explicit sketch matrix is not initialized.")
            return (self._mat @ v).astype(float, copy=False)
        return self._matvec_operator(v)

    def rmatvec(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.mode == "explicit":
            if self._mat is None:
                raise ValueError("Explicit sketch matrix is not initialized.")
            return (self._mat.T @ u).astype(float, copy=False)
        return self._rmatvec_operator(u)

    def _matvec_operator(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        m, n = self.shape
        if v.shape[0] != n:
            raise ValueError(f"matvec expects v shape ({n},), got {v.shape}")
        out = np.zeros(m, dtype=float)
        rng = np.random.default_rng(self.seed)
        bs = max(1, int(self.block_size))
        row = 0
        while row < m:
            rows = min(bs, m - row)
            block = rng.standard_normal(size=(rows, n)).astype(self.dtype, copy=False)
            block = (self.scale * block).astype(self.dtype, copy=False)
            out[row : row + rows] = block @ v
            row += rows
        return out

    def _rmatvec_operator(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        m, n = self.shape
        if u.shape[0] != m:
            raise ValueError(f"rmatvec expects u shape ({m},), got {u.shape}")
        out = np.zeros(n, dtype=float)
        rng = np.random.default_rng(self.seed)
        bs = max(1, int(self.block_size))
        row = 0
        while row < m:
            rows = min(bs, m - row)
            block = rng.standard_normal(size=(rows, n)).astype(self.dtype, copy=False)
            block = (self.scale * block).astype(self.dtype, copy=False)
            out += block.T @ u[row : row + rows]
            row += rows
        return out
