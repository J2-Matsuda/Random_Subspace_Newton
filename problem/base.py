"""Problem protocol definitions for unconstrained minimisation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Problem(Protocol):
    """Interface all benchmark problems must expose."""

    dim: int

    def value(self, x: np.ndarray) -> float: ...

    def gradient(self, x: np.ndarray) -> np.ndarray: ...

    def hessian(self, x: np.ndarray) -> np.ndarray: ...

    def initial_point(self) -> np.ndarray: ...
