from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Problem(ABC):
    @property
    @abstractmethod
    def n(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def f(self, x: NDArray[np.floating]) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    @abstractmethod
    def hvp(self, x: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError
