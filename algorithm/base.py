"""Common scaffolding shared across optimisation algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

import numpy as np

from problem.base import Problem


class Logger(Protocol):
    def log(self, row: Dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


@dataclass
class AlgorithmResult:
    x: np.ndarray
    f: float
    grad_norm: float
    iterations: int
    converged: bool
    history: List[Dict[str, Any]] = field(default_factory=list)


class AlgorithmBase:
    """Abstract base class."""

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def run(self, problem: Problem, logger: Logger) -> AlgorithmResult:
        raise NotImplementedError
