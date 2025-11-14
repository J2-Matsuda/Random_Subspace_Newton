"""Common scaffolding shared across optimisation algorithms."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Protocol

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
        self._progress_marks: Deque[int] | None = None
        self._progress_total: int = 0

    def run(self, problem: Problem, logger: Logger) -> AlgorithmResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Progress reporting helpers
    # ------------------------------------------------------------------
    def _init_progress_tracker(self, max_iters: int) -> None:
        """Pre-compute iteration counts (10%, 20%, ..., 100%) for logging."""
        if max_iters <= 0:
            self._progress_marks = deque()
            self._progress_total = 0
            return
        marks: List[int] = []
        for i in range(1, 11):
            mark = math.ceil(max_iters * i / 10)
            mark = min(mark, max_iters)
            if not marks or mark != marks[-1]:
                marks.append(mark)
        self._progress_marks = deque(marks)
        self._progress_total = max_iters

    def _report_progress(self, iteration: int) -> None:
        """Emit a console message when we cross a precomputed milestone."""
        if not self._progress_marks:
            return
        current = iteration + 1
        updated = False
        while self._progress_marks and current >= self._progress_marks[0]:
            mark = self._progress_marks.popleft()
            percent = (
                100.0 * mark / self._progress_total if self._progress_total else 0.0
            )
            print(
                f"[{self.__class__.__name__}] progress: "
                f"{mark}/{self._progress_total} iterations ({percent:.0f}%)"
            )
            updated = True
        if updated and not self._progress_marks:
            # Ensure deque is falsy after exhausting marks
            self._progress_marks = deque()
