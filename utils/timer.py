"""Simple timing utilities (wall-clock and CPU)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def time_block() -> Iterator[float]:
    start = time.perf_counter()
    duration = -1.0
    try:
        yield lambda: duration
    finally:
        duration = time.perf_counter() - start


class CPUTimer:
    """Utility for measuring CPU time per block."""

    def __init__(self) -> None:
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.process_time()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.process_time() - self._start
        self._start = None
        return float(elapsed)
