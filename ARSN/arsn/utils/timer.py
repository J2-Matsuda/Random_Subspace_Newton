from __future__ import annotations

import time


class Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0
