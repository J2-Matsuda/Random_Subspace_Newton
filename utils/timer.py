"""Simple context manager for timing code blocks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block() -> Iterator[float]:
    start = time.perf_counter()
    duration = -1.0
    try:
        yield lambda: duration
    finally:
        duration = time.perf_counter() - start
