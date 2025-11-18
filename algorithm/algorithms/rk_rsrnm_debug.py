"""Debug wrapper around the standard RK+RSRNM implementation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

from algorithm.base import AlgorithmBase, AlgorithmResult, Logger
from algorithm.algorithms.rk_rsrnm import RKRsrnm
from utils.io import ensure_parent_dir


class _DebugLogger(Logger):
    def __init__(
        self,
        base_logger: Logger,
        csv_path: Optional[str],
        enable_print: bool,
    ) -> None:
        self._base = base_logger
        self._enable_print = enable_print
        self._csv_path = Path(csv_path).expanduser().resolve() if csv_path else None
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        if self._csv_path:
            ensure_parent_dir(self._csv_path)
            self._file = self._csv_path.open("w", newline="", encoding="utf-8")

    def log(self, row: Dict[str, float]) -> None:
        if self._enable_print:
            self._print_row(row)
        if self._file:
            self._write_row(row)
        self._base.log(row)

    def flush(self) -> None:
        self._base.flush()
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _print_row(self, row: Dict[str, float]) -> None:
        k = row.get("k", "?")
        f_val = float(row.get("f", 0.0))
        grad = float(row.get("grad_norm", 0.0))
        step = float(row.get("t_k", 0.0))
        rk_res = float(row.get("rk_residual", 0.0))
        print(
            "[rk_rsrnm_debug] "
            f"k={k}, f={f_val:.4e} ||g||={grad:.4e} "
            f"t_k={step:.3e} rk_res={rk_res:.3e}"
        )

    def _write_row(self, row: Dict[str, float]) -> None:
        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)


class RKRsrnmDebug(AlgorithmBase):
    """Wraps the standard RKRsrnm to add debug printing / CSV dumps."""

    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        self._base = RKRsrnm(params)

    def run(self, problem, logger: Logger) -> AlgorithmResult:
        debug_print = self.params.get("debug_print", True)
        debug_csv = self.params.get("debug_csv")
        wrapper = _DebugLogger(logger, debug_csv, debug_print)
        try:
            return self._base.run(problem, wrapper)
        finally:
            wrapper.close()
