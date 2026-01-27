from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ALL_COLUMNS: list[str] = [
    "iter",
    "f",
    "grad_norm",
    "step_norm",
    "gTd",
    "alpha",
    "armijo_iters",
    "time_iter_sec",
    "time_cum_sec",
    "hvp_calls_iter",
    "hvp_calls_cum",
    "yk_resid_rel",
    "inner_T",
    "inner_r",
    "inner_minres_total_iters",
    "eta",
    "lambda_min_PHPT",
    "Lambda_shift",
    "seed_outer",
    "seed_Sk",
    "seed_Rk_base",
]


def _normalize_value(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, np.generic):
        return val.item()
    return val


class CSVLogger:
    def __init__(self, path: str | Path, record_columns: Iterable[str] | None = None) -> None:
        self.path = Path(path)
        self.columns = list(record_columns) if record_columns is not None else list(ALL_COLUMNS)
        unknown = [c for c in self.columns if c not in ALL_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown record_columns: {unknown}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        out = {col: _normalize_value(row.get(col, "")) for col in self.columns}
        self._writer.writerow(out)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
