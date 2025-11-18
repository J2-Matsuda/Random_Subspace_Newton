"""Simple experiment logger that emits CSV and JSONL outputs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.io import ensure_parent_dir


class ExperimentLogger:
    def __init__(
        self,
        csv_path: str | Path,
        jsonl_path: str | Path | None = None,
        csv_columns: List[str] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.jsonl_path = (
            Path(jsonl_path).expanduser().resolve() if jsonl_path else None
        )
        self.rows: List[Dict[str, Any]] = []
        self._fixed_columns = csv_columns is not None and len(csv_columns) > 0
        self.columns: List[str] = list(csv_columns) if csv_columns else []
        if self._fixed_columns:
            if "cpu_time" not in self.columns:
                self.columns.append("cpu_time")
            if "cpu_time_sum" not in self.columns:
                self.columns.append("cpu_time_sum")
        self._last_cpu_time = time.process_time()
        self._cpu_time_sum = 0.0

    def log(self, row: Dict[str, Any]) -> None:
        if self._fixed_columns:
            filtered = {col: row.get(col, None) for col in self.columns}
        else:
            filtered = dict(row)
            for key in filtered:
                if key not in self.columns:
                    self.columns.append(key)
            if "cpu_time_sum" not in self.columns:
                self.columns.append("cpu_time_sum")
        now = time.process_time()
        delta = now - self._last_cpu_time
        self._last_cpu_time = now
        if "cpu_time" not in filtered or filtered["cpu_time"] in (None, ""):
            filtered["cpu_time"] = delta
        self._cpu_time_sum += float(filtered["cpu_time"])
        if "cpu_time_sum" not in filtered or filtered["cpu_time_sum"] in (None, ""):
            filtered["cpu_time_sum"] = self._cpu_time_sum
        self.rows.append(filtered)

    def flush(self) -> None:
        if not self.rows:
            return
        ensure_parent_dir(self.csv_path)
        self._write_csv()
        if self.jsonl_path:
            ensure_parent_dir(self.jsonl_path)
            self._write_jsonl()

    def _write_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_jsonl(self) -> None:
        assert self.jsonl_path is not None
        with self.jsonl_path.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row))
                fh.write("\n")
