"""Utility helpers for file I/O and configuration loading."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    """Load a YAML configuration file into a mapping."""
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write iterable rows to a JSONL file."""
    target = Path(path).expanduser().resolve()
    ensure_parent_dir(target)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def ensure_parent_dir(path: str | Path) -> None:
    """Ensure the parent of the given path exists."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return it."""
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def timestamp_tag() -> str:
    """Return a sortable timestamp tag."""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
