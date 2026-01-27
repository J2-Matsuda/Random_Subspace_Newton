from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    text = cfg_path.read_text()
    suffix = cfg_path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)

    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_path(maybe_path: str | None, base_dir: Path) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    cand1 = (base_dir / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (base_dir.parent / p).resolve()
    return cand2


def save_config_copy(cfg: dict[str, Any], out_path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write config_copy.yml")
    out = Path(out_path)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
