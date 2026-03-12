from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_path_env() -> dict[str, str]:
    project_root = str(PROJECT_ROOT)
    return {
        "PROJECT_ROOT": os.environ.get("PROJECT_ROOT", project_root),
        "MODEL_DIR": os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models")),
        "DATA_DIR": os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")),
        "OUTPUT_DIR": os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")),
    }


def expand_path_vars(value: str) -> str:
    expanded = value
    for key, default in default_path_env().items():
        expanded = expanded.replace(f"${{{key}}}", default)
    return os.path.expandvars(expanded)


def expand_nested_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_nested_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_nested_paths(v) for v in value]
    if isinstance(value, str):
        return expand_path_vars(value)
    return value

