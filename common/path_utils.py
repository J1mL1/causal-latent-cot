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


def resolve_pretrained_source(path: str) -> tuple[str, bool]:
    """
    Resolve a Hugging Face ``from_pretrained`` path.

    Returns (path_for_from_pretrained, local_files_only).

    If ``path`` is an absolute filesystem path that does not exist, raises
    FileNotFoundError with a clear message (avoids HF hub treating it as a
    repo id and raising a confusing OSError).

    If ``path`` points at a local directory containing ``config.json``, sets
    ``local_files_only=True`` so HF never tries to download when the folder
    exists but is incomplete.
    """
    p = Path(path).expanduser()
    if p.is_absolute() and not p.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {path}. "
            "Set MODEL_DIR to the directory that contains the base checkpoint, "
            "or download the model into the path used in your YAML "
            "(e.g. models/Qwen3-4B-Instruct-2507 with config.json inside)."
        )
    if not p.exists():
        # Likely a Hub repo id (e.g. org/name); let transformers resolve it.
        return path, False
    resolved = p.resolve()
    if resolved.is_dir() and (resolved / "config.json").is_file():
        return str(resolved), True
    if resolved.is_dir() and p.is_absolute():
        raise FileNotFoundError(
            f"Local model directory has no config.json: {path}"
        )
    return path, False

