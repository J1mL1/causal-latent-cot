#!/usr/bin/env python3
"""
Sweep multiple patchscope ``target_template`` values for **Coconut GPT-2** and **CODI GPT-2**
on GSM8K. Writes under ``outputs/rq4/patchscope/`` with filenames and ``.meta.json`` that
include ``prompt_label``.

Requires ``experiments/rq4/run_patchscope.py`` support for ``--patchscope_target_template_file``
and ``--prompt_label`` (see that script).

Usage (from repo root)::

    python scripts/rq4/patchscope/run_patchscope_prompt_ablation_gpt2.py
    MAX_SAMPLES=32 python scripts/rq4/patchscope/run_patchscope_prompt_ablation_gpt2.py --dry_run

Env (optional): ``CONDA_ENV``, ``BATCH_SIZE``, ``STEPS``, ``MAX_SAMPLES``, ``ONLY_MODEL`` (coconut|codi).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Repo root: scripts/rq4/patchscope/this_file.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.analysis.patchscope import (  # noqa: E402
    DEFAULT_PAIR_TARGET_TEMPLATE,
    DEFAULT_PHRASE_TARGET_TEMPLATE,
)

OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "outputs" / "rq4" / "patchscope"

# Bullet-list + "Your phrase:" (no PAIR arrows); for comparison with phrase_arrow.
BULLET_PHRASE_TARGET_TEMPLATE = (
    "In one short phrase, describe what the latent represents.\n"
    "Examples:\n"
    '- "Comparing two quantities before combining them."\n'
    '- "An intermediate value in a multi-step reasoning step."\n'
    "Your phrase:\n"
    "?"
)

# Degenerate: only placeholder (strong prior from LM only).
MINIMAL_TARGET_TEMPLATE = "?\n"


def _prompt_variants() -> List[Tuple[str, str]]:
    """(prompt_label, target_template text). Labels are filesystem-safe slugs."""
    return [
        ("pair_token_identity", DEFAULT_PAIR_TARGET_TEMPLATE),
        ("phrase_arrow_fewshot", DEFAULT_PHRASE_TARGET_TEMPLATE),
        ("bullet_phrase_fewshot", BULLET_PHRASE_TARGET_TEMPLATE),
        ("minimal_placeholder_only", MINIMAL_TARGET_TEMPLATE),
    ]


def _models() -> List[Tuple[str, Path]]:
    return [
        ("coconut", PROJECT_ROOT / "configs" / "rq4" / "coconut" / "gpt2-gsm8k.yaml"),
        ("codi", PROJECT_ROOT / "configs" / "rq4" / "codi" / "gpt2-gsm8k.yaml"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run patchscope prompt ablation for Coconut/CODI GPT-2 on GSM8K."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"JSONL / .meta.json root (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument(
        "--only_model",
        choices=("coconut", "codi"),
        default=None,
        help="If set, only run this registry model.",
    )
    args = parser.parse_args()

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size if args.batch_size is not None else int(os.environ.get("BATCH_SIZE", "4"))
    steps = args.steps if args.steps is not None else os.environ.get("STEPS", "1,2,3,4,5,6")
    max_samples = args.max_samples
    if max_samples is None and os.environ.get("MAX_SAMPLES", "").strip():
        max_samples = int(os.environ["MAX_SAMPLES"])

    only = args.only_model or os.environ.get("ONLY_MODEL", "").strip() or None
    if only:
        only = only.lower()
        models = [m for m in _models() if m[0] == only]
        if not models:
            raise SystemExit(f"Unknown ONLY_MODEL / --only_model: {only!r}")
    else:
        models = _models()

    run_py = PROJECT_ROOT / "experiments" / "rq4" / "run_patchscope.py"
    variants = _prompt_variants()

    print(f"Output root: {out_root.resolve()}")
    print(f"Models: {[m[0] for m in models]}")
    print(f"Prompt labels: {[v[0] for v in variants]}")

    for model_name, config_path in models:
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")
        for label, template in variants:
            stem = f"gsm8k_{model_name}_gpt2_patchscope_prompt_{label}"
            output_path = out_root / f"{stem}.jsonl"

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(template)
                tmp_path = tmp.name

            cmd = [
                sys.executable,
                str(run_py),
                "--model_name",
                model_name,
                "--config_path",
                str(config_path),
                "--output_path",
                str(output_path),
                "--steps",
                steps,
                "--batch_size",
                str(batch_size),
                "--num_workers",
                "0",
                "--patchscope_target_template_file",
                tmp_path,
                "--prompt_label",
                label,
            ]
            if args.dry_run:
                cmd.append("--dry_run")
            if max_samples is not None:
                cmd.extend(["--max_samples", str(max_samples)])

            print("\n" + "=" * 72)
            print(" ".join(cmd[:6]), "...")
            print(f" -> {output_path.name}")

            try:
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    print(
        "\nDone. Each run: ``*.meta.json`` (``prompt_label``, full ``target_template``) and "
        "each JSONL line includes ``prompt_label`` when set."
    )


if __name__ == "__main__":
    main()
