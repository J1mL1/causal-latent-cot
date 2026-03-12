from __future__ import annotations

"""
Quick Multiplex Thinking inference demo.

Usage:
  python scripts/inference/test_multiplex_infer.py \
    --config_path configs/rq1/multiplex/multiplex-1.5b-gsm8k.yaml \
    --question "If you have 2 apples and eat one, how many left?"
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from common.model_registry import load_model


def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text()
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise
        return yaml.safe_load(text)


def get_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == args.idx:
                    sample = json.loads(line)
                    return sample.get("question") or sample.get("prompt") or line
        raise IndexError(f"Index {args.idx} not found in {args.jsonl}")
    raise ValueError("Provide --question or --jsonl with --idx.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to Multiplex config (YAML/JSON).")
    parser.add_argument("--question", default=None, help="Direct question string.")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL to pick question from.")
    parser.add_argument("--idx", type=int, default=0, help="Index into JSONL when using --jsonl.")
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = config.get("model", config)
    prompt_template = config.get("prompt_template", "Question: {question}\nAnswer:")
    question = get_question(args)
    prompt = prompt_template.replace("{question}", question)

    model = load_model("multiplex", model_cfg)
    outputs = model.run_baseline(prompt)

    print("Question:\n", question)
    print("\nPrompt:\n", prompt)
    print("\nOutputs:")
    if isinstance(outputs, dict):
        print("Thought preview:\n", outputs.get("thought_preview"))
        print("Answer:\n", outputs.get("text"))
    else:
        print(outputs)


if __name__ == "__main__":
    main()
