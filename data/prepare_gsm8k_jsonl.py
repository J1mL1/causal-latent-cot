"""
Utility to dump GSM8K (or similar) into a local JSONL file the ablation runner can read.

Usage:
  python data/prepare_gsm8k_jsonl.py --split test --subset main --out data/gsm8k_local.jsonl

Requires the `datasets` package and network access to pull HF datasets; otherwise,
adapt this script to read your local copy and emit {"question","answer","id"} per line.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from datasets import load_dataset


def parse_answer(answer: str) -> Dict[str, Any]:
    """Extract a clean text/float answer from GSM8K style answer string."""
    text = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    text = text.replace(",", "")
    try:
        value = float(text)
    except Exception:
        value = None
    return {"answer_clean": text, "answer_value": value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/gsm8k_local.jsonl")
    args = parser.parse_args()

    ds = load_dataset("gsm8k", args.subset, split=args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for ex in ds:
            rec = {
                "question": ex.get("question", ""),
                "answer": ex.get("answer", ""),
                "id": str(ex.get("id")) if ex.get("id") is not None else None,
            }
            rec.update(parse_answer(rec["answer"]))
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
