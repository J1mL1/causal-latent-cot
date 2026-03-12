"""
Utility to dump ProntoQA JSON into a local JSONL file with fields
compatible with the ablation runner: {"question","answer","steps","id"}.

Usage:
  python data/prepare_prontoqa_jsonl.py \
    --input external/coconut/data/prontoqa_test.json \
    --out data/protoqa_local.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_record(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(ex.get("question", "")).strip()
    answer = ex.get("answer", None)
    steps = ex.get("steps", None)
    rec: Dict[str, Any] = {
        "question": question,
        "answer": str(answer) if answer is not None else None,
        "id": str(ex.get("id")) if ex.get("id") is not None else str(idx),
    }
    if steps is not None:
        rec["steps"] = steps
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="external/coconut/data/prontoqa_test.json",
        help="Input ProntoQA JSON file (list of dicts with question/steps/answer).",
    )
    parser.add_argument(
        "--out",
        default="data/protoqa_local.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    with in_path.open("r", encoding="utf-8") as f:
        ds = json.load(f)
    if not isinstance(ds, list):
        raise ValueError("Expected a JSON list of records.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            rec = build_record(ex, idx)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
