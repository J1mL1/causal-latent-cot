"""
Utility to dump CommonsenseQA into a local JSONL file with fields
compatible with the GSM8K ablation runner: {"question","answer","id"}.

Usage:
  python data/prepare_commonsenseqa_jsonl.py \
    --input datasets/CommonsenseQA-GPT4omini/commonsense_cot_dev.json \
    --out data/commonsenseqa_local.jsonl

The input JSON is expected to be a list of dicts with keys:
  - question: str (already formatted with "Question:" and "Choices:")
  - answer: str (correct option label, e.g. "A")
  - cot: str (optional; ignored here)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_record(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Convert a raw CommonsenseQA example into a flat dict with keys:
      - question: str (question + choices)
      - answer: str (correct label, e.g. "A")
      - id: str or None
    """
    qid = ex.get("id")
    question_str = str(ex.get("question", "")).strip()
    answer_key = ex.get("answer", None)

    rec: Dict[str, Any] = {
        "question": question_str,
        "answer": str(answer_key) if answer_key is not None else None,
        "id": str(qid) if qid is not None else str(idx),
    }
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/CommonsenseQA-GPT4omini/commonsense_cot_dev.json",
        help="Input JSON file (list of dicts with question/answer fields)",
    )
    parser.add_argument(
        "--out",
        default="data/commonsenseqa_local.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    with in_path.open("r", encoding="utf-8") as f:
        ds = json.load(f)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            rec = build_record(ex, idx)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
