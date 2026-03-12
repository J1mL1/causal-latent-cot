#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from common.path_utils import expand_path_vars


def _normalize_answer(ans: Any) -> str | None:
    if ans is None:
        return None
    if isinstance(ans, bool):
        return "yes" if ans else "no"
    s = str(ans).strip()
    if s.lower() in {"true", "false"}:
        return "yes" if s.lower() == "true" else "no"
    if s == "":
        return None
    return s


def _iter_records(raw: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for idx, ex in enumerate(raw):
        question = ex.get("question")
        if question is None:
            continue
        answer = _normalize_answer(ex.get("answer"))
        rec: Dict[str, Any] = {
            "question": str(question),
            "answer": answer,
            "id": str(ex.get("id", idx)),
        }
        # Keep answer_clean/answer_value when answer is present (for evaluation compatibility).
        if answer is not None:
            rec["answer_clean"] = answer
        yield rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare StrategyQA test-style jsonl for local evaluation."
    )
    parser.add_argument(
        "--input",
        default="${PROJECT_ROOT}/datasets/StrategyQA_CoT_GPT4o/strategyqa_cot_dev.json",
        help="Path to StrategyQA json (list). Default uses dev as test.",
    )
    parser.add_argument(
        "--output",
        default="${PROJECT_ROOT}/data/strategyqa_local.jsonl",
        help="Output jsonl path.",
    )
    args = parser.parse_args()

    in_path = Path(expand_path_vars(args.input))
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON list of examples.")

    out_path = Path(expand_path_vars(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in _iter_records(raw):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
