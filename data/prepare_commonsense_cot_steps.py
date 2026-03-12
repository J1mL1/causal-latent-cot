#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_cot_steps(text: str) -> List[str]:
    if not text:
        return []
    chunks = [c.strip() for c in text.strip().splitlines() if c.strip()]
    steps: List[str] = []
    for chunk in chunks:
        for sent in _SENT_SPLIT.split(chunk):
            sent = sent.strip()
            if sent:
                steps.append(sent)
    return steps


def transform_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    cot = rec.get("cot", "")
    steps = split_cot_steps(str(cot))
    out = dict(rec)
    out["steps"] = steps
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add sentence-level CoT steps for CommonsenseQA GPT4omini.")
    parser.add_argument("--input_path", required=True, help="Input JSON file (list of records).")
    parser.add_argument("--output_path", required=True, help="Output JSON file with steps list.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of records.")

    transformed = [transform_record(rec) for rec in data]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=True)


if __name__ == "__main__":
    main()
