from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re

@dataclass
class Gsm8kRecord:
    """Canonical structure for a GSM8K example."""

    prompt: str
    question: str
    answer: str
    answer_clean: str
    answer_value: Optional[float]
    id: Optional[str]
    raw: Dict[str, Any]


def parse_answer(answer: str) -> tuple[str, Optional[float]]:
    """
    Extract the numeric/text answer from GSM8K-formatted string.
    Prefers lm-eval style: "#### <num>" when present; otherwise the last number in the text.
    """
    if answer is None:
        return "", None

    # 1) Strict pattern: #### <answer>
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
    if m:
        text = m.group(1).replace(",", "")
    else:
        # 2) Fallback: last number anywhere in the string
        nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        if nums:
            text = nums[-1].replace(",", "")
        else:
            text = answer.strip().replace(",", "")

    try:
        value = float(text)
    except Exception:
        value = None
    return text, value


def to_record(example: Dict[str, Any], prompt: str) -> Gsm8kRecord:
    answer = example.get("answer", "")
    answer_clean, answer_value = parse_answer(answer)
    return Gsm8kRecord(
        prompt=prompt,
        question=example.get("question", ""),
        answer=answer,
        answer_clean=answer_clean,
        answer_value=answer_value,
        id=str(example.get("id")) if example.get("id") is not None else None,
        raw=example,
    )
