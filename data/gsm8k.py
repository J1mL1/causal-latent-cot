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


def extract_golden_cot(answer: str) -> str:
    """
    GSM8K-style solutions: reasoning lines followed by a final line '#### <number>'.
    Return the reasoning (chain-of-thought) text only, without the #### line.
    """
    if answer is None or not str(answer).strip():
        return ""
    parts = str(answer).split("####", 1)
    return parts[0].strip()


def extract_golden_cot_at_step(
    answer: Optional[str],
    step: int,
    *,
    split: str = "line",
) -> str:
    """
    Return only the segment of the official CoT aligned with 1-based step index ``step``.

    ``split="line"`` (default): non-empty lines of text before ``####``; step ``t`` uses line ``t``
    (1-based). If ``t`` exceeds the number of lines, returns "".

    This is a **heuristic** alignment to latent step indices for case-study tables; GSM8K
    reference text is not explicitly labeled by the same step boundaries as model latents.
    """
    if answer is None or step < 1:
        return ""
    full = extract_golden_cot(str(answer))
    if split != "line":
        raise ValueError(f"Unsupported gold_cot split: {split!r}")
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    if step > len(lines):
        return ""
    return lines[step - 1]


def count_golden_cot_lines(answer: Optional[str]) -> int:
    """Number of non-empty lines in official CoT (before ####)."""
    if answer is None:
        return 0
    full = extract_golden_cot(str(answer))
    return len([ln for ln in full.splitlines() if ln.strip()])


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
