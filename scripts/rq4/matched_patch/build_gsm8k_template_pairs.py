#!/usr/bin/env python3
"""
Build GSM8K arithmetic “same-structure” pairs for RQ4 matched-patch.

Strict digit-template deduplication rarely yields duplicates on GSM8K (questions are unique).
We score pairs by Jaccard similarity on word sets after normalizing digit tokens to <NUM>,
then take high-similarity pairs with different gold answers:

  - Default: greedy **disjoint** matching (each index in at most one pair).
  - --allow_overlap: take top unique edges; same index may appear in multiple pairs (larger N, dependent samples).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from data.gsm8k import parse_answer


def _norm_word_set(question: str) -> Set[str]:
    s = re.sub(r"[-+]?\d[\d,]*\.?\d*", "<NUM>", question.lower())
    return set(s.split())


def _all_scored_pairs(
    rows: List[Dict[str, Any]], min_jaccard: float
) -> List[Tuple[float, int, int]]:
    sets = [_norm_word_set(str(r.get("question", ""))) for r in rows]
    answers: List[str] = []
    for r in rows:
        ac = r.get("answer_clean")
        if ac is not None and str(ac).strip():
            answers.append(str(ac).strip())
        else:
            t, _ = parse_answer(str(r.get("answer", "")))
            answers.append(t.strip() if t else "")

    n = len(rows)
    scored: List[Tuple[float, int, int]] = []
    for i in range(n):
        if not answers[i]:
            continue
        ti = sets[i]
        for j in range(i + 1, n):
            if not answers[j] or answers[i] == answers[j]:
                continue
            tj = sets[j]
            uni = ti | tj
            if not uni:
                continue
            jacc = len(ti & tj) / len(uni)
            if jacc >= min_jaccard:
                scored.append((jacc, i, j))
    scored.sort(reverse=True, key=lambda t: t[0])
    return scored


def _greedy_disjoint(
    scored: List[Tuple[float, int, int]], num_pairs: int, rng: random.Random
) -> List[Dict[str, Any]]:
    used: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for jacc, i, j in scored:
        if i in used or j in used:
            continue
        if rng.random() < 0.5:
            src, tgt = i, j
        else:
            src, tgt = j, i
        used.add(i)
        used.add(j)
        out.append(
            {
                "pair_id": len(out),
                "jaccard": jacc,
                "source_idx": src,
                "target_idx": tgt,
            }
        )
        if len(out) >= num_pairs:
            break
    return out


def _greedy_overlap_ok(
    scored: List[Tuple[float, int, int]], num_pairs: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """
    Take top unique (i,j) edges by Jaccard; the same index may appear in multiple pairs.
    Increases N when disjoint matching saturates.
    """
    seen_edge: Set[Tuple[int, int]] = set()
    out: List[Dict[str, Any]] = []
    for jacc, i, j in scored:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen_edge:
            continue
        seen_edge.add((a, b))
        if rng.random() < 0.5:
            src, tgt = i, j
        else:
            src, tgt = j, i
        out.append(
            {
                "pair_id": len(out),
                "jaccard": jacc,
                "source_idx": src,
                "target_idx": tgt,
            }
        )
        if len(out) >= num_pairs:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GSM8K structure-matched pairs JSONL.")
    ap.add_argument(
        "--dataset_path",
        type=str,
        default="data/gsm8k_local.jsonl",
        help="GSM8K JSONL (one object per line).",
    )
    ap.add_argument(
        "--num_pairs",
        type=int,
        default=60,
        help="Target number of pairs (see --allow_overlap).",
    )
    ap.add_argument(
        "--min_jaccard",
        type=float,
        default=0.32,
        help="Minimum word-set Jaccard (digits -> <NUM>) for a candidate pair.",
    )
    ap.add_argument(
        "--allow_overlap",
        action="store_true",
        help="Allow the same problem index in multiple pairs (more pairs; samples are dependent).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output",
        type=str,
        default="data/rq4/gsm8k_template_pairs_pilot.jsonl",
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    path = Path(args.dataset_path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rng = random.Random(args.seed)
    scored = _all_scored_pairs(rows, args.min_jaccard)
    if args.allow_overlap:
        pairs = _greedy_overlap_ok(scored, args.num_pairs, rng)
        mode = "overlap-allowed"
    else:
        pairs = _greedy_disjoint(scored, args.num_pairs, rng)
        mode = "disjoint"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Candidate edges (>= {args.min_jaccard}): {len(scored)}; "
        f"wrote {len(pairs)} {mode} pairs to {out}"
    )
    if len(pairs) < args.num_pairs:
        print(
            "Warning: fewer pairs than requested — lower --min_jaccard, add --allow_overlap, "
            "or use a larger dataset_path.",
            flush=True,
        )


if __name__ == "__main__":
    main()
