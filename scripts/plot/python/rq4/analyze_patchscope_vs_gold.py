#!/usr/bin/env python
"""
Analyze RQ4 Patchscope decodes **against GSM8K supervision**, not against the unpatched baseline.

**Primary (recommended): numbers and operators** — largely unaffected by boilerplate like
``###`` or ``The answer is:``, which can spuriously align substring metrics with gold text.

For each JSONL row (when ``patchscope_text`` is present), computes:

  - **Final answer (numeric only)**
    - ``gold_number_hit``: float value of ``gold_answer`` appears among numbers parsed from decode
      (regex numbers → float equality; **primary** match to final gold)

  - **Step-aligned official CoT line** (``golden_cot_step``)
    - ``cot_number_hit``: at least one **numeric value** from the CoT line appears as an extracted
      number in decode (**strict float set**, no digit substring tricks)
    - ``cot_op_recall``: operators in ``<<...>>``, ``+ - * / = ( )``, ``<<``, ``>>`` — fraction of
      **distinct** operator tokens present on the CoT line that also appear in decode
      (0 if CoT line has no operators)

  - **Secondary / noisy**
    - ``gold_substring``: token-safe substring for numeric gold (``4`` does **not** match inside
      ``64``); non-numeric gold still uses plain substring
    - ``cot_word_jaccard``: token Jaccard vs CoT line (lexical)

Aggregates counts and rates **overall**, **by latent step**, and optionally **by run** when you pass
multiple ``--jsonl`` paths.

Usage:
  python scripts/plot/python/rq4/analyze_patchscope_vs_gold.py \\
    --jsonl outputs/rq4/patchscope/gsm8k_codi_gpt2_patchscope.jsonl

  python scripts/plot/python/rq4/analyze_patchscope_vs_gold.py \\
    --jsonl a.jsonl --jsonl b.jsonl --out_csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# GSM8K-style: << >> and single-char arithmetic / punctuation
_OP_RE = re.compile(r"<<|>>|[+\-*/=()^×÷%]")


def _extract_operator_set(s: str) -> Set[str]:
    return set(_OP_RE.findall(s or ""))


def _cot_operator_recall(cot_line: str, decode: str) -> float:
    """Fraction of distinct operators on the CoT line that appear in decode (0 if CoT has no ops)."""
    co = _extract_operator_set(cot_line)
    if not co:
        return 0.0
    de = _extract_operator_set(decode)
    return len(co & de) / len(co)


def _extract_number_strings(s: str) -> List[str]:
    """Digit runs that look like GSM8K numbers (allow commas, decimals)."""
    if not s:
        return []
    parts = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", s)
    out: List[str] = []
    for p in parts:
        norm = p.replace(",", "")
        if norm not in out:
            out.append(norm)
    return out


def _to_float_safe(x: str) -> Optional[float]:
    try:
        return float(x.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _tokens(s: str) -> Set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", s) if len(t) >= 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _gold_substring_token_safe(gold: str, text: str) -> bool:
    """
    If ``gold`` looks like a single number, require it to appear as a **numeric token**:
    ``4`` must not count as matching inside ``64`` (``\"4\" in \"64\"`` is True in Python otherwise).
    """
    g = str(gold).strip().replace(",", "")
    if not g:
        return False
    d = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", g):
        pat = r"(?<![0-9])" + re.escape(g) + r"(?![0-9])"
        return re.search(pat, d) is not None
    return g in d


def _cot_line_number_hit(decode: str, cot_line: str) -> bool:
    """True if any float parsed from cot_line appears in decode's extracted number set (strict)."""
    if not cot_line.strip():
        return False
    dset = {_to_float_safe(x) for x in _extract_number_strings(decode)}
    dset.discard(None)
    for cn in _extract_number_strings(cot_line):
        cf = _to_float_safe(cn)
        if cf is not None and cf in dset:
            return True
    return False


def _gold_hit(decode: str, gold_answer: Optional[str]) -> Tuple[bool, bool]:
    """
    Returns (substring_hit, numeric_hit).
    Substring uses :func:`_gold_substring_token_safe` so ``4`` is not found inside ``64``.
    """
    if not decode or gold_answer is None:
        return False, False
    g = str(gold_answer).strip().replace(",", "")
    if not g:
        return False, False
    dnorm = decode.replace(",", "")
    sub = _gold_substring_token_safe(g, dnorm)
    gv = _to_float_safe(g)
    dnums = [_to_float_safe(x) for x in _extract_number_strings(decode)]
    dnums = [x for x in dnums if x is not None]
    num_hit = gv is not None and any(abs(gv - x) < 1e-5 for x in dnums)
    return sub, num_hit


def _row_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    dec = rec.get("patchscope_text")
    if dec is None or rec.get("patch_error"):
        return {}
    dec = str(dec)
    ga = rec.get("gold_answer")
    cot = rec.get("golden_cot_step") or ""

    g_sub, g_num = _gold_hit(dec, ga)
    cot_nonempty = bool(str(cot).strip())
    cot_s = str(cot)
    cot_num = _cot_line_number_hit(dec, cot_s) if cot_nonempty else False
    j = _jaccard(_tokens(dec), _tokens(cot_s)) if cot_nonempty else 0.0
    cot_op_rec = _cot_operator_recall(cot_s, dec) if cot_nonempty else 0.0
    cot_has_ops = bool(_extract_operator_set(cot_s)) if cot_nonempty else False

    return {
        "gold_substring": int(g_sub),
        "gold_number_hit": int(g_num),
        "cot_nonempty": int(cot_nonempty),
        "cot_number_hit": int(cot_num),
        "cot_op_recall": round(cot_op_rec, 4),
        "cot_has_ops": int(cot_has_ops),
        "cot_word_jaccard": round(j, 4),
        "decode_num_distinct": len(
            {x for x in (_to_float_safe(t) for t in _extract_number_strings(dec)) if x is not None}
        ),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Overall stats and per-step stats."""
    n = 0
    n_gold = 0
    sum_gsub = 0
    sum_gnum = 0
    n_cot_eligible = 0
    sum_cot_hit = 0
    sum_jacc = 0.0
    sum_cot_op_rec = 0.0
    n_cot_with_ops = 0
    sum_cot_op_rec_has_ops = 0.0
    sum_decode_num_distinct = 0

    per_step: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "gold_substring": 0,
            "gold_number_hit": 0,
            "cot_nonempty": 0,
            "cot_number_hit": 0,
            "sum_jacc": 0.0,
            "sum_cot_op_rec": 0.0,
            "cot_with_ops": 0,
            "sum_cot_op_rec_has_ops": 0.0,
            "sum_decode_num_distinct": 0,
        }
    )

    for r in rows:
        m = _row_metrics(r)
        if not m:
            continue
        st = int(r.get("step", -1))
        n += 1
        sum_decode_num_distinct += m["decode_num_distinct"]
        if r.get("gold_answer") is not None and str(r.get("gold_answer")).strip():
            n_gold += 1
            sum_gsub += m["gold_substring"]
            sum_gnum += m["gold_number_hit"]

        per_step[st]["n"] += 1
        per_step[st]["gold_substring"] += m["gold_substring"]
        per_step[st]["gold_number_hit"] += m["gold_number_hit"]
        per_step[st]["cot_nonempty"] += m["cot_nonempty"]
        per_step[st]["cot_number_hit"] += m["cot_number_hit"]
        per_step[st]["sum_jacc"] += m["cot_word_jaccard"]
        per_step[st]["sum_cot_op_rec"] += m["cot_op_recall"]
        per_step[st]["sum_decode_num_distinct"] += m["decode_num_distinct"]
        if m["cot_has_ops"]:
            per_step[st]["cot_with_ops"] += 1
            per_step[st]["sum_cot_op_rec_has_ops"] += m["cot_op_recall"]

        if m["cot_nonempty"]:
            n_cot_eligible += 1
            sum_cot_hit += m["cot_number_hit"]
            sum_jacc += m["cot_word_jaccard"]
            sum_cot_op_rec += m["cot_op_recall"]
            if m["cot_has_ops"]:
                n_cot_with_ops += 1
                sum_cot_op_rec_has_ops += m["cot_op_recall"]

    overall = {
        "rows_with_decode": n,
        "rows_with_gold_answer": n_gold,
        "rate_gold_substring": sum_gsub / n_gold if n_gold else None,
        "rate_gold_number_hit": sum_gnum / n_gold if n_gold else None,
        "rows_cot_nonempty": n_cot_eligible,
        "rate_cot_number_if_cot_nonempty": sum_cot_hit / n_cot_eligible if n_cot_eligible else None,
        "mean_cot_op_recall_if_cot_nonempty": sum_cot_op_rec / n_cot_eligible if n_cot_eligible else None,
        "mean_cot_op_recall_if_cot_has_ops": (
            sum_cot_op_rec_has_ops / n_cot_with_ops if n_cot_with_ops else None
        ),
        "rows_cot_with_ops": n_cot_with_ops,
        "mean_cot_word_jaccard_if_cot_nonempty": sum_jacc / n_cot_eligible if n_cot_eligible else None,
        "mean_decode_num_distinct": sum_decode_num_distinct / n if n else None,
    }

    per_step_out: Dict[int, Dict[str, Any]] = {}
    for st in sorted(per_step.keys()):
        d = per_step[st]
        nc = d["cot_nonempty"]
        cw = d["cot_with_ops"]
        per_step_out[st] = {
            "n": d["n"],
            "rate_gold_substring": d["gold_substring"] / d["n"] if d["n"] else None,
            "rate_gold_number": d["gold_number_hit"] / d["n"] if d["n"] else None,
            "rate_cot_number_among_nonempty_cot": (
                d["cot_number_hit"] / nc if nc else None
            ),
            "mean_cot_op_rec_if_cot_nonempty": d["sum_cot_op_rec"] / nc if nc else None,
            "mean_cot_op_rec_if_cot_has_ops": (
                d["sum_cot_op_rec_has_ops"] / cw if cw else None
            ),
            "mean_cot_jaccard_among_nonempty_cot": (d["sum_jacc"] / nc if nc else None),
            "mean_decode_num_distinct": d["sum_decode_num_distinct"] / d["n"] if d["n"] else None,
        }

    return overall, per_step_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Patchscope decode vs GSM8K gold / CoT line.")
    ap.add_argument(
        "--jsonl",
        action="append",
        required=True,
        help="Path to patchscope JSONL (repeat for multiple runs)",
    )
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional per-row CSV")
    args = ap.parse_args()

    for jsonl_path in args.jsonl:
        path = Path(jsonl_path)
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        overall, per_step = _aggregate(rows)

        print("=" * 88)
        print(path)
        print("=" * 88)
        print("— Primary: numbers & operators (not confounded by ### / The answer is:) —")
        print(
            f"  Gold final answer — numeric hit only: {overall['rate_gold_number_hit']!s}  "
            f"(n with gold_answer={overall['rows_with_gold_answer']})"
        )
        print(
            f"  CoT line — any CoT number in decode (strict float): "
            f"{overall['rate_cot_number_if_cot_nonempty']!s}  "
            f"(golden_cot_step nonempty: {overall['rows_cot_nonempty']})"
        )
        print(
            f"  CoT line — mean operator recall vs decode: "
            f"if cot nonempty: {overall['mean_cot_op_recall_if_cot_nonempty']!s}  |  "
            f"if cot has ops: {overall['mean_cot_op_recall_if_cot_has_ops']!s}  "
            f"(rows with ops on CoT line: {overall['rows_cot_with_ops']})"
        )
        print(f"  Decode — mean distinct float count per row: {overall['mean_decode_num_distinct']!s}")
        print()
        print("— Secondary (substring can track templates; Jaccard = lexical) —")
        print(f"  gold substring rate: {overall['rate_gold_substring']!s}")
        print(f"  mean token Jaccard(decode, cot line) if cot nonempty: {overall['mean_cot_word_jaccard_if_cot_nonempty']!s}")
        print()
        print(
            f"{'step':>5} {'n':>6} {'P_g#':>8} {'cot#':>8} {'op|cot':>8} {'op|ops':>8} "
            f"{'#dist':>8} {'sub':>8} {'jac':>8}"
        )
        def _pf(x: Optional[float], w: int = 8, prec: int = 4) -> str:
            if x is None:
                return " " * (w - 3) + "—"
            return f"{x:{w}.{prec}f}"

        for st in sorted(per_step.keys()):
            d = per_step[st]
            print(
                f"{st:5d} {d['n']:6d} "
                f"{_pf(d['rate_gold_number'])} "
                f"{_pf(d['rate_cot_number_among_nonempty_cot'])} "
                f"{_pf(d['mean_cot_op_rec_if_cot_nonempty'])} "
                f"{_pf(d['mean_cot_op_rec_if_cot_has_ops'])} "
                f"{_pf(d['mean_decode_num_distinct'], prec=2)} "
                f"{_pf(d['rate_gold_substring'])} "
                f"{_pf(d['mean_cot_jaccard_among_nonempty_cot'])}"
            )
        print()

        if args.out_csv:
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "jsonl",
                "sample_id",
                "step",
                "gold_substring",
                "gold_number_hit",
                "cot_nonempty",
                "cot_number_hit",
                "cot_op_recall",
                "cot_has_ops",
                "cot_word_jaccard",
                "decode_num_distinct",
            ]
            # Only write once for first file if single path — append mode messy; write per-file name
            out_path = args.out_csv
            if len(args.jsonl) > 1:
                out_path = args.out_csv.parent / f"{path.stem}.vs_gold.csv"
            with out_path.open("w", newline="", encoding="utf-8") as cf:
                w = csv.DictWriter(cf, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    m = _row_metrics(r)
                    if not m:
                        continue
                    w.writerow(
                        {
                            "jsonl": str(path),
                            "sample_id": r.get("sample_id"),
                            "step": r.get("step"),
                            **m,
                        }
                    )
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
