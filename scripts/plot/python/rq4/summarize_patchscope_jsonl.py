#!/usr/bin/env python
"""
Summarize RQ4 Patchscope JSONL + optional .meta.json (global baseline).

**Scope:** compares each patched decode to the **unpatched greedy baseline** on the same
target prompt (in ``.meta.json``). It does **not** measure overlap with GSM8K gold answers or
reference CoT; for that use ``analyze_patchscope_vs_gold.py``.

Quantifies how much each patched decode differs from the **single-run** baseline
(stored once in meta) and aggregates by step.

Metrics (per row, when baseline and patchscope_text are present):
  - seq_match_ratio: difflib.SequenceMatcher ratio in [0, 1] (higher = more similar to baseline)
  - diff_from_baseline: 1 if texts differ (after strip), else 0
  - len_patched / len_baseline: character lengths

Aggregates:
  - by step: mean seq_match_ratio, fraction diff_from_baseline, counts

Usage:
  python scripts/plot/python/rq4/summarize_patchscope_jsonl.py \\
    --jsonl outputs/rq4/patchscope/foo.jsonl \\
    --out_csv outputs/rq4/patchscope/foo.per_row.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _meta_path_for_jsonl(jsonl_path: Path) -> Path:
    """Map foo.jsonl / foo.jsonl.rank0 -> foo.meta.json (rank0 writes meta next to logical output)."""
    name = jsonl_path.name
    if name.endswith(".jsonl"):
        base = jsonl_path.stem
    elif ".jsonl." in name:
        base = name.split(".jsonl")[0]
    else:
        base = jsonl_path.stem
    return jsonl_path.parent / f"{base}.meta.json"


def _load_meta(jsonl_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    meta_path = _meta_path_for_jsonl(jsonl_path)
    if not meta_path.is_file():
        return None, None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f), meta_path


def _row_metrics(
    patched: Optional[str],
    baseline: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "seq_match_ratio": None,
        "diff_from_baseline": None,
        "len_patched": None,
        "len_baseline": None,
    }
    if patched is None or baseline is None:
        return out
    p = str(patched).strip()
    b = str(baseline).strip()
    out["len_patched"] = len(p)
    out["len_baseline"] = len(b)
    out["diff_from_baseline"] = int(p != b)
    out["seq_match_ratio"] = SequenceMatcher(None, p, b).ratio()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize RQ4 patchscope JSONL.")
    ap.add_argument("--jsonl", required=True, type=Path, help="Path to patchscope .jsonl")
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Override path to .meta.json (default: sibling foo.meta.json for foo.jsonl or foo.jsonl.rank*)",
    )
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional per-row CSV output")
    args = ap.parse_args()

    jsonl_path = args.jsonl
    meta, meta_path = _load_meta(jsonl_path)
    if args.meta is not None:
        with args.meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_path = args.meta

    baseline_text: Optional[str] = None
    if meta is not None:
        baseline_text = meta.get("baseline_patchscope_text")
        if baseline_text is not None:
            baseline_text = str(baseline_text)

    per_step_ratios: Dict[int, List[float]] = defaultdict(list)
    per_step_diff: Dict[int, List[int]] = defaultdict(list)
    n_ok = 0
    n_missing_text = 0
    rows_out: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step")
            if step is None:
                continue
            try:
                step_i = int(step)
            except (TypeError, ValueError):
                continue
            pt = rec.get("patchscope_text")
            if pt is None or rec.get("patch_error"):
                n_missing_text += 1
                rows_out.append(
                    {
                        "sample_id": rec.get("sample_id"),
                        "step": step_i,
                        "seq_match_ratio": None,
                        "diff_from_baseline": None,
                        "len_patched": None,
                        "len_baseline": len(baseline_text) if baseline_text else None,
                    }
                )
                continue

            m = _row_metrics(pt, baseline_text)
            if m["seq_match_ratio"] is not None:
                per_step_ratios[step_i].append(float(m["seq_match_ratio"]))
                per_step_diff[step_i].append(int(m["diff_from_baseline"] or 0))
                n_ok += 1
            rows_out.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "step": step_i,
                    **m,
                }
            )

    # Print summary
    print(f"JSONL: {jsonl_path}")
    if meta_path:
        print(f"Meta:  {meta_path}")
    else:
        print("Meta:  (not found — seq_match_ratio vs baseline unavailable)")
    print(f"Rows with usable patchscope_text: {n_ok}")
    print(f"Rows missing patch / error:       {n_missing_text}")
    print()

    if baseline_text is None:
        print("No baseline_patchscope_text in meta; only per-row lengths / missing stats apply.")
        return

    print("By step (vs global baseline in meta):")
    print(f"{'step':>6} {'n':>6} {'mean_sim':>10} {'std_sim':>10} {'frac_diff':>10}")
    for step_i in sorted(per_step_ratios.keys()):
        rs = per_step_ratios[step_i]
        ds = per_step_diff[step_i]
        mean_r = statistics.mean(rs) if rs else float("nan")
        std_r = statistics.stdev(rs) if len(rs) > 1 else 0.0
        frac_diff = sum(ds) / len(ds) if ds else float("nan")
        print(f"{step_i:6d} {len(rs):6d} {mean_r:10.4f} {std_r:10.4f} {frac_diff:10.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "step",
            "seq_match_ratio",
            "diff_from_baseline",
            "len_patched",
            "len_baseline",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"\nWrote per-row CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
