#!/usr/bin/env python
"""
Quick analyzer for ablation JSONL outputs.

It streams the JSONL file and reports:
  - numeric accuracy for baseline/ablated texts per (mode, step)
  - flip rates: baseline correct -> ablated wrong and baseline wrong -> ablated correct
  - teacher-forced log-prob deltas per (mode, step):
        * teacher_forced_delta_sum       (Δ log p(gold sequence) = base - ablt)
        * delta_logp_final_token         (Δ log p(final gold token) = base - ablt)

Assumes answers are of the form:
    {answer} <eos>
and gold_answer is a string containing the gold number (e.g., "70000").

Usage:
  python scripts/plot/python/rq1/analyze_ablation_jsonl.py --path outputs/rq1/ablation/gsm8k_coconut_gpt2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common.experiment_utils import step_order_key

MODE_DESCRIPTIONS: Dict[str, str] = {
    "zero": "Replace latent with zeros",
    "mean": "Replace latent with global mean",
    "gaussian_h": "Add Gaussian noise to latent h_t",
    "gaussian_mu": "Add Gaussian noise around global mean",
    "mean_step": "Replace latent with step-specific mean",
    "gaussian_mu_step": "Add Gaussian noise around step-specific mean",
}


# --------------------------
# Parsing helpers
# --------------------------

def extract_number(text: Any) -> Optional[float]:
    """Extract the LAST number in the answer tail (optionally after <|end-latent|>)."""
    if text is None:
        return None
    s = str(text)
    if "<|end-latent|>" in s:
        s = s.split("<|end-latent|>")[-1]
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def parse_gold(rec: Dict[str, Any]) -> Optional[float]:
    raw = rec.get("gold_answer", "")
    return extract_number(raw)


def normalize_mode_step(mode: Any, step: Any) -> Optional[Tuple[str, int]]:
    """
    Return (mode:str, step:int) with stable labels for sorting/plotting.
    Supports numeric strings/ints -> int.
    """
    if mode is None or step is None:
        return None
    try:
        m = str(mode)

        if isinstance(step, str):
            s = step.strip().lower()
            if s.lstrip("-").isdigit():
                return m, int(s)

        return m, int(step)
    except Exception:
        return None


# --------------------------
# Stats helpers
# --------------------------

def update_acc_stats(
    stats: Dict[Tuple[str, int], Dict[str, float]],
    key: Tuple[str, int],
    gold: Optional[float],
    pred: Optional[float],
) -> None:
    bucket = stats.setdefault(key, {"total": 0.0, "correct": 0.0})
    bucket["total"] += 1.0
    if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
        bucket["correct"] += 1.0


def format_bucket(bucket: Dict[str, float]) -> str:
    total = bucket.get("total", 0.0)
    correct = bucket.get("correct", 0.0)
    acc = correct / total if total else 0.0
    return f"acc={acc:.3f} (correct={int(correct)}/{int(total)})"


def flip_bucket_default() -> Dict[str, float]:
    return {"total": 0.0, "correct_to_wrong": 0.0, "wrong_to_correct": 0.0}


def flip_rates(bucket: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    total = bucket.get("total", 0.0)
    w2r = bucket.get("wrong_to_correct", 0.0)
    r2w = bucket.get("correct_to_wrong", 0.0)
    rate = (w2r + r2w) / total if total else 0.0
    w2r_rate = w2r / total if total else 0.0
    r2w_rate = r2w / total if total else 0.0
    return rate, w2r_rate, r2w_rate, total, w2r, r2w


def format_flip(bucket: Dict[str, float]) -> str:
    rate, w2r_rate, r2w_rate, total, w2r, r2w = flip_rates(bucket)
    if total == 0:
        return "flip rate=n/a"
    return (
        f"flip rate={rate:.3f} "
        f"(wrong->right={w2r_rate:.3f} ({int(w2r)}/{int(total)}) "
        f"right->wrong={r2w_rate:.3f} ({int(r2w)}/{int(total)}))"
    )


# --------------------------
# Heatmap summarizers
# --------------------------

def summarize_delta_heatmap(
    ablated_stats: Dict[Tuple[str, int], Dict[str, float]],
    baseline_acc: float,
) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
    modes = sorted({m for (m, _) in ablated_stats.keys()})
    steps = sorted({s for (_, s) in ablated_stats.keys()}, key=step_order_key)

    grid: List[List[float]] = []
    ann: List[List[str]] = []
    for m in modes:
        row: List[float] = []
        ann_row: List[str] = []
        for s in steps:
            b = ablated_stats.get((m, s))
            if not b or b.get("total", 0.0) == 0.0:
                row.append(float("nan"))
                ann_row.append("")
                continue
            acc = b["correct"] / b["total"]
            delta = acc - baseline_acc
            row.append(delta)
            ann_row.append(f"{delta:+.3f}")
        grid.append(row)
        ann.append(ann_row)
    return modes, steps, grid, ann


def summarize_flip_heatmap(
    flip_stats: Dict[Tuple[str, int], Dict[str, float]],
) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
    modes = sorted({m for (m, _) in flip_stats.keys()})
    steps = sorted({s for (_, s) in flip_stats.keys()}, key=step_order_key)

    grid: List[List[float]] = []
    ann: List[List[str]] = []
    for m in modes:
        row: List[float] = []
        ann_row: List[str] = []
        for s in steps:
            b = flip_stats.get((m, s), {})
            rate, _, _, total, _, _ = flip_rates(b)
            if total == 0:
                row.append(float("nan"))
                ann_row.append("")
            else:
                row.append(rate)
                ann_row.append(f"{rate:.2f}")
        grid.append(row)
        ann.append(ann_row)
    return modes, steps, grid, ann


def summarize_logp_heatmap(
    logp_stats: Dict[Tuple[str, int], Dict[str, float]],
    kind: str,
) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
    assert kind in {"seq", "final"}
    modes = sorted({m for (m, _) in logp_stats.keys()})
    steps = sorted({s for (_, s) in logp_stats.keys()}, key=step_order_key)

    grid: List[List[float]] = []
    ann: List[List[str]] = []
    for m in modes:
        row: List[float] = []
        ann_row: List[str] = []
        for s in steps:
            b = logp_stats.get((m, s))
            if not b:
                row.append(float("nan"))
                ann_row.append("")
                continue

            if kind == "seq":
                c = b.get("count_seq", 0.0)
                if c == 0:
                    row.append(float("nan"))
                    ann_row.append("")
                    continue
                val = b["sum_delta_seq"] / c
            else:
                c = b.get("count_final", 0.0)
                if c == 0:
                    row.append(float("nan"))
                    ann_row.append("")
                    continue
                val = b["sum_delta_final"] / c

            row.append(val)
            ann_row.append(f"{val:+.2f}")
        grid.append(row)
        ann.append(ann_row)
    return modes, steps, grid, ann


# --------------------------
# Plot helper
# --------------------------

def plot_heatmap(
    data: List[List[float]],
    modes: List[str],
    steps: List[int],
    title: str,
    cmap: str,
    annotations: List[List[str]],
    out_path: str,
    center: Optional[float] = None,
) -> None:
    arr = np.array(data, dtype=float)
    mask = ~np.isfinite(arr)
    arr_masked = np.ma.array(arr, mask=mask)

    norm = None
    if center is not None:
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size:
            max_abs = float(np.max(np.abs(finite_vals)))
            if max_abs > 0:
                norm = TwoSlopeNorm(vcenter=center, vmin=-max_abs, vmax=max_abs)

    figsize = (max(8.0, len(steps) * 1.2), max(4.0, len(modes) * 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr_masked, cmap=cmap, norm=norm)

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("step")
    ax.set_ylabel("mode")
    ax.set_title(title)

    for i, row in enumerate(annotations):
        for j, text in enumerate(row):
            if not mask[i, j] and text:
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to ablation JSONL.")
    args = parser.parse_args()

    baseline_stats: Dict[str, float] = {"total": 0.0, "correct": 0.0}
    baseline_seen: Set[Any] = set()
    baseline_outcomes: Dict[Any, Optional[bool]] = {}  # sid -> correct? (None if unparseable)

    ablated_stats: Dict[Tuple[str, int], Dict[str, float]] = {}
    flip_stats: Dict[Tuple[str, int], Dict[str, float]] = {}
    logp_stats: Dict[Tuple[str, int], Dict[str, float]] = {}

    with open(args.path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            sid = rec.get("sample_id")
            gold_val = parse_gold(rec)

            batch_idx = rec.get("batch_idx", 0)

            # ---- baseline (once per sample_id) ----
            baseline_texts = (rec.get("baseline") or {}).get("text") or []
            if baseline_texts and sid not in baseline_seen:
                text_idx = batch_idx if isinstance(batch_idx, int) else 0
                if text_idx < 0 or text_idx >= len(baseline_texts):
                    text_idx = 0
                pred_val = extract_number(baseline_texts[text_idx])

                baseline_stats["total"] += 1.0
                correct: Optional[bool] = None
                if pred_val is not None and gold_val is not None:
                    correct = abs(pred_val - gold_val) < 1e-6
                    if correct:
                        baseline_stats["correct"] += 1.0
                baseline_outcomes[sid] = correct
                baseline_seen.add(sid)

            # ---- ablated stats per (mode, step) ----
            ms = normalize_mode_step(rec.get("mode"), rec.get("step"))
            if ms is None:
                continue
            mode, step = ms
            key = (mode, step)

            ablated_texts = (rec.get("ablated") or {}).get("text") or []
            if ablated_texts:
                text_idx = batch_idx if isinstance(batch_idx, int) else 0
                if text_idx < 0 or text_idx >= len(ablated_texts):
                    text_idx = 0
                pred_val = extract_number(ablated_texts[text_idx])
                update_acc_stats(ablated_stats, key, gold_val, pred_val)

                # flips
                base_correct = baseline_outcomes.get(sid)
                ablt_correct: Optional[bool] = None
                if pred_val is not None and gold_val is not None:
                    ablt_correct = abs(pred_val - gold_val) < 1e-6

                if base_correct is not None and ablt_correct is not None:
                    fb = flip_stats.setdefault(key, flip_bucket_default())
                    fb["total"] += 1.0
                    if base_correct and not ablt_correct:
                        fb["correct_to_wrong"] += 1.0
                    elif (not base_correct) and ablt_correct:
                        fb["wrong_to_correct"] += 1.0

            # ---- teacher-forced logp deltas ----
            delta_seq = rec.get("teacher_forced_delta_sum", None)
            delta_final = rec.get("delta_logp_final_token", None)

            if delta_seq is not None:
                lb = logp_stats.setdefault(
                    key,
                    {"count_seq": 0.0, "sum_delta_seq": 0.0, "count_final": 0.0, "sum_delta_final": 0.0},
                )
                lb["count_seq"] += 1.0
                lb["sum_delta_seq"] += float(delta_seq)

            if delta_final is not None:
                lb = logp_stats.setdefault(
                    key,
                    {"count_seq": 0.0, "sum_delta_seq": 0.0, "count_final": 0.0, "sum_delta_final": 0.0},
                )
                lb["count_final"] += 1.0
                lb["sum_delta_final"] += float(delta_final)

    # ---- print summaries ----
    print("=== Baseline ===")
    print(format_bucket(baseline_stats))

    baseline_acc = baseline_stats["correct"] / baseline_stats["total"] if baseline_stats["total"] else float("nan")

    print("\n=== Ablated (greedy accuracy & flips) ===")
    modes_present = sorted({m for (m, _) in ablated_stats.keys()})
    if modes_present:
        print("Mode descriptions:")
        for m in modes_present:
            print(f"  {m}: {MODE_DESCRIPTIONS.get(m, 'N/A')}")

    for (mode, step) in sorted(ablated_stats.keys(), key=lambda t: (t[0], step_order_key(t[1]))):
        print(
            f"{mode:15s} step={step}: "
            f"{format_bucket(ablated_stats[(mode, step)])} | "
            f"{format_flip(flip_stats.get((mode, step), flip_bucket_default()))}"
        )

    if logp_stats:
        print("\n=== Teacher-forced logp deltas (base - ablated) ===")
        print("Positive: ablation hurts gold (lower p). Negative: ablation helps gold (higher p).")
        for (mode, step) in sorted(logp_stats.keys(), key=lambda t: (t[0], step_order_key(t[1]))):
            b = logp_stats[(mode, step)]
            seq_msg = "n/a"
            fin_msg = "n/a"
            if b.get("count_seq", 0.0) > 0:
                seq_msg = f"{(b['sum_delta_seq']/b['count_seq']):+.3f} (n={int(b['count_seq'])})"
            if b.get("count_final", 0.0) > 0:
                fin_msg = f"{(b['sum_delta_final']/b['count_final']):+.3f} (n={int(b['count_final'])})"
            print(f"{mode:15s} step={step}: mean Δlogp_seq={seq_msg}, mean Δlogp_final={fin_msg}")

    # ---- heatmaps ----
    prefix = str(Path(args.path).with_suffix(""))

    if ablated_stats:
        rows, steps, grid, ann = summarize_delta_heatmap(ablated_stats, baseline_acc)
        plot_heatmap(
            grid, rows, steps,
            title="Delta accuracy vs baseline",
            cmap="RdBu_r",
            annotations=ann,
            out_path=f"{prefix}_delta_acc_heatmap.png",
            center=0.0,
        )

    if flip_stats:
        rows, steps, grid, ann = summarize_flip_heatmap(flip_stats)
        plot_heatmap(
            grid, rows, steps,
            title="Flip rate (w->r + r->w) by mode/step",
            cmap="plasma",
            annotations=ann,
            out_path=f"{prefix}_flip_heatmap.png",
            center=None,
        )

    if logp_stats:
        rows, steps, grid, ann = summarize_logp_heatmap(logp_stats, kind="seq")
        plot_heatmap(
            grid, rows, steps,
            title="Mean Δ log p(gold sequence) (base - ablt)",
            cmap="RdBu_r",
            annotations=ann,
            out_path=f"{prefix}_delta_logp_seq_heatmap.png",
            center=0.0,
        )

        rows, steps, grid, ann = summarize_logp_heatmap(logp_stats, kind="final")
        plot_heatmap(
            grid, rows, steps,
            title="Mean Δ log p(final gold token) (base - ablt)",
            cmap="RdBu_r",
            annotations=ann,
            out_path=f"{prefix}_delta_logp_final_heatmap.png",
            center=0.0,
        )


if __name__ == "__main__":
    main()
