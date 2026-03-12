#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def softmax_pair(a: float, b: float, tau: float) -> Tuple[float, float]:
    if tau <= 0:
        return (1.0, 0.0) if a >= b else (0.0, 1.0)
    ea = math.exp(a / tau)
    eb = math.exp(b / tau)
    denom = ea + eb
    if denom <= 0:
        return 0.0, 0.0
    return ea / denom, eb / denom


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def entropy(p_a: float, p_b: float) -> float:
    ent = 0.0
    for p in (p_a, p_b):
        if p > 0:
            ent -= p * math.log(p)
    return ent


def get_norm_scores(rec: Dict) -> Tuple[float, float]:
    if rec.get("cos_A") is not None and rec.get("cos_B") is not None:
        cos_a = float(rec["cos_A"])
        cos_b = float(rec["cos_B"])
        # Map cosine similarity from [-1, 1] to [0, 1].
        return (cos_a + 1.0) * 0.5, (cos_b + 1.0) * 0.5
    return float(rec.get("score_A", 0.0)), float(rec.get("score_B", 0.0))


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem}


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RQ3 metrics from projection/intervention outputs.")
    parser.add_argument("--projection_jsonl", required=True)
    parser.add_argument("--probes_jsonl", required=True)
    parser.add_argument("--interventions_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--teacher_forced_jsonl", default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--p_mode", choices=["softmax", "sigmoid", "given"], default="softmax")
    parser.add_argument("--delta", type=float, default=0.5)
    args = parser.parse_args()

    proj = load_jsonl(Path(args.projection_jsonl))
    probes = load_jsonl(Path(args.probes_jsonl))
    interventions = load_jsonl(Path(args.interventions_jsonl))
    tf_records: List[Dict] = []
    if args.teacher_forced_jsonl:
        tf_path = Path(args.teacher_forced_jsonl)
        if tf_path.exists():
            tf_records = load_jsonl(tf_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-step metric aggregation
    metrics: Dict[Tuple[int, str, str], List[float]] = {}
    delta_by_sample: Dict[int, Dict[int, float]] = {}
    for rec in proj:
        step = rec.get("step")
        phase = rec.get("phase", "unknown")
        if not isinstance(step, int):
            continue
        norm_a, norm_b = get_norm_scores(rec)
        p_a = rec.get("p_A")
        p_b = rec.get("p_B")
        if args.p_mode == "given":
            if p_a is None or p_b is None:
                p_a, p_b = softmax_pair(norm_a, norm_b, args.tau)
            else:
                p_a = float(p_a)
                p_b = float(p_b)
        elif args.p_mode == "softmax":
            p_a, p_b = softmax_pair(norm_a, norm_b, args.tau)
        else:
            logit = (norm_a - norm_b) / args.tau if args.tau > 0 else float("inf")
            p_a = sigmoid(logit)
            p_b = 1.0 - p_a

        m_val = p_a + p_b
        ss = min(norm_a, norm_b)
        dp = abs(p_a - p_b)
        signed_dp = p_a - p_b
        ent = entropy(p_a / m_val if m_val > 0 else 0.0, p_b / m_val if m_val > 0 else 0.0)

        for name, val in (("SS", ss), ("SignedDeltaP", signed_dp), ("Entropy", ent), ("DeltaP", dp)):
            key = (step, phase, name)
            metrics.setdefault(key, []).append(float(val))

        sid = rec.get("sample_id")
        if isinstance(sid, int):
            delta_by_sample.setdefault(sid, {})[step] = float(dp)

    rows = []
    for (step, phase, name), vals in sorted(metrics.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        stats = summarize(vals)
        rows.append(
            {
                "step": step,
                "phase": phase,
                "metric": name,
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "sem": stats["sem"],
            }
        )
    # AFR + AFR by step (overall and per mode when available)
    afr_total = 0
    afr_flip = 0
    afr_by_step: Dict[int, List[int]] = {}
    afr_by_step_mode: Dict[Tuple[int, str], List[int]] = {}
    for rec in interventions:
        afr_total += 1
        if rec.get("flip_to_B"):
            afr_flip += 1
        step = rec.get("ablate_step")
        mode = rec.get("mode", "all")
        if isinstance(step, int):
            afr_by_step.setdefault(step, [0, 0])
            afr_by_step[step][0] += 1
            if rec.get("flip_to_B"):
                afr_by_step[step][1] += 1
            key = (step, str(mode))
            afr_by_step_mode.setdefault(key, [0, 0])
            afr_by_step_mode[key][0] += 1
            if rec.get("flip_to_B"):
                afr_by_step_mode[key][1] += 1
    afr = afr_flip / afr_total if afr_total else float("nan")
    for step in sorted(afr_by_step.keys()):
        total, flip = afr_by_step[step]
        rows.append(
            {
                "step": step,
                "phase": "all",
                "metric": "AFR",
                "n": total,
                "mean": flip / total if total else float("nan"),
                "std": float("nan"),
                "sem": float("nan"),
            }
        )
    for (step, mode), (total, flip) in sorted(afr_by_step_mode.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "step": step,
                "phase": mode,
                "metric": "AFR",
                "n": total,
                "mean": flip / total if total else float("nan"),
                "std": float("nan"),
                "sem": float("nan"),
            }
        )

    # Decision time: first step where DeltaP >= delta
    decision_steps = []
    for sid, step_map in delta_by_sample.items():
        for step in sorted(step_map.keys()):
            if step_map[step] >= args.delta:
                decision_steps.append(step)
                break

    # AFR summary already computed above

    # Orthogonality
    dots = [float(r.get("dot")) for r in probes if r.get("dot") is not None]
    dot_stats = summarize(dots)

    decision_stats = summarize(decision_steps)
    summary = {
        "afr_total": afr_total,
        "afr_flip": afr_flip,
        "afr": afr,
        "orthogonality_mean_dot": dot_stats["mean"],
        "orthogonality_std_dot": dot_stats["std"],
        "orthogonality_n": dot_stats["n"],
        "decision_delta": args.delta,
        "decision_step_mean": decision_stats["mean"],
        "decision_step_std": decision_stats["std"],
        "decision_step_n": decision_stats["n"],
    }

    # Teacher-forced competition metrics (optional)
    if tf_records:
        tf_metrics: Dict[Tuple[int, str], List[float]] = {}
        tf_delta_by_sample: Dict[int, Dict[int, float]] = {}
        for rec in tf_records:
            if rec.get("record_type") != "per_step":
                continue
            step = rec.get("step")
            if not isinstance(step, int):
                continue
            ss = rec.get("ss")
            dp = rec.get("delta_p")
            signed = None
            if rec.get("s_yes") is not None and rec.get("s_no") is not None:
                signed = float(rec["s_yes"]) - float(rec["s_no"])
            for name, val in (("SS", ss), ("DeltaP", dp), ("SignedDeltaP", signed)):
                if val is None:
                    continue
                key = (step, name)
                tf_metrics.setdefault(key, []).append(float(val))
            sid = rec.get("sample_id")
            if isinstance(sid, int) and dp is not None:
                tf_delta_by_sample.setdefault(sid, {})[step] = float(dp)

        tf_rows = []
        for (step, name), vals in sorted(tf_metrics.items(), key=lambda x: (x[0][0], x[0][1])):
            stats = summarize(vals)
            tf_rows.append(
                {
                    "step": step,
                    "metric": name,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "sem": stats["sem"],
                }
            )

        tf_csv = out_dir / "rq3_metrics_teacher_forced.csv"
        with tf_csv.open("w") as f:
            f.write("step,metric,n,mean,std,sem\n")
            for row in tf_rows:
                f.write(
                    f"{row['step']},{row['metric']},{row['n']},"
                    f"{row['mean']},{row['std']},{row['sem']}\n"
                )
        print("Saved:", tf_csv)

        tf_decision_steps = []
        for sid, step_map in tf_delta_by_sample.items():
            for step in sorted(step_map.keys()):
                if step_map[step] >= args.delta:
                    tf_decision_steps.append(step)
                    break
        tf_decision_stats = summarize(tf_decision_steps)
        summary["tf_decision_delta"] = args.delta
        summary["tf_decision_step_mean"] = tf_decision_stats["mean"]
        summary["tf_decision_step_std"] = tf_decision_stats["std"]
        summary["tf_decision_step_n"] = tf_decision_stats["n"]

    # Save outputs
    metrics_csv = out_dir / "rq3_metrics_per_step.csv"
    with metrics_csv.open("w") as f:
        f.write("step,phase,metric,n,mean,std,sem\n")
        for row in rows:
            f.write(
                f"{row['step']},{row['phase']},{row['metric']},{row['n']},"
                f"{row['mean']},{row['std']},{row['sem']}\n"
            )

    summary_path = out_dir / "rq3_metrics_summary.json"
    summary_path.write_text(json.dumps(sanitize_json(summary), indent=2))

    print("Saved:", metrics_csv)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
