from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

META_KEYS = {"sample_id", "rank", "step_i", "step_j", "mode"}


def parse_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    suffix = "_explicit_graph"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_int(val) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def compute_metrics(edges: Dict[Tuple[int, int], float], num_steps: int) -> Dict[str, float]:
    total = sum(edges.values())
    if total <= 0:
        return {
            "total_weight": total,
            "locality": float("nan"),
            "early_out": float("nan"),
            "late_in": float("nan"),
            "span": float("nan"),
        }

    local_sum = sum(edges.get((t, t + 1), 0.0) for t in range(1, max(num_steps, 2)))
    early_sum = sum(edges.get((1, s), 0.0) for s in range(2, num_steps + 1))
    late_sum = sum(edges.get((t, num_steps), 0.0) for t in range(1, num_steps))
    span_sum = sum((s - t) * w for (t, s), w in edges.items())

    return {
        "total_weight": total,
        "locality": local_sum / total,
        "early_out": early_sum / total,
        "late_in": late_sum / total,
        "span": span_sum / total if total > 0 else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute structure metrics for explicit graphs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("*.jsonl")):
        model = parse_name(path)
        metrics: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
        max_step = 0
        for rec in iter_jsonl(path):
            step_i = _as_int(rec.get("step_i"))
            step_j = _as_int(rec.get("step_j"))
            if step_i is None or step_j is None:
                continue
            max_step = max(max_step, step_i, step_j)
            for k, v in rec.items():
                if k in META_KEYS:
                    continue
                if not isinstance(v, (int, float)):
                    continue
                metrics[k][(step_i, step_j)] += float(v)
        if max_step <= 0:
            continue
        for metric, edges in metrics.items():
            stat = compute_metrics(edges, max_step)
            rows.append(
                {
                    "dataset": args.dataset,
                    "model": model,
                    "metric": metric,
                    "mode": "zero",
                    **stat,
                }
            )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "metric",
                "mode",
                "total_weight",
                "locality",
                "early_out",
                "late_in",
                "span",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
