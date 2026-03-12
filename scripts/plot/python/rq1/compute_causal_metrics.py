from __future__ import annotations

"""
Summarize causal-graph JSONL outputs into scalar metrics.

Usage:
  python scripts/plot/python/rq1/compute_causal_metrics.py \
    --input_dir outputs/rq2/latent_graph \
    --output_csv outputs/rq2/latent_graph/causal_metrics.csv \
    --plot_dir outputs/rq2/latent_graph/plots
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


META_KEYS = {"sample_id", "rank", "step_i", "step_j", "mode"}


def parse_name(path: Path) -> Tuple[str, str]:
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    suffix = "_latent_graph"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    parts = name.split("_")
    if len(parts) < 2:
        return "unknown", name
    dataset = parts[0]
    model = "_".join(parts[1:])
    return dataset, model


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(
    edges: Dict[Tuple[int, int], float],
    num_steps: int,
) -> Dict[str, float]:
    total = sum(edges.values())
    if total <= 0:
        return {
            "total_weight": total,
            "locality": float("nan"),
            "early_out": float("nan"),
            "late_in": float("nan"),
            "span": float("nan"),
        }

    local_sum = sum(
        edges.get((t, t + 1), 0.0) for t in range(1, max(num_steps, 2))
    )
    early_sum = sum(edges.get((1, s), 0.0) for s in range(2, num_steps + 1))
    late_sum = sum(edges.get((t, num_steps), 0.0) for t in range(1, num_steps))
    span_sum = sum((s - t) * w for (t, s), w in edges.items())

    return {
        "total_weight": total,
        "locality": local_sum / total,
        "early_out": early_sum / total,
        "late_in": late_sum / total,
        "span": span_sum / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of causal JSONL files.")
    parser.add_argument("--output_csv", required=True, help="CSV path to write metrics table.")
    parser.add_argument("--plot_dir", default=None, help="Optional directory to save bar plots.")
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use signed edge weights (default: absolute value).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(p for p in input_dir.glob("*.jsonl") if not p.name.endswith(".rank0") and not p.name.endswith(".rank1") and not p.name.endswith(".rank2") and not p.name.endswith(".rank3"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found under {input_dir}")

    rows: List[Dict[str, object]] = []
    for path in paths:
        if path.stat().st_size == 0:
            continue
        dataset, model = parse_name(path)
        edge_sums: Dict[Tuple[str, str, int, int], float] = defaultdict(float)
        edge_counts: Dict[Tuple[str, str, int, int], int] = defaultdict(int)
        metric_names: List[str] = []
        max_step = 0

        for record in iter_jsonl(path):
            step_i_raw = record.get("step_i")
            step_j_raw = record.get("step_j")
            if step_i_raw is None or step_j_raw is None:
                continue
            try:
                step_i = int(step_i_raw)
                step_j = int(step_j_raw)
            except (ValueError, TypeError):
                continue
            if step_i >= step_j:
                continue
            mode = str(record.get("mode", "unknown"))
            max_step = max(max_step, step_j)
            if not metric_names:
                metric_names = [k for k in record.keys() if k not in META_KEYS]
            for metric in metric_names:
                val = record.get(metric)
                if val is None:
                    continue
                try:
                    weight = float(val)
                except (TypeError, ValueError):
                    continue
                if not args.signed:
                    weight = abs(weight)
                edge_sums[(mode, metric, step_i, step_j)] += weight
                edge_counts[(mode, metric, step_i, step_j)] += 1

        if not metric_names or max_step <= 1:
            continue

        for metric in metric_names:
            modes = {k[0] for k in edge_sums.keys() if k[1] == metric}
            if not modes:
                continue
            for mode in sorted(modes):
                edges: Dict[Tuple[int, int], float] = {}
                for (edge_mode, edge_metric, step_i, step_j), total in edge_sums.items():
                    if edge_mode != mode or edge_metric != metric:
                        continue
                    count = edge_counts[(edge_mode, edge_metric, step_i, step_j)]
                    if count > 0:
                        edges[(step_i, step_j)] = total / count
                stats = compute_metrics(edges, max_step)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "mode": mode,
                        **stats,
                    }
                )

    if not rows:
        raise RuntimeError("No metrics computed; check input files.")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "model", "metric", "mode", "total_weight", "locality", "early_out", "late_in", "span"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.plot_dir:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            print("matplotlib not available; skipping plots.")
            return

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        features = ["locality", "early_out", "late_in", "span"]

        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["metric"])].append(row)

        for metric, metric_rows in grouped.items():
            labels = [
                f"{r['dataset']}:{r['model']}:{r['mode']}" for r in metric_rows
            ]
            x = list(range(len(labels)))
            for feature in features:
                values = [float(r[feature]) if r[feature] is not None else math.nan for r in metric_rows]
                plt.figure(figsize=(max(6, len(labels) * 0.5), 4))
                plt.bar(x, values)
                plt.xticks(x, labels, rotation=45, ha="right")
                plt.title(f"{metric} - {feature}")
                plt.tight_layout()
                out_path = plot_dir / f"{metric}_{feature}.png"
                plt.savefig(out_path)
                plt.close()


if __name__ == "__main__":
    main()
