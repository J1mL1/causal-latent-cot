#!/usr/bin/env python
"""
Plot latent-to-latent causal graph and heatmap from JSONL produced by
experiments/rq2/run_latent_causal_graph.py.

Usage:
  python scripts/plot/python/rq1/plot_causal_graph.py --path outputs/rq2/latent_graph/gsm8k_coconut_gpt2_latent_graph.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional
    nx = None  # type: ignore


def _coerce_step(val: Any) -> Any:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return val
    return val


def load_records(path: Path) -> Tuple[List[Dict], int]:
    """Load causal JSONL; return list of records and latent count (T)."""
    records: List[Dict] = []
    max_step = 0
    with path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            rec["step_i"] = _coerce_step(rec.get("step_i"))
            rec["step_j"] = _coerce_step(rec.get("step_j"))
            records.append(rec)
            step_i = rec.get("step_i")
            step_j = rec.get("step_j")
            if isinstance(step_i, int) and step_i > 0:
                max_step = max(max_step, step_i)
            if isinstance(step_j, int) and step_j > 0:
                max_step = max(max_step, step_j)
    return records, max_step


def plot_heatmap(M: np.ndarray, out_path: Path, label: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="magma", origin="lower")
    plt.colorbar(label=label)
    plt.xlabel("step j")
    plt.ylabel("step i")
    steps = np.arange(1, M.shape[0] + 1)
    plt.xticks(ticks=np.arange(len(steps)), labels=steps)
    plt.yticks(ticks=np.arange(len(steps)), labels=steps)
    plt.title(f"Latent-to-latent {label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_graph(
    edges: List[Tuple[int, int, float]],
    T: int,
    tau: float,
    topk: int,
    out_path: Path,
    answer_edges: Dict[int, float] | None = None,
    answer_label: str = "Y",
    edge_metric_name: str = "",
    answer_metric_name: str = "",
    topk_direction: str = "out",
) -> None:
    if nx is None:
        print("networkx not installed; skip graph plot.")
        return
    G = nx.DiGraph()
    for i in range(1, T + 1):
        G.add_node(i)
    if answer_edges:
        G.add_node(answer_label)
    if topk_direction == "in":
        per_j: dict[int, List[Tuple[int, float]]] = {}
        for i, j, w in edges:
            per_j.setdefault(j, []).append((i, w))
        for j, lst in per_j.items():
            for i, w in sorted(lst, key=lambda x: -x[1])[:topk]:
                if w >= tau:
                    G.add_edge(i, j, weight=w)
    else:
        per_i: dict[int, List[Tuple[int, float]]] = {}
        for i, j, w in edges:
            per_i.setdefault(i, []).append((j, w))
        for i, lst in per_i.items():
            for j, w in sorted(lst, key=lambda x: -x[1])[:topk]:
                if w >= tau:
                    G.add_edge(i, j, weight=w)
    # optional latent -> answer edges using answer_metric aggregated per i
    if answer_edges:
        max_ans = max(answer_edges.values()) if answer_edges else 1.0
        for i, w in answer_edges.items():
            G.add_edge(i, answer_label, weight=w, ans_edge=True, norm_w=w / (max_ans + 1e-8))

    # Fixed layout: nodes 0..T-1 on a line, answer node at the end.
    pos = {i: (i, 0) for i in range(1, T + 1)}
    if answer_edges:
        pos[answer_label] = (T + 1, 0)

    plt.figure(figsize=(8, 3))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos)
    edges_data = list(G.edges(data=True))
    weights = [d.get("weight", 0.0) for _, _, d in edges_data]
    colors = []
    widths = []
    if weights:
        max_w = max(weights)
        for u, v, d in edges_data:
            if d.get("ans_edge"):
                colors.append("#d62728")  # red for answer edges
                widths.append(2 + 6 * d.get("norm_w", 0.0))
            else:
                colors.append("#1f77b4")  # blue for latent-latent
                widths.append(2 + 4 * (d.get("weight", 0.0) / (max_w + 1e-8)))
        # Slight curvature so edges don't overlap completely.
        nx.draw_networkx_edges(
            G,
            pos,
            width=widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            edge_color=colors,
            connectionstyle="arc3,rad=0.0",
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): f"{d.get('weight', 0.0):.2f}" for u, v, d in edges_data},
            font_size=8,
        )
    plt.axis("off")
    title_edge = edge_metric_name if edge_metric_name else "edges"
    title_ans = answer_metric_name if answer_metric_name else "answer"
    plt.title(f"Latent edges ({title_edge}), answer edges ({title_ans}), τ={tau:.3f}, top{topk}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot causal graph from latent causal JSONL.")
    parser.add_argument("--path", required=True, help="Path to JSONL from run_latent_causal_graph.py")
    parser.add_argument("--out_dir", default=None, help="Directory to save plots (default: alongside input)")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile threshold for KL edges")
    parser.add_argument("--topk", type=int, default=2, help="Top-k edges per source step")
    parser.add_argument(
        "--topk_direction",
        default="out",
        choices=["out", "in"],
        help="Choose top-k outgoing edges per source step, or top-k incoming edges per target step.",
    )
    parser.add_argument(
        "--edge_metric",
        default="kl_mean",
        choices=["kl_mean", "delta_logp_final_token", "teacher_forced_delta_sum"],
        help="Metric for latent->latent edges and heatmap.",
    )
    parser.add_argument(
        "--answer_metric",
        default="delta_logp_final_token",
        choices=["delta_logp_final_token", "teacher_forced_delta_sum"],
        help="Metric for latent->answer edges.",
    )
    parser.add_argument(
        "--add_answer",
        action="store_true",
        help="Add edges from each latent step to a virtual answer node using answer_metric.",
    )
    parser.add_argument(
        "--answer_mode",
        default="mean",
        choices=["mean", "last"],
        help="How to aggregate answer_metric for latent->answer edges: mean over all step_j, or only last step_j.",
    )
    args = parser.parse_args()

    in_path = Path(args.path)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records, T = load_records(in_path)
    if T == 0:
        raise ValueError("No positive integer steps found in records.")
    edges = []
    for rec in records:
        step_i = rec.get("step_i")
        step_j = rec.get("step_j")
        if not isinstance(step_i, int) or not isinstance(step_j, int):
            continue
        if step_i <= 0 or step_j <= 0:
            continue
        w = rec.get(args.edge_metric, 0.0)
        edges.append((step_i, step_j, w))
    vals = np.array([w for _, _, w in edges])
    print(f"{args.edge_metric} stats: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    answer_edges: Dict[int, float] | None = None
    if args.add_answer:
        answer_edges = {}
        per_i: Dict[int, List[float]] = {}
        target_j = None
        if args.answer_mode == "last":
            target_j = T
        for rec in records:
            step_i = rec.get("step_i")
            step_j = rec.get("step_j")
            if not isinstance(step_i, int) or not isinstance(step_j, int):
                continue
            if step_i <= 0 or step_j <= 0:
                continue
            if target_j is not None and step_j != target_j:
                continue
            per_i.setdefault(step_i, []).append(rec.get(args.answer_metric, 0.0))
        for i, lst in per_i.items():
            if lst:
                answer_edges[i] = float(np.mean(lst))
        if answer_edges:
            ans_vals = np.array(list(answer_edges.values()))
            print(
                f"{args.answer_metric} (latent->answer) stats: "
                f"mean={ans_vals.mean():.4f}, min={ans_vals.min():.4f}, max={ans_vals.max():.4f}"
            )

    M = np.zeros((T, T))
    counts = np.zeros((T, T))
    for i, j, w in edges:
        M[i - 1, j - 1] += w
        counts[i - 1, j - 1] += 1
    M = np.divide(M, counts, out=np.zeros_like(M), where=counts > 0)

    heat_path = out_dir / (in_path.stem + ".heatmap.png")
    label = {
        "kl_mean": "KL mean",
        "delta_logp_final_token": "Δ log p(final token)",
        "teacher_forced_delta_sum": "Δ log p(sequence)",
    }.get(args.edge_metric, args.edge_metric)
    plot_heatmap(M, heat_path, label=label)

    tau = float(np.quantile(vals, args.quantile))
    graph_path = out_dir / (in_path.stem + ".graph.png")
    plot_graph(
        edges,
        T,
        tau,
        args.topk,
        graph_path,
        answer_edges=answer_edges,
        answer_label="Y",
        edge_metric_name=args.edge_metric,
        answer_metric_name=args.answer_metric if args.add_answer else "",
        topk_direction=args.topk_direction,
    )
    print("Saved plots:", heat_path, graph_path)


if __name__ == "__main__":
    main()
