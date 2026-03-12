#!/usr/bin/env python
"""
Plot explicit step causal graph from JSONL produced by run_explicit_causal_graph.py.

Usage:
  python scripts/plot/python/rq2/plot_explicit_causal_graph.py --input outputs/rq2/explicit_graph/gsm8k_cot_gpt2_explicit_graph.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional
    nx = None  # type: ignore


def load_records(path: Path, max_steps: int | None) -> Tuple[List[Dict], int]:
    records: List[Dict] = []
    max_step = 0
    with path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            step_i = rec.get("step_i")
            step_j = rec.get("step_j")
            if isinstance(step_i, int):
                max_step = max(max_step, step_i)
            if isinstance(step_j, int):
                max_step = max(max_step, step_j)
    if max_steps is not None:
        max_step = min(max_step, max_steps)
    return records, max_step


def plot_heatmap(M: np.ndarray, out_path: Path, label: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="magma", origin="lower")
    plt.colorbar(label=label)
    plt.xlabel("step j")
    plt.ylabel("step i")
    K = M.shape[0]
    ticks = np.arange(K)
    labels = np.arange(1, K + 1)
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.title(label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_graph(
    edges: List[Tuple[int, int, float]],
    K: int,
    tau: float,
    topk: int,
    out_path: Path,
    answer_edges: Dict[int, float] | None = None,
    edge_metric_name: str = "",
) -> None:
    if nx is None:
        print("networkx not installed; skip graph plot.")
        return
    G = nx.DiGraph()
    for i in range(1, K + 1):
        G.add_node(i)
    if answer_edges:
        G.add_node("Y")

    per_i: dict[int, List[Tuple[int, float]]] = {}
    for i, j, w in edges:
        per_i.setdefault(i, []).append((j, w))
    for i, lst in per_i.items():
        for j, w in sorted(lst, key=lambda x: -x[1])[:topk]:
            if w >= tau:
                G.add_edge(i, j, weight=w)
    if answer_edges:
        max_ans = max(answer_edges.values()) if answer_edges else 1.0
        for i, w in answer_edges.items():
            G.add_edge(i, "Y", weight=w, ans_edge=True, norm_w=w / (max_ans + 1e-8))

    pos = {i: (i, 0) for i in range(1, K + 1)}
    if answer_edges:
        pos["Y"] = (K + 1, 0)

    plt.figure(figsize=(8, 3))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos)
    edges_data = list(G.edges(data=True))
    weights = [d.get("weight", 0.0) for _, _, d in edges_data]
    colors = []
    widths = []
    if weights:
        max_w = max(weights)
        for _, _, d in edges_data:
            if d.get("ans_edge"):
                colors.append("#d62728")
                widths.append(2 + 6 * d.get("norm_w", 0.0))
            else:
                colors.append("#1f77b4")
                widths.append(2 + 4 * (d.get("weight", 0.0) / (max_w + 1e-8)))
        nx.draw_networkx_edges(
            G,
            pos,
            width=widths,
            arrows=True,
            edge_color=colors,
            connectionstyle="arc3,rad=0.15",
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): f"{d.get('weight', 0.0):.2f}" for u, v, d in edges_data},
            font_size=8,
        )
    plt.axis("off")
    title_edge = edge_metric_name if edge_metric_name else "edges"
    plt.title(f"Explicit causal graph ({title_edge}), tau={tau:.3f}, top{topk}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot explicit causal graph from JSONL.")
    parser.add_argument("--input", required=True, help="Path to JSONL from run_explicit_causal_graph.py")
    parser.add_argument("--out_dir", default=None, help="Directory to save plots (default: alongside input)")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile threshold for edges")
    parser.add_argument("--topk", type=int, default=2, help="Top-k edges per source step")
    parser.add_argument("--max_steps", type=int, default=6, help="Max step index to plot")
    parser.add_argument(
        "--metric",
        default="delta_logp_last",
        choices=["delta_logp_seq", "delta_logp_last"],
        help="Metric for explicit edges.",
    )
    parser.add_argument(
        "--add_answer",
        action="store_true",
        help="Add edges from steps to answer node using same metric.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records, K = load_records(in_path, args.max_steps)
    edges = []
    answer_edges: Dict[int, List[float]] = {}
    for rec in records:
        w = rec.get(args.metric, 0.0)
        step_i = rec.get("step_i")
        step_j = rec.get("step_j")
        if isinstance(step_i, int) and isinstance(step_j, int):
            if step_i > K or step_j > K:
                continue
            edges.append((step_i, step_j, w))
        if args.add_answer and step_j == "Y" and isinstance(step_i, int):
            if step_i > K:
                continue
            answer_edges.setdefault(step_i, []).append(w)

    vals = np.array([w for _, _, w in edges]) if edges else np.array([0.0])
    M = np.zeros((K, K))
    counts = np.zeros((K, K))
    for i, j, w in edges:
        M[i - 1, j - 1] += w
        counts[i - 1, j - 1] += 1
    M = np.divide(M, counts, out=np.zeros_like(M), where=counts > 0)

    heat_path = out_dir / (in_path.stem + ".heatmap.png")
    plot_heatmap(M, heat_path, label=args.metric)

    tau = float(np.quantile(vals, args.quantile))
    graph_path = out_dir / (in_path.stem + ".graph.png")
    ans_edge_vals = None
    if args.add_answer:
        ans_edge_vals = {k: float(np.mean(v)) for k, v in answer_edges.items() if v}
    plot_graph(
        edges,
        K,
        tau,
        args.topk,
        graph_path,
        answer_edges=ans_edge_vals,
        edge_metric_name=args.metric,
    )
    print("Saved plots:", heat_path, graph_path)


if __name__ == "__main__":
    main()
