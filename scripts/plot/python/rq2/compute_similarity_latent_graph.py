#!/usr/bin/env python3
"""
Compute similarity of latent influence structure across models/datasets.

We vectorize the upper-triangular entries of W̄ (mean edge weights) and compute
cosine similarity and Spearman rank correlation between settings.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


MODEL_SPECS = [
    ("Coconut-GPT2", "coconut_gpt2"),
    ("Coconut-Llama3-1B", "coconut_llama1b"),
    ("Coconut-Qwen3-4B", "coconut_qwen3_4b"),
    ("CODI-GPT2", "codi_gpt2"),
    ("CODI-Llama3-1B", "codi_llama1b"),
    ("CODI-Qwen3-4B", "codi_qwen3_4b"),
]

DATASETS = ["commonsenseqa", "gsm8k"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--metric", default="kl_mean")
    parser.add_argument("--max_steps", type=int, default=6)
    return parser.parse_args()


def find_input_path(input_dir: str, dataset: str, model_key: str) -> str | None:
    candidates = [
        f"{dataset}_{model_key}_latent_causal_graph.jsonl",
        f"{dataset}_{model_key}_latent_graph.jsonl",
    ]
    # Handle inconsistent naming for CODI Llama on GSM8K.
    if dataset == "gsm8k" and model_key == "codi_llama1b":
        candidates = [
            "gsm8k_codi_llama_latent_causal_graph.jsonl",
            "gsm8k_codi_llama_latent_graph.jsonl",
        ] + candidates
    for name in candidates:
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_vector(path: str, metric: str, max_steps: int) -> np.ndarray:
    df = pd.read_json(path, lines=True)
    if "mode" in df.columns:
        df = df[df["mode"] == "zero"]
    df["step_i"] = pd.to_numeric(df["step_i"], errors="coerce")
    df["step_j"] = pd.to_numeric(df["step_j"], errors="coerce")
    df = df[df["step_i"].between(1, max_steps) & df["step_j"].between(1, max_steps)]
    if metric not in df.columns:
        raise ValueError(f"Missing metric '{metric}' in {path}")

    agg = (
        df.groupby(["step_i", "step_j"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "weight"})
    )
    weight_map: Dict[Tuple[int, int], float] = {
        (int(r.step_i), int(r.step_j)): float(r.weight)
        for r in agg.itertuples(index=False)
    }

    vec: List[float] = []
    for i in range(1, max_steps + 1):
        for j in range(1, max_steps + 1):
            if j > i:
                vec.append(weight_map.get((i, j), 0.0))
    return np.asarray(vec, dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def pairwise_metrics(keys: List[str], vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i:]:
            va = vectors[a]
            vb = vectors[b]
            rows.append(
                {
                    "setting_a": a,
                    "setting_b": b,
                    "cosine": cosine_similarity(va, vb),
                    "spearman": spearman_correlation(va, vb),
                }
            )
    return pd.DataFrame(rows)


def matrix_from_pairs(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    keys = sorted(set(df["setting_a"]).union(df["setting_b"]))
    mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for row in df.itertuples(index=False):
        mat.loc[row.setting_a, row.setting_b] = getattr(row, value_col)
        mat.loc[row.setting_b, row.setting_a] = getattr(row, value_col)
    return mat


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vectors: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for dataset in DATASETS:
        for label, key in MODEL_SPECS:
            path = find_input_path(args.input_dir, dataset, key)
            setting = f"{dataset}:{label}"
            if not path:
                missing.append(setting)
                continue
            vectors[setting] = load_vector(path, args.metric, args.max_steps)

    if missing:
        print("Missing inputs:")
        for m in missing:
            print(f"  - {m}")

    keys = sorted(vectors.keys())
    if not keys:
        raise SystemExit("No usable inputs found.")

    pairs = pairwise_metrics(keys, vectors)
    pairs_path = os.path.join(args.out_dir, "latent_similarity_pairs.csv")
    pairs.to_csv(pairs_path, index=False)

    cosine_mat = matrix_from_pairs(pairs, "cosine")
    spearman_mat = matrix_from_pairs(pairs, "spearman")
    cosine_path = os.path.join(args.out_dir, "latent_similarity_cosine_matrix.csv")
    spearman_path = os.path.join(args.out_dir, "latent_similarity_spearman_matrix.csv")
    cosine_mat.to_csv(cosine_path)
    spearman_mat.to_csv(spearman_path)

    print(f"Saved: {pairs_path}")
    print(f"Saved: {cosine_path}")
    print(f"Saved: {spearman_path}")


if __name__ == "__main__":
    main()
