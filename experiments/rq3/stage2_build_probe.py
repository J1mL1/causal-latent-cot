#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Stage 2: build probes vA/vB.")
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--traj_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_step", default="final", choices=["final", "penultimate"])
    parser.add_argument("--method", default="gram_schmidt", choices=["gram_schmidt", "mean_sub"])
    args = parser.parse_args()

    samples = {r["sample_id"]: r for r in load_jsonl(Path(args.samples_jsonl))}
    trajs = load_jsonl(Path(args.traj_jsonl))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / "probes.jsonl"

    per_sample: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for rec in trajs:
        sid = rec["sample_id"]
        if sid not in samples:
            continue
        cluster = rec["cluster"]
        latent_path = Path(rec["latent_path"])
        if not latent_path.exists():
            continue
        latent = np.load(latent_path)
        idx = -1 if args.probe_step == "final" else -2
        vec = latent[idx]
        bucket = per_sample.setdefault(sid, {"A": [], "B": []})
        if cluster in bucket:
            bucket[cluster].append(vec)

    with probe_path.open("w") as writer:
        for sid, buckets in per_sample.items():
            if not buckets["A"] or not buckets["B"]:
                continue
            mu_a = np.mean(np.stack(buckets["A"], axis=0), axis=0)
            mu_b = np.mean(np.stack(buckets["B"], axis=0), axis=0)
            mu_center = 0.5 * (mu_a + mu_b)
            if args.method == "mean_sub":
                v_a = normalize_vec(mu_a - mu_center)
                v_b = normalize_vec(mu_b - mu_center)
            else:
                v_a = normalize_vec(mu_a)
                b_unique = mu_b - float(np.dot(mu_b, v_a)) * v_a
                v_b = normalize_vec(b_unique)

            v_a_path = output_dir / f"{sid}_vA.npy"
            v_b_path = output_dir / f"{sid}_vB.npy"
            center_path = output_dir / f"{sid}_mu_center.npy"
            np.save(v_a_path, v_a)
            np.save(v_b_path, v_b)
            np.save(center_path, mu_center)

            writer.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "answer_A": samples[sid]["answer_A"],
                        "answer_B": samples[sid]["answer_B"],
                        "vA_path": str(v_a_path),
                        "vB_path": str(v_b_path),
                        "mu_center_path": str(center_path),
                        "dot": float(np.dot(v_a, v_b)),
                        "method": args.method,
                        "probe_step": args.probe_step,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
