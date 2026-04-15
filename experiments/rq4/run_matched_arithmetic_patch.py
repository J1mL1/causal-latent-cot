"""
RQ4 pilot: single-model matched arithmetic patching on Coconut (or any LatentReasoningModel).

For each (source, target) pair, at latent step t, replace the target's natural latent h_t with:
  - matched: h_t from the paired source problem
  - random:  h_t from a random third problem

Metrics (teacher-forced on the continuation after full latent rollout):
  - Target gold-answer log-prob drop: baseline log p(y_target) minus patched (sum over gold tokens).
  - Source-answer log-prob increase: log p(y_source | matched) - log p(y_source | random).
Optional:
  - Greedy decode after rollout; parse numeric answer and compare to target/source gold (flip heuristics).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import (
    _clone_teacher_state,
    build_dataset,
    build_teacher_state,
    load_config,
)
from common.model_registry import load_model
from data.gsm8k import parse_answer


def _parse_steps_arg(raw: Optional[str], config: Dict[str, Any]) -> List[int]:
    if raw is None:
        raw = config.get("matched_patch", {}).get("latent_steps") or config.get("steps")
    if raw is None:
        return [2, 3]
    if isinstance(raw, list):
        out = [int(x) for x in raw]
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        out = [int(p) for p in parts]
    return out


def _answer_clean(sample: Dict[str, Any]) -> str:
    ac = sample.get("answer_clean")
    if ac is not None and str(ac).strip():
        return str(ac).strip()
    text, _ = parse_answer(str(sample.get("answer", "")))
    return text.strip() if text else ""


def _teacher_seq_logp_sum(
    model: Any,
    h_ref: torch.Tensor,
    state: Dict[str, Any],
    target_ids: Optional[torch.Tensor],
) -> Optional[float]:
    if target_ids is None or target_ids.numel() == 0:
        return None
    if not hasattr(model, "compute_logits"):
        return None
    st = _clone_teacher_state(state)
    if st is None:
        return None
    logits = model.compute_logits(h_ref, st, target_ids)
    logp = F.log_softmax(logits, dim=-1)
    t = target_ids.to(logp.device)
    per = logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    return float(per.sum().item())


def _pred_answer_str(decoded: Optional[str]) -> str:
    if not decoded:
        return ""
    text, _ = parse_answer(decoded)
    return text.strip() if text else ""


def _pick_random_idx(n: int, forbidden: set, rng: random.Random) -> int:
    if n <= len(forbidden):
        raise ValueError("Not enough indices for random control.")
    while True:
        j = rng.randint(0, n - 1)
        if j not in forbidden:
            return j


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ4 matched arithmetic latent patch pilot.")
    parser.add_argument("--model_name", default="coconut", help="Registry name (default: coconut).")
    parser.add_argument("--config_path", required=True, help="YAML/JSON config (dataset + model).")
    parser.add_argument("--pairs_path", required=True, help="JSONL from build_gsm8k_template_pairs.py.")
    parser.add_argument("--output_path", required=True, help="JSONL output path.")
    parser.add_argument("--steps", default=None, help="Comma-separated latent steps (default: config matched_patch.latent_steps or 2,3).")
    parser.add_argument("--random_seed", type=int, default=0, help="RNG for random-source control.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_pairs", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    cfg_device = str(model_cfg.get("device", "")).lower()
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        model_cfg["device"] = "cpu"
    elif not cfg_device:
        model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"] = model_cfg

    model = load_model(args.model_name, config.get("model", config))
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Model has no tokenizer.")

    dataset_rows = build_dataset(config, tokenizer=tokenizer)
    n = len(dataset_rows)
    if n < 3:
        raise ValueError("Dataset must have at least 3 examples for random control.")

    pairs: List[Dict[str, Any]] = []
    with Path(args.pairs_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    if args.dry_run:
        pairs = pairs[: args.dry_run_pairs]

    steps = _parse_steps_arg(args.steps, config)
    rng = random.Random(args.random_seed)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bm = getattr(model, "base_model", None)
    if bm is not None and hasattr(bm, "eval"):
        bm.eval()
    device = getattr(model, "device", torch.device("cpu"))

    meta = {
        "record_type": "matched_arithmetic_patch_meta",
        "model_name": args.model_name,
        "config_path": str(args.config_path),
        "pairs_path": str(args.pairs_path),
        "latent_steps": steps,
        "random_seed": args.random_seed,
        "n_dataset": n,
        "metrics_note": {
            "target_gold_logp_drop_matched": "logp_target_gold_baseline_sum - logp_target_gold_matched_sum",
            "target_gold_logp_drop_random": "logp_target_gold_baseline_sum - logp_target_gold_random_sum",
            "source_ans_logp_increase_matched_minus_random": "logp_source_ans_matched_sum - logp_source_ans_random_sum",
            "greedy_matched_neq_target_gold": "greedy decode (matched) parsed answer != target gold; not baseline→intervened flip",
            "greedy_matched_eq_source_gold": "greedy decode (matched) parsed answer == source gold",
        },
    }
    (out_path.parent / f"{out_path.stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with out_path.open("w", encoding="utf-8") as w:
        for rec in tqdm(pairs, desc="matched_patch"):
            src_i = int(rec["source_idx"])
            tgt_i = int(rec["target_idx"])
            if src_i >= n or tgt_i >= n:
                row = {"error": "index out of range", "pair": rec}
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            source = dataset_rows[src_i]
            target = dataset_rows[tgt_i]
            rand_i = _pick_random_idx(n, {src_i, tgt_i}, rng)
            rand_s = dataset_rows[rand_i]

            src_ans = _answer_clean(source)
            tgt_ans = _answer_clean(target)
            tgt_ids_gold = model.build_teacher_target_ids(tgt_ans) if tgt_ans else None
            src_ids_gold = model.build_teacher_target_ids(src_ans) if src_ans else None
            if tgt_ids_gold is not None:
                tgt_ids_gold = tgt_ids_gold.to(device)
            if src_ids_gold is not None:
                src_ids_gold = src_ids_gold.to(device)

            pair_out: Dict[str, Any] = {
                "pair_id": rec.get("pair_id"),
                "jaccard": rec.get("jaccard"),
                "source_idx": src_i,
                "target_idx": tgt_i,
                "random_idx": rand_i,
                "source_answer_clean": src_ans,
                "target_answer_clean": tgt_ans,
            }

            for step in steps:
                row = dict(pair_out)
                row["latent_step"] = step
                try:
                    h_src, _ = model.forward_until_step(source["prompt"], step)
                    h_rand, _ = model.forward_until_step(rand_s["prompt"], step)
                    # Three independent target prefixes (deepcopy on state is unsafe for HF caches).
                    h_tgt_b, st_b = model.forward_until_step(target["prompt"], step)
                    h_tgt_m, st_m = model.forward_until_step(target["prompt"], step)
                    h_tgt_r, st_r = model.forward_until_step(target["prompt"], step)
                    out_b = model.rollout_from_step(h_tgt_b, st_b)
                    out_m = model.rollout_from_step(h_src, st_m)
                    out_r = model.rollout_from_step(h_rand, st_r)
                except Exception as exc:
                    row["error"] = str(exc)
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue

                state_b = build_teacher_state(out_b, st_b)
                state_m = build_teacher_state(out_m, st_m)
                state_r = build_teacher_state(out_r, st_r)

                lp_t_base = _teacher_seq_logp_sum(model, h_tgt_b, state_b, tgt_ids_gold)
                lp_t_m = _teacher_seq_logp_sum(model, h_tgt_m, state_m, tgt_ids_gold)
                lp_t_r = _teacher_seq_logp_sum(model, h_tgt_r, state_r, tgt_ids_gold)

                lp_s_m = _teacher_seq_logp_sum(model, h_tgt_m, state_m, src_ids_gold)
                lp_s_r = _teacher_seq_logp_sum(model, h_tgt_r, state_r, src_ids_gold)

                row["logp_target_gold_baseline_sum"] = lp_t_base
                row["logp_target_gold_matched_sum"] = lp_t_m
                row["logp_target_gold_random_sum"] = lp_t_r
                row["logp_source_ans_matched_sum"] = lp_s_m
                row["logp_source_ans_random_sum"] = lp_s_r

                if lp_t_base is not None and lp_t_m is not None:
                    row["target_gold_logp_drop_matched"] = lp_t_base - lp_t_m
                if lp_t_base is not None and lp_t_r is not None:
                    row["target_gold_logp_drop_random"] = lp_t_base - lp_t_r
                if lp_s_m is not None and lp_s_r is not None:
                    row["source_ans_logp_increase_matched_minus_random"] = lp_s_m - lp_s_r

                # Optional: greedy decode text + parsed answers
                def _text(o: Any) -> str:
                    if isinstance(o, dict) and o.get("text"):
                        t = o["text"]
                        return t[0] if isinstance(t, list) else str(t)
                    return ""

                tb, tm, tr = _text(out_b), _text(out_m), _text(out_r)
                row["greedy_text_baseline"] = tb
                row["greedy_text_matched"] = tm
                row["greedy_text_random"] = tr
                row["pred_answer_baseline"] = _pred_answer_str(tb)
                row["pred_answer_matched"] = _pred_answer_str(tm)
                row["pred_answer_random"] = _pred_answer_str(tr)
                row["greedy_matched_neq_target_gold"] = (
                    _pred_answer_str(tm) != tgt_ans if tgt_ans else None
                )
                row["greedy_matched_eq_source_gold"] = (
                    _pred_answer_str(tm) == src_ans if (src_ans and tm) else None
                )

                w.write(json.dumps(row, ensure_ascii=False) + "\n")

    if os.environ.get("RANK", "0") == "0":
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
