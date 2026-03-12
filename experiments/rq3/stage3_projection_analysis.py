#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _parse_csv_ints(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    return [int(s) for s in str(raw).split(",") if s.strip() != ""]


def _parse_csv_floats(raw: Optional[str]) -> List[float]:
    if raw is None:
        return []
    return [float(s) for s in str(raw).split(",") if s.strip() != ""]


def _logp_for_answer(
    model: Any, h_t: torch.Tensor, state: Dict, target_ids: torch.Tensor
) -> float:
    # Clone state to avoid cache mutation across repeated teacher-forced calls.
    try:
        from common.experiment_utils import _clone_teacher_state  # type: ignore
        state = _clone_teacher_state(state)
    except Exception:
        state = dict(state)
    logits = model.compute_logits(h_t, state, target_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids_dev = target_ids.to(log_probs.device)
    if target_ids_dev.numel() == 0:
        return float("-inf")
    seq_len = target_ids_dev.size(1)
    if seq_len == 0:
        return float("-inf")
    # Use the last answer token (penultimate if EOS is appended).
    time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
    last_targets = target_ids_dev[:, time_index]
    last_logp = log_probs[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    return float(last_logp.mean().item())


def _last_logp_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids_dev = target_ids.to(log_probs.device)
    if target_ids_dev.numel() == 0:
        return torch.full((logits.size(0),), float("-inf"), device=logits.device)
    # Expand targets to batch size if needed.
    if target_ids_dev.size(0) == 1 and logits.size(0) > 1:
        target_ids_dev = target_ids_dev.expand(logits.size(0), -1)
    seq_len = target_ids_dev.size(1)
    time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
    last_targets = target_ids_dev[:, time_index]
    last_logp = log_probs[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    return last_logp


def _teacher_text(model: Any, answer: str) -> str:
    template = getattr(model, "teacher_target_template", None)
    if template:
        try:
            return template.format(answer=answer)
        except Exception:
            return f"{template} {answer}"
    return answer


def softmax_pair(a: float, b: float, tau: float) -> tuple[float, float]:
    if tau <= 0:
        return (1.0, 0.0) if a >= b else (0.0, 1.0)
    a_scaled = a / tau
    b_scaled = b / tau
    m = a_scaled if a_scaled >= b_scaled else b_scaled
    ea = math.exp(a_scaled - m)
    eb = math.exp(b_scaled - m)
    denom = ea + eb
    if denom <= 0:
        return 0.0, 0.0
    return ea / denom, eb / denom


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Stage 3: projection analysis.")
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--traj_jsonl", required=True)
    parser.add_argument("--probes_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--early_steps", default="2,3")
    parser.add_argument("--late_step", default="final", choices=["final", "penultimate"])
    parser.add_argument("--p_mode", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--tau", type=float, default=1.0)
    # Teacher-forced answer competition (optional).
    parser.add_argument("--config_path", default=None, help="Config for teacher-forced competition.")
    parser.add_argument("--tf_output_jsonl", default=None, help="Write teacher-forced competition metrics here.")
    parser.add_argument("--tf_use_latent_path", action="store_true", default=True, help="Rebuild cache per trajectory using latent_path.")
    parser.add_argument("--tf_no_latent_path", dest="tf_use_latent_path", action="store_false", help="Ignore latent_path and reuse prompt-only cache.")
    parser.add_argument("--tf_steps", default="1,2,3,4,5,6")
    parser.add_argument("--tf_answers", default=None, help="Override answers, e.g., Yes,No or True,False")
    parser.add_argument("--tf_deltas", default="0.4,0.5")
    parser.add_argument("--tf_debug_samples", type=int, default=5)
    parser.add_argument("--tf_debug_topk", type=int, default=5)
    args = parser.parse_args()

    # Distributed setup (optional)
    dist = None
    rank = 0
    world_size = 1
    local_rank = args.local_rank
    if args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        try:
            import torch.distributed as torch_dist
            dist = torch_dist
        except Exception as exc:
            raise RuntimeError("Distributed requested but torch.distributed is unavailable") from exc
        if local_rank < 0:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    samples = {r["sample_id"]: r for r in load_jsonl(Path(args.samples_jsonl))}
    probes = {r["sample_id"]: r for r in load_jsonl(Path(args.probes_jsonl))}
    trajs = load_jsonl(Path(args.traj_jsonl))

    early_steps = [int(s) for s in args.early_steps.split(",") if s != ""]
    late_idx = -1 if args.late_step == "final" else -2

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajs_shard = trajs[rank::world_size] if dist is not None and world_size > 1 else trajs
    rank_output = output_path if dist is None else output_path.with_suffix(output_path.suffix + f".rank{rank}")

    with rank_output.open("w") as writer:
        for rec in tqdm(trajs_shard, desc="projection", total=len(trajs_shard), disable=rank != 0):
            sid = rec["sample_id"]
            if sid not in probes or sid not in samples:
                continue
            latent_path = Path(rec["latent_path"])
            if not latent_path.exists():
                continue
            latent = np.load(latent_path)
            v_a = np.load(Path(probes[sid]["vA_path"]))
            v_b = np.load(Path(probes[sid]["vB_path"]))
            center_path = probes[sid].get("mu_center_path")
            mu_center = np.load(Path(center_path)) if center_path else None
            v_a_norm = float(np.linalg.norm(v_a)) + 1e-8
            v_b_norm = float(np.linalg.norm(v_b)) + 1e-8

            for t in early_steps:
                if t < 1 or t > latent.shape[0]:
                    continue
                h = latent[t - 1]
                h_proj = h - mu_center if mu_center is not None else h
                h_norm = float(np.linalg.norm(h_proj)) + 1e-8
                score_a = float(np.dot(h_proj, v_a))
                score_b = float(np.dot(h_proj, v_b))
                cos_a = score_a / (h_norm * v_a_norm)
                cos_b = score_b / (h_norm * v_b_norm)
                if args.p_mode == "sigmoid":
                    logit = (score_a - score_b) / args.tau if args.tau > 0 else float("inf")
                    p_a = sigmoid(logit)
                    p_b = 1.0 - p_a
                else:
                    p_a, p_b = softmax_pair(score_a, score_b, args.tau)
                writer.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            "traj_id": rec["traj_id"],
                            "cluster": rec["cluster"],
                            "step": t,
                            "score_A": score_a,
                            "score_B": score_b,
                            "cos_A": cos_a,
                            "cos_B": cos_b,
                            "p_A": p_a,
                            "p_B": p_b,
                            "phase": "early",
                        }
                    )
                    + "\n"
                )

            h_late = latent[late_idx]
            h_proj = h_late - mu_center if mu_center is not None else h_late
            h_norm = float(np.linalg.norm(h_proj)) + 1e-8
            score_a = float(np.dot(h_proj, v_a))
            score_b = float(np.dot(h_proj, v_b))
            cos_a = score_a / (h_norm * v_a_norm)
            cos_b = score_b / (h_norm * v_b_norm)
            if args.p_mode == "sigmoid":
                logit = (score_a - score_b) / args.tau if args.tau > 0 else float("inf")
                p_a = sigmoid(logit)
                p_b = 1.0 - p_a
            else:
                p_a, p_b = softmax_pair(score_a, score_b, args.tau)
            writer.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "traj_id": rec["traj_id"],
                        "cluster": rec["cluster"],
                        "step": latent.shape[0] if late_idx == -1 else latent.shape[0] - 1,
                        "score_A": score_a,
                        "score_B": score_b,
                        "cos_A": cos_a,
                        "cos_B": cos_b,
                        "p_A": p_a,
                        "p_B": p_b,
                        "phase": "late",
                    }
                )
                + "\n"
            )

    # Teacher-forced answer competition (no probe training).
    if args.config_path is None:
        raise ValueError("--config_path required for teacher-forced competition.")
    if args.tf_output_jsonl is None:
        stem = output_path.stem
        tf_name = f"{stem}_teacher_forced.jsonl"
        args.tf_output_jsonl = str(output_path.with_name(tf_name))
    from common.experiment_utils import build_dataset, load_config, prepare_target_ids
    from common.model_registry import load_model

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    cfg_device = str(model_cfg.get("device", "")).lower()
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        model_cfg["device"] = "cpu"
    elif not cfg_device:
        model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"] = model_cfg

    model_name = str(config.get("model_name", "coconut")).lower()
    model = load_model(model_name, config.get("model", config))
    dataset = build_dataset(config, tokenizer=getattr(model, "tokenizer", None))
    dataset_by_id: Dict[Any, Dict] = {}
    for idx, d in enumerate(dataset):
        if not isinstance(d, dict):
            continue
        if d.get("id") is not None:
            dataset_by_id[d["id"]] = d
        dataset_by_id[idx] = d

    tf_steps = _parse_csv_ints(args.tf_steps)
    tf_deltas = _parse_csv_floats(args.tf_deltas)
    if args.tf_answers:
        answer_list = [a.strip() for a in str(args.tf_answers).split(",") if a.strip()]
    else:
        answer_list = ["yes", "no"]
        if "codi" in model_name:
            answer_list = ["True", "False"]
    if len(answer_list) != 2:
        raise ValueError("--tf_answers must provide exactly two answers, e.g., Yes,No")
    ans_pos, ans_neg = answer_list
    target_yes = prepare_target_ids(model, ans_pos, False)
    target_no = prepare_target_ids(model, ans_neg, False)
    if target_yes is None or target_no is None:
        raise RuntimeError("Failed to build target ids for teacher-forced answers.")

    tf_path = Path(args.tf_output_jsonl)
    tf_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = str(config.get("dataset_name", ""))
    debug_limit = max(0, int(args.tf_debug_samples))
    debug_topk = max(1, int(args.tf_debug_topk))
    debug_count = 0

    tf_recs = []
    for rec in trajs_shard:
        sid = rec.get("sample_id")
        if sid is None or sid not in samples:
            continue
        sample = samples[sid]
        sample_uid = sample.get("sample_uid")
        data = dataset_by_id.get(sample_uid, dataset_by_id.get(sid))
        if not data or not isinstance(data, dict):
            continue
        prompt = data.get("prompt")
        if not prompt:
            continue
        tf_recs.append(
            {
                "rec": rec,
                "prompt": prompt,
                "gold_answer": data.get("answer_clean") or data.get("answer") or sample.get("gold_answer"),
                "sample_uid": sample_uid,
                "latent_path": rec.get("latent_path"),
            }
        )

    def _clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from common.experiment_utils import _clone_teacher_state  # type: ignore
            return _clone_teacher_state(state)
        except Exception:
            return dict(state)

    base_state_cache: Dict[str, Dict[str, Any]] = {}

    def _get_base_state(prompt: str, sample_uid: Any) -> Dict[str, Any]:
        key = str(sample_uid)
        cached = base_state_cache.get(key)
        if cached is None:
            _, base_state = model.forward_until_step(prompt, 1)
            base_state_cache[key] = base_state
            cached = base_state
        return _clone_state(cached)

    rank_tf = tf_path if dist is None else tf_path.with_suffix(tf_path.suffix + f".rank{rank}")
    with rank_tf.open("w") as tf_writer:
        for i in tqdm(range(0, len(tf_recs), max(1, args.batch_size)), desc="tf-competition", total=math.ceil(len(tf_recs) / max(1, args.batch_size)), disable=rank != 0):
            batch = tf_recs[i : i + max(1, args.batch_size)]
            prompts = [b["prompt"] for b in batch]
            delta_maps = [dict() for _ in batch]
            traj_runtime = []
            if args.tf_use_latent_path:
                for entry in batch:
                    latent_path = entry.get("latent_path")
                    latent_tensor = None
                    if latent_path:
                        try:
                            latent_np = np.load(latent_path)
                            latent_tensor = torch.from_numpy(latent_np).to(model.device)
                        except Exception:
                            latent_tensor = None
                    state = None
                    if latent_tensor is not None:
                        state = _get_base_state(entry["prompt"], entry["sample_uid"])
                    traj_runtime.append(
                        {
                            "latent": latent_tensor,
                            "state": state,
                        }
                    )
            for step in tf_steps:
                if args.tf_use_latent_path:
                    per_states = True
                else:
                    with torch.no_grad():
                        h_t, state = model.forward_until_step(prompts, step)
                    per_states = state.get("per_sample_states") if isinstance(state, dict) else None
                    # Batch path: compute logits once when the state is shared across the batch.
                    if per_states is None:
                        batch_size = h_t.size(0)
                        target_yes_batch = target_yes
                        target_no_batch = target_no
                        if target_yes.size(0) == 1 and batch_size > 1:
                            target_yes_batch = target_yes.expand(batch_size, -1)
                        if target_no.size(0) == 1 and batch_size > 1:
                            target_no_batch = target_no.expand(batch_size, -1)
                        try:
                            from common.experiment_utils import _clone_teacher_state  # type: ignore
                            state_yes = _clone_teacher_state(state)
                            state_no = _clone_teacher_state(state)
                        except Exception:
                            state_yes = dict(state)
                            state_no = dict(state)
                        logits_yes = model.compute_logits(h_t, state_yes, target_yes_batch)
                        logits_no = model.compute_logits(h_t, state_no, target_no_batch)
                        logp_yes_all = _last_logp_from_logits(logits_yes, target_yes_batch)
                        logp_no_all = _last_logp_from_logits(logits_no, target_no_batch)
                for idx, entry in enumerate(batch):
                    rec = entry["rec"]
                    sid = rec.get("sample_id")
                    traj_id = rec.get("traj_id")
                    cluster = rec.get("cluster")
                    sample_uid = entry["sample_uid"]
                    gold_answer = entry["gold_answer"]
                    if args.tf_use_latent_path:
                        runtime = traj_runtime[idx]
                        latent_tensor = runtime.get("latent")
                        state_i = runtime.get("state")
                        if latent_tensor is None or state_i is None:
                            continue
                        if step < 1 or step > latent_tensor.size(0):
                            continue
                        h_i = latent_tensor[step - 1].unsqueeze(0)
                        with torch.no_grad():
                            h_i, state_i = model.rollout_to_step(h_i, state_i, target_step=step)
                        runtime["state"] = state_i
                        logp_yes = _logp_for_answer(model, h_i, state_i, target_yes)
                        logp_no = _logp_for_answer(model, h_i, state_i, target_no)
                    else:
                        if per_states is None:
                            logp_yes = float(logp_yes_all[idx].item())
                            logp_no = float(logp_no_all[idx].item())
                            state_i = state
                            h_i = h_t[idx : idx + 1]
                        else:
                            state_i = per_states[idx]
                            h_i = h_t[idx : idx + 1]
                            logp_yes = _logp_for_answer(model, h_i, state_i, target_yes)
                            logp_no = _logp_for_answer(model, h_i, state_i, target_no)
                    # Debug: show top-k probs for the last target position for first few samples.
                    if debug_count < debug_limit and rank == 0:
                        def _topk_info(target_ids: torch.Tensor, label: str, logits_override: torch.Tensor | None) -> None:
                            if logits_override is None:
                                try:
                                    from common.experiment_utils import _clone_teacher_state  # type: ignore
                                    debug_state = _clone_teacher_state(state_i)
                                except Exception:
                                    debug_state = dict(state_i)
                                logits_local = model.compute_logits(h_i, debug_state, target_ids)
                            else:
                                logits_local = logits_override[idx : idx + 1]
                            log_probs = F.log_softmax(logits_local, dim=-1)
                            tgt = target_ids.to(log_probs.device)
                            seq_len = tgt.size(1)
                            time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
                            probs = log_probs[:, time_index, :].exp().mean(dim=0)
                            top_vals, top_idx = torch.topk(probs, k=min(debug_topk, probs.size(0)))
                            tokens = []
                            if hasattr(model, "tokenizer"):
                                tokens = model.tokenizer.convert_ids_to_tokens(top_idx.tolist())  # type: ignore[attr-defined]
                            print(f"[tf-debug] sample_id={sid} traj_id={traj_id} step={step} label={label}")
                            print(f"[tf-debug] prompt: {entry['prompt']}")
                            teacher_text = _teacher_text(model, label)
                            print(f"[tf-debug] teacher_text: {teacher_text}")
                            print(f"[tf-debug] target_len={seq_len} time_index={time_index}")
                            if hasattr(model, "tokenizer"):
                                tok = model.tokenizer  # type: ignore[attr-defined]
                                tgt_ids = tgt[0].tolist()
                                tgt_tokens = tok.convert_ids_to_tokens(tgt_ids)
                                scored_id = tgt_ids[time_index] if tgt_ids else None
                                scored_tok = tgt_tokens[time_index] if tgt_tokens else None
                                print(f"[tf-debug] target_ids: {tgt_ids}")
                                print(f"[tf-debug] target_toks: {tgt_tokens}")
                                print(f"[tf-debug] scored_token: {scored_tok} (id={scored_id})")
                            for tok, val in zip(tokens, top_vals.tolist()):
                                print(f"[tf-debug]   {tok}: {val:.6f}")

                        if args.tf_use_latent_path:
                            _topk_info(target_yes, ans_pos, None)
                            _topk_info(target_no, ans_neg, None)
                        else:
                            _topk_info(
                                target_yes_batch if per_states is None else target_yes,
                                ans_pos,
                                logits_yes if per_states is None else None,
                            )
                            _topk_info(
                                target_no_batch if per_states is None else target_no,
                                ans_neg,
                                logits_no if per_states is None else None,
                            )
                        debug_count += 1
                    max_lp = max(logp_yes, logp_no)
                    denom = math.exp(logp_yes - max_lp) + math.exp(logp_no - max_lp)
                    s_yes = math.exp(logp_yes - max_lp) / denom if denom > 0 else 0.5
                    s_no = 1.0 - s_yes
                    ss = min(s_yes, s_no)
                    delta_p = abs(s_yes - s_no)
                    delta_maps[idx][step] = delta_p
                    tf_writer.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "traj_id": traj_id,
                                "cluster": cluster,
                                "sample_uid": sample_uid,
                                "dataset": dataset_name,
                                "gold_answer": gold_answer,
                                "step": step,
                                "logp_yes": logp_yes,
                                "logp_no": logp_no,
                                "s_yes": s_yes,
                                "s_no": s_no,
                                "ss": ss,
                                "delta_p": delta_p,
                                "record_type": "per_step",
                            }
                        )
                        + "\n"
                    )

            # summary per trajectory
            for entry, delta_p_by_step in zip(batch, delta_maps):
                rec = entry["rec"]
                sid = rec.get("sample_id")
                traj_id = rec.get("traj_id")
                cluster = rec.get("cluster")
                sample_uid = entry["sample_uid"]
                gold_answer = entry["gold_answer"]
                for delta in tf_deltas:
                    d_step = None
                    for step in sorted(delta_p_by_step.keys()):
                        if delta_p_by_step[step] >= delta:
                            d_step = step
                            break
                    tf_writer.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "traj_id": traj_id,
                                "cluster": cluster,
                                "sample_uid": sample_uid,
                                "dataset": dataset_name,
                                "gold_answer": gold_answer,
                                "delta": delta,
                                "d_step": d_step,
                                "record_type": "summary",
                            }
                        )
                        + "\n"
                    )

    if dist is not None and world_size > 1:
        dist.barrier()
        if rank == 0:
            with output_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = output_path.with_suffix(output_path.suffix + f".rank{r}")
                    if not shard_path.exists():
                        continue
                    with shard_path.open("r") as shard:
                        for line in shard:
                            merged.write(line)
                    shard_path.unlink(missing_ok=True)
            with tf_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = tf_path.with_suffix(tf_path.suffix + f".rank{r}")
                    if not shard_path.exists():
                        continue
                    with shard_path.open("r") as shard:
                        for line in shard:
                            merged.write(line)
                    shard_path.unlink(missing_ok=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
