#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import (
    build_dataset,
    create_dataloader,
    load_config,
    parse_step_tokens,
    step_order_key,
    prepare_target_ids,
    build_teacher_state,
)
from common.model_registry import load_model
from common.analysis.step_ablation import ablate_gaussian_on_h, ablate_gaussian_noise, ablate_zero
from common.metrics.grad_sensitivity import compute_grad_sensitivity


def apply_ablation(h_t: torch.Tensor, mode: str, sigma_scale: float) -> torch.Tensor:
    if mode == "zero":
        return ablate_zero(h_t)
    if mode == "gaussian_h":
        return ablate_gaussian_on_h(h_t, sigma_scale=sigma_scale)
    if mode == "gaussian":
        return ablate_gaussian_noise(h_t, sigma_scale=sigma_scale)
    raise ValueError(f"Unsupported ablation mode for causal graph: {mode}")


def compute_kl_and_delta(
    logits_clean: torch.Tensor,
    logits_ablt: torch.Tensor,
    target_ids: torch.Tensor,
    eos_id: int | None,
) -> Dict[str, float]:
    logp_clean = F.log_softmax(logits_clean, dim=-1)
    logp_ablt = F.log_softmax(logits_ablt, dim=-1)
    p_clean = logp_clean.exp()
    kl_token = (p_clean * (logp_clean - logp_ablt)).sum(dim=-1)  # [B, L_ans]
    kl_mean = kl_token.mean().item()

    target_ids_dev = target_ids.to(logits_clean.device)
    base_tok = logp_clean.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    ablt_tok = logp_ablt.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    delta_sum = (base_tok - ablt_tok).sum(dim=-1).mean().item()

    seq_len = target_ids_dev.size(1)
    time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
    last_targets = target_ids_dev[:, time_index]
    base_last = logp_clean[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    ablt_last = logp_ablt[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    delta_last = (base_last - ablt_last).mean().item()

    return {
        "kl_mean": kl_mean,
        "teacher_forced_delta_sum": delta_sum,
        "delta_logp_final_token": delta_last,
    }


def compute_kl_and_delta_batch(
    logits_clean: torch.Tensor,
    logits_ablt: torch.Tensor,
    target_ids: torch.Tensor,
    eos_id: int | None,
) -> Dict[str, torch.Tensor]:
    logp_clean = F.log_softmax(logits_clean, dim=-1)
    logp_ablt = F.log_softmax(logits_ablt, dim=-1)
    p_clean = logp_clean.exp()
    kl_token = (p_clean * (logp_clean - logp_ablt)).sum(dim=-1)  # [B, L_ans]
    kl_mean = kl_token.mean(dim=-1)

    target_ids_dev = target_ids.to(logits_clean.device)
    base_tok = logp_clean.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    ablt_tok = logp_ablt.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    delta_sum = (base_tok - ablt_tok).sum(dim=-1)

    seq_len = target_ids_dev.size(1)
    time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
    last_targets = target_ids_dev[:, time_index]
    base_last = logp_clean[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    ablt_last = logp_ablt[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
    delta_last = base_last - ablt_last

    return {
        "kl_mean": kl_mean,
        "teacher_forced_delta_sum": delta_sum,
        "delta_logp_final_token": delta_last,
    }


def compute_logit_kl_from_hidden_batch(
    model: Any,
    h_clean: torch.Tensor,
    h_ablt: torch.Tensor,
) -> torch.Tensor | None:
    if not hasattr(model, "logits_from_latent"):
        return None
    try:
        logits_clean = model.logits_from_latent(h_clean)
        logits_ablt = model.logits_from_latent(h_ablt)
    except Exception:
        return None
    logp_clean = F.log_softmax(logits_clean, dim=-1)
    logp_ablt = F.log_softmax(logits_ablt, dim=-1)
    p_clean = logp_clean.exp()
    kl = (p_clean * (logp_clean - logp_ablt)).sum(dim=-1)
    return kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate latent-to-latent causal edges via KL.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--steps", default=None, help="Comma separated list or 'all'.")
    parser.add_argument("--mode", default="zero", choices=["zero", "gaussian_h", "gaussian"], help="Ablation mode for i-step intervention.")
    parser.add_argument("--sigma_scale", type=float, default=0.5)
    parser.add_argument(
        "--include_self",
        action="store_true",
        help="Include i==j ablations (e.g., ablate final latent itself before decoding).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for input loading.")
    parser.add_argument("--pin_memory", default="auto", choices=["auto", "on", "off"], help="DataLoader pin_memory setting.")
    parser.add_argument(
        "--persistent_workers",
        default="auto",
        choices=["auto", "on", "off"],
        help="Keep DataLoader workers alive between batches.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (requires num_workers > 0).")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed to shard data across ranks (torchrun).")
    parser.add_argument("--dist_backend", default="nccl", help="torch.distributed backend (default: nccl).")
    parser.add_argument("--dist_url", default="env://", help="Init method for torch.distributed (default: env:// for torchrun).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torchrun.")
    parser.add_argument(
        "--model_type",
        default="coconut",
        choices=["coconut", "codi", "softthinking"],
        help="Latent model type for grad metrics.",
    )
    parser.add_argument("--save_adj", action="store_true", help="Save adjacency matrix as .npy alongside output jsonl.")
    parser.add_argument("--save_graph", action="store_true", help="Save a causal graph image (PNG) with edge weights.")
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
        except Exception as exc:  # pragma: no cover - optional dependency
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

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    if dist is not None:
        model_cfg["device"] = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        cfg_device = str(model_cfg.get("device", "")).lower()
        if cfg_device.startswith("cuda") and not torch.cuda.is_available():
            model_cfg["device"] = "cpu"
        elif not cfg_device:
            model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"] = model_cfg

    model = load_model(args.model_name, config.get("model", config))
    tokenizer = getattr(model, "tokenizer", None)

    dataset_full = build_dataset(config, tokenizer=tokenizer)
    if not dataset_full:
        raise ValueError("Dataset is empty; check config dataset_path/name.")
    dataset = dataset_full
    if dist is not None and world_size > 1:
        dataset = dataset_full[rank::world_size]
    pin_memory = None if args.pin_memory == "auto" else args.pin_memory == "on"
    persistent_workers = None if args.persistent_workers == "auto" else args.persistent_workers == "on"
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    steps = parse_step_tokens(args.steps, config.get("num_steps"))
    nodes: List[Any] = sorted(steps, key=step_order_key)
    numeric_nodes = [s for s in nodes if isinstance(s, int)]
    invalid_zero_based = [s for s in numeric_nodes if s < 1]
    if invalid_zero_based:
        raise ValueError(f"Causal graph steps must be 1-based, got invalid: {invalid_zero_based}")
    if rank == 0:
        print(f"[causal graph] nodes (sorted): {nodes}")
    use_gsm8k_parse = config.get("dataset_name") == "gsm8k"
    eos_id = getattr(getattr(model, "tokenizer", None), "eos_token_id", None)

    metric_keys = [
        "kl_mean",
        "teacher_forced_delta_sum",
        "delta_logp_final_token",
        "kl_logit_ht",
        "grad_logprob",
    ]
    adj_values: Dict[str, Dict[tuple[Any, Any], List[float]]] = {
        key: {(i, j): [] for i in nodes for j in nodes} for key in metric_keys
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = output_path if dist is None else output_path.with_suffix(output_path.suffix + f".rank{rank}")

    total_samples = len(dataset)

    with rank_output_path.open("w") as writer:
        pbar = tqdm(total=total_samples, desc="samples", disable=rank != 0)
        sample_counter = 0
        for batch in dataloader:
            batch_samples = batch if isinstance(batch, list) else [batch]
            batched_by_len: Dict[int, list[Dict[str, Any]]] = {}
            for sample in batch_samples:
                prompt = sample["prompt"] if isinstance(sample, Dict) else sample
                gold_answer = sample.get("answer_clean") or sample.get("answer") if isinstance(sample, Dict) else None
                if gold_answer is not None and hasattr(model, "build_teacher_target_ids"):
                    target_ids = model.build_teacher_target_ids(gold_answer)
                else:
                    target_ids = prepare_target_ids(model, gold_answer, use_gsm8k_parse)

                sample_id = sample_counter if dist is None else sample_counter * world_size + rank
                sample_counter += 1

                if target_ids is None:
                    continue
                if target_ids.dim() == 1:
                    target_ids = target_ids.unsqueeze(0)
                target_ids = target_ids.to(model.device)
                target_len = int(target_ids.size(1))
                batched_by_len.setdefault(target_len, []).append(
                    {
                        "prompt": prompt,
                        "target_ids": target_ids,
                        "sample_id": sample_id,
                    }
                )

            for target_len, items in batched_by_len.items():
                prompts = [item["prompt"] for item in items]
                target_ids = torch.cat([item["target_ids"] for item in items], dim=0)

                grad_steps = [n for n in nodes if isinstance(n, int)]
                grad_logprob_cache: Dict[Any, Dict[int, torch.Tensor]] = {}
                if grad_steps:
                    for node_j in tqdm(nodes, desc="steps-j", leave=False, disable=rank != 0):
                        if isinstance(node_j, int):
                            step_s = int(node_j)
                        elif isinstance(node_j, str) and node_j.lower() == "end":
                            step_s = max(grad_steps) if grad_steps else None
                        else:
                            continue
                        if step_s is None:
                            continue
                        with torch.enable_grad():
                            grad_logprob_cache[node_j] = compute_grad_sensitivity(
                                model,
                                prompts,
                                None,
                                target_ids,
                                grad_steps,
                                metric="gold_logprob",
                                step_s=step_s,
                                model_type=args.model_type,
                            )

                def _build_state_for_logits(
                    h: torch.Tensor,
                    state: Dict[str, Any] | None,
                    fallback: Dict[str, Any] | None = None,
                ) -> Dict[str, Any] | None:
                    if state is None:
                        return None
                    if isinstance(state, dict) and "per_sample_states" in state:
                        rollout = model.rollout_from_step(h, state)
                        return build_teacher_state(rollout, state)
                    return build_teacher_state(state, fallback)

                # cache baseline logits per j-node to avoid recomputation
                baseline_cache: Dict[Any, torch.Tensor] = {}
                for node_j in tqdm(nodes, desc="steps-j", leave=False, disable=rank != 0):
                    h_clean, state_clean = model.forward_until_step(prompts, node_j)
                    base_state_clean = _build_state_for_logits(h_clean, state_clean)
                    if base_state_clean is None:
                        sample_ids = [item["sample_id"] for item in items]
                        raise RuntimeError(
                            f"base_state_clean is None (rank={rank}, node_j={node_j}, "
                            f"target_len={target_len}, sample_ids={sample_ids})"
                        )
                    logits_clean = model.compute_logits(h_clean, base_state_clean, target_ids)
                    baseline_cache[node_j] = logits_clean

                    for node_i in tqdm(nodes, desc="steps-i", leave=False, disable=rank != 0):
                        if step_order_key(node_i) > step_order_key(node_j):
                            continue
                        if not args.include_self and step_order_key(node_i) >= step_order_key(node_j):
                            continue
                        if node_i == node_j:
                            if isinstance(node_j, str) and node_j.lower() == "end":
                                continue
                            # For i==j, match ablation experiment semantics: run to end before teacher forcing.
                            base_out = model.rollout_from_step(h_clean, state_clean)
                            base_state = build_teacher_state(base_out, state_clean)
                            if base_state is None:
                                sample_ids = [item["sample_id"] for item in items]
                                raise RuntimeError(
                                    f"base_state is None (rank={rank}, node_j={node_j}, "
                                    f"target_len={target_len}, sample_ids={sample_ids})"
                                )
                            logits_clean = model.compute_logits(h_clean, base_state, target_ids)

                            h_i_mod = apply_ablation(h_clean, args.mode, args.sigma_scale)
                            ablt_out = model.rollout_from_step(h_i_mod, state_clean)
                            ablt_state = build_teacher_state(ablt_out, state_clean)
                            if ablt_state is None:
                                sample_ids = [item["sample_id"] for item in items]
                                raise RuntimeError(
                                    f"ablt_state is None (rank={rank}, node_j={node_j}, "
                                    f"target_len={target_len}, sample_ids={sample_ids})"
                                )
                            logits_ablt = model.compute_logits(h_i_mod, ablt_state, target_ids)
                            kl_logit_ht = compute_logit_kl_from_hidden_batch(model, h_clean, h_i_mod)
                        else:
                            h_i, state_i = model.forward_until_step(prompts, node_i)
                            h_i_mod = apply_ablation(h_i, args.mode, args.sigma_scale)
                            if isinstance(node_j, int):
                                if not hasattr(model, "rollout_to_step"):
                                    raise AttributeError("Model lacks rollout_to_step required for causal graph.")
                                if isinstance(state_i, dict) and "per_sample_states" in state_i:
                                    per_states = state_i["per_sample_states"]
                                    h_list = []
                                    state_list = []
                                    for idx, s in enumerate(per_states):
                                        h_j, s_j = model.rollout_to_step(  # type: ignore[attr-defined]
                                            h_i_mod[idx : idx + 1], s, target_step=int(node_j)
                                        )
                                        h_list.append(h_j)
                                        state_list.append(s_j)
                                    h_j_ablt = torch.cat(h_list, dim=0)
                                    state_j_ablt = {"per_sample_states": state_list}
                                else:
                                    h_j_ablt, state_j_ablt = model.rollout_to_step(  # type: ignore[attr-defined]
                                        h_i_mod, state_i, target_step=int(node_j)
                                    )
                                ablt_state = _build_state_for_logits(h_j_ablt, state_j_ablt, state_i)
                                logits_ablt = model.compute_logits(h_j_ablt, ablt_state, target_ids)
                                kl_logit_ht = compute_logit_kl_from_hidden_batch(model, h_clean, h_j_ablt)
                            elif isinstance(node_j, str) and node_j.lower() == "end":
                                if not numeric_nodes:
                                    raise ValueError("Cannot rollout to 'end' without numeric latent steps.")
                                max_lat_step = max(numeric_nodes)
                                if not hasattr(model, "rollout_to_step"):
                                    raise AttributeError("Model lacks rollout_to_step required for causal graph.")
                                if isinstance(state_i, dict) and "per_sample_states" in state_i:
                                    per_states = state_i["per_sample_states"]
                                    h_list = []
                                    state_list = []
                                    for idx, s in enumerate(per_states):
                                        h_j, s_j = model.rollout_to_step(  # type: ignore[attr-defined]
                                            h_i_mod[idx : idx + 1], s, target_step=max_lat_step
                                        )
                                        h_list.append(h_j)
                                        state_list.append(s_j)
                                    h_last = torch.cat(h_list, dim=0)
                                    state_last = {"per_sample_states": state_list}
                                else:
                                    h_last, state_last = model.rollout_to_step(  # type: ignore[attr-defined]
                                        h_i_mod, state_i, target_step=max_lat_step
                                    )
                                state_last = dict(state_last)
                                if isinstance(state_clean, dict) and "next_compute_range" in state_clean:
                                    state_last["next_compute_range"] = state_clean.get("next_compute_range", state_last.get("next_compute_range"))
                                state_last["step_label"] = state_clean.get("step_label", "end") if isinstance(state_clean, dict) else "end"
                                ablt_state = _build_state_for_logits(h_last, state_last, state_i)
                                logits_ablt = model.compute_logits(h_last, ablt_state, target_ids)
                                kl_logit_ht = None
                            else:
                                raise ValueError(f"Unsupported node_j label: {node_j}")

                        base_logits = logits_clean if node_i == node_j else baseline_cache[node_j]
                        metrics = compute_kl_and_delta_batch(base_logits, logits_ablt, target_ids, eos_id)
                        if kl_logit_ht is not None:
                            metrics["kl_logit_ht"] = kl_logit_ht

                        grad_logprob_vals = grad_logprob_cache.get(node_j, {}).get(node_i)

                        for idx, item in enumerate(items):
                            record = {
                                "sample_id": item["sample_id"],
                                "rank": rank,
                                "step_i": str(node_i),
                                "step_j": str(node_j),
                                "mode": args.mode,
                            }
                            for key, tensor_vals in metrics.items():
                                val = float(tensor_vals[idx].item())
                                record[key] = val
                                adj_values[key][(node_i, node_j)].append(val)
                            if grad_logprob_vals is not None:
                                grad_val = float(grad_logprob_vals[idx].item())
                                record["grad_logprob"] = grad_val
                                adj_values["grad_logprob"][(node_i, node_j)].append(grad_val)
                            writer.write(json.dumps(record) + "\n")
            pbar.update(len(batch_samples))

    # Save adjacency summary (single-rank only; multi-rank users should merge manually)
    if dist is None or rank == 0:
        node_to_idx = {n: idx for idx, n in enumerate(nodes)}
        edge_perc: Dict[str, Dict[tuple[Any, Any], List[float]]] = {k: {} for k in metric_keys}

        for metric_key in metric_keys:
            adj_mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
            for (i_node, j_node), vals in adj_values[metric_key].items():
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                med = np.nanmedian(arr)
                adj_mat[node_to_idx[i_node], node_to_idx[j_node]] = med
                edge_perc[metric_key][(i_node, j_node)] = np.percentile(arr, [25, 50, 75]).tolist()
            if args.save_adj:
                np.save(output_path.with_suffix(f".{metric_key}.adj.npy"), adj_mat)

            if args.save_graph:
                g = nx.DiGraph()
                g.add_nodes_from([str(n) for n in nodes])
                for (i_node, j_node), vals in adj_values[metric_key].items():
                    if not vals:
                        continue
                    w = float(np.nanmedian(np.array(vals, dtype=float)))
                    if np.isnan(w):
                        continue
                    g.add_edge(str(i_node), str(j_node), weight=w, label=f"{w:.3f}")
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(g, seed=42)
                edge_labels = nx.get_edge_attributes(g, "label")
                nx.draw_networkx(g, pos, with_labels=True, node_color="#aed1e9", edge_color="#4a6fa5", arrows=True)
                nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(output_path.with_suffix(f".{metric_key}.png"), dpi=200)
                plt.close()

    # Merge rank outputs
    if dist is not None and world_size > 1:
        dist.barrier()
        if rank == 0:
            with output_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = output_path.with_suffix(output_path.suffix + f".rank{r}")
                    with shard_path.open("r") as shard:
                        for line in shard:
                            merged.write(line)
                    shard_path.unlink(missing_ok=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
