from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as torch_dist
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.analysis.step_ablation import (
    estimate_global_mean_latent,
    estimate_step_mean_latents,
    run_step_ablation,
)
from common.model_registry import load_model
from common.experiment_utils import (
    build_dataset,
    create_dataloader,
    load_config,
    parse_step_tokens,
    summarize_output,
    parse_modes,
    prepare_target_ids,
    strip_caches,
    build_teacher_state,
    compute_teacher_forced_metrics,
    compute_teacher_forced_metrics_batch,
    compute_teacher_forced_metrics_from_logits,
)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent step ablations.")
    parser.add_argument("--model_name", required=True, help="Model registry name.")
    parser.add_argument("--config_path", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--modes", default="zero,mean,gaussian_h,gaussian_mu,mean_step,gaussian_mu_step", help="Comma separated list. If omitted, read from config.")
    parser.add_argument("--steps", default=None, help="Comma separated list or 'all'.")
    parser.add_argument("--output_path", required=True, help="Where to save JSONL.")
    parser.add_argument("--max_mean_steps", type=int, default=None)
    parser.add_argument(
        "--mean_cache_path",
        default=None,
        help="Optional path to cache/load mean latents (.pt).",
    )
    parser.add_argument(
        "--only_estimate_mean",
        action="store_true",
        help="Only compute and cache mean latents, then exit (no ablation).",
    )
    parser.add_argument("--sigma_scale", type=float, default=0.5)
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
    parser.add_argument("--dry_run", action="store_true", help="Process a small subset and print records for quick inspection.")
    parser.add_argument("--dry_run_samples", type=int, default=10, help="Number of samples to keep when --dry_run is set.")
    args = parser.parse_args()

    # Distributed setup (optional)
    dist = None
    rank = 0
    world_size = 1
    local_rank = args.local_rank
    if args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        
        dist = torch_dist
        if local_rank < 0:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank,
        )

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    # If distributed, pin model to local device to avoid all ranks piling onto cuda:0
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
    if args.dry_run:
        dataset_full = dataset_full[: args.dry_run_samples]
    if not dataset_full:
        raise ValueError(f"Dataset is empty. Check dataset_path or inputs in config: {args.config_path}")
    # Shard dataset across ranks if distributed
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

    modes = parse_modes(args.modes)
    steps_arg = args.steps if args.steps is not None else config.get("steps")
    steps = parse_step_tokens(steps_arg, config.get("num_steps"))
    numeric_steps = [s for s in steps if isinstance(s, int)]
    max_config_steps = config.get("num_steps")
    if max_config_steps is not None:
        max_config_steps = int(max_config_steps)
        invalid_steps = [s for s in numeric_steps if s < 1 or s > max_config_steps]
        if invalid_steps:
            raise ValueError(
                f"Numeric steps must be within [1, {max_config_steps}], got invalid: {invalid_steps}"
            )
    use_gsm8k_parse = config.get("dataset_name") == "gsm8k"

    mu = None
    mu_by_step = None
    mean_cache = Path(args.mean_cache_path) if args.mean_cache_path else None
    if mean_cache and mean_cache.exists():
        payload = torch.load(mean_cache, map_location="cpu")
        mu = payload.get("mu")
        mu_by_step = payload.get("mu_by_step")

    if any(mode in {"mean", "gaussian_mu"} for mode in modes) and mu is None:
        mean_dataset = [
            sample["prompt"] if isinstance(sample, dict) and "prompt" in sample else sample
            for sample in dataset_full
        ]
        if dist is None or rank == 0:
            mu = estimate_global_mean_latent(
                model,
                create_dataloader(
                    mean_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=args.prefetch_factor,
                ),
                max_steps=args.max_mean_steps,
            )
    if any(mode in {"mean_step", "gaussian_mu_step"} for mode in modes) and mu_by_step is None:
        mean_dataset = [
            sample["prompt"] if isinstance(sample, dict) and "prompt" in sample else sample
            for sample in dataset_full
        ]
        max_step_to_collect = max(numeric_steps) if numeric_steps else args.max_mean_steps
        if dist is None or rank == 0:
            mu_by_step = estimate_step_mean_latents(
                model,
                create_dataloader(
                    mean_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=args.prefetch_factor,
                ),
                max_steps=max_step_to_collect,
                include_start=False,
            )

    if mean_cache and (mu is not None or mu_by_step is not None) and (dist is None or rank == 0):
        mean_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": mu, "mu_by_step": mu_by_step}, mean_cache)

    if dist is not None and world_size > 1 and mean_cache is not None:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        if mean_cache.exists() and (mu is None or mu_by_step is None):
            payload = torch.load(mean_cache, map_location="cpu")
            mu = payload.get("mu", mu)
            mu_by_step = payload.get("mu_by_step", mu_by_step)

    if args.only_estimate_mean:
        if mean_cache is None:
            raise ValueError("--only_estimate_mean requires --mean_cache_path.")
        if dist is not None and world_size > 1:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
            dist.destroy_process_group()
        return

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = output_path if dist is None else output_path.with_suffix(output_path.suffix + f".rank{rank}")

    num_samples = len(dataset)
    with rank_output_path.open("w") as writer:
        pbar = tqdm(total=num_samples, desc="samples", disable=rank != 0)
        sample_counter = 0
        for batch in dataloader:
            batch_samples = batch if isinstance(batch, list) else [batch]
            batched_by_len: Dict[int, list[Dict[str, Any]]] = {}
            per_sample: list[Dict[str, Any]] = []
            for sample in batch_samples:
                prompt = sample["prompt"] if isinstance(sample, dict) else sample
                gold_answer = sample.get("answer_clean") or sample.get("answer") if isinstance(sample, dict) else None
                question = sample.get("question") if isinstance(sample, dict) else None
                sample_id_field = sample.get("id") if isinstance(sample, dict) else None
                if gold_answer is not None and hasattr(model, "build_teacher_target_ids"):
                    target_ids = model.build_teacher_target_ids(gold_answer)
                else:
                    target_ids = prepare_target_ids(model, gold_answer, use_gsm8k_parse)

                sample_id = sample_counter if dist is None else sample_counter * world_size + rank
                sample_counter += 1

                payload = {
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "question": question,
                    "sample_id_field": sample_id_field,
                    "sample_id": sample_id,
                    "target_ids": target_ids,
                }
                if target_ids is None:
                    per_sample.append(payload)
                else:
                    if target_ids.dim() == 1:
                        target_ids = target_ids.unsqueeze(0)
                    payload["target_ids"] = target_ids.to(model.device)
                    target_len = int(target_ids.size(1))
                    batched_by_len.setdefault(target_len, []).append(payload)

            for target_len, items in batched_by_len.items():
                prompts = [item["prompt"] for item in items]
                target_ids = torch.cat([item["target_ids"] for item in items], dim=0)
                for step in tqdm(steps, desc="steps", leave=False, disable=rank != 0):
                    with torch.no_grad():
                        h_t, other_state = model.forward_until_step(prompts, step)
                        baseline_rollout = model.rollout_from_step(h_t, other_state)
                        baseline_state = build_teacher_state(baseline_rollout, other_state)
                        baseline_logits = None
                        if baseline_state is not None:
                            try:
                                baseline_logits = model.compute_logits(h_t, baseline_state, target_ids)
                            except Exception:
                                baseline_logits = None

                    for mode in tqdm(modes, desc="modes", leave=False, disable=rank != 0):
                        # Skip per-step mean ablations when we do not have an estimated mean for that step.
                        if mode in {"mean_step", "gaussian_mu_step"} and (
                            mu_by_step is None or step not in mu_by_step
                        ):
                            continue
                        result = run_step_ablation(
                            model,
                            prompts,
                            step_t=step,
                            mode=mode,
                            mu=mu,
                            mu_by_step=mu_by_step,
                            sigma_scale=args.sigma_scale,
                            state=(h_t, other_state),
                            baseline_rollout=baseline_rollout,
                        )
                        ablated_state = build_teacher_state(result["ablated"], result.get("state"))
                        if rank == 0 and sample_counter <= 1 and step == steps[0] and mode == modes[0]:
                            print(f"DEBUG: using cache source baseline={baseline_state.get('_cache_source') if baseline_state else None} ablated={ablated_state.get('_cache_source') if ablated_state else None}")
                            if baseline_state and "past_key_values_latents" in result.get("baseline", {}):
                                print("DEBUG: baseline includes past_key_values_latents")
                            if ablated_state and "past_key_values_latents" in result.get("ablated", {}):
                                print("DEBUG: ablated includes past_key_values_latents")
                        tf_metrics = {}
                        if baseline_logits is not None and ablated_state is not None:
                            try:
                                ablt_logits = model.compute_logits(result["h_t_modified"], ablated_state, target_ids)
                                tf_metrics = compute_teacher_forced_metrics_from_logits(
                                    baseline_logits, ablt_logits, target_ids
                                )
                            except Exception:
                                tf_metrics = {}
                        else:
                            tf_metrics = compute_teacher_forced_metrics_batch(
                                model,
                                target_ids,
                                result["h_t"],
                                result["h_t_modified"],
                                baseline_state,
                                ablated_state,
                            )
                        for idx, item in enumerate(items):
                            def _select_sample_output(output: Any, sample_idx: int) -> Any:
                                if not isinstance(output, dict):
                                    return output
                                selected = dict(output)
                                text_val = selected.get("text")
                                if isinstance(text_val, list) and len(text_val) > sample_idx:
                                    selected["text"] = [text_val[sample_idx]]
                                return selected

                            baseline_clean = strip_caches(_select_sample_output(result["baseline"], idx))
                            ablated_clean = strip_caches(_select_sample_output(result["ablated"], idx))
                            record = {
                                "sample_id": item["sample_id"],
                                "sample_uid": item["sample_id_field"],
                                "batch_idx": idx,
                                "rank": rank,
                                "question": item["question"],
                                "gold_answer": item["gold_answer"],
                                "step": step,
                                "mode": mode,
                                "baseline": summarize_output(baseline_clean),
                                "ablated": summarize_output(ablated_clean),
                                "h_t_shape": list(result["h_t"].shape)
                                if isinstance(result.get("h_t"), torch.Tensor)
                                else None,
                            }
                            for key, val in tf_metrics.items():
                                record[key] = float(val[idx].item())
                            line = json.dumps(record)
                            writer.write(line + "\n")
                            if args.dry_run and rank == 0:
                                print(line)

            for item in per_sample:
                prompt = item["prompt"]
                gold_answer = item["gold_answer"]
                question = item["question"]
                target_ids = item["target_ids"]
                for step in tqdm(steps, desc="steps", leave=False, disable=rank != 0):
                    with torch.no_grad():
                        h_t, other_state = model.forward_until_step(prompt, step)
                        baseline_rollout = model.rollout_from_step(h_t, other_state)
                        baseline_state = build_teacher_state(baseline_rollout, other_state)
                        baseline_logits = None
                        if baseline_state is not None:
                            try:
                                baseline_logits = model.compute_logits(h_t, baseline_state, target_ids)
                            except Exception:
                                baseline_logits = None

                    for mode in tqdm(modes, desc="modes", leave=False, disable=rank != 0):
                        if mode in {"mean_step", "gaussian_mu_step"} and (
                            mu_by_step is None or step not in mu_by_step
                        ):
                            continue
                        result = run_step_ablation(
                            model,
                            prompt,
                            step_t=step,
                            mode=mode,
                            mu=mu,
                            mu_by_step=mu_by_step,
                            sigma_scale=args.sigma_scale,
                            state=(h_t, other_state),
                            baseline_rollout=baseline_rollout,
                        )
                        ablated_state = build_teacher_state(result["ablated"], result.get("state"))
                        tf_metrics = {}
                        if baseline_logits is not None and ablated_state is not None:
                            try:
                                ablt_logits = model.compute_logits(result["h_t_modified"], ablated_state, target_ids)
                                tf_metrics = compute_teacher_forced_metrics_from_logits(
                                    baseline_logits, ablt_logits, target_ids
                                )
                                tf_metrics = {k: float(v.mean().item()) for k, v in tf_metrics.items()}
                            except Exception:
                                tf_metrics = {}
                        else:
                            tf_metrics = compute_teacher_forced_metrics(
                                model,
                                target_ids,
                                result["h_t"],
                                result["h_t_modified"],
                                baseline_state,
                                ablated_state,
                            )
                        baseline_clean = strip_caches(result["baseline"])
                        ablated_clean = strip_caches(result["ablated"])
                        record = {
                            "sample_id": item["sample_id"],
                            "sample_uid": item["sample_id_field"],
                            "batch_idx": 0,
                            "rank": rank,
                            "question": question,
                            "gold_answer": gold_answer,
                            "step": step,
                            "mode": mode,
                            "baseline": summarize_output(baseline_clean),
                            "ablated": summarize_output(ablated_clean),
                            "h_t_shape": list(result["h_t"].shape)
                            if isinstance(result.get("h_t"), torch.Tensor)
                            else None,
                        }
                        record.update(tf_metrics)
                        line = json.dumps(record)
                        writer.write(line + "\n")
                        if args.dry_run and rank == 0:
                            print(line)
            pbar.update(len(batch_samples))

    # Merge rank files on rank 0
    if dist is not None and world_size > 1:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        if rank == 0:
            with output_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = output_path.with_suffix(output_path.suffix + f".rank{r}")
                    with shard_path.open("r") as shard:
                        for line in shard:
                            merged.write(line)
                    shard_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
