from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Avoid protobuf binary compatibility issues (e.g., onnx imports) in some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.model_registry import load_model
from common.experiment_utils import (
    build_dataset,
    create_dataloader,
    load_config,
    parse_step_tokens,
    summarize_output,
    parse_modes,
    prepare_target_ids,
    build_teacher_state,
    _clone_teacher_state,
    strip_caches,
)
from data.gsm8k import parse_answer


def _slice_batched_value(value: Any, indices: List[int], batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.dim() > 0 and value.size(0) == batch_size:
            idx = torch.as_tensor(indices, device=value.device, dtype=torch.long)
            return value.index_select(0, idx)
        return value
    if isinstance(value, dict):
        return {k: _slice_batched_value(v, indices, batch_size) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == batch_size:
            return [value[i] for i in indices]
        if value and isinstance(value[0], tuple):
            sliced: List[Any] = []
            idx = None
            for entry in value:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and torch.is_tensor(entry[0])
                    and entry[0].dim() > 0
                    and entry[0].size(0) == batch_size
                ):
                    if idx is None:
                        idx = torch.as_tensor(indices, device=entry[0].device, dtype=torch.long)
                    k = entry[0].index_select(0, idx)
                    v = entry[1].index_select(0, idx)
                    sliced.append((k, v))
                else:
                    sliced.append(entry)
            return sliced
        return value
    return value


def _run_baseline_batch(model: Any, prompts: List[str]) -> tuple[Optional[Any], Optional[str]]:
    try:
        return model.run_baseline(prompts), None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step sufficiency / logit-lens experiment.")
    parser.add_argument("--model_name", required=True, help="Model registry name.")
    parser.add_argument("--config_path", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--steps", default=None, help="Comma separated list or 'all'.")
    parser.add_argument("--output_path", required=True, help="Where to save JSONL.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for input loading.")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed to shard data across ranks (torchrun).")
    parser.add_argument("--dist_backend", default="nccl", help="torch.distributed backend (default: nccl).")
    parser.add_argument("--dist_url", default="env://", help="Init method for torch.distributed (default: env:// for torchrun).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torchrun.")
    parser.add_argument(
        "--modes",
        default="decode,logit_lens_single,logit_lens_teacher,baseline",
        help=(
            "Comma separated modes to run: decode (early decode), "
            "logit_lens_single (single-step lm_head), logit_lens_teacher (teacher-forced gold continuation), "
            "baseline (no-latent generation)."
        ),
    )
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
        raise ValueError(f"Dataset is empty. Check dataset_path or inputs in config: {args.config_path}")
    dataset = dataset_full
    if dist is not None and world_size > 1:
        dataset = dataset_full[rank::world_size]
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    use_gsm8k_parse = config.get("dataset_name") == "gsm8k"
    step_arg = args.steps if args.steps is not None else config.get("steps")
    steps = parse_step_tokens(step_arg, config.get("num_steps"))
    modes = parse_modes(args.modes)
    do_decode = "decode" in modes
    do_logit_single = "logit_lens_single" in modes
    do_logit_teacher = "logit_lens_teacher" in modes
    do_baseline = "baseline" in modes

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = output_path if dist is None else output_path.with_suffix(output_path.suffix + f".rank{rank}")

    total_samples = len(dataset)
    with rank_output_path.open("w") as writer:
        pbar = tqdm(total=total_samples, desc="samples", disable=rank != 0)
        sample_counter = 0
        for batch in dataloader:
            batch_samples = batch if isinstance(batch, list) else [batch]
            items: List[Dict[str, Any]] = []
            for sample in batch_samples:
                prompt = sample["prompt"] if isinstance(sample, dict) else sample
                if not prompt:
                    sample_counter += 1
                    continue
                gold_answer = sample.get("answer") if isinstance(sample, dict) else None
                target_answer = parse_answer(gold_answer)[0] if gold_answer and use_gsm8k_parse else gold_answer
                question = sample.get("question") if isinstance(sample, dict) else None
                sample_id = sample_counter if dist is None else sample_counter * world_size + rank
                sample_counter += 1

                target_ids = None
                target_error = None
                if do_logit_teacher and hasattr(model, "compute_logits") and target_answer:
                    try:
                        # target_answer 已经过 parse_answer 处理过，这里不再重复 gsm8k 解析。
                        target_ids = prepare_target_ids(model, target_answer, use_gsm8k_parse=False)
                    except Exception as exc:
                        target_ids = None
                        target_error = str(exc)

                items.append(
                    {
                        "prompt": prompt,
                        "question": question,
                        "gold_answer": gold_answer,
                        "sample_id": sample_id,
                        "target_ids": target_ids,
                        "target_error": target_error,
                    }
                )

            if not items:
                pbar.update(len(batch_samples))
                continue

            prompts = [item["prompt"] for item in items]

            if do_baseline:
                batch_size = len(items)
                baseline_outputs: Dict[int, Dict[str, Any]] = {}
                baseline_errors: Dict[int, str] = {}
                baseline_batch, baseline_batch_error = _run_baseline_batch(model, prompts)
                if baseline_batch_error is None and baseline_batch is not None:
                    for idx in range(batch_size):
                        baseline_slice = _slice_batched_value(baseline_batch, [idx], batch_size)
                        baseline_outputs[idx] = summarize_output(strip_caches(baseline_slice))
                else:
                    for idx, item in enumerate(items):
                        try:
                            baseline = model.run_baseline(item["prompt"])
                            baseline_outputs[idx] = summarize_output(strip_caches(baseline))
                        except Exception as exc:  # pragma: no cover - best effort
                            baseline_errors[idx] = str(exc)
                for idx, item in enumerate(items):
                    record = {
                        "sample_id": item["sample_id"],
                        "rank": rank,
                        "question": item["question"],
                        "gold_answer": item["gold_answer"],
                        "step": "baseline",
                        "modes": modes,
                    }
                    if idx in baseline_outputs:
                        record["decode"] = baseline_outputs[idx]
                    if idx in baseline_errors:
                        record["decode_error"] = baseline_errors[idx]
                    writer.write(json.dumps(record) + "\n")

            for step in tqdm(steps, desc="steps", leave=False, disable=rank != 0):
                h_t, state = model.forward_until_step(prompts, step)
                batch_size = h_t.size(0)

                decode_outputs = None
                decode_error = None
                if do_decode:
                    try:
                        state_dec = dict(state)
                        state_dec["decode_continue_latents"] = False
                        decode_outputs = model.decode_from_state(h_t, state_dec)  # type: ignore[arg-type]
                    except Exception as exc:  # pragma: no cover - best effort
                        decode_error = str(exc)

                logit_single_outputs = None
                logit_single_error = None
                if do_logit_single and hasattr(model, "logits_from_latent"):
                    try:
                        logit_single_outputs = model.logits_from_latent(h_t)
                    except Exception as exc:  # pragma: no cover - best effort
                        logit_single_error = str(exc)

                teacher_mean: Dict[int, float] = {}
                teacher_final: Dict[int, float] = {}
                teacher_summary: Dict[int, Dict[str, Any]] = {}
                teacher_error: Dict[int, str] = {}

                if do_logit_teacher:
                    grouped: Dict[int, List[int]] = {}
                    for idx, item in enumerate(items):
                        target_ids = item["target_ids"]
                        if target_ids is None:
                            if item["target_error"] is not None:
                                teacher_error[idx] = item["target_error"]
                            continue
                        grouped.setdefault(int(target_ids.size(1)), []).append(idx)

                    for _, indices in grouped.items():
                        target_batch = torch.cat([items[i]["target_ids"] for i in indices], dim=0)
                        h_sub = h_t.index_select(
                            0,
                            torch.as_tensor(indices, device=h_t.device, dtype=torch.long),
                        )
                        state_sub = _slice_batched_value(state, indices, batch_size)
                        try:
                            teacher_state = _clone_teacher_state(build_teacher_state(state_sub))
                            logits_tf = model.compute_logits(h_sub, teacher_state, target_batch)  # type: ignore[arg-type]
                            log_probs = torch.log_softmax(logits_tf, dim=-1)
                            labels = target_batch.to(log_probs.device)
                            per_tok = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                            mean_logprob = per_tok.mean(dim=1)

                            seq_len = labels.size(1)
                            final_pos = seq_len - 2 if seq_len >= 2 else seq_len - 1
                            tgt = labels[:, final_pos]
                            gold_lp = log_probs[:, final_pos, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                            for offset, idx in enumerate(indices):
                                teacher_mean[idx] = mean_logprob[offset].item()
                                teacher_final[idx] = gold_lp[offset].item()
                                teacher_summary[idx] = summarize_output(logits_tf[offset : offset + 1])
                        except Exception as exc:  # pragma: no cover
                            for idx in indices:
                                teacher_error[idx] = str(exc)

                for idx, item in enumerate(items):
                    record: Dict[str, Any] = {
                        "sample_id": item["sample_id"],
                        "rank": rank,
                        "question": item["question"],
                        "gold_answer": item["gold_answer"],
                        "step": step,
                        "h_t_shape": list(h_t[idx : idx + 1].shape),
                        "modes": modes,
                    }

                    if do_decode:
                        if decode_error is not None:
                            record["decode_error"] = decode_error
                        elif decode_outputs is not None:
                            decode_slice = _slice_batched_value(decode_outputs, [idx], batch_size)
                            record["decode"] = summarize_output(strip_caches(decode_slice))

                    if do_logit_single and hasattr(model, "logits_from_latent"):
                        if logit_single_error is not None:
                            record["logit_lens_single_error"] = logit_single_error
                        elif logit_single_outputs is not None:
                            logit_slice = _slice_batched_value(logit_single_outputs, [idx], batch_size)
                            record["logit_lens_single"] = summarize_output(logit_slice)

                    if do_logit_teacher:
                        if idx in teacher_mean:
                            record["logit_lens_teacher_mean_logprob"] = teacher_mean[idx]
                            record["logit_lens_teacher_final_logprob"] = teacher_final[idx]
                            record["logit_lens_teacher"] = teacher_summary[idx]
                        elif idx in teacher_error:
                            record["logit_lens_teacher_error"] = teacher_error[idx]

                    writer.write(json.dumps(record) + "\n")
            pbar.update(len(batch_samples))

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
