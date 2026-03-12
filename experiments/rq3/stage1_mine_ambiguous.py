#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import build_dataset, create_dataloader, load_config, parse_step_tokens
from common.model_registry import load_model
from data.gsm8k import parse_answer


def normalize_answer(text: str | None, dataset_name: str, model_name: str | None = None) -> str | None:
    if text is None:
        return None
    if dataset_name.lower() == "gsm8k":
        val, _ = parse_answer(text)
        return str(val) if val is not None else None
    if "strategyqa" in dataset_name.lower():
        s = str(text).lower()
        if "<|end-latent|>" in s:
            s = s.split("<|end-latent|>")[-1]
        if "###" in s:
            s = s.split("###")[-1]
        if "the answer is:" in s:
            s = s.split("the answer is:")[-1]
        if "answer:" in s:
            s = s.split("answer:")[-1]
        s = s.replace("\r", " ").replace("\n", " ").strip()
        if not s:
            return None
        tok = s.split()[0]
        tok = re.sub(r"^[^a-z]+|[^a-z]+$", "", tok)
        if tok in {"true", "false", "yes", "no"}:
            if model_name is not None and "codi" in model_name.lower():
                return "true" if tok in {"true", "yes"} else "false"
            return tok
        return None
    s = str(text).strip().lower()
    return s if s else None


def cosine_delta_mean(h_list: List[np.ndarray]) -> float:
    if len(h_list) < 2:
        return 0.0
    deltas = []
    for i in range(len(h_list) - 1):
        a = h_list[i]
        b = h_list[i + 1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cos = float(np.dot(a, b) / denom)
        deltas.append(1.0 - cos)
    return float(np.mean(deltas)) if deltas else 0.0


def _sanitize_token_ids(tokenizer: Any, token_ids: List[int]) -> List[int]:
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    cleaned: List[int] = []
    for tok in token_ids:
        try:
            tok_id = int(tok)
        except Exception:
            continue
        if vocab_size is not None and (tok_id < 0 or tok_id >= vocab_size):
            if unk_id is None:
                continue
            tok_id = int(unk_id)
        cleaned.append(tok_id)
    return cleaned


def safe_decode(tokenizer: Any, token_ids: List[int]) -> str:
    cleaned = _sanitize_token_ids(tokenizer, token_ids)
    try:
        return tokenizer.decode(cleaned, skip_special_tokens=True)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(cleaned)
        unk_tok = getattr(tokenizer, "unk_token", "") or ""
        safe_tokens = [t if isinstance(t, str) else unk_tok for t in tokens]
        safe_tokens = [t for t in safe_tokens if t]
        if not safe_tokens:
            return ""
        return tokenizer.convert_tokens_to_string(safe_tokens)


def sample_coconut(
    model: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    if not hasattr(model, "coconut_model") or not hasattr(model, "tokenizer"):
        raise RuntimeError("Coconut model/tokenizer missing.")
    tokens = model._prepare_inputs(prompt) if hasattr(model, "_prepare_inputs") else None
    if tokens is None:
        tokens = model.tokenizer(prompt, return_tensors="pt")  # type: ignore[attr-defined]
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))

    labels = input_ids.clone()
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).reshape(1, -1)
    outputs = model.coconut_model.forward(  # type: ignore[attr-defined]
        input_ids,
        attention_mask,
        labels,
        position_ids,
    )
    inputs_embeds = outputs.inputs_embeds
    logits = outputs.logits[0, -1, :]

    def sample_token(logits_t: torch.Tensor) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits_t).item())
        probs = torch.softmax(logits_t / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    tokens_list = input_ids[0].detach().tolist()
    next_token = sample_token(logits)
    tokens_list.append(next_token)
    new_token_embed = model.coconut_model.embedding(  # type: ignore[attr-defined]
        torch.tensor(next_token, device=input_ids.device)
    ).view(1, 1, -1)
    new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

    for _ in range(max_new_tokens - 1):
        outputs = model.base_model(inputs_embeds=new_inputs_embeds)  # type: ignore[attr-defined]
        next_token = sample_token(outputs.logits[0, -1, :])
        if next_token == getattr(model.tokenizer, "eos_token_id", None):  # type: ignore[attr-defined]
            break
        tokens_list.append(next_token)
        new_token_embed = model.coconut_model.embedding(  # type: ignore[attr-defined]
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

    return safe_decode(model.tokenizer, tokens_list)  # type: ignore[attr-defined]


def sample_coconut_batch(
    model: Any,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    if not prompts:
        return []
    if not hasattr(model, "coconut_model") or not hasattr(model, "tokenizer"):
        raise RuntimeError("Coconut model/tokenizer missing.")

    tokens = model._prepare_inputs(prompts) if hasattr(model, "_prepare_inputs") else None
    if tokens is None:
        tokens = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)  # type: ignore[attr-defined]
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))

    labels = input_ids.clone()
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).unsqueeze(0).expand(input_ids.size(0), -1)
    outputs = model.coconut_model.forward(  # type: ignore[attr-defined]
        input_ids,
        attention_mask,
        labels,
        position_ids,
    )
    inputs_embeds = outputs.inputs_embeds
    logits = outputs.logits[:, -1, :]

    def sample_tokens(logits_t: torch.Tensor) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits_t, dim=-1)
        probs = torch.softmax(logits_t / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    tokens_list = [input_ids[i].detach().tolist() for i in range(input_ids.size(0))]
    next_token_ids = sample_tokens(logits)
    for i in range(len(tokens_list)):
        tokens_list[i].append(int(next_token_ids[i].item()))

    eos_id = getattr(model.tokenizer, "eos_token_id", None)  # type: ignore[attr-defined]
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    if eos_id is not None:
        finished |= next_token_ids.eq(eos_id)

    new_token_embed = model.coconut_model.embedding(  # type: ignore[attr-defined]
        next_token_ids.to(input_ids.device)
    ).view(input_ids.size(0), 1, -1)
    new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break
        outputs = model.base_model(inputs_embeds=new_inputs_embeds)  # type: ignore[attr-defined]
        next_token_ids = sample_tokens(outputs.logits[:, -1, :])
        if eos_id is not None:
            next_token_ids = torch.where(finished, torch.tensor(eos_id, device=input_ids.device), next_token_ids)
        for i in range(len(tokens_list)):
            if not finished[i]:
                tokens_list[i].append(int(next_token_ids[i].item()))
        if eos_id is not None:
            finished |= next_token_ids.eq(eos_id)
        new_token_embed = model.coconut_model.embedding(  # type: ignore[attr-defined]
            next_token_ids.to(input_ids.device)
        ).view(input_ids.size(0), 1, -1)
        new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

    return [safe_decode(model.tokenizer, tokens) for tokens in tokens_list]  # type: ignore[attr-defined]


def sample_model(
    model: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    model_name: str,
) -> str:
    if model_name == "coconut":
        return sample_coconut(model, prompt, temperature, max_new_tokens)

    gen_kwargs = getattr(model, "generation_kwargs", None)
    restore = None
    if isinstance(gen_kwargs, dict):
        restore = dict(gen_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["greedy"] = False
            else:
                gen_kwargs["do_sample"] = False
                gen_kwargs["greedy"] = True
        setattr(model, "generation_kwargs", gen_kwargs)

    try:
        out = model.run_baseline(prompt)
    finally:
        if restore is not None:
            setattr(model, "generation_kwargs", restore)

    if isinstance(out, list):
        return out[0] if out else ""
    if isinstance(out, dict):
        text = out.get("text")
        if isinstance(text, list):
            return text[0] if text else ""
        if isinstance(text, str):
            return text
    text_attr = getattr(out, "text", None)
    if isinstance(text_attr, list):
        return text_attr[0] if text_attr else ""
    if isinstance(text_attr, str):
        return text_attr
    return str(out)


def sample_model_batch(
    model: Any,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    model_name: str,
) -> List[str]:
    if not prompts:
        return []
    if model_name == "coconut":
        return sample_coconut_batch(model, prompts, temperature, max_new_tokens)

    gen_kwargs = getattr(model, "generation_kwargs", None)
    restore = None
    if isinstance(gen_kwargs, dict):
        restore = dict(gen_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["greedy"] = False
            else:
                gen_kwargs["do_sample"] = False
                gen_kwargs["greedy"] = True
        setattr(model, "generation_kwargs", gen_kwargs)

    try:
        out = model.run_baseline(prompts)
    finally:
        if restore is not None:
            setattr(model, "generation_kwargs", restore)

    if isinstance(out, list):
        return [str(t) for t in out]
    if isinstance(out, dict):
        text = out.get("text")
        if isinstance(text, list):
            return [str(t) for t in text]
        if isinstance(text, str):
            return [text]
    text_attr = getattr(out, "text", None)
    if isinstance(text_attr, list):
        return [str(t) for t in text_attr]
    if isinstance(text_attr, str):
        return [text_attr]
    return [str(out)]


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Stage 1: mine ambiguous samples.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--activity_epsilon", type=float, default=None)
    parser.add_argument("--min_cover", type=float, default=None)
    parser.add_argument("--min_ratio", type=float, default=None)
    parser.add_argument("--max_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_dropout", type=float, default=0.0, help="Dropout probability applied to latent h_t during sampling.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--distributed", action="store_true", help="Shard dataset across ranks (torchrun).")
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config_path)
    rq3_cfg = config.get("rq3", {})
    temperature = args.temperature if args.temperature is not None else float(rq3_cfg.get("temperature", 0.7))
    num_samples = args.num_samples if args.num_samples is not None else int(rq3_cfg.get("num_samples", 20))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(rq3_cfg.get("max_new_tokens", 64))
    activity_epsilon = args.activity_epsilon if args.activity_epsilon is not None else float(rq3_cfg.get("activity_epsilon", 0.05))
    min_cover = args.min_cover if args.min_cover is not None else float(rq3_cfg.get("ambiguity_min_cover", 0.9))
    min_ratio = args.min_ratio if args.min_ratio is not None else float(rq3_cfg.get("ambiguity_min_ratio", 0.3))
    max_ratio = args.max_ratio if args.max_ratio is not None else float(rq3_cfg.get("ambiguity_max_ratio", 0.7))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    model_name = (args.model_name or config.get("model_name") or "coconut").lower()
    model = load_model(model_name, config.get("model", config))

    dataset_full = build_dataset(config, tokenizer=getattr(model, "tokenizer", None))
    dataset = dataset_full
    if dist is not None and world_size > 1:
        dataset = dataset_full[rank::world_size]
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    steps = parse_step_tokens(args.steps or config.get("steps"), config.get("num_steps"))
    latent_steps = [s for s in steps if isinstance(s, int)]
    if not latent_steps:
        raise ValueError("No latent steps available; set --steps or config steps.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "ambiguous_samples.jsonl"
    traj_path = output_dir / "ambiguous_trajectories.jsonl"
    rank_sample_path = sample_path if dist is None else sample_path.with_suffix(sample_path.suffix + f".rank{rank}")
    rank_traj_path = traj_path if dist is None else traj_path.with_suffix(traj_path.suffix + f".rank{rank}")

    dataset_name = str(config.get("dataset_name", ""))
    with rank_sample_path.open("w") as sample_writer, rank_traj_path.open("w") as traj_writer:
        pbar = tqdm(total=len(dataset), desc="samples", disable=rank != 0)
        sample_counter = 0
        for sample in dataloader:
            batch_samples = sample if isinstance(sample, list) else [sample]
            entries: List[Dict[str, Any] | None] = []
            for rec in batch_samples:
                prompt = rec.get("prompt") if isinstance(rec, dict) else None
                question = rec.get("question") if isinstance(rec, dict) else None
                gold = rec.get("answer") if isinstance(rec, dict) else None
                sample_uid = rec.get("id") if isinstance(rec, dict) else None
                sample_id = sample_counter if dist is None else sample_counter * world_size + rank
                sample_counter += 1
                if not prompt:
                    entries.append(None)
                    continue
                entries.append(
                    {
                        "prompt": prompt,
                        "question": question,
                        "gold": gold,
                        "sample_uid": sample_uid,
                        "sample_id": sample_id,
                    }
                )

            valid = [(idx, entry) for idx, entry in enumerate(entries) if entry is not None]
            if not valid:
                pbar.update(len(batch_samples))
                continue
            prompts = [entry["prompt"] for _, entry in valid]

            trajectories_per_sample: List[List[Tuple[int, str, str | None]]] = [
                [] for _ in valid
            ]
            gen_iter = range(num_samples)
            if rank == 0:
                gen_iter = tqdm(gen_iter, desc="gen", leave=False)
            for k in gen_iter:
                texts = sample_model_batch(model, prompts, temperature, max_new_tokens, model_name)
                if len(texts) != len(prompts):
                    texts = (texts + [""] * len(prompts))[: len(prompts)]
                for idx, text in enumerate(texts):
                    answer_norm = normalize_answer(text, dataset_name, model_name)
                    trajectories_per_sample[idx].append((k, text, answer_norm))

            latent_paths_per_sample: List[List[Path]] = [[] for _ in valid]
            latent_iter = range(num_samples)
            if rank == 0:
                latent_iter = tqdm(latent_iter, desc="latent", leave=False)
            for k in latent_iter:
                h_lists = [[] for _ in valid]
                for t in latent_steps:
                    h_t, _ = model.forward_until_step(prompts, t)
                    if args.latent_dropout > 0:
                        h_t = F.dropout(h_t, p=args.latent_dropout, training=True)
                    for idx in range(len(valid)):
                        h_lists[idx].append(h_t[idx].detach().cpu().numpy())
                for idx, (_, entry) in enumerate(valid):
                    latent_arr = np.stack(h_lists[idx], axis=0)
                    sample_id = entry["sample_id"]
                    latent_file = traj_dir / f"{sample_id}_traj{k}_r{rank}.npy"
                    np.save(latent_file, latent_arr)
                    latent_paths_per_sample[idx].append(latent_file)

            for idx, (_, entry) in enumerate(valid):
                trajectories = trajectories_per_sample[idx]
                counts: Dict[str, int] = {}
                for _, _, ans in trajectories:
                    if ans is None:
                        continue
                    counts[ans] = counts.get(ans, 0) + 1
                if not counts:
                    continue
                sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
                if len(sorted_counts) >= 2:
                    (ans_a, cnt_a), (ans_b, cnt_b) = sorted_counts[0], sorted_counts[1]
                else:
                    (ans_a, cnt_a) = sorted_counts[0]
                    ans_b, cnt_b = None, 0

                # For CODI StrategyQA, require opposite boolean answers; skip otherwise.
                if "strategyqa" in dataset_name.lower() and "codi" in model_name.lower():
                    if ans_b is None or ans_a not in {"true", "false"} or ans_b not in {"true", "false"} or ans_a == ans_b:
                        print(
                            f"[ambiguous-skip] sample_id={entry['sample_id']} ans_a={ans_a} ans_b={ans_b} counts={sorted_counts}"
                        )
                        continue
                total_cnt = cnt_a + cnt_b
                ratio = (cnt_b / total_cnt) if total_cnt > 0 else 0.0

                if "strategyqa" in dataset_name.lower():
                    if ans_b is None:
                        continue
                else:
                    if ans_b is None:
                        continue
                    if (cnt_a + cnt_b) < math.ceil(min_cover * num_samples):
                        continue
                    if ratio <= min_ratio or ratio >= max_ratio:
                        continue

                v_vals = []
                for latent_file in latent_paths_per_sample[idx]:
                    arr = np.load(latent_file)
                    v_vals.append(cosine_delta_mean([arr[i] for i in range(arr.shape[0])]))
                v_mean = float(np.mean(v_vals)) if v_vals else 0.0
                if "strategyqa" not in dataset_name.lower():
                    if v_mean < activity_epsilon:
                        continue

                sample_writer.write(
                    json.dumps(
                        {
                            "sample_id": entry["sample_id"],
                            "sample_uid": entry["sample_uid"],
                            "question": entry["question"],
                            "prompt": entry["prompt"],
                            "gold_answer": entry["gold"],
                            "answer_A": ans_a,
                            "answer_B": ans_b,
                            "count_A": cnt_a,
                            "count_B": cnt_b,
                            "ratio_B": ratio,
                            "activity_v": v_mean,
                            "latent_steps": latent_steps,
                        }
                    )
                    + "\n"
                )

                for k, text, ans in trajectories:
                    if ans is None:
                        continue
                    label = "A" if ans == ans_a else ("B" if ans_b is not None and ans == ans_b else "other")
                    if label == "other":
                        continue
                    traj_writer.write(
                        json.dumps(
                            {
                                "sample_id": entry["sample_id"],
                                "sample_uid": entry["sample_uid"],
                                "traj_id": k,
                                "answer_norm": ans,
                                "cluster": label,
                                "answer_text": text,
                                "latent_path": str(traj_dir / f"{entry['sample_id']}_traj{k}_r{rank}.npy"),
                            }
                        )
                        + "\n"
                    )

            pbar.update(len(batch_samples))

    if dist is not None and world_size > 1:
        dist.barrier()
        if rank == 0:
            with sample_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = sample_path.with_suffix(sample_path.suffix + f".rank{r}")
                    if not shard_path.exists():
                        continue
                    with shard_path.open("r") as shard:
                        for line in shard:
                            merged.write(line)
                    shard_path.unlink(missing_ok=True)
            with traj_path.open("w") as merged:
                for r in range(world_size):
                    shard_path = traj_path.with_suffix(traj_path.suffix + f".rank{r}")
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
