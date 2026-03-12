from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers.cache_utils import DynamicCache  # type: ignore

from data.gsm8k import parse_answer, to_record
from common.path_utils import expand_nested_paths

from datasets import load_dataset  # type: ignore



class SimpleDataset(Dataset):
    """Tiny helper to wrap a Python list so DataLoader can iterate it."""

    def __init__(self, data: Sequence[Any]) -> None:
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        text = f.read()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed."
            ) from exc
        config = yaml.safe_load(text)
    return expand_nested_paths(config)


def _apply_prompt(
    question: str,
    template: Optional[str],
    tokenizer: Any | None = None,
    system_instruction: Optional[str] = None,
    use_chat_template: bool = False,
) -> str:
    """
    Build a prompt string, optionally using the tokenizer's chat template to prepend a system instruction.
    """
    user_content = template.format(question=question) if template else f"Question: {question}\nAnswer:"

    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_content if template else question})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fall back to plain text below if chat templating fails.
            pass

    if system_instruction:
        return f"{system_instruction}\n\n{user_content}"
    return user_content


def _load_gsm8k(config: Dict[str, Any], tokenizer: Any | None = None) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("datasets package is required to load gsm8k automatically.")
    split = config.get("dataset_split", "test")
    subset = config.get("dataset_subset", "main")
    prompt_template = config.get("prompt_template")
    system_instruction = config.get("system_instruction")
    use_chat_template = bool(config.get("use_chat_template", False))
    ds = load_dataset(config.get("dataset_path"), subset, split=split)
    formatted = []
    for ex in ds:
        q = ex.get("question", "")
        prompt = _apply_prompt(
            q, prompt_template, tokenizer=tokenizer, system_instruction=system_instruction, use_chat_template=use_chat_template
        )
        rec = to_record(ex, prompt)
        formatted.append(
            {
                "prompt": rec.prompt,
                "question": rec.question,
                "answer": rec.answer,
                "answer_clean": rec.answer_clean,
                "answer_value": rec.answer_value,
                "id": rec.id,
            }
        )
    return formatted


def _normalize_steps(steps: Any) -> Optional[List[str]]:
    if steps is None:
        return None
    if isinstance(steps, list):
        return [str(s) for s in steps if str(s).strip()]
    # Numpy array or pandas series
    try:
        import numpy as np  # type: ignore

        if isinstance(steps, np.ndarray):
            return [str(s) for s in steps.tolist() if str(s).strip()]
    except Exception:
        pass
    if isinstance(steps, str):
        parts = [s.strip() for s in steps.split("\n") if s.strip()]
        return parts if parts else [steps.strip()]
    return [str(steps)]


def _load_gsm8k_aug_local(config: Dict[str, Any], tokenizer: Any | None = None) -> List[Dict[str, Any]]:
    dataset_path = config.get("dataset_path")
    if dataset_path is None:
        raise ValueError("gsm8k-aug loader requires dataset_path.")
    path = Path(dataset_path)
    split = config.get("dataset_split", "validation")
    prompt_template = config.get("prompt_template")
    system_instruction = config.get("system_instruction")
    use_chat_template = bool(config.get("use_chat_template", False))

    if path.is_dir():
        data_dir = path / "data"
        if data_dir.is_dir():
            candidates = sorted(data_dir.glob(f"{split}-*.parquet"))
        else:
            candidates = sorted(path.glob(f"{split}-*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet files for split '{split}' under {path}")
        parquet_path = candidates[0]
    else:
        parquet_path = path
        if not parquet_path.exists():
            raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required to load local gsm8k-aug parquet.") from exc

    df = pd.read_parquet(parquet_path)
    formatted = []
    for _, row in df.iterrows():
        q = str(row.get("question", ""))
        prompt = _apply_prompt(
            q, prompt_template, tokenizer=tokenizer, system_instruction=system_instruction, use_chat_template=use_chat_template
        )
        steps = _normalize_steps(row.get("steps") if "steps" in row else row.get("cot"))
        formatted.append(
            {
                "prompt": prompt,
                "question": q,
                "answer": row.get("answer"),
                "steps": steps,
            }
        )
    return formatted


def build_dataset(config: Dict[str, Any], tokenizer: Any | None = None) -> List[Any]:
    system_instruction = config.get("system_instruction")
    use_chat_template = bool(config.get("use_chat_template", False))
    prompt_template = config.get("prompt_template")
    dataset_path = config.get("dataset_path")
    if dataset_path:
        path = Path(dataset_path)
        if path.is_dir():
            dataset_name = config.get("dataset_name", "").lower()
            if dataset_name in {"gsm8k-aug", "gsm8k_aug", "gsm8kaug"}:
                return _load_gsm8k_aug_local(config, tokenizer=tokenizer)
        if path.exists() and path.is_file():
            data: List[Any] = []
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        sample = json.loads(line)
                        # Prefer explicit question field if available
                        base_rec = {
                            "answer": sample.get("answer"),
                            "answer_clean": sample.get("answer_clean"),
                            "answer_value": sample.get("answer_value"),
                            "id": sample.get("id"),
                        }
                        steps = sample.get("steps") or sample.get("cot")
                        if steps is not None:
                            base_rec["steps"] = _normalize_steps(steps)
                        if "question" in sample:
                            rec = {
                                "prompt": _apply_prompt(
                                    sample["question"],
                                    prompt_template,
                                    tokenizer=tokenizer,
                                    system_instruction=system_instruction,
                                    use_chat_template=use_chat_template,
                                ),
                                "question": sample["question"],
                                **base_rec,
                            }
                        else:
                            rec = {
                                "prompt": sample.get("input", sample),
                                "question": sample.get("question"),
                                **base_rec,
                            }
                        data.append(rec)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"dataset_path not found: {dataset_path}") from exc
            return data
    if config.get("dataset_name") == "gsm8k":
        return _load_gsm8k(config, tokenizer=tokenizer)
    if config.get("dataset_name") in {"gsm8k-aug", "gsm8k_aug", "gsm8kaug"}:
        return _load_gsm8k_aug_local(config, tokenizer=tokenizer)
    return []


def parse_steps(steps_arg: str | None, fallback: int | None) -> List[int]:
    if steps_arg:
        if steps_arg.lower() == "all" and fallback is not None:
            return list(range(1, fallback + 1))
        return [int(s) for s in steps_arg.split(",") if s != ""]
    if fallback is not None:
        return list(range(1, fallback + 1))
    return [1]

def parse_modes(modes_str: str) -> List[str]:
    """
    Accept a comma-separated list of experiment modes.
    Supported: decode, logit_lens_single (single-step), logit_lens_teacher (teacher-forced).
    """
    return [m.strip() for m in modes_str.split(",") if m.strip()]

def parse_step_tokens(steps_arg: str | None, fallback: int | None) -> List[int]:
    """
    Extended step parser that accepts numeric tokens.
    Numeric tokens can be provided as strings and are returned as ints.
    """
    def _norm(token: str) -> int | None:
        tok = token.strip()
        if tok == "":
            return None
        try:
            return int(tok)
        except ValueError:
            return None

    if steps_arg:
        if steps_arg.lower() == "all" and fallback is not None:
            return list(range(1, fallback + 1))
        parsed = [_norm(s) for s in steps_arg.split(",")]
        return [p for p in parsed if p is not None]
    if fallback is not None:
        return list(range(1, fallback + 1))
    return [1]


def step_order_key(step: int) -> Tuple[int, int]:
    """
    Provide a sortable key for step comparisons.
    Order: numeric (ascending), non-numeric last.
    """
    if isinstance(step, str):
        try:
            num = int(step)
            return (num, num)
        except ValueError:
            return (10**9 + 1, 10**9 + 1)
    return (int(step), int(step))

def summarize_output(output: Any) -> Dict[str, Any]:
    """Reduce heavy outputs to lightweight summaries for logging."""
    summary: Dict[str, Any] = {}
    if isinstance(output, list):
        summary["text"] = output
        return summary
    if isinstance(output, dict):
        for key, value in output.items():
            if key == "text" and isinstance(value, list) and all(isinstance(v, str) for v in value):
                summary[key] = value
                continue
            if isinstance(value, torch.Tensor):
                summary[key] = {
                    "shape": list(value.shape),
                    "mean": value.detach().float().mean().item(),
                }
            else:
                summary[key] = str(value)
        return summary
    if hasattr(output, "logits"):
        logits = output.logits
        summary["logits_mean"] = logits.detach().float().mean().item()
        summary["logits_std"] = logits.detach().float().std().item()
    summary["repr"] = str(output)
    return summary


def create_dataloader(
    dataset: List[Any],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Iterable[Any]:
    """
    DataLoader that preserves raw samples (dict/str) without stacking.
    Returns a list of samples for each batch to allow manual per-sample handling.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 2
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": lambda x: x,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(SimpleDataset(dataset), **loader_kwargs)


def prepare_target_ids(model: Any, answer: Optional[str], use_gsm8k_parse: bool) -> Optional[torch.Tensor]:
    """Build target ids for teacher-forced scoring using model-provided template if available."""
    if answer is None:
        return None
    target_text = parse_answer(answer)[0] if use_gsm8k_parse else answer
    if hasattr(model, "build_teacher_target_ids"):
        return model.build_teacher_target_ids(target_text)  # type: ignore[attr-defined]
    if not hasattr(model, "tokenizer"):
        return None
    try:
        tokenized = model.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")  # type: ignore[attr-defined]
    except Exception:
        return None
    ids = tokenized.get("input_ids")
    if ids is None:
        return None
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


# -------- Teacher-forcing helpers shared across experiments --------
CACHE_KEYS = {"past_key_values", "past_key_values_latents"}


def strip_caches(output: Any) -> Any:
    """Drop heavy cache entries before summarization/logging."""
    if isinstance(output, dict):
        return {k: v for k, v in output.items() if k not in CACHE_KEYS}
    return output


def _clone_cache(cache: Any) -> Any:
    """
    Defensive copy for kv caches so repeated teacher-forced evaluations do not
    mutate the same object (DynamicCache accumulates sequence length in-place).
    """
    if cache is None:
        return None

    # New HF cache interface (DynamicCache)
    if isinstance(cache, DynamicCache) or hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()  # type: ignore[attr-defined]
        except Exception:
            return cache
        cloned_layers = []
        for layer in legacy:
            if layer is None:
                cloned_layers.append(None)
                continue
            k, v = layer
            cloned_layers.append((k.clone(), v.clone()))
        try:
            return type(cache).from_legacy_cache(tuple(cloned_layers))  # type: ignore[attr-defined]
        except Exception:
            return tuple(cloned_layers)

    # Legacy tuple/list caches
    if isinstance(cache, (list, tuple)):
        cloned_layers = []
        for layer in cache:
            if layer is None:
                cloned_layers.append(None)
                continue
            if isinstance(layer, (list, tuple)) and len(layer) == 2:
                k, v = layer
                cloned_layers.append((k.clone(), v.clone()))
            else:
                try:
                    cloned_layers.append(layer.clone())
                except Exception:
                    cloned_layers.append(torch.clone(layer) if torch.is_tensor(layer) else layer)
        return tuple(cloned_layers) if isinstance(cache, tuple) else cloned_layers

    try:
        return cache.clone()
    except Exception:
        try:
            import copy

            return copy.deepcopy(cache)
        except Exception:
            return cache


def _clone_teacher_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Clone caches in the teacher-forcing state to avoid in-place growth."""
    if state is None:
        return None
    cloned = dict(state)
    for key in ("past_key_values", "past_key_values_latents"):
        if key in cloned:
            cloned[key] = _clone_cache(cloned[key])
    if "first_logit" in cloned and isinstance(cloned["first_logit"], torch.Tensor):
        cloned["first_logit"] = cloned["first_logit"].clone()
    return cloned


def build_teacher_state(output: Any, fallback: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Extract a lightweight state dict usable by model.compute_logits.
    Prefers cache snapshots after latent rollout when available.
    """
    if output is None and fallback is None:
        return None
    source = output if isinstance(output, dict) else {}
    fallback = fallback or {}
    state: Dict[str, Any] = {}

    pkv_latents = source.get("past_key_values_latents") or fallback.get("past_key_values_latents")
    pkv = pkv_latents or source.get("past_key_values") or fallback.get("past_key_values")
    if pkv is None:
        return None
    state["past_key_values"] = _clone_cache(pkv)
    state["_cache_source"] = "past_key_values_latents" if pkv_latents is not None else "past_key_values"

    # Preserve optional helpers when present.
    if "first_logit" in source:
        state["first_logit"] = source["first_logit"]
    elif "first_logit" in fallback:
        state["first_logit"] = fallback["first_logit"]

    for key in ("tokens", "input_ids", "attention_mask", "position_ids"):
        if key in source:
            state[key] = source[key]
        elif key in fallback:
            state[key] = fallback[key]
    return state


def compute_teacher_forced_metrics(
    model: Any,
    target_ids: Optional[torch.Tensor],
    h_t: torch.Tensor,
    h_t_modified: torch.Tensor,
    baseline_state: Optional[Dict[str, Any]],
    ablated_state: Optional[Dict[str, Any]],
) -> Dict[str, float | str]:
    """
    Compute Δ log p(y) and distribution deltas using teacher-forced logits.
    Returns empty dict when unavailable (e.g., missing tokenizer/compute_logits).
    """
    metrics: Dict[str, float | str] = {}
    if target_ids is None or not hasattr(model, "compute_logits"):
        return metrics
    if baseline_state is None or ablated_state is None:
        return metrics
    baseline_state = _clone_teacher_state(baseline_state)
    ablated_state = _clone_teacher_state(ablated_state)
    try:
        logits_base = model.compute_logits(h_t, baseline_state, target_ids)
        logits_ablt = model.compute_logits(h_t_modified, ablated_state, target_ids)
    except Exception as exc:  # pragma: no cover - best effort metrics
        metrics["teacher_forced_error"] = str(exc)
        return metrics

    log_probs_base = F.log_softmax(logits_base, dim=-1)
    log_probs_ablt = F.log_softmax(logits_ablt, dim=-1)
    target_ids_dev = target_ids.to(log_probs_base.device)
    if target_ids_dev.numel() == 0:
        return metrics

    # Per-token log-probs along the gold answer sequence.
    if target_ids_dev.size(1) > 0:
        base_tok_logp = log_probs_base.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
        ablt_tok_logp = log_probs_ablt.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
        seq_delta = (base_tok_logp - ablt_tok_logp).sum(dim=-1).mean()
        metrics["teacher_forced_delta_sum"] = seq_delta.item()

    # Final-token Δ log p(y*): by default target_ids include an eos; use the penultimate token when available.
    seq_len = target_ids_dev.size(1)
    if seq_len > 0:
        time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
        last_targets = target_ids_dev[:, time_index]
        base_last = log_probs_base[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
        ablt_last = log_probs_ablt[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
        metrics["delta_logp_final_token"] = (base_last - ablt_last).mean().item()

    # Distributional shifts on that same token position.
    if seq_len > 0:
        final_pos = seq_len - 2 if seq_len >= 2 else -1
        base_final_lp = log_probs_base[:, final_pos, :]
        ablt_final_lp = log_probs_ablt[:, final_pos, :]
        metrics["kl_final_baseline_to_ablated"] = F.kl_div(
            base_final_lp, ablt_final_lp.exp(), reduction="batchmean"
        ).item()
        metrics["l2_final_prob_dist"] = torch.norm(
            base_final_lp.exp() - ablt_final_lp.exp(), p=2, dim=-1
        ).mean().item()
    return metrics


def compute_teacher_forced_metrics_from_logits(
    logits_base: torch.Tensor,
    logits_ablt: torch.Tensor,
    target_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample teacher-forced deltas from precomputed logits.
    Returns tensors with shape [B].
    """
    metrics: Dict[str, torch.Tensor] = {}
    log_probs_base = F.log_softmax(logits_base, dim=-1)
    log_probs_ablt = F.log_softmax(logits_ablt, dim=-1)
    target_ids_dev = target_ids.to(log_probs_base.device)
    if target_ids_dev.numel() == 0:
        return metrics

    base_tok_logp = log_probs_base.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    ablt_tok_logp = log_probs_ablt.gather(-1, target_ids_dev.unsqueeze(-1)).squeeze(-1)
    metrics["teacher_forced_delta_sum"] = (base_tok_logp - ablt_tok_logp).sum(dim=-1)

    seq_len = target_ids_dev.size(1)
    if seq_len > 0:
        time_index = seq_len - 2 if seq_len >= 2 else seq_len - 1
        last_targets = target_ids_dev[:, time_index]
        base_last = log_probs_base[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
        ablt_last = log_probs_ablt[:, time_index, :].gather(-1, last_targets.unsqueeze(-1)).squeeze(-1)
        metrics["delta_logp_final_token"] = base_last - ablt_last

        base_final_lp = log_probs_base[:, time_index, :]
        ablt_final_lp = log_probs_ablt[:, time_index, :]
        base_final_p = base_final_lp.exp()
        metrics["kl_final_baseline_to_ablated"] = (base_final_p * (base_final_lp - ablt_final_lp)).sum(dim=-1)
        metrics["l2_final_prob_dist"] = torch.norm(base_final_p - ablt_final_lp.exp(), p=2, dim=-1)
    return metrics


def compute_teacher_forced_metrics_batch(
    model: Any,
    target_ids: Optional[torch.Tensor],
    h_t: torch.Tensor,
    h_t_modified: torch.Tensor,
    baseline_state: Optional[Dict[str, Any]],
    ablated_state: Optional[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Batched variant of compute_teacher_forced_metrics.
    Returns per-sample tensors (shape [B]) when available.
    """
    metrics: Dict[str, torch.Tensor] = {}
    if target_ids is None or not hasattr(model, "compute_logits"):
        return metrics
    if baseline_state is None or ablated_state is None:
        return metrics
    baseline_state = _clone_teacher_state(baseline_state)
    ablated_state = _clone_teacher_state(ablated_state)
    try:
        logits_base = model.compute_logits(h_t, baseline_state, target_ids)
        logits_ablt = model.compute_logits(h_t_modified, ablated_state, target_ids)
    except Exception:
        return metrics

    return compute_teacher_forced_metrics_from_logits(logits_base, logits_ablt, target_ids)
