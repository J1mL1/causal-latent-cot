#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import build_dataset, create_dataloader, load_config
from common.model_registry import load_model


def _format_step(text: str, template: str) -> str:
    return template.format(text=text)


def _normalize_steps(raw_steps: Any) -> List[str]:
    if raw_steps is None:
        return []
    if isinstance(raw_steps, list):
        return [str(s) for s in raw_steps if str(s).strip()]
    try:
        import numpy as np  # type: ignore

        if isinstance(raw_steps, np.ndarray):
            return [str(s) for s in raw_steps.tolist() if str(s).strip()]
    except Exception:
        pass
    if isinstance(raw_steps, str):
        parts = [s.strip() for s in raw_steps.split("\n") if s.strip()]
        return parts if parts else [raw_steps.strip()]
    return [str(raw_steps)]


def _join_context(prompt: str, steps: List[str], sep: str) -> str:
    if not steps:
        return prompt
    if prompt.endswith(sep):
        return prompt + sep.join(steps)
    return prompt + sep + sep.join(steps)


def _append_eos(text: str, tokenizer: Any) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if not eos:
        return text
    return text if text.endswith(eos) else text + eos


def _tokenize_target(model: Any, text: str) -> Optional[torch.Tensor]:
    if not hasattr(model, "tokenizer"):
        return None
    try:
        tokenized = model.tokenizer(text, add_special_tokens=False, return_tensors="pt")  # type: ignore[attr-defined]
    except Exception:
        return None
    ids = tokenized.get("input_ids")
    if ids is None:
        return None
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(model.device)


def _get_backend_model(model: Any) -> Any:
    return getattr(model, "base_model", None) or getattr(model, "model", None)


def _generate_text(model: Any, text: str, max_new_tokens: int) -> Optional[str]:
    tokenizer = getattr(model, "tokenizer", None)
    backend_model = _get_backend_model(model)
    if tokenizer is None or backend_model is None:
        return None
    tokens = tokenizer(text, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(backend_model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    with torch.no_grad():
        out = backend_model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def _compute_log_probs_and_hidden(
    model: Any,
    context: str,
    target_text: str,
    debug_token: bool = False,
    debug_label: str = "",
) -> Optional[Dict[str, torch.Tensor]]:
    backend_model = getattr(model, "base_model", None) or getattr(model, "model", None)
    if not hasattr(model, "tokenizer") or backend_model is None:
        return None

    context_ids = _tokenize_target(model, context)
    target_ids = _tokenize_target(model, target_text)
    if target_ids is None or target_ids.numel() == 0:
        return None

    if context_ids is None or context_ids.numel() == 0:
        input_ids = target_ids
        start_idx = 0
    else:
        input_ids = torch.cat([context_ids, target_ids], dim=1)
        start_idx = context_ids.size(1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = backend_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    # Align logits to predict the next token (causal LM shift).
    logits_all = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits_all, dim=-1)
    hidden_all = outputs.hidden_states[-1][:, :-1, :]
    target_ids = target_ids.to(log_probs.device)
    if target_ids.numel() == 0:
        return None

    if start_idx == 0:
        # No context: skip the first target token since there's no preceding token to predict it from.
        if target_ids.size(1) <= 1:
            return None
        target_ids = target_ids[:, 1:]
        start = 0
    else:
        start = start_idx - 1

    end = start + target_ids.size(1)
    log_probs = log_probs[:, start:end, :]
    if log_probs.size(1) != target_ids.size(1):
        return None
    hidden_all = hidden_all[:, start:end, :]

    if debug_token and hasattr(model, "tokenizer"):
        last_idx = -1
        tok_id = int(target_ids[0, last_idx].item())
        tok_text = model.tokenizer.decode([tok_id])  # type: ignore[attr-defined]
        label = f" {debug_label}" if debug_label else ""
        print(f"DEBUG logp_last{label} token_id={tok_id} token={tok_text!r}")
    return {"log_probs": log_probs, "target_ids": target_ids, "hidden": hidden_all}


def _compute_logp_stats(log_probs: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, float]:
    per_tok = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    seq_logp = per_tok.sum(dim=-1).mean().item()
    last_idx = -1
    last_tok = per_tok[:, last_idx].mean().item()
    return {"logp_seq": seq_logp, "logp_last": last_tok}


def _compute_kl_mean(base_logp: torch.Tensor, ablt_logp: torch.Tensor) -> float:
    p_base = base_logp.exp()
    kl_token = (p_base * (base_logp - ablt_logp)).sum(dim=-1)
    return kl_token.mean().item()


def _compute_kl_last_token(base_logp: torch.Tensor, ablt_logp: torch.Tensor) -> float:
    base_last = base_logp[:, -1, :]
    ablt_last = ablt_logp[:, -1, :]
    p_base = base_last.exp()
    kl_last = (p_base * (base_last - ablt_last)).sum(dim=-1)
    return kl_last.mean().item()


def _get_lm_head(backend_model: Any) -> Optional[Any]:
    if hasattr(backend_model, "lm_head"):
        return backend_model.lm_head
    if hasattr(backend_model, "get_output_embeddings"):
        return backend_model.get_output_embeddings()
    return None


def _compute_hidden_kl(
    lm_head: Any,
    base_hidden: torch.Tensor,
    ablt_hidden: torch.Tensor,
) -> Dict[str, float]:
    base_last = base_hidden[:, -1, :]
    ablt_last = ablt_hidden[:, -1, :]
    base_last_logp = F.log_softmax(lm_head(base_last), dim=-1)
    ablt_last_logp = F.log_softmax(lm_head(ablt_last), dim=-1)
    kl_last = _compute_kl_mean(base_last_logp, ablt_last_logp)

    base_mean = base_hidden.mean(dim=1)
    ablt_mean = ablt_hidden.mean(dim=1)
    base_mean_logp = F.log_softmax(lm_head(base_mean), dim=-1)
    ablt_mean_logp = F.log_softmax(lm_head(ablt_mean), dim=-1)
    kl_mean = _compute_kl_mean(base_mean_logp, ablt_mean_logp)

    return {"kl_logit_last_hidden": kl_last, "kl_logit_mean_hidden": kl_mean}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute explicit step causal graph from GSM8k-Aug.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Optional Coconut checkpoint to override config.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_steps", type=int, default=None, help="Truncate explicit steps to first K.")
    parser.add_argument("--step_template", default="{text}", help="Template for explicit steps.")
    parser.add_argument("--answer_template", default="### {text}", help="Template for final answer target.")
    parser.add_argument("--sep", default="\n", help="Separator between prompt and steps.")
    parser.add_argument(
        "--edge_metric",
        default="delta_logp_seq",
        choices=[
            "delta_logp_seq",
            "delta_logp_last",
            "kl_mean",
            "kl_last_token",
            "kl_logit_last_hidden",
            "kl_logit_mean_hidden",
        ],
        help="Edge weight metric for adjacency.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_adj", action="store_true", help="Save averaged adjacency matrix as .npy.")
    parser.add_argument("--debug_samples", type=int, default=0, help="Print target-token debug for first N samples.")
    parser.add_argument("--debug_generate_samples", type=int, default=0, help="Generate text for first N samples.")
    parser.add_argument("--debug_generate_max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_type = str(config.get("model_type", "hf-auto")).lower()
    model_cfg = config.get("model", {})
    if model_type == "coconut":
        raise ValueError("Coconut explicit mode has been removed; use hf-auto.")
    if args.checkpoint_path:
        model_cfg["checkpoint_path"] = args.checkpoint_path
    cfg_device = str(model_cfg.get("device", "")).lower()
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        model_cfg["device"] = "cpu"
    elif not cfg_device:
        model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"] = model_cfg
    model = load_model(model_type, config.get("model", config))
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Model tokenizer not available for explicit causal graph.")
    backend_model = _get_backend_model(model)
    lm_head = _get_lm_head(backend_model) if backend_model is not None else None

    dataset = build_dataset(config, tokenizer=tokenizer)
    if not dataset:
        raise ValueError("Dataset is empty; check dataset_path/name.")
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    max_steps = args.max_steps or config.get("num_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adj_sum: Optional[np.ndarray] = None
    adj_count: Optional[np.ndarray] = None
    ans_sum: Optional[np.ndarray] = None
    ans_count: Optional[np.ndarray] = None

    with output_path.open("w") as writer, torch.no_grad():
        pbar = tqdm(total=len(dataset), desc="samples")
        debug_left = args.debug_samples
        debug_gen_left = args.debug_generate_samples
        sample_counter = 0
        for batch in dataloader:
            batch_samples = batch if isinstance(batch, list) else [batch]
            for sample in batch_samples:
                sample_id = None
                if isinstance(sample, dict):
                    sample_id = sample.get("id") or sample.get("sample_id")
                if sample_id is None:
                    sample_id = sample_counter
                sample_counter += 1

                prompt = sample.get("prompt") if isinstance(sample, dict) else None
                if not prompt:
                    pbar.update(1)
                    continue
                raw_steps = sample.get("steps") if isinstance(sample, dict) else None
                steps = _normalize_steps(raw_steps)
                if max_steps is not None:
                    steps = steps[:max_steps]
                if not steps:
                    pbar.update(1)
                    continue
                answer = sample.get("answer") if isinstance(sample, dict) else None
                if answer is None:
                    pbar.update(1)
                    continue

                formatted_steps = [_format_step(s, args.step_template) for s in steps]
                full_context = _join_context(prompt, formatted_steps, args.sep)
                target_answer = args.sep.join([_format_step(str(answer), args.answer_template)])

                K = len(formatted_steps)
                if adj_sum is None:
                    adj_sum = np.zeros((K, K), dtype=np.float64)
                    adj_count = np.zeros((K, K), dtype=np.int64)
                    ans_sum = np.zeros((K,), dtype=np.float64)
                    ans_count = np.zeros((K,), dtype=np.int64)
                elif adj_sum.shape[0] < K:
                    pad = K - adj_sum.shape[0]
                    adj_sum = np.pad(adj_sum, ((0, pad), (0, pad)), mode="constant")
                    adj_count = np.pad(adj_count, ((0, pad), (0, pad)), mode="constant")
                    ans_sum = np.pad(ans_sum, (0, pad), mode="constant")
                    ans_count = np.pad(ans_count, (0, pad), mode="constant")

                if debug_gen_left > 0:
                    print(
                        f"DEBUG sample_id={sample_id} "
                        f"prompt={prompt!r}"
                    )
                    for i in range(K):
                        base_context = _join_context(prompt, formatted_steps[: i + 1], args.sep)
                        ablt_context = _join_context(prompt, formatted_steps[:i], args.sep)
                        base_gen = _generate_text(model, base_context, args.debug_generate_max_new_tokens)
                        ablt_gen = _generate_text(model, ablt_context, args.debug_generate_max_new_tokens)
                        print(f"DEBUG generate i={i+1} base_context -> {base_gen!r}")
                        print(f"DEBUG generate i={i+1} ablt_context -> {ablt_gen!r}")

                for i in range(K):
                    # Step i -> Answer: target is steps after i plus answer.
                    target_parts = formatted_steps[i + 1 :] + [_format_step(str(answer), args.answer_template)]
                    if not target_parts:
                        continue
                    target_text = args.sep.join(target_parts)
                    if debug_left > 0:
                        target_ids = _tokenize_target(model, target_text)
                        print(
                            f"DEBUG sample_id={sample_id} "
                            f"i={i+1} -> answer target_len={target_ids.size(1) if target_ids is not None else None} "
                            f"target_text={target_text!r}"
                        )
                    base_context = _join_context(prompt, formatted_steps[: i + 1], args.sep)
                    ablt_context = _join_context(prompt, formatted_steps[:i], args.sep)
                    base_pack = _compute_log_probs_and_hidden(
                        model,
                        base_context,
                        target_text,
                        debug_token=debug_left > 0,
                        debug_label="base_answer",
                    )
                    ablt_pack = _compute_log_probs_and_hidden(
                        model,
                        ablt_context,
                        target_text,
                        debug_token=debug_left > 0,
                        debug_label="ablt_answer",
                    )
                    if base_pack and ablt_pack:
                        base_logp = base_pack["log_probs"]
                        base_ids = base_pack["target_ids"]
                        base_hidden = base_pack["hidden"]
                        ablt_logp = ablt_pack["log_probs"]
                        ablt_ids = ablt_pack["target_ids"]
                        ablt_hidden = ablt_pack["hidden"]
                        if base_ids.size(1) != ablt_ids.size(1):
                            continue
                        base_stats = _compute_logp_stats(base_logp, base_ids)
                        ablt_stats = _compute_logp_stats(ablt_logp, ablt_ids)
                        delta_seq = base_stats["logp_seq"] - ablt_stats["logp_seq"]
                        delta_last = base_stats["logp_last"] - ablt_stats["logp_last"]
                        kl_mean = _compute_kl_mean(base_logp, ablt_logp)
                        kl_last_token = _compute_kl_last_token(base_logp, ablt_logp)
                        hidden_kl = (
                            _compute_hidden_kl(lm_head, base_hidden, ablt_hidden)
                            if lm_head is not None
                            else {}
                        )
                        record = {
                            "sample_id": sample_id,
                            "step_i": i + 1,
                            "step_j": "Y",
                            "delta_logp_seq": delta_seq,
                            "delta_logp_last": delta_last,
                            "base_logp_seq": base_stats["logp_seq"],
                            "ablt_logp_seq": ablt_stats["logp_seq"],
                            "kl_mean": kl_mean,
                            "kl_last_token": kl_last_token,
                        }
                        record.update(hidden_kl)
                        writer.write(json.dumps(record) + "\n")
                        if ans_sum is not None and ans_count is not None:
                            if args.edge_metric == "kl_mean":
                                ans_sum[i] += kl_mean
                            elif args.edge_metric == "kl_last_token":
                                ans_sum[i] += kl_last_token
                            elif args.edge_metric == "delta_logp_last":
                                ans_sum[i] += delta_last
                            elif args.edge_metric == "kl_logit_last_hidden":
                                if "kl_logit_last_hidden" in hidden_kl:
                                    ans_sum[i] += hidden_kl["kl_logit_last_hidden"]
                                else:
                                    continue
                            elif args.edge_metric == "kl_logit_mean_hidden":
                                if "kl_logit_mean_hidden" in hidden_kl:
                                    ans_sum[i] += hidden_kl["kl_logit_mean_hidden"]
                                else:
                                    continue
                            else:
                                ans_sum[i] += delta_seq
                            ans_count[i] += 1

                # Step i -> Step j (i < j)
                for j in range(1, K):
                    for i in range(j):
                        # Step i -> Step j: target is steps i+1..j (inclusive).
                        target_parts = formatted_steps[i + 1 : j + 1]
                        if not target_parts:
                            continue
                        target_text = args.sep.join(target_parts)
                        if debug_left > 0:
                            target_ids = _tokenize_target(model, target_text)
                            print(
                                f"DEBUG sample_id={sample_id} "
                                f"i={i+1} -> j={j+1} target_len={target_ids.size(1) if target_ids is not None else None} "
                                f"target_text={target_text!r}"
                            )
                        base_context = _join_context(prompt, formatted_steps[: i + 1], args.sep)
                        ablt_context = _join_context(prompt, formatted_steps[:i], args.sep)
                        base_pack = _compute_log_probs_and_hidden(
                            model,
                            base_context,
                            target_text,
                            debug_token=debug_left > 0,
                            debug_label="base_step",
                        )
                        if base_pack is None:
                            continue
                        ablt_pack = _compute_log_probs_and_hidden(
                            model,
                            ablt_context,
                            target_text,
                            debug_token=debug_left > 0,
                            debug_label="ablt_step",
                        )
                        if ablt_pack and adj_sum is not None and adj_count is not None:
                            base_logp = base_pack["log_probs"]
                            base_ids = base_pack["target_ids"]
                            base_hidden = base_pack["hidden"]
                            ablt_logp = ablt_pack["log_probs"]
                            ablt_ids = ablt_pack["target_ids"]
                            ablt_hidden = ablt_pack["hidden"]
                            if base_ids.size(1) != ablt_ids.size(1):
                                continue
                            base_stats = _compute_logp_stats(base_logp, base_ids)
                            ablt_stats = _compute_logp_stats(ablt_logp, ablt_ids)
                            delta_seq = base_stats["logp_seq"] - ablt_stats["logp_seq"]
                            delta_last = base_stats["logp_last"] - ablt_stats["logp_last"]
                            kl_mean = _compute_kl_mean(base_logp, ablt_logp)
                            kl_last_token = _compute_kl_last_token(base_logp, ablt_logp)
                            hidden_kl = (
                                _compute_hidden_kl(lm_head, base_hidden, ablt_hidden)
                                if lm_head is not None
                                else {}
                            )
                            if args.edge_metric == "kl_mean":
                                adj_sum[i, j] += kl_mean
                            elif args.edge_metric == "kl_last_token":
                                adj_sum[i, j] += kl_last_token
                            elif args.edge_metric == "delta_logp_last":
                                adj_sum[i, j] += delta_last
                            elif args.edge_metric == "kl_logit_last_hidden":
                                if "kl_logit_last_hidden" in hidden_kl:
                                    adj_sum[i, j] += hidden_kl["kl_logit_last_hidden"]
                                else:
                                    continue
                            elif args.edge_metric == "kl_logit_mean_hidden":
                                if "kl_logit_mean_hidden" in hidden_kl:
                                    adj_sum[i, j] += hidden_kl["kl_logit_mean_hidden"]
                                else:
                                    continue
                            else:
                                adj_sum[i, j] += delta_seq
                            adj_count[i, j] += 1
                            record = {
                                "sample_id": sample_id,
                                "step_i": i + 1,
                                "step_j": j + 1,
                                "delta_logp_seq": delta_seq,
                                "delta_logp_last": delta_last,
                                "base_logp_seq": base_stats["logp_seq"],
                                "ablt_logp_seq": ablt_stats["logp_seq"],
                                "kl_mean": kl_mean,
                                "kl_last_token": kl_last_token,
                            }
                            record.update(hidden_kl)
                            writer.write(json.dumps(record) + "\n")
                if debug_left > 0:
                    debug_left -= 1
                if debug_gen_left > 0:
                    debug_gen_left -= 1
            pbar.update(len(batch_samples))

    if args.save_adj and adj_sum is not None and adj_count is not None:
        adj = np.divide(adj_sum, adj_count, out=np.zeros_like(adj_sum), where=adj_count > 0)
        out_adj = output_path.with_suffix(".adj.npy")
        np.save(out_adj, adj)
        if ans_sum is not None and ans_count is not None:
            ans = np.divide(ans_sum, ans_count, out=np.zeros_like(ans_sum), where=ans_count > 0)
            out_ans = output_path.with_suffix(".ans.npy")
            np.save(out_ans, ans)


if __name__ == "__main__":
    main()
