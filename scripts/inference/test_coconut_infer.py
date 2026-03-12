from __future__ import annotations

"""
Quick Coconut inference demo (mirrors run_baseline).

Usage:
  python scripts/inference/test_coconut_infer.py --config_path configs/rq1/coconut/deepseek-r1-qwen1.5b-gsm8k.yaml --question "If you have 2 apples and eat one, how many left?"

Or pick a question from a JSONL:
  python scripts/inference/test_coconut_infer.py --config_path configs/rq1/coconut/deepseek-r1-qwen1.5b-gsm8k.yaml --jsonl data/gsm8k_local.jsonl --idx 0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from common.model_registry import load_model


def load_config(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text()
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise
        return yaml.safe_load(text)


def get_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == args.idx:
                    sample = json.loads(line)
                    return sample.get("question") or sample.get("prompt") or line
        raise IndexError(f"Index {args.idx} not found in {args.jsonl}")
    raise ValueError("Provide --question or --jsonl with --idx.")


def stepwise_decode(
    base_model: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 20,
    top_k: int = 10,
    prompt_ids: torch.Tensor | None = None,
    past_cache: Any | None = None,
    initial_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """
    Greedy step-by-step decode: prints each generated token with prob and top-k.
    If past_cache/initial_logits are provided, continues from that state; otherwise primes
    with the provided prompt_ids.
    """
    trace: list[dict[str, Any]] = []
    with torch.no_grad():
        if past_cache is None or initial_logits is None:
            if prompt_ids is None:
                raise ValueError("prompt_ids required when past_cache is not provided.")
            prompt_ids = prompt_ids.to(device)
            out = base_model(input_ids=prompt_ids, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        else:
            past = past_cache
            logits = initial_logits.to(device)
        for _ in range(max_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.argmax(probs, dim=-1).item())
            top_p, top_i = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            step_entry = {
                "token_id": next_id,
                "token": tokenizer.decode([next_id]),
                "prob": float(probs[0, next_id].item()),
                "top_tokens": [
                    (tokenizer.decode([int(i.item())]), float(p.item()))
                    for i, p in zip(top_i[0], top_p[0])
                ],
            }
            trace.append(step_entry)
            if next_id == getattr(tokenizer, "eos_token_id", None):
                break
            # Next step: feed the chosen token
            next_ids = torch.tensor([[next_id]], device=device)
            out = base_model(input_ids=next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to Coconut config (YAML/JSON).")
    parser.add_argument("--question", default=None, help="Direct question string.")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL to pick question from.")
    parser.add_argument("--idx", type=int, default=0, help="Index into JSONL when using --jsonl.")
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = config.get("model", config)
    prompt_template = config.get("prompt_template", "Question: {question}\nAnswer:")
    question = get_question(args)
    prompt = prompt_template.format(question=question)

    model = load_model("coconut", model_cfg)
    # Use forward_until_step + rollout_from_step to get logits on the final decode.
    with torch.no_grad():
        h_t, state = model.forward_until_step(prompt, step=1)
        outputs = model.rollout_from_step(h_t, state)

    # Full Coconut baseline decode (with latent iterations + KV cache updates).
    baseline_outputs = model.run_baseline(prompt)

    # Plain autoregressive decode from the prompt for comparison.
    ar_text = None
    ar_cont = None
    if hasattr(model, "tokenizer") and getattr(model, "base_model", None) is not None:
        tokens = model.tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore[attr-defined]
        gen_ids = model.base_model.generate(  # type: ignore[attr-defined]
            **tokens, max_new_tokens=32, do_sample=False
        )
        decoded_full = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)  # type: ignore[attr-defined]
        prompt_len = tokens["input_ids"].size(1)
        ar_cont = model.tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)  # type: ignore[attr-defined]
        ar_text = decoded_full

    print("Question:\n", question)
    print("\nPrompt:\n", prompt)
    print("\nOutputs:")
    if isinstance(outputs, dict) and "text" in outputs:
        outputs_list = outputs["text"]
    else:
        outputs_list = outputs if isinstance(outputs, list) else [str(outputs)]
    for i, out in enumerate(outputs_list):
        print(f"[{i}] {out}")
        if isinstance(out, str):
            import re

            nums = re.findall(r"-?\\d+\\.?\\d*", out.replace(",", ""))
            if nums:
                print(f"  -> Parsed answer: {nums[-1]}")

    # Coconut baseline decode (with latent path)
    print("\nBaseline (run_baseline):")
    if isinstance(baseline_outputs, dict) and "text" in baseline_outputs:
        for i, out in enumerate(baseline_outputs["text"]):
            print(f"[{i}] {out}")
    elif isinstance(baseline_outputs, list):
        for i, out in enumerate(baseline_outputs):
            print(f"[{i}] {out}")
    else:
        print(baseline_outputs)

    if ar_text:
        print("\nAutoregressive decode from prompt (greedy):")
        print(ar_text)
        print(f"\nContinuation only: {ar_cont}")

    # Step-by-step greedy decode after prompt to show per-token choices.
    if getattr(model, "base_model", None) is not None and hasattr(model, "tokenizer"):
        tokens = model.tokenizer(prompt, return_tensors="pt")  # type: ignore[attr-defined]
        past_cache = None
        init_logits = None
        if isinstance(outputs, dict):
            cache_key = "kv_cache_latents" if "kv_cache_latents" in outputs else "kv_cache"
            if cache_key in outputs:
                ensure_fn = getattr(model, "_ensure_cache", None)  # type: ignore[attr-defined]
                past_cache = ensure_fn(outputs[cache_key]) if ensure_fn else outputs[cache_key]
            if isinstance(outputs.get("logits"), torch.Tensor):
                init_logits = outputs["logits"][:, -1, :]
        trace = stepwise_decode(
            model.base_model,
            model.tokenizer,
            model.device,
            prompt_ids=tokens["input_ids"],
            past_cache=past_cache,
            initial_logits=init_logits,
            top_k=10,
        )  # type: ignore[arg-type]
        print("\nStepwise greedy decode (token, prob, top-10) until eos/max steps:")
        for idx, step in enumerate(trace):
            tops = ", ".join([f"{tok!r}:{p:.3f}" for tok, p in step["top_tokens"]])
            print(f"  {idx}: {step['token']!r} (p={step['prob']:.3f}) | topk: {tops}")

    # If logits are available, inspect last-step distribution and top tokens.
    logits = None
    if isinstance(outputs, dict) and isinstance(outputs.get("logits"), torch.Tensor):
        logits = outputs["logits"]
    if logits is not None:
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        top_k = min(10, probs.size(-1))
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
        print("\nLast-step top tokens:")
        for b in range(top_ids.size(0)):
            decoded = [model.tokenizer.decode([tid]) for tid in top_ids[b].tolist()]  # type: ignore[attr-defined]
            probs_list = top_probs[b].tolist()
            print(f"  batch {b}:")
            for tok, p in zip(decoded, probs_list):
                print(f"    {tok!r}: {p:.4f}")


if __name__ == "__main__":
    main()
