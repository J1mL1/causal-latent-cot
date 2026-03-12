from __future__ import annotations

"""
Quick CODI inference demo that uses the latent-iteration path (same as run_baseline).

Usage:
  python scripts/test_codi_infer.py --config_path configs/rq1/codi/gpt2-gsm8k.yaml --question "If you have 2 apples and eat one, how many left?"

Or pick a question from a JSONL:
  python scripts/test_codi_infer.py --config_path ... --jsonl data/gsm8k_local.jsonl --idx 0
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
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 20,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Greedy step-by-step decode after priming on prompt_ids.
    """
    prompt_ids = prompt_ids.to(device)
    trace: list[dict[str, Any]] = []
    with torch.no_grad():
        out = base_model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
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
            next_ids = torch.tensor([[next_id]], device=device)
            out = base_model(input_ids=next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return trace


def stepwise_decode_from_latent(
    model: Any,
    tokens: Dict[str, torch.Tensor],
    past_key_values: Any,
    max_new_tokens: int = 20,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Stepwise greedy decode starting from a cached state after latent iterations.
    Mirrors the CODI rollout decode loop.
    """
    trace: list[dict[str, Any]] = []
    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    training_args = model.model.training_args  # type: ignore[attr-defined]

    # Build EOT embedding (same as rollout).
    if getattr(training_args, "remove_eos", False):
        eot_ids = torch.tensor([model.model.eot_id], device=model.device)  # type: ignore[attr-defined]
    else:
        eot_ids = torch.tensor(
            [model.model.eot_id, tokenizer.eos_token_id], device=model.device  # type: ignore[attr-defined]
        )
    eot_emb = model.model.get_embd(model.model.codi, model.model.model_name)(eot_ids).unsqueeze(0)  # type: ignore[attr-defined]
    eot_emb = eot_emb.expand(tokens["input_ids"].size(0), -1, -1)

    output = eot_emb
    finished = torch.zeros(tokens["input_ids"].size(0), dtype=torch.bool, device=model.device)
    past_key_values = past_key_values

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model.model.codi(  # type: ignore[attr-defined]
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.model.codi.config.vocab_size - 1]  # type: ignore[attr-defined]

            probs = torch.softmax(logits, dim=-1)
            next_token_ids = torch.argmax(logits, dim=-1)
            if next_token_ids.dim() == 0:
                next_token_ids = next_token_ids.view(1)

            top_p, top_i = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            step_entry = {
                "token_id": int(next_token_ids[0].item()),
                "token": tokenizer.decode([int(next_token_ids[0].item())]),
                "prob": float(probs[0, next_token_ids[0]].item()),
                "top_tokens": [
                    (tokenizer.decode([int(i.item())]), float(p.item()))
                    for i, p in zip(top_i[0], top_p[0])
                ],
            }
            trace.append(step_entry)

            for b in range(len(next_token_ids)):
                if not finished[b] and next_token_ids[b] == tokenizer.eos_token_id:
                    finished[b] = True
            if finished.all():
                break

            output = model.model.get_embd(model.model.codi, model.model.model_name)(next_token_ids).unsqueeze(1).to(model.device)  # type: ignore[attr-defined]
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to CODI config (YAML/JSON).")
    parser.add_argument("--question", default=None, help="Direct question string.")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL to pick question from.")
    parser.add_argument("--idx", type=int, default=0, help="Index into JSONL when using --jsonl.")
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = config.get("model", config)
    prompt_template = config.get("prompt_template", "Question: {question}\nAnswer:")
    question = get_question(args)
    prompt = prompt_template.format(question=question)

    model = load_model("codi", model_cfg)
    # Baseline decode with latent iterations
    baseline_outputs = model.run_baseline(prompt)
    # Rollout via forward_until_step -> rollout_from_step to surface caches/logits if present
    with torch.no_grad():
        h_t, state = model.forward_until_step(prompt, step=1)
        rollout_outputs = model.rollout_from_step(h_t, state)

    print("Question:\n", question)
    print("\nPrompt:\n", prompt)
    print("\nRollout (forward_until_step + rollout_from_step):")
    roll_texts = rollout_outputs["text"] if isinstance(rollout_outputs, dict) and "text" in rollout_outputs else rollout_outputs
    if isinstance(roll_texts, list):
        for i, out in enumerate(roll_texts):
            print(f"[{i}] {out}")
    else:
        print(roll_texts)

    print("\nBaseline (run_baseline):")
    if isinstance(baseline_outputs, list):
        for i, out in enumerate(baseline_outputs):
            print(f"[{i}] {out}")
    else:
        print(baseline_outputs)

    # Stepwise greedy decode from prompt (no latent iterations)
    if hasattr(model, "tokenizer") and getattr(model, "model", None) is not None:
        tokens = model.tokenizer(prompt, return_tensors="pt")  # type: ignore[attr-defined]
        base = getattr(model.model, "codi", None)
        if base is not None:
            trace = stepwise_decode(base, model.tokenizer, tokens["input_ids"], model.device)  # type: ignore[arg-type,attr-defined]
            print("\nStepwise greedy decode from prompt (no latent iterations):")
            for idx, step in enumerate(trace):
                tops = ", ".join([f"{tok!r}:{p:.3f}" for tok, p in step["top_tokens"]])
                print(f"  {idx}: {step['token']!r} (p={step['prob']:.3f}) | topk: {tops}")

    # Stepwise decode starting after latent iterations (uses rollout cache)
    if isinstance(rollout_outputs, dict) and "past_key_values_latents" in rollout_outputs:
        pkv = rollout_outputs.get("past_key_values_latents")
        lat_trace = stepwise_decode_from_latent(model, state["tokens"], pkv)  # type: ignore[arg-type]
        print("\nStepwise greedy decode after latent iterations (uses rollout cache):")
        for idx, step in enumerate(lat_trace):
            tops = ", ".join([f"{tok!r}:{p:.3f}" for tok, p in step["top_tokens"]])
            print(f"  {idx}: {step['token']!r} (p={step['prob']:.3f}) | topk: {tops}")


if __name__ == "__main__":
    main()
