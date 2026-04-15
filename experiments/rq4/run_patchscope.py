"""
RQ4: Step-level Patchscope-style decoding — inject source step latent h_t into a
fixed target prompt (last transformer block), then greedy-generate text.

Default target is **PAIR-style ``a -> b`` lines** but with **phrase** pairs (not
``cat -> cat`` token identity), then ``?`` (see ``DEFAULT_PHRASE_TARGET_TEMPLATE``).
Optionally add ``Question: {question}`` in YAML for per-sample prompts. Use
``DEFAULT_PAIR_TARGET_TEMPLATE`` for exact PAIR reproduction.

See common/analysis/patchscope.py and Patchscopes (Ghandeharioun et al., ICML 2024).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as torch_dist
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.analysis.patchscope import (
    DEFAULT_PATCHSCOPE_TARGET_TEMPLATE,
    find_placeholder_token_index,
    get_causal_lm_backbone,
    greedy_generate_after_patch,
    greedy_generate_baseline_no_patch,
)
from common.experiment_utils import build_dataset, create_dataloader, load_config, parse_step_tokens
from common.model_registry import load_model
from data.gsm8k import (
    count_golden_cot_lines,
    extract_golden_cot_at_step,
    parse_answer,
)


def _parse_steps(steps_arg: Optional[str], config: Dict[str, Any]) -> List[int]:
    raw = steps_arg if steps_arg is not None else config.get("steps")
    if isinstance(raw, list):
        raw = ",".join(str(x) for x in raw)
    elif raw is not None:
        raw = str(raw)
    steps = parse_step_tokens(raw, config.get("num_steps"))
    out: List[int] = []
    for s in steps:
        if isinstance(s, int):
            out.append(s)
        else:
            raise ValueError(f"Patchscope RQ4 expects numeric steps only, got {s!r}")
    return out


def _forward_h_stack(model: Any, prompts: List[Any], step: int) -> torch.Tensor:
    """Return h_t ``[B, H]``. Try batched ``forward_until_step`` first, then per-prompt."""
    bsz = len(prompts)
    try:
        h_t, _ = model.forward_until_step(prompts, step)
        if (
            isinstance(h_t, torch.Tensor)
            and h_t.dim() >= 2
            and h_t.size(0) == bsz
        ):
            return h_t
    except Exception:
        pass
    parts: List[torch.Tensor] = []
    for p in prompts:
        h_i, _ = model.forward_until_step(p, step)
        parts.append(h_i)
    return torch.cat(parts, dim=0)


def _gsm8k_gold_static(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Question + numeric gold answer (no full CoT; use golden_cot_step per row)."""
    raw_answer = sample.get("answer")
    out: Dict[str, Any] = {
        "question": sample.get("question"),
        "gold_answer": sample.get("answer_clean") if sample.get("answer_clean") is not None else None,
    }
    if out["gold_answer"] is None and raw_answer is not None:
        text, _val = parse_answer(str(raw_answer))
        out["gold_answer"] = text if text else None
    return out


def _target_template_uses_question(template: str) -> bool:
    return "{question}" in template


def _format_patchscope_target(template: str, sample: Dict[str, Any]) -> str:
    """Replace ``{question}`` with the dataset question (use ``.replace`` to avoid ``str.format`` on user braces)."""
    if not _target_template_uses_question(template):
        return template
    q = sample.get("question")
    if q is None or not str(q).strip():
        raise ValueError(
            "patchscope.target_template contains {question} but sample has no non-empty 'question' field."
        )
    return template.replace("{question}", str(q))


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ4 Patchscope step-level decoding.")
    parser.add_argument("--model_name", required=True, help="Registry name, e.g. coconut, codi.")
    parser.add_argument("--config_path", required=True, help="YAML/JSON config.")
    parser.add_argument("--output_path", required=True, help="JSONL output path.")
    parser.add_argument("--steps", default=None, help="Comma-separated steps (default: config).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="DataLoader batch size; forward_until_step + patch decode run on the same batch.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Truncate dataset for smoke tests.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_samples", type=int, default=4)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--patchscope_target_template",
        default=None,
        help="Override config patchscope.target_template (multiline). Prefer --patchscope_target_template_file from shell.",
    )
    parser.add_argument(
        "--patchscope_target_template_file",
        default=None,
        help="UTF-8 file whose contents override patchscope.target_template (wins over --patchscope_target_template).",
    )
    parser.add_argument(
        "--patchscope_placeholder",
        default=None,
        help="Override patchscope.placeholder (e.g. ?).",
    )
    parser.add_argument(
        "--prompt_label",
        default=None,
        help="Short label for this run (written to .meta.json as prompt_label; use in filenames when sweeping prompts).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

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
        dist.init_process_group(backend="nccl", init_method="env://")

    config = load_config(args.config_path)
    ps_override: Dict[str, Any] = {}
    if args.patchscope_target_template_file:
        tpath = Path(args.patchscope_target_template_file)
        ps_override["target_template"] = tpath.read_text(encoding="utf-8")
    elif args.patchscope_target_template is not None:
        ps_override["target_template"] = args.patchscope_target_template
    if args.patchscope_placeholder is not None:
        ps_override["placeholder"] = args.patchscope_placeholder
    if ps_override:
        config.setdefault("patchscope", {}).update(ps_override)

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
    if tokenizer is None:
        raise RuntimeError("Model has no tokenizer.")

    dataset_full = build_dataset(config, tokenizer=tokenizer)
    if args.dry_run:
        dataset_full = dataset_full[: args.dry_run_samples]
    if args.max_samples is not None:
        dataset_full = dataset_full[: args.max_samples]
    if not dataset_full:
        raise ValueError("Empty dataset.")

    if dist is not None and world_size > 1:
        dataset_full = dataset_full[rank::world_size]

    ps_cfg = config.get("patchscope") or {}
    target_template = ps_cfg.get("target_template", DEFAULT_PATCHSCOPE_TARGET_TEMPLATE)
    placeholder = ps_cfg.get("placeholder", "?")
    max_new_tokens = int(ps_cfg.get("max_new_tokens", 48))
    include_baseline = bool(ps_cfg.get("include_baseline", True))
    gold_cot_split = str(ps_cfg.get("gold_cot_split", "line"))
    skip_final_ln = bool(ps_cfg.get("skip_final_ln", True))

    uses_question = _target_template_uses_question(target_template)
    # Fixed prompt + single patch_pos when template has no {question}; else per-sample below.
    target_text_fixed: Optional[str] = None
    patch_pos_fixed: Optional[int] = None
    if not uses_question:
        target_text_fixed = target_template
        patch_pos_fixed = find_placeholder_token_index(
            target_text_fixed, tokenizer, placeholder
        )

    steps = _parse_steps(args.steps, config)
    causal_lm = get_causal_lm_backbone(model)
    causal_lm.eval()
    device = next(causal_lm.parameters()).device

    dataloader = create_dataloader(
        dataset_full,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = output_path if dist is None else output_path.with_suffix(
        output_path.suffix + f".rank{rank}"
    )
    meta_path = output_path.parent / f"{output_path.stem}.meta.json"

    baseline_once: Optional[str] = None
    baseline_err: Optional[str] = None
    if include_baseline and rank == 0 and not uses_question:
        assert target_text_fixed is not None and patch_pos_fixed is not None
        try:
            baseline_once = greedy_generate_baseline_no_patch(
                causal_lm,
                tokenizer,
                target_text_fixed,
                patch_pos_fixed,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        except Exception as exc:
            baseline_err = str(exc)
    elif include_baseline and uses_question and rank == 0:
        baseline_err = (
            "skipped: target_template contains {question}; baseline is not a single global prompt."
        )

    if rank == 0:
        meta_payload: Dict[str, Any] = {
            "record_type": "patchscope_meta",
            "prompt_label": args.prompt_label,
            "baseline_patchscope_text": baseline_once,
            "baseline_error": baseline_err,
            "target_template": target_template,
            "target_uses_question": uses_question,
            "target_prompt": (
                target_text_fixed if not uses_question else "(resolved per row from target_template + question)"
            ),
            "placeholder": placeholder,
            "patch_pos": patch_pos_fixed if not uses_question else "(per row; see each JSONL record)",
            "skip_final_ln": skip_final_ln,
            "patch_injection": (
                "final_norm" if skip_final_ln else "last_block_pre_norm"
            ),
            "max_new_tokens": max_new_tokens,
            "include_baseline": include_baseline,
            "model_name": args.model_name,
            "config_path": str(args.config_path),
            "output_path": str(output_path),
            "batch_size": args.batch_size,
            "note": (
                "When target_template has no {question}, baseline is one unpatched greedy on that fixed prompt. "
                "When {question} is present, baseline_patchscope_text is omitted (per-sample prompts)."
            ),
            "jsonl_fields": {
                "prompt_label": "optional sweep label when using --prompt_label",
                "question": "GSM8K question text",
                "target_prompt": "resolved patchscope prompt (add Question: {question} in YAML for per-row)",
                "patch_pos": "token index of placeholder (per row only if template uses {question})",
                "gold_answer": "parsed final numeric answer (answer_clean when available)",
                "golden_cot_step": "official CoT line aligned to latent step (1-based line index before ####; heuristic)",
                "golden_cot_num_lines": "count of non-empty CoT lines (before ####) for this problem",
                "gold_cot_split": gold_cot_split,
            },
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample_index = 0
    with rank_output_path.open("w") as writer:
        pbar = tqdm(dataloader, desc="patchscope", disable=rank != 0)
        for batch in pbar:
            batch_samples = batch if isinstance(batch, list) else [batch]
            prompts = [s["prompt"] if isinstance(s, dict) else s for s in batch_samples]
            bsz = len(batch_samples)

            for step in steps:
                gold_blocks: List[Dict[str, Any]] = []
                for sample in batch_samples:
                    gold_static = _gsm8k_gold_static(sample) if isinstance(sample, dict) else {
                        "question": None,
                        "gold_answer": None,
                    }
                    raw_answer = sample.get("answer") if isinstance(sample, dict) else None
                    cot_num_lines = count_golden_cot_lines(raw_answer)
                    cot_step = extract_golden_cot_at_step(
                        raw_answer, int(step), split=gold_cot_split
                    )
                    gold_blocks.append(
                        {
                            **gold_static,
                            "golden_cot_step": cot_step,
                            "golden_cot_num_lines": cot_num_lines,
                        }
                    )

                try:
                    with torch.no_grad():
                        h_t = _forward_h_stack(model, prompts, step)
                except Exception as exc:
                    for i, sample in enumerate(batch_samples):
                        sample_id_field = sample.get("id") if isinstance(sample, dict) else None
                        record = {
                            "sample_id": sample_index + i,
                            "sample_uid": sample_id_field,
                            "step": step,
                            "prompt_label": args.prompt_label,
                            "error": str(exc),
                            "patchscope_text": None,
                            **gold_blocks[i],
                        }
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                if not isinstance(h_t, torch.Tensor) or h_t.size(0) != bsz:
                    msg = f"expected h_t [{bsz}, H], got {getattr(h_t, 'shape', type(h_t))}"
                    for i, sample in enumerate(batch_samples):
                        sample_id_field = sample.get("id") if isinstance(sample, dict) else None
                        record = {
                            "sample_id": sample_index + i,
                            "sample_uid": sample_id_field,
                            "step": step,
                            "prompt_label": args.prompt_label,
                            "error": msg,
                            "patchscope_text": None,
                            **gold_blocks[i],
                        }
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                row_target_texts: List[str] = []
                row_patch_positions: List[int] = []
                row_patchscope: List[Optional[str]] = []
                row_patch_err: List[Optional[str]] = []

                if uses_question:
                    for i in range(bsz):
                        try:
                            txt_i = _format_patchscope_target(
                                target_template, batch_samples[i]
                            )
                            pp_i = find_placeholder_token_index(
                                txt_i, tokenizer, placeholder
                            )
                            with torch.no_grad():
                                out_i = greedy_generate_after_patch(
                                    causal_lm,
                                    tokenizer,
                                    txt_i,
                                    pp_i,
                                    h_t[i : i + 1].detach(),
                                    max_new_tokens=max_new_tokens,
                                    device=device,
                                    skip_final_ln=skip_final_ln,
                                )
                            row_target_texts.append(txt_i)
                            row_patch_positions.append(pp_i)
                            row_patchscope.append(out_i[0])
                            row_patch_err.append(None)
                        except Exception as exc:
                            row_patchscope.append(None)
                            row_patch_err.append(str(exc))
                            try:
                                txt_i = _format_patchscope_target(
                                    target_template, batch_samples[i]
                                )
                                pp_i = find_placeholder_token_index(
                                    txt_i, tokenizer, placeholder
                                )
                                row_target_texts.append(txt_i)
                                row_patch_positions.append(pp_i)
                            except Exception:
                                row_target_texts.append("")
                                row_patch_positions.append(-1)
                else:
                    assert target_text_fixed is not None and patch_pos_fixed is not None
                    try:
                        with torch.no_grad():
                            batched = greedy_generate_after_patch(
                                causal_lm,
                                tokenizer,
                                target_text_fixed,
                                patch_pos_fixed,
                                h_t.detach(),
                                max_new_tokens=max_new_tokens,
                                device=device,
                                skip_final_ln=skip_final_ln,
                            )
                        row_patch_err = [None] * bsz
                        row_patchscope = list(batched)
                    except Exception as exc:
                        row_patchscope = [None] * bsz
                        row_patch_err = [str(exc)] * bsz
                    row_target_texts = [target_text_fixed] * bsz
                    row_patch_positions = [patch_pos_fixed] * bsz

                for i in range(bsz):
                    sample = batch_samples[i]
                    sample_id_field = sample.get("id") if isinstance(sample, dict) else None
                    record = {
                        "sample_id": sample_index + i,
                        "sample_uid": sample_id_field,
                        "rank": rank,
                        "step": step,
                        "prompt_label": args.prompt_label,
                        "target_prompt": row_target_texts[i],
                        "placeholder": placeholder,
                        "patch_pos": row_patch_positions[i],
                        "skip_final_ln": skip_final_ln,
                        "max_new_tokens": max_new_tokens,
                        "patchscope_text": row_patchscope[i],
                        "patch_error": row_patch_err[i],
                        **gold_blocks[i],
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if args.dry_run and rank == 0 and i == 0:
                        print(json.dumps(record, ensure_ascii=False))

            sample_index += bsz

    if dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
