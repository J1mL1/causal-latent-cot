"""
Step-level Patchscope-style inspection: patch a source step latent h_t into a
target prompt, then greedy-decode.

When ``skip_final_ln`` is True (PAIR reference behavior for last-layer patches),
the vector is written at the **output** of the backbone final norm (``model.norm`` /
``ln_f``), matching ``patchscopes_utils.set_hs_patch_hooks_*(..., skip_final_ln=True)``.

When False, the vector is written after the **last transformer block** (pre-norm),
then the model applies final norm + LM head as usual.

Layer is fixed to the last stack block; not swept. Same ``hidden_states[-1]`` space
as RQ1 ``h_t`` (pre-norm residual).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn


def get_causal_lm_backbone(model: Any) -> nn.Module:
    """Return the HF CausalLM used for Patchscope target forward (Coconut or CODI)."""
    if hasattr(model, "base_model") and model.base_model is not None:
        return model.base_model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "codi"):
        return inner.codi
    raise TypeError(
        "Patchscope requires CoconutWrapper (base_model) or CodiWrapper (model.codi)."
    )


# PAIR ``next_token_prediction.ipynb`` (token-identity few-shot); use for paper reproduction.
DEFAULT_PAIR_TARGET_TEMPLATE = (
    "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
)

# Same layout as ``DEFAULT_PAIR_TARGET_TEMPLATE`` (``a -> b`` lines + ``?``), but few-shot
# slots are short phrases instead of single-token identities. Fixed prompt, no ``{question}``.
DEFAULT_PHRASE_TARGET_TEMPLATE = (
    "Comparing two quantities -> Comparing two quantities before combining them.\n"
    "An intermediate value -> An intermediate value in a multi-step reasoning step.\n"
    "?"
)

# Default when YAML omits ``patchscope.target_template``.
DEFAULT_PATCHSCOPE_TARGET_TEMPLATE = DEFAULT_PHRASE_TARGET_TEMPLATE


def get_final_norm_module(causal_lm: nn.Module) -> nn.Module:
    """
    Final norm before LM head: Llama/Qwen ``model.model.norm``, GPT-2 ``transformer.ln_f``.
    Used for PAIR-style ``skip_final_ln`` patching.
    """
    m: nn.Module = causal_lm
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if hasattr(m, "model") and hasattr(m.model, "norm"):
        return m.model.norm
    if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"):
        return m.transformer.ln_f
    raise ValueError(
        f"Unsupported causal LM for final-norm hook (need model.norm or transformer.ln_f): {type(causal_lm)}"
    )


def get_last_transformer_block(causal_lm: nn.Module) -> nn.Module:
    """
    Last transformer block output is patched (same space as HF hidden_states[-1]
    before final norm in many configs; matches Coconut latent gather).
    """
    m: nn.Module = causal_lm
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers[-1]
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h[-1]
    raise ValueError(f"Unsupported causal LM for last-block hook: {type(causal_lm)}")


def _unpack_block_output(out: Any) -> Tuple[torch.Tensor, Tuple[Any, ...]]:
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, ()


def _repack_block_output(hidden: torch.Tensor, rest: Tuple[Any, ...], was_tuple: bool) -> Any:
    if was_tuple:
        return (hidden,) + rest
    return hidden


def patch_hidden_at_position(
    hidden: torch.Tensor,
    patch_pos: int,
    patch_vector: torch.Tensor,
) -> torch.Tensor:
    """Replace hidden[:, patch_pos, :] with patch_vector (batch must match)."""
    h = hidden.clone()
    pv = patch_vector.to(device=h.device, dtype=h.dtype)
    if pv.dim() == 1:
        pv = pv.unsqueeze(0)
    h[:, patch_pos, :] = pv
    return h


def make_last_block_patch_hook(patch_pos: int, patch_vector: torch.Tensor):
    """Forward hook on last block: patch residual stream at patch_pos."""

    def hook_fn(module: nn.Module, inp: Any, out: Any) -> Any:
        hidden, rest = _unpack_block_output(out)
        was_tuple = isinstance(out, tuple)
        hidden = patch_hidden_at_position(hidden, patch_pos, patch_vector)
        return _repack_block_output(hidden, rest, was_tuple)

    return hook_fn


def _norm_forward_output_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out


def make_final_norm_patch_hook(patch_pos: int, patch_vector: torch.Tensor):
    """
    Forward hook on final RMSNorm / LayerNorm (PAIR ``skip_final_ln`` path):
    replace ``hidden[:, patch_pos, :]`` with ``patch_vector`` in-place on the norm output.
    """

    def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
        t = _norm_forward_output_tensor(out)
        pv = patch_vector.to(device=t.device, dtype=t.dtype)
        if pv.dim() == 1:
            pv = pv.unsqueeze(0)
        bsz = int(pv.size(0))
        for b in range(bsz):
            t[b, patch_pos, :] = pv[b]
        return None

    return hook_fn


def find_placeholder_token_index(
    target_text: str,
    tokenizer: Any,
    placeholder: str,
) -> int:
    """
    Find the token index of the placeholder in `target_text`.

    BPE often merges boundaries (e.g. ``decode: ?`` is not the same as ``encode('?')``),
    so we align by character offsets via ``offset_mapping`` (or backend_tokenizer offsets),
    not by matching the standalone placeholder token id in the full sequence.
    """
    if placeholder not in target_text:
        raise ValueError(f"Placeholder {placeholder!r} not found in target text.")
    char_start = target_text.rindex(placeholder)

    def _normalize_offset_spans(offsets: Any) -> list:
        """HF may return unbatched [(s,e), ...] or batched [[(s,e), ...], ...]."""
        if not offsets:
            return []
        first = offsets[0]
        # Batched: first row is a list of (s, e) spans
        if isinstance(first, list) and len(first) > 0:
            inner0 = first[0]
            if isinstance(inner0, (list, tuple)) and len(inner0) == 2:
                try:
                    int(inner0[0])
                    return list(first)
                except (TypeError, ValueError):
                    pass
        # Unbatched: list of (s, e)
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                int(first[0])
                return list(offsets)
            except (TypeError, ValueError):
                pass
        return []

    def _from_offsets(offsets: Any) -> int:
        seq = _normalize_offset_spans(offsets)
        if not seq:
            raise ValueError("Empty offset list.")
        for i, span in enumerate(seq):
            if not span or len(span) < 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start <= char_start < end:
                return i
        raise ValueError(
            f"Could not map placeholder at char {char_start} to a token. "
            f"Try a different target_template or placeholder."
        )

    try:
        enc = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        enc = None

    if enc is not None and enc.get("offset_mapping"):
        return _from_offsets(enc["offset_mapping"])

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            enc_bt = backend.encode(target_text)
            for i, (s, e) in enumerate(enc_bt.offsets):
                if s <= char_start < e:
                    return i
        except Exception:
            pass

    raise ValueError(
        "Tokenizer does not support offset_mapping or backend offsets; "
        "cannot locate placeholder. Use a Fast tokenizer or set patchscope.target_template "
        "so the placeholder is a single token when tokenized in isolation and in context, "
        "or install tokenizers with offset support."
    )


def greedy_generate_after_patch(
    causal_lm: nn.Module,
    tokenizer: Any,
    target_text: str,
    patch_pos: int,
    patch_vector: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    skip_final_ln: bool = True,
) -> List[str]:
    """
    Patched forward on the target prompt (same template for every row), then greedy decode.

    ``patch_vector``: ``[B, H]`` or ``[H]`` (single row). Returns ``B`` decoded strings
    (new tokens only, same as single-sample behavior).

    ``skip_final_ln``: If True (PAIR default for last-layer source), register the hook on
    the backbone **final norm** module; if False, hook the **last transformer block** output
    (pre-norm), then norm + LM head run normally.
    """
    enc = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("Target prompt must encode to a single sequence (batch expand internally).")
    prompt_len = int(input_ids.size(1))
    if patch_pos < 0 or patch_pos >= prompt_len:
        raise ValueError(f"patch_pos {patch_pos} out of range for length {prompt_len}")

    pv = patch_vector.to(device)
    if pv.dim() == 1:
        pv = pv.unsqueeze(0)
    bsz = int(pv.size(0))
    input_ids = input_ids.expand(bsz, -1).contiguous()
    attention_mask = attention_mask.expand(bsz, -1).contiguous()

    if skip_final_ln:
        patch_module = get_final_norm_module(causal_lm)
        hook = make_final_norm_patch_hook(patch_pos, pv)
    else:
        patch_module = get_last_transformer_block(causal_lm)
        hook = make_last_block_patch_hook(patch_pos, pv)
    handle = patch_module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        next_ids = out.logits[:, patch_pos, :].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([input_ids, next_ids], dim=1)
        attn = torch.cat(
            [
                attention_mask,
                torch.ones(bsz, 1, device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = causal_lm(input_ids=cur_ids, attention_mask=attn, use_cache=False)
            next_t = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_t], dim=1)
            attn = torch.cat(
                [attn, torch.ones(bsz, 1, device=device, dtype=attn.dtype)],
                dim=1,
            )
    finally:
        handle.remove()

    gen_only = cur_ids[:, prompt_len:]
    out_texts: List[str] = []
    for i in range(bsz):
        row = gen_only[i].detach().cpu().tolist()
        out_texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return out_texts


def greedy_generate_baseline_no_patch(
    causal_lm: nn.Module,
    tokenizer: Any,
    target_text: str,
    patch_pos: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Unpatched greedy decode (batch size 1); used once for global baseline in meta."""
    enc = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("Baseline expects a single encoded target sequence.")
    prompt_len = int(input_ids.size(1))

    with torch.no_grad():
        out = causal_lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    next_ids = out.logits[:, patch_pos, :].argmax(dim=-1, keepdim=True)
    cur_ids = torch.cat([input_ids, next_ids], dim=1)
    attn = torch.cat(
        [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
        dim=1,
    )
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = causal_lm(input_ids=cur_ids, attention_mask=attn, use_cache=False)
        next_t = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_t], dim=1)
        attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=attn.dtype)], dim=1)

    gen_only = cur_ids[:, prompt_len:]
    return tokenizer.decode(gen_only[0].detach().cpu().tolist(), skip_special_tokens=True)
