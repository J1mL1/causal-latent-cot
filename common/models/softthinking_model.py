from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_interface import LatentReasoningModel
from common.model_registry import register_model


@dataclass
class SoftThinkingConfig:
    temperature_think: float = 0.6
    top_k_think: int = 30
    top_p_think: float = 0.95
    concept_top_n: int = 15
    cold_stop_entropy: float = 0.1
    cold_stop_len: int = 256
    max_think_steps: int = 2048
    max_answer_tokens: int = 512
    think_end_str: str = "</think>"
    do_sample_answer: bool = True
    temperature_answer: float = 0.6
    top_k_answer: int = 30
    top_p_answer: float = 0.95
    answer_stop_strings: List[str] = field(
        default_factory=lambda: ["\nQuestion:", "\n\nQuestion:"]
    )


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-12)
    return -(p * p.log()).sum()


def _top_k_top_p_filter_1d(
    probs: torch.Tensor, top_k: int, top_p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab = probs.shape[0]
    k = min(max(int(top_k), 1), vocab)
    topk_probs, topk_idx = torch.topk(probs, k=k, largest=True, sorted=True)
    cumsum = torch.cumsum(topk_probs, dim=0)
    keep = cumsum <= top_p
    keep[0] = True
    kept_probs = topk_probs[keep]
    kept_idx = topk_idx[keep]
    kept_probs = kept_probs / kept_probs.sum()
    return kept_probs, kept_idx


def _sample_from_probs_1d(
    probs: torch.Tensor, do_sample: bool, temperature: float, top_k: int, top_p: float
) -> int:
    if temperature == 0.0 or not do_sample:
        return int(torch.argmax(probs).item())
    kept_probs, kept_idx = _top_k_top_p_filter_1d(probs, top_k=top_k, top_p=top_p)
    draw = torch.multinomial(kept_probs, num_samples=1)
    return int(kept_idx[draw].item())


def _truncate_on_stop(text: str, stop_strings: List[str]) -> Tuple[str, bool]:
    if not stop_strings:
        return text, False
    earliest = None
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text, False
    return text[:earliest], True


@register_model("softthinking")
class SoftThinkingWrapper(LatentReasoningModel):
    """
    SoftThinking wrapper that exposes concept embeddings as latent steps.

    Steps are 1-based concept embeddings derived from the current token distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.emb = None
        self.cfg = SoftThinkingConfig()
        self.think_end_ids: List[int] = []
        self.eos_id: Optional[int] = None

    def load_from_config(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_name_or_path") or config.get("model_path")
        if model_path is None:
            raise ValueError("SoftThinkingWrapper requires 'model_name_or_path'.")

        tokenizer_name = config.get("tokenizer_name_or_path", model_path)
        self.teacher_target_template = config.get("teacher_target_template", self.teacher_target_template)
        self.device = torch.device(config.get("device", self.device))

        cfg = SoftThinkingConfig()
        for field in cfg.__dataclass_fields__:
            if field in config:
                setattr(cfg, field, config[field])
        self.cfg = cfg

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        torch_dtype = config.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            else:
                model_kwargs["torch_dtype"] = torch_dtype
        attn_impl = config.get("attn_implementation")
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.emb = self.model.get_input_embeddings().weight
        self.think_end_ids = self.tokenizer.encode(self.cfg.think_end_str, add_special_tokens=False)
        if not self.think_end_ids:
            raise ValueError(f"think_end_str tokenizes to empty: {self.cfg.think_end_str!r}")
        self.eos_id = self.tokenizer.eos_token_id

    def _prepare_inputs(self, inputs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("inputs list is empty.")
            if all(isinstance(x, str) for x in inputs):
                tokenizer_inputs: Any = list(inputs)
            else:
                raise TypeError("inputs list must contain strings only.")
        else:
            tokenizer_inputs = inputs

        if isinstance(tokenizer_inputs, str) or isinstance(tokenizer_inputs, list):
            tokens = self.tokenizer(
                tokenizer_inputs, return_tensors="pt", padding=True, truncation=True
            )
        elif isinstance(tokenizer_inputs, Dict):
            tokens = tokenizer_inputs
        else:
            raise TypeError("inputs must be a string, list[str], or a dict of tensors.")
        tokens = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()
        }
        if "attention_mask" not in tokens:
            tokens["attention_mask"] = torch.ones_like(tokens["input_ids"])
        # Precompute position ids so padding behaves consistently across models.
        if "position_ids" not in tokens:
            attention_mask = tokens["attention_mask"].long()
            pos = attention_mask.cumsum(dim=-1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)
            tokens["position_ids"] = pos
        return tokens

    def _concept_embedding_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        logits = logits / max(cfg.temperature_think, 1e-8)
        probs = torch.softmax(logits, dim=-1)
        concept_embeds: List[torch.Tensor] = []
        for row in probs:
            kept_probs, kept_idx = _top_k_top_p_filter_1d(
                row, top_k=max(cfg.top_k_think, cfg.concept_top_n), top_p=cfg.top_p_think
            )
            kept_probs = kept_probs[: cfg.concept_top_n]
            kept_idx = kept_idx[: cfg.concept_top_n]
            kept_probs = kept_probs / kept_probs.sum()
            emb = (kept_probs.unsqueeze(-1) * self.emb[kept_idx]).sum(dim=0)
            concept_embeds.append(emb)
        return torch.stack(concept_embeds, dim=0)

    def forward_until_step(
        self, inputs: Any, step: int, allow_grad: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if step < 1:
            raise ValueError(f"Latent steps are 1-based; got {step}")
        if step > self.cfg.max_think_steps:
            raise ValueError(
                f"Requested step {step} exceeds max_think_steps={self.cfg.max_think_steps}"
            )

        tokens = self._prepare_inputs(inputs)
        attention_mask = tokens["attention_mask"]
        position_ids = tokens.get("position_ids")
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                **tokens, use_cache=True, output_hidden_states=False
            )
        past_key_values = outputs.past_key_values
        last_indices = attention_mask.long().sum(dim=1) - 1
        last_indices = last_indices.clamp(min=0)
        logits = outputs.logits.gather(
            1, last_indices.view(-1, 1, 1).expand(-1, 1, outputs.logits.size(-1))
        ).squeeze(1)
        past_lengths = attention_mask.sum(dim=1)

        for _ in range(1, step):
            concept_emb = self._concept_embedding_from_logits(logits)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model(
                    inputs_embeds=concept_emb.unsqueeze(1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        concept_emb = self._concept_embedding_from_logits(logits)
        state = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "past_lengths": past_lengths,
            "current_step": step,
            "think_stop_step": None,
        }
        return concept_emb, state

    def _insert_latent(
        self,
        latent_emb: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        allow_grad: bool = False,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        position_ids = past_lengths.view(-1, 1)
        past_lengths = past_lengths + 1
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                inputs_embeds=latent_emb.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
        return outputs.past_key_values, attention_mask, past_lengths, outputs.logits[:, -1, :]

    def rollout_from_step(
        self, h_t_modified: torch.Tensor, other_state: Dict[str, Any], allow_grad: bool = False
    ) -> Any:
        past_key_values = other_state["past_key_values"]
        attention_mask = other_state["attention_mask"]
        past_lengths = other_state["past_lengths"]
        start_step = int(other_state.get("current_step", 1))
        continue_latents = bool(other_state.get("decode_continue_latents", True))
        think_stop_step = other_state.get("think_stop_step")

        past_key_values, attention_mask, past_lengths, logits = self._insert_latent(
            h_t_modified, past_key_values, attention_mask, past_lengths, allow_grad=allow_grad
        )

        if continue_latents:
            if think_stop_step is None:
                remaining = max(self.cfg.max_think_steps - start_step, 0)
            else:
                # Align ablation to the baseline think length; avoid injecting extra latents.
                remaining = max(int(think_stop_step) - start_step, 0)
            low_entropy_steps = torch.zeros(
                attention_mask.size(0), dtype=torch.long, device=self.device
            )
            think_end_single = len(self.think_end_ids) == 1
            think_end_id = self.think_end_ids[0] if think_end_single else None
            for _ in range(remaining):
                probs = torch.softmax(logits / max(self.cfg.temperature_think, 1e-8), dim=-1)
                ent = -(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log()).sum(dim=-1)
                top1_ids = torch.argmax(probs, dim=-1)
                low_entropy_steps = torch.where(
                    ent < self.cfg.cold_stop_entropy,
                    low_entropy_steps + 1,
                    torch.zeros_like(low_entropy_steps),
                )
                if think_end_single:
                    hit_think_end = top1_ids == think_end_id
                else:
                    hit_think_end = torch.zeros_like(low_entropy_steps, dtype=torch.bool)
                cold_stop = low_entropy_steps >= self.cfg.cold_stop_len
                if torch.all(hit_think_end | cold_stop):
                    break
                concept_emb = self._concept_embedding_from_logits(logits)
                past_key_values, attention_mask, past_lengths, logits = self._insert_latent(
                    concept_emb, past_key_values, attention_mask, past_lengths, allow_grad=allow_grad
                )

        for tid in self.think_end_ids:
            tid_tensor = torch.full((attention_mask.size(0), 1), tid, device=self.device, dtype=torch.long)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model(
                    input_ids=tid_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        answer_ids: List[List[int]] = [[] for _ in range(attention_mask.size(0))]
        answer_texts = [""] * attention_mask.size(0)
        finished = torch.zeros(attention_mask.size(0), dtype=torch.bool, device=self.device)
        use_stop_strings = bool(self.cfg.answer_stop_strings)

        for _ in range(self.cfg.max_answer_tokens):
            if self.cfg.temperature_answer == 0.0 or not self.cfg.do_sample_answer:
                next_ids = torch.argmax(logits, dim=-1)
            else:
                logits_scaled = logits / max(self.cfg.temperature_answer, 1e-8)
                probs = torch.softmax(logits_scaled, dim=-1)
                next_ids = []
                for row in probs:
                    next_ids.append(
                        _sample_from_probs_1d(
                            row,
                            do_sample=self.cfg.do_sample_answer,
                            temperature=self.cfg.temperature_answer,
                            top_k=self.cfg.top_k_answer,
                            top_p=self.cfg.top_p_answer,
                        )
                    )
                next_ids = torch.tensor(next_ids, device=self.device, dtype=torch.long)

            for idx, tok in enumerate(next_ids.tolist()):
                if finished[idx]:
                    continue
                if self.eos_id is not None and tok == self.eos_id:
                    finished[idx] = True
                    continue
                answer_ids[idx].append(tok)
                if use_stop_strings:
                    answer_texts[idx] += self.tokenizer.decode([tok], skip_special_tokens=False)
                    truncated, hit = _truncate_on_stop(
                        answer_texts[idx], self.cfg.answer_stop_strings
                    )
                    if hit:
                        answer_texts[idx] = truncated
                        finished[idx] = True
            if finished.all():
                break

            if self.eos_id is not None:
                next_ids = next_ids.clone()
                next_ids[finished] = self.eos_id
            next_tensor = next_ids.view(-1, 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model(
                    input_ids=next_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        if use_stop_strings:
            text = answer_texts
        else:
            text = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in answer_ids]
        return {
            "text": text,
            "past_key_values": past_key_values,
            "logits": logits,
        }

    def rollout_to_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        target_step: int,
        allow_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        start_step = int(other_state.get("current_step", 1))
        if target_step < start_step:
            raise ValueError(f"target_step {target_step} precedes current_step {start_step}")
        if target_step == start_step:
            return h_t_modified, other_state

        past_key_values = other_state["past_key_values"]
        attention_mask = other_state["attention_mask"]
        past_lengths = other_state["past_lengths"]

        past_key_values, attention_mask, past_lengths, logits = self._insert_latent(
            h_t_modified, past_key_values, attention_mask, past_lengths, allow_grad=allow_grad
        )

        current_step = start_step + 1
        while current_step < target_step:
            concept_emb = self._concept_embedding_from_logits(logits)
            past_key_values, attention_mask, past_lengths, logits = self._insert_latent(
                concept_emb, past_key_values, attention_mask, past_lengths, allow_grad=allow_grad
            )
            current_step += 1

        concept_emb = self._concept_embedding_from_logits(logits)
        state = {
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "past_lengths": past_lengths,
            "current_step": target_step,
        }
        return concept_emb, state

    def logits_from_latent(self, h_t: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "lm_head"):
            raise AttributeError("Model lacks lm_head for projecting latents.")
        return self.model.lm_head(h_t)

    def decode_from_state(
        self, h_t: torch.Tensor, other_state: Dict[str, Any]
    ) -> Any:
        return self.rollout_from_step(h_t, other_state)

    def compute_logits(
        self,
        h_t: torch.Tensor,
        other_state: Dict[str, Any],
        target_ids: torch.Tensor,
        allow_grad: bool = False,
    ) -> torch.Tensor:
        target_ids = target_ids.to(self.device)
        past_key_values = other_state.get("past_key_values")
        attention_mask = other_state.get("attention_mask")
        past_lengths = other_state.get("past_lengths")
        if attention_mask is None and "tokens" in other_state:
            attention_mask = other_state["tokens"].get("attention_mask")
        if attention_mask is None:
            raise ValueError("SoftThinking compute_logits requires attention_mask in state.")
        if past_lengths is None:
            past_lengths = attention_mask.sum(dim=1)

        past_key_values, attention_mask, past_lengths, logits = self._insert_latent(
            h_t, past_key_values, attention_mask, past_lengths, allow_grad=allow_grad
        )

        for tid in self.think_end_ids:
            tid_tensor = torch.full((attention_mask.size(0), 1), tid, device=self.device, dtype=torch.long)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model(
                    input_ids=tid_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values

        target_len = target_ids.size(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), target_len), device=self.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        position_ids = past_lengths.view(-1, 1) + torch.arange(
            target_len, device=self.device, dtype=past_lengths.dtype
        ).view(1, -1)
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                input_ids=target_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=False,
            )
        return outputs.logits

    def run_baseline(self, inputs: Any) -> Any:
        if isinstance(inputs, (list, tuple)):
            outputs = [self._run_baseline_single(text) for text in inputs]
            return {
                "text": [out["text"] for out in outputs],
                "thought_preview": [out["thought_preview"] for out in outputs],
                "think_stop_step": [out["think_stop_step"] for out in outputs],
            }
        return self._run_baseline_single(inputs)

    def _run_baseline_single(self, prompt: str) -> Dict[str, Any]:
        tokens = self._prepare_inputs(prompt)
        attention_mask = tokens["attention_mask"]
        position_ids = tokens.get("position_ids")
        with torch.no_grad():
            outputs = self.model(
                **tokens, use_cache=True, output_hidden_states=False
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        past_lengths = attention_mask.sum(dim=1)

        thought_preview_tokens: List[int] = []
        low_entropy_steps = 0
        think_end_single = len(self.think_end_ids) == 1
        think_end_id = self.think_end_ids[0] if think_end_single else None
        # Track completed latent steps for ablation length alignment.
        think_steps_done = 0

        for _ in range(self.cfg.max_think_steps):
            probs = torch.softmax(logits / max(self.cfg.temperature_think, 1e-8), dim=-1)
            ent = float(_entropy(probs[0]).item())
            top1_id = int(torch.argmax(probs, dim=-1).item())
            thought_preview_tokens.append(top1_id)

            if ent < self.cfg.cold_stop_entropy:
                low_entropy_steps += 1
            else:
                low_entropy_steps = 0

            if think_end_single and top1_id == think_end_id:
                break
            if low_entropy_steps >= self.cfg.cold_stop_len:
                break

            concept_emb = self._concept_embedding_from_logits(logits)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=concept_emb.unsqueeze(1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            think_steps_done += 1

        for tid in self.think_end_ids:
            tid_tensor = torch.full((attention_mask.size(0), 1), tid, device=self.device, dtype=torch.long)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tid_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        answer_ids: List[int] = []
        answer_text = ""
        use_stop_strings = bool(self.cfg.answer_stop_strings)
        for _ in range(self.cfg.max_answer_tokens):
            if self.cfg.temperature_answer == 0.0 or not self.cfg.do_sample_answer:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits / max(self.cfg.temperature_answer, 1e-8), dim=-1)[0]
                next_id = _sample_from_probs_1d(
                    probs,
                    do_sample=self.cfg.do_sample_answer,
                    temperature=self.cfg.temperature_answer,
                    top_k=self.cfg.top_k_answer,
                    top_p=self.cfg.top_p_answer,
                )
            if self.eos_id is not None and next_id == self.eos_id:
                break
            answer_ids.append(next_id)
            if use_stop_strings:
                answer_text += self.tokenizer.decode([next_id], skip_special_tokens=False)
                truncated, hit = _truncate_on_stop(answer_text, self.cfg.answer_stop_strings)
                if hit:
                    answer_text = truncated
                    break
            next_tensor = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = past_lengths.view(-1, 1)
            past_lengths = past_lengths + 1
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        thought_preview = self.tokenizer.decode(thought_preview_tokens, skip_special_tokens=False)
        if use_stop_strings:
            text_answer = answer_text
        else:
            text_answer = self.tokenizer.decode(answer_ids, skip_special_tokens=False)
        return {
            "text": text_answer,
            "thought_preview": thought_preview,
            "think_stop_step": think_steps_done,
        }
