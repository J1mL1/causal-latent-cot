from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_interface import LatentReasoningModel
from common.model_registry import register_model


@dataclass
class MultiplexThinkingConfig:
    # Soft-thinking / multiplex parameters.
    temperature_think: float = 1.0
    top_k_think: int = 0
    top_p_think: float = 1.0
    min_p_think: float = 0.0
    multiplex_width: int = 3
    enable_unweighting: bool = False
    enable_replacement: bool = False
    enable_gumbel: bool = False
    gumbel_tau: float = 1.0
    dirichlet_alpha: float = 0.0
    cold_stop_entropy: float = 0.1
    cold_stop_len: int = 256
    max_think_steps: int = 2048
    think_end_str: str = "</think>"

    # Answer decoding parameters.
    max_answer_tokens: int = 512
    do_sample_answer: bool = True
    temperature_answer: float = 0.6
    top_k_answer: int = 30
    top_p_answer: float = 0.95
    answer_stop_strings: List[str] = field(
        default_factory=lambda: ["\nQuestion:", "\n\nQuestion:"]
    )


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-12)
    return -(p * p.log()).sum(dim=-1)


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


def _sample_from_probs_1d(
    probs: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature == 0.0 or not do_sample:
        return int(torch.argmax(probs).item())
    vocab = probs.shape[0]
    if top_k > 0 and top_k < vocab:
        topk_probs, topk_idx = torch.topk(probs, k=top_k, largest=True, sorted=True)
        probs = torch.zeros_like(probs).scatter(0, topk_idx, topk_probs)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        keep = cumsum <= top_p
        keep[0] = True
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
    probs = probs / probs.sum()
    draw = torch.multinomial(probs, num_samples=1)
    return int(draw.item())


@register_model("multiplex")
class MultiplexThinkingWrapper(LatentReasoningModel):
    """
    HF implementation of Multiplex Thinking-style implicit reasoning.

    Latent steps are soft tokens formed by sampling multiple token ids and
    merging their embeddings. Returned h_t values correspond to the last
    hidden state after inserting that soft token.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.emb = None
        self.cfg = MultiplexThinkingConfig()
        self.think_end_ids: List[int] = []
        self.eos_id: Optional[int] = None

    def load_from_config(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_name_or_path") or config.get("model_path")
        if model_path is None:
            raise ValueError("MultiplexThinkingWrapper requires 'model_name_or_path'.")

        tokenizer_name = config.get("tokenizer_name_or_path", model_path)
        self.teacher_target_template = config.get("teacher_target_template", self.teacher_target_template)
        self.device = torch.device(config.get("device", self.device))

        cfg = MultiplexThinkingConfig()
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
        if "position_ids" not in tokens:
            attention_mask = tokens["attention_mask"].long()
            pos = attention_mask.cumsum(dim=-1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)
            tokens["position_ids"] = pos
        return tokens

    def _apply_top_k_top_p_min_p(
        self,
        probs: torch.Tensor,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> torch.Tensor:
        if top_k and top_k > 0:
            k = min(int(top_k), probs.size(-1))
            topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
            masked = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)
            probs = masked
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= top_p
            keep[:, 0] = True
            sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        if min_p > 0.0:
            max_prob = probs.max(dim=-1, keepdim=True).values
            threshold = max_prob * min_p
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        denom = probs.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        return probs / denom

    def _maybe_apply_dirichlet(self, probs: torch.Tensor) -> torch.Tensor:
        if self.cfg.dirichlet_alpha <= 0:
            return probs
        conc = probs * self.cfg.dirichlet_alpha
        conc = torch.clamp(conc, min=torch.finfo(conc.dtype).min)
        gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
        gamma_samples = gamma_dist.sample()
        return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    def _maybe_apply_gumbel(self, probs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_gumbel:
            return probs
        noise = torch.rand_like(probs)
        noise.clamp_(min=1e-20, max=1.0 - 1e-7)
        noise.log_()
        noise.mul_(-1.0)
        noise.log_()
        noise.mul_(-1.0)
        gumbel_logits = (noise + torch.log(probs.clamp(min=1e-20))) / max(self.cfg.gumbel_tau, 1e-8)
        return torch.softmax(gumbel_logits, dim=-1)

    def _latent_embedding_from_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = logits / max(self.cfg.temperature_think, 1e-8)
        probs = torch.softmax(scaled, dim=-1)
        probs = self._apply_top_k_top_p_min_p(
            probs, self.cfg.top_k_think, self.cfg.top_p_think, self.cfg.min_p_think
        )
        probs = self._maybe_apply_dirichlet(probs)
        probs = self._maybe_apply_gumbel(probs)

        vocab = probs.size(-1)
        width = max(int(self.cfg.multiplex_width), 1)
        width = min(width, vocab)
        topk_indices = torch.multinomial(
            probs, num_samples=width, replacement=self.cfg.enable_replacement
        )
        if self.cfg.enable_unweighting:
            topk_probs = torch.ones_like(topk_indices, dtype=probs.dtype, device=probs.device)
        else:
            topk_probs = torch.gather(probs, dim=1, index=topk_indices)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        latent_emb = (topk_probs.unsqueeze(-1) * self.emb[topk_indices]).sum(dim=1)
        return latent_emb, probs

    def _insert_latent(
        self,
        latent_emb: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        allow_grad: bool = False,
        output_hidden: bool = True,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                output_hidden_states=output_hidden,
            )
        last_hidden = outputs.hidden_states[-1][:, -1, :] if output_hidden else None
        return outputs.past_key_values, attention_mask, past_lengths, outputs.logits[:, -1, :], last_hidden

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
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(**tokens, use_cache=True, output_hidden_states=False)
        past_key_values = outputs.past_key_values
        past_lengths = attention_mask.sum(dim=1)
        logits = outputs.logits[:, -1, :]
        if step == 1:
            latent_emb, _ = self._latent_embedding_from_logits(logits)
            other_state = {
                "tokens": tokens,
                "attention_mask": attention_mask,
                "past_lengths": past_lengths,
                "past_key_values": past_key_values,
                "current_step": 0,
            }
            return latent_emb, other_state

        for _ in range(step - 1):
            latent_emb, _ = self._latent_embedding_from_logits(logits)
            past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
                latent_emb,
                past_key_values,
                attention_mask,
                past_lengths,
                allow_grad=allow_grad,
                output_hidden=False,
            )

        latent_emb, _ = self._latent_embedding_from_logits(logits)
        other_state = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "past_lengths": past_lengths,
            "past_key_values": past_key_values,
            "current_step": step - 1,
        }
        return latent_emb, other_state

    def rollout_to_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        target_step: int,
        allow_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        current_step = int(other_state.get("current_step", 0))
        if target_step <= current_step:
            return h_t_modified, other_state
        if target_step == current_step + 1:
            return h_t_modified, other_state
        if target_step > self.cfg.max_think_steps:
            raise ValueError(
                f"target_step {target_step} exceeds max_think_steps={self.cfg.max_think_steps}"
            )

        attention_mask = other_state["attention_mask"]
        past_lengths = other_state["past_lengths"]
        past_key_values = other_state["past_key_values"]
        past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
            h_t_modified,
            past_key_values,
            attention_mask,
            past_lengths,
            allow_grad=allow_grad,
            output_hidden=False,
        )
        current_step += 1

        while current_step < target_step:
            latent_emb, _ = self._latent_embedding_from_logits(logits)
            if current_step + 1 == target_step:
                state = {
                    "attention_mask": attention_mask,
                    "past_lengths": past_lengths,
                    "past_key_values": past_key_values,
                    "current_step": current_step,
                }
                return latent_emb, state
            past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
                latent_emb,
                past_key_values,
                attention_mask,
                past_lengths,
                allow_grad=allow_grad,
                output_hidden=False,
            )
            current_step += 1

        raise RuntimeError(f"Failed to reach target_step {target_step}.")

    def rollout_from_step(
        self, h_t_modified: torch.Tensor, other_state: Dict[str, Any], allow_grad: bool = False
    ) -> Any:
        attention_mask = other_state["attention_mask"]
        past_lengths = other_state["past_lengths"]
        past_key_values = other_state["past_key_values"]
        start_step = int(other_state.get("current_step", 0))
        think_stop_step = other_state.get("think_stop_step")
        past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
            h_t_modified,
            past_key_values,
            attention_mask,
            past_lengths,
            allow_grad=allow_grad,
            output_hidden=False,
        )

        if think_stop_step is None:
            remaining = max(self.cfg.max_think_steps - (start_step + 1), 0)
        else:
            remaining = max(int(think_stop_step) - (start_step + 1), 0)

        low_entropy_steps = torch.zeros(
            attention_mask.size(0), dtype=torch.long, device=self.device
        )
        think_end_single = len(self.think_end_ids) == 1
        think_end_id = self.think_end_ids[0] if think_end_single else None

        for _ in range(remaining):
            latent_emb, probs = self._latent_embedding_from_logits(logits)
            ent = _entropy(probs)
            top1_ids = torch.argmax(probs, dim=-1)
            low_entropy_steps = torch.where(
                ent < self.cfg.cold_stop_entropy,
                low_entropy_steps + 1,
                torch.zeros_like(low_entropy_steps),
            )
            hit_think_end = top1_ids == think_end_id if think_end_single else torch.zeros_like(low_entropy_steps, dtype=torch.bool)
            cold_stop = low_entropy_steps >= self.cfg.cold_stop_len
            if torch.all(hit_think_end | cold_stop):
                break
            past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
                latent_emb,
                past_key_values,
                attention_mask,
                past_lengths,
                allow_grad=allow_grad,
                output_hidden=False,
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

    def logits_from_latent(self, h_t: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "lm_head"):
            raise AttributeError("Model lacks lm_head for projecting latents.")
        return self.model.lm_head(h_t)

    def decode_from_state(self, h_t: torch.Tensor, other_state: Dict[str, Any]) -> Any:
        return self.rollout_from_step(h_t, other_state)

    def compute_logits(
        self,
        h_t: torch.Tensor,
        other_state: Dict[str, Any],
        target_ids: torch.Tensor,
        allow_grad: bool = False,
    ) -> torch.Tensor:
        target_ids = target_ids.to(self.device)
        attention_mask = other_state.get("attention_mask")
        past_lengths = other_state.get("past_lengths")
        past_key_values = other_state.get("past_key_values")
        if attention_mask is None and "tokens" in other_state:
            attention_mask = other_state["tokens"].get("attention_mask")
        if attention_mask is None or past_lengths is None:
            raise ValueError("Multiplex compute_logits requires attention_mask and past_lengths in state.")
        past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
            h_t,
            past_key_values,
            attention_mask,
            past_lengths,
            allow_grad=allow_grad,
            output_hidden=False,
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
        with torch.no_grad():
            outputs = self.model(**tokens, use_cache=True, output_hidden_states=False)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        past_lengths = attention_mask.sum(dim=1)

        thought_preview_tokens: List[int] = []
        low_entropy_steps = 0
        think_end_single = len(self.think_end_ids) == 1
        think_end_id = self.think_end_ids[0] if think_end_single else None
        think_steps_done = 0

        for _ in range(self.cfg.max_think_steps):
            latent_emb, probs = self._latent_embedding_from_logits(logits)
            ent = float(_entropy(probs)[0].item())
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

            past_key_values, attention_mask, past_lengths, logits, _ = self._insert_latent(
                latent_emb,
                past_key_values,
                attention_mask,
                past_lengths,
                allow_grad=False,
                output_hidden=False,
            )
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
            if self.cfg.answer_stop_strings:
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

        if not self.cfg.answer_stop_strings:
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=False)

        return {
            "text": answer_text,
            "thought_preview": self.tokenizer.decode(thought_preview_tokens, skip_special_tokens=False),
            "think_stop_step": think_steps_done,
        }
