from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_interface import LatentReasoningModel
from common.model_registry import register_model


@register_model("hf-auto")
class HFAutoregressiveModel(LatentReasoningModel):
    """
    Generic HuggingFace causal LM wrapper.

    For models without explicit latent steps, `step` refers to a token position
    (default: last token when the requested index exceeds sequence length).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.generation_kwargs: Dict[str, Any] = {}

    def load_from_config(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_name_or_path") or config.get("model_path")
        if model_path is None:
            raise ValueError("HFAutoregressiveModel requires 'model_name_or_path'.")
        tokenizer_name = config.get("tokenizer_name_or_path", model_path)
        self.teacher_target_template = config.get("teacher_target_template")
        self.device = torch.device(config.get("device", self.device))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            _ = self.model.load_state_dict(state_dict, strict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_kwargs = config.get("generation_kwargs", {})

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
        return tokens

    def forward_until_step(
        self, inputs: Any, step: int, allow_grad: bool = False
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        tokens = self._prepare_inputs(inputs)
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                **tokens, output_hidden_states=True, use_cache=True
            )
        hidden_states = outputs.hidden_states[-1]
        seq_len = hidden_states.size(1)
        # Interpret steps as 1-based token indices.
        if step < 1:
            raise ValueError(f"Step must be >= 1, got {step}")
        step_idx = step - 1
        if step_idx < 0 or step_idx >= seq_len:
            raise ValueError(
                f"Requested step index {step_idx} (step={step}) outside sequence length {seq_len}"
            )
        h_t = hidden_states[:, step_idx, :]
        other_state = {
            "tokens": tokens,
            "step_idx": step_idx,
            "hidden_states": hidden_states,
            "past_key_values": getattr(outputs, "past_key_values", None),
        }
        return h_t, other_state

    def rollout_from_step(
        self, h_t_modified: torch.Tensor, other_state: Dict[str, Any], allow_grad: bool = False
    ) -> Any:
        tokens = other_state["tokens"]
        step_idx = other_state["step_idx"]
        inputs_embeds = self.model.get_input_embeddings()(tokens["input_ids"])
        inputs_embeds = inputs_embeds.to(self.device)
        if step_idx < 0 or step_idx >= inputs_embeds.size(1):
            raise ValueError(f"Invalid step_idx {step_idx} for sequence length {inputs_embeds.size(1)}")
        # Replace the embedding for the target position with the ablated latent.
        inputs_embeds[:, step_idx, :] = h_t_modified.to(self.device)
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=tokens.get("attention_mask"),
                output_hidden_states=True,
                use_cache=True,
            )
        return outputs

    def rollout_to_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        target_step: int,
        allow_grad: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        tokens = dict(other_state["tokens"])
        step_idx = other_state["step_idx"]
        inputs_embeds = self.model.get_input_embeddings()(tokens["input_ids"])
        inputs_embeds = inputs_embeds.to(self.device)
        inputs_embeds[:, step_idx, :] = h_t_modified.to(self.device)
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=tokens.get("attention_mask"),
                output_hidden_states=True,
                use_cache=True,
            )
        hidden_states = outputs.hidden_states[-1]
        if target_step < 1:
            raise ValueError(f"target_step must be >=1, got {target_step}")
        target_idx = target_step - 1
        if target_idx >= hidden_states.size(1):
            raise ValueError(
                f"target_step {target_step} exceeds sequence length {hidden_states.size(1)}"
            )
        h_target = hidden_states[:, target_idx, :]
        new_state = {
            "tokens": tokens,
            "step_idx": target_idx,
            "hidden_states": hidden_states,
            "past_key_values": getattr(outputs, "past_key_values", None),
        }
        return h_target, new_state

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
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model(
                input_ids=target_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=False,
            )
        return outputs.logits

    def run_baseline(self, inputs: Any) -> Any:
        tokens = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.generation_kwargs:
                generated = self.model.generate(
                    **tokens, **self.generation_kwargs
                )
                return self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            return self.model(**tokens, output_hidden_states=True)
