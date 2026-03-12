from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from common.model_interface import LatentReasoningModel
from common.model_registry import register_model

# Qwen2 tokenizers can return None for unknown ids; sanitize before decoding.
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


def _safe_decode(tokenizer: Any, token_ids: List[int]) -> str:
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

# Ensure external/codi is importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CODI_SRC = REPO_ROOT / "external" / "codi" / "src"
if CODI_SRC.as_posix() not in sys.path:
    sys.path.append(CODI_SRC.as_posix())

try:
    from model import CODI, ModelArguments, TrainingArguments  # type: ignore
except Exception as exc:  # pragma: no cover - soft import failure for read-only environments
    CODI = None  # type: ignore
    ModelArguments = None  # type: ignore
    TrainingArguments = None  # type: ignore


@register_model("codi")
class CodiWrapper(LatentReasoningModel):
    """
    CODI latent wrapper built on the original implementation in external/codi.

    Latents are produced exactly the same way as training: repeatedly feeding
    the last hidden state back for `num_latent` iterations (with optional proj).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[CODI] = None
        self.tokenizer = None
        self.num_latent = 0
        self.generation_kwargs: Dict[str, Any] = {}
        self.use_prj: bool = False

    def _ensure_cache(self, past_key_values: Any) -> Any:
        """Convert legacy tuple/list caches to the new HF Cache interface."""
        if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def _clone_cache(self, past_key_values: Any) -> Any:
        """Clone a cache so downstream calls do not share tensors."""
        if past_key_values is None:
            return None
        cache = self._ensure_cache(past_key_values)
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

    def load_from_config(self, config: Dict[str, Any]) -> None:
        if CODI is None:
            raise ImportError("Failed to import CODI from external/codi/src/model.py")

        model_args_dict = dict(config.get("model_args", {}))
        tokenizer_override = config.get("tokenizer_name_or_path")
        if tokenizer_override and "tokenizer_name_or_path" not in model_args_dict:
            model_args_dict["tokenizer_name_or_path"] = tokenizer_override

        model_args = ModelArguments(**model_args_dict)
        train_args_dict = config.get("training_args", {})
        # Provide defaults for key fields to avoid HF TrainingArguments CLI parsing
        train_args = TrainingArguments(
            output_dir=train_args_dict.get("output_dir", "./codi-out"),
            per_device_train_batch_size=train_args_dict.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=train_args_dict.get("per_device_eval_batch_size", 1),
            **{k: v for k, v in train_args_dict.items() if k not in {"output_dir", "per_device_train_batch_size", "per_device_eval_batch_size"}},
        )

        # LoRA config is managed internally; pass a dummy dict when not used.
        lora_config = None
        if train_args.use_lora:
            from peft import LoraConfig, TaskType  # type: ignore

            target_modules = ["c_attn", "c_proj", "c_fc"]
            if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=target_modules,
                init_lora_weights=True,
            )

        self.model = CODI(model_args, train_args, lora_config)
        self.device = torch.device(config.get("device", self.device))
        self.teacher_target_template = config.get("teacher_target_template", self.teacher_target_template)
        self.model.to(self.device)
        if model_args.ckpt_dir:
            # Load checkpoint if provided
            ckpt_path_sft = os.path.join(model_args.ckpt_dir, "model.safetensors")
            ckpt_path_bin = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
            if os.path.exists(ckpt_path_sft):
                state_dict = torch.load(ckpt_path_sft, map_location="cpu")
            elif os.path.exists(ckpt_path_bin):
                state_dict = torch.load(ckpt_path_bin, map_location="cpu")
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_args.ckpt_dir}")
            self.model.load_state_dict(state_dict, strict=False)
        # Honor precision flag: keep fp32 when full_precision is requested.
        if not model_args.full_precision and not train_args.bf16:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

        tokenizer_name = config.get("tokenizer_name_or_path", model_args.model_name_or_path)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model, "tokenizer", None) is None:
            self.model.tokenizer = self.tokenizer

        self.num_latent = int(train_args.num_latent)
        self.use_prj = bool(train_args.use_prj)
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
        # Ensure attention mask exists
        if "attention_mask" not in tokens:
            tokens["attention_mask"] = torch.ones_like(tokens["input_ids"])

        # Append eos + BOT tokens to match CODI inference format
        bot_tensor = torch.tensor([self.model.bot_id], dtype=torch.long).expand(
            tokens["input_ids"].size(0), 1
        )
        if getattr(self.model.training_args, "remove_eos", False):
            concat = bot_tensor
        else:
            eos_id = self.tokenizer.eos_token_id
            eos_bot = torch.tensor(
                [eos_id, self.model.bot_id], dtype=torch.long
            ).expand(tokens["input_ids"].size(0), 2)
            concat = eos_bot

        tokens["input_ids"] = torch.cat((tokens["input_ids"], concat), dim=1)
        tokens["attention_mask"] = torch.cat(
            (tokens["attention_mask"], torch.ones_like(concat)), dim=1
        )
        # Finally, move tensors to device
        tokens = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()
        }
        return tokens

    def forward_until_step(
        self, inputs: Any, step: int, allow_grad: bool = False
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        tokens = self._prepare_inputs(inputs)
        # Normalize step: define step=t as the latent *input* vector written to cache at iteration t.
        if isinstance(step, str):
            raise ValueError(f"Unsupported step label: {step}")
        target_step = int(step)
        if target_step < 1:
            raise ValueError(f"Latent steps are 1-based; got {step}")
        if target_step > self.num_latent:
            raise ValueError(
                f"Requested step {target_step} exceeds configured num_latent={self.num_latent}"
            )
        if self.num_latent <= 0:
            raise ValueError("Model has no latent steps.")

        # Encode the question and get initial latent
        with torch.set_grad_enabled(allow_grad):
            outputs = self.model.codi(
                **tokens, use_cache=True, output_hidden_states=True
            )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :]  # [bs, dim]
        if self.use_prj and hasattr(self.model, "prj"):
            latent_embd = self.model.prj(latent_embd.unsqueeze(1)).squeeze(1)

        if target_step == 1:
            other_state = {
                "tokens": tokens,
                "past_key_values": past_key_values,
                "past_key_values_before_step": past_key_values,
                "current_step": 0,
            }
            return latent_embd, other_state

        for idx in range(self.num_latent):
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd.unsqueeze(1),
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :]
            if self.use_prj and hasattr(self.model, "prj"):
                latent_embd = self.model.prj(latent_embd.unsqueeze(1)).squeeze(1)
            if idx + 2 == target_step:
                other_state = {
                    "tokens": tokens,
                    "past_key_values": past_key_values,
                    "past_key_values_before_step": past_key_values,
                    "current_step": idx + 1,
                }
                return latent_embd, other_state
        raise RuntimeError(f"Failed to reach target latent step {target_step}.")

    def rollout_to_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        target_step: int,
        allow_grad: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Continue latent iterations from the current step up to target_step (inclusive of target_step).
        This mirrors the latent loop in forward_until_step but allows early stopping for causal probes.
        """
        current_step = other_state.get("current_step", 0)
        if target_step < 0:
            raise ValueError(f"target_step must be non-negative, got {target_step}")
        if target_step > self.num_latent:
            raise ValueError(
                f"target_step {target_step} exceeds configured num_latent={self.num_latent}"
            )
        if current_step < 0 or current_step > self.num_latent:
            raise ValueError(
                f"Invalid current_step in state: {current_step} (num_latent={self.num_latent})"
            )
        past_key_values = self._clone_cache(other_state.get("past_key_values"))
        tokens = other_state.get("tokens", {})
        latent_embd = h_t_modified
        past_key_values = past_key_values

        if target_step <= current_step or self.num_latent == 0:
            return latent_embd, other_state

        for idx in range(current_step, min(target_step, self.num_latent)):
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd.unsqueeze(1),
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :]
            if self.use_prj and hasattr(self.model, "prj"):
                latent_embd = self.model.prj(latent_embd.unsqueeze(1)).squeeze(1)

        new_state = {
            "tokens": tokens,
            "past_key_values": past_key_values,
            "current_step": target_step,
        }
        return latent_embd, new_state

    def rollout_from_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        continue_latents: bool = True,
        allow_grad: bool = False,
    ) -> Any:
        tokens = other_state["tokens"]
        past_key_values = self._clone_cache(other_state.get("past_key_values"))
        start_step = other_state.get("current_step", 0)
        if start_step < 0 or start_step > self.num_latent:
            raise ValueError(
                f"Invalid current_step in state: {start_step} (num_latent={self.num_latent})"
            )
        latent_embd = h_t_modified
        # We already ran `start_step` iterations in forward_until_step; finish out the
        # configured total so the ablated latent still flows into the cache before decoding.
        remaining = 0 if not continue_latents else max(self.num_latent - start_step, 0)

        # If we don't continue latents, still write the current latent into the cache once.
        if not continue_latents:
            pre_pkv = self._clone_cache(
                other_state.get("past_key_values_before_step", past_key_values)
            )
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd.unsqueeze(1),
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=pre_pkv,
                )
            past_key_values = outputs.past_key_values
            kv_after_latents = past_key_values
        else:
            for _ in range(remaining):
                with torch.set_grad_enabled(allow_grad):
                    outputs = self.model.codi(
                        inputs_embeds=latent_embd.unsqueeze(1),
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                    )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :]
                if self.use_prj and hasattr(self.model, "prj"):
                    latent_embd = self.model.prj(latent_embd.unsqueeze(1)).squeeze(1)

            kv_after_latents = past_key_values


        # Decode final answer tokens if a decoder input is provided.
        decoder_ids = tokens.get("decoder_input_ids")
        if decoder_ids is not None:
            decoder_ids = decoder_ids.to(self.device)
            with torch.set_grad_enabled(allow_grad):
                outputs = self.model.codi(
                    input_ids=decoder_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
            return outputs

        # Otherwise, generate text from the current cached state (mirror run_baseline).
        gen_kwargs = {
            "max_new_tokens": self.generation_kwargs.get("max_new_tokens", 64),
            "temperature": self.generation_kwargs.get("temperature", 0.7),
            "top_k": self.generation_kwargs.get("top_k", 0),
            "top_p": self.generation_kwargs.get("top_p", 1.0),
            "greedy": self.generation_kwargs.get("greedy", False),
        }
        training_args = self.model.training_args

        if getattr(training_args, "remove_eos", False):
            eot_ids = torch.tensor([self.model.eot_id], device=self.device)
        else:
            eot_ids = torch.tensor(
                [self.model.eot_id, self.tokenizer.eos_token_id],
                device=self.device,
            )
        eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(eot_ids).unsqueeze(0)
        eot_emb = eot_emb.expand(tokens["input_ids"].size(0), -1, -1)

        output = eot_emb
        finished = torch.zeros(tokens["input_ids"].size(0), dtype=torch.bool, device=self.device)
        pred_tokens = [[] for _ in range(tokens["input_ids"].size(0))]
        last_logits = None

        for _ in range(gen_kwargs["max_new_tokens"]):
            with torch.set_grad_enabled(allow_grad):
                out = self.model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
            past_key_values = out.past_key_values
            last_logits = out.logits
            logits = last_logits[:, -1, : self.model.codi.config.vocab_size - 1]

            if gen_kwargs["greedy"]:
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
            else:
                logits = logits / gen_kwargs["temperature"]
                if gen_kwargs["top_k"] and gen_kwargs["top_k"] > 1:
                    top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    logits[logits < min_top_k_value] = -float("inf")
                if gen_kwargs["top_p"] < 1.0:
                    sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                    if sorted_indices_to_remove.any():
                        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                        sorted_indices_to_remove[:, 0] = False
                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Ensure 1D tensor of shape [batch]
            if next_token_ids.dim() == 0:
                next_token_ids = next_token_ids.view(1)

            for b in range(len(pred_tokens)):
                if not finished[b]:
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == self.tokenizer.eos_token_id:
                        finished[b] = True
            if finished.all():
                break

            output = self.model.get_embd(self.model.codi, self.model.model_name)(
                next_token_ids
                ).unsqueeze(1).to(self.device)

        decoded = []
        for tokens in pred_tokens:
            try:
                decoded.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
            except Exception:
                # Fallback when tokenizer returns None for some token ids.
                tok_list = self.tokenizer.convert_ids_to_tokens(tokens)
                tok_list = [t for t in tok_list if isinstance(t, str)]
                decoded.append(self.tokenizer.convert_tokens_to_string(tok_list))
        return {
            "text": decoded,
            "latent": latent_embd,
            "past_key_values": past_key_values,
            "past_key_values_latents": kv_after_latents,
            "logits": last_logits,
        }

    def build_teacher_target_ids(self, answer_text: str | None) -> torch.Tensor | None:
        """
        CODI answers are typically formatted as 'The answer is: {answer}'.
        Use a configurable template when provided.
        """
        if answer_text is None:
            return None
        # Align with CODI generation style: "<bos> answer is {answer} <eos>"
        template = self.teacher_target_template or "answer is {answer}"
        text = template.format(answer=answer_text)
        try:
            tokenized = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        except Exception:
            return None
        ids = tokenized.get("input_ids")
        if ids is None:
            return None
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_tensor = torch.tensor([[eos_id]], device=ids.device)
            ids = torch.cat([ids, eos_tensor], dim=-1)
        return ids.to(self.device)

    def logits_from_latent(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Project a latent through the LM head (logit lens style).
        Note: CODI latents share the same dimensionality as the base model hidden.
        """
        if not hasattr(self.model.codi, "lm_head"):
            raise AttributeError("codi model lacks lm_head for logit lens.")
        return self.model.codi.lm_head(h_t)

    def decode_from_state(self, h_t: torch.Tensor, other_state: Dict[str, Any]) -> Any:
        """
        Decode from the current latent. Honor optional flag `decode_continue_latents`:
        - True (default): run remaining latent iterations before decoding.
        - False: inject current latent once into cache, then decode immediately.
        """
        cont = other_state.get("decode_continue_latents", True)
        return self.rollout_from_step(h_t, other_state, continue_latents=cont)

    def compute_logits(
        self,
        h_t: torch.Tensor,
        other_state: Dict[str, Any],
        target_ids: torch.Tensor,
        allow_grad: bool = False,
    ) -> torch.Tensor:
        """
        Teacher-forcing logits for a target continuation given cached prefix.
        """
        target_ids = target_ids.to(self.device)
        if target_ids.size(1) == 0:
            vocab = getattr(self.model.codi.config, "vocab_size", 0)
            return torch.empty(target_ids.size(0), 0, vocab, device=self.device)

        past_key_values = self._ensure_cache(other_state.get("past_key_values"))
        # Seed decode with the same EOT embedding used in rollout_from_step, then
        # teacher-force the gold tokens so positions align with free decoding.
        training_args = self.model.training_args
        if getattr(training_args, "remove_eos", False):
            eot_ids = torch.tensor([self.model.eot_id], device=self.device)
        else:
            eot_ids = torch.tensor(
                [self.model.eot_id, self.tokenizer.eos_token_id],
                device=self.device,
            )
        eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(eot_ids).unsqueeze(0)
        eot_emb = eot_emb.expand(target_ids.size(0), -1, -1)

        with torch.set_grad_enabled(allow_grad):
            seed_out = self.model.codi(
                inputs_embeds=eot_emb,
                use_cache=True,
                output_hidden_states=False,
                past_key_values=past_key_values,
            )
        first_logit = seed_out.logits[:, -1, :]
        if target_ids.size(1) == 1:
            return first_logit.unsqueeze(1)

        input_ids = target_ids[:, :-1]

        past_seq_len = None
        try:
            if hasattr(seed_out.past_key_values, "get_seq_length"):
                past_seq_len = seed_out.past_key_values.get_seq_length()  # type: ignore[attr-defined]
            else:
                past_seq_len = seed_out.past_key_values[0][0].shape[-2]
        except Exception:
            past_seq_len = None

        attention_mask = None
        position_ids = None
        if past_seq_len is not None:
            attention_mask = torch.ones(
                target_ids.size(0),
                past_seq_len + input_ids.size(1),
                device=self.device,
                dtype=torch.long,
            )
            position_ids = torch.arange(
                past_seq_len,
                past_seq_len + input_ids.size(1),
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)

        with torch.set_grad_enabled(allow_grad):
            outputs = self.model.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=seed_out.past_key_values,
                use_cache=False,
                output_hidden_states=False,
            )

        return torch.cat([first_logit.unsqueeze(1), outputs.logits], dim=1)
        

    def run_baseline(self, inputs: Any) -> Any:
        tokens = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_new_tokens": self.generation_kwargs.get("max_new_tokens", 64),
            "temperature": self.generation_kwargs.get("temperature", 0.7),
            "top_k": self.generation_kwargs.get("top_k", 0),
            "top_p": self.generation_kwargs.get("top_p", 1.0),
            "greedy": self.generation_kwargs.get("greedy", False),
        }
        training_args = self.model.training_args
        self.model.eval()

        with torch.no_grad():
            # encode question
            outputs = self.model.codi(
                input_ids=tokens["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=None,
                attention_mask=tokens["attention_mask"],
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if self.use_prj and hasattr(self.model, "prj"):
                latent_embd = self.model.prj(latent_embd)

            # latent iterations
            for _ in range(training_args.inf_latent_iterations):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj and hasattr(self.model, "prj"):
                    latent_embd = self.model.prj(latent_embd)

            # prepare EOT embedding
            if getattr(training_args, "remove_eos", False):
                eot_ids = torch.tensor([self.model.eot_id], device=self.device)
            else:
                eot_ids = torch.tensor(
                    [self.model.eot_id, self.tokenizer.eos_token_id],
                    device=self.device,
                )
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                eot_ids
            ).unsqueeze(0)
            eot_emb = eot_emb.expand(tokens["input_ids"].size(0), -1, -1)

            output = eot_emb
            finished = torch.zeros(tokens["input_ids"].size(0), dtype=torch.bool, device=self.device)
            pred_tokens = [[] for _ in range(tokens["input_ids"].size(0))]

            for _ in range(gen_kwargs["max_new_tokens"]):
                out = self.model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : self.model.codi.config.vocab_size - 1]

                if gen_kwargs["greedy"]:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits = logits / gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] and gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Ensure 1D tensor of shape [batch]
                if next_token_ids.dim() == 0:
                    next_token_ids = next_token_ids.view(1)

                for b in range(len(pred_tokens)):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == self.tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break

                output = self.model.get_embd(self.model.codi, self.model.model_name)(
                    next_token_ids
                ).unsqueeze(1).to(self.device)

            decoded = [_safe_decode(self.tokenizer, tokens) for tokens in pred_tokens]
            return decoded
