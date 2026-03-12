from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from common.model_interface import LatentReasoningModel
from common.model_registry import register_model
from external.coconut.coconut import Coconut


@register_model("coconut")
class CoconutWrapper(LatentReasoningModel):
    """
    Wrapper around the Coconut latent generation procedure to expose h_t.

    The wrapper follows the iterative latent filling scheme from `external/coconut/coconut.py`.
    At `forward_until_step`, latents before the requested step are materialized and cached,
    while the target step is returned for ablation without being injected yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = None
        self.base_model = None
        self.coconut_model: Coconut | None = None
        self.latent_token_id: int | None = None
        self.start_latent_id: int | None = None
        self.end_latent_id: int | None = None
        self.eos_token_id: int | None = None
        self.generation_kwargs: Dict[str, Any] = {}
        self.num_latent_placeholders: int = 0
        self.use_coconut_question_only: bool = False
        self.align_latent_padding: bool = False

    def _ensure_cache(self, past_key_values: Any) -> Any:
        """Convert legacy tuple/list caches to the new HF Cache interface."""
        if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def load_from_config(self, config: Dict[str, Any]) -> None:
        base_path = config.get("base_model_name_or_path") or config.get(
            "model_name_or_path"
        )
        if base_path is None:
            raise ValueError("CoconutWrapper requires 'base_model_name_or_path'.")
        tokenizer_name = config.get("tokenizer_name_or_path", base_path)
        self.device = torch.device(config.get("device", self.device))
        self.teacher_target_template = config.get("teacher_target_template", self.teacher_target_template)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_path, trust_remote_code=True
        ).to(self.device)
        # Disable dropout for reproducible logits during teacher-forcing/ablations.
        self.base_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.generation_kwargs = config.get("generation_kwargs", {})
        self.num_latent_placeholders = int(config.get("num_latent_placeholders", 0))
        self.use_coconut_question_only = bool(config.get("use_coconut_question_only", False))
        self.align_latent_padding = bool(config.get("align_latent_padding", False))

        # Ensure pad_token exists (HF padding needs it); default to eos_token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Match official recipe: right padding, eos as pad token.
        self.tokenizer.padding_side = "right"

        # Add Coconut special tokens if requested (matches official training script).
        add_latent_tokens = config.get("add_latent_tokens", True)
        latent_token_str = config.get("latent_token_str", "<|latent|>")
        start_latent_token_str = config.get("start_latent_token_str", "<|start-latent|>")
        end_latent_token_str = config.get("end_latent_token_str", "<|end-latent|>")
        # Stash strings for downstream template construction
        self.latent_token_str = latent_token_str
        self.start_latent_token_str = start_latent_token_str
        self.end_latent_token_str = end_latent_token_str
        newly_added_tokens: List[int] = []
        if add_latent_tokens:
            new_tokens = []
            for tok in [start_latent_token_str, end_latent_token_str, latent_token_str]:
                if tok not in self.tokenizer.get_vocab():
                    new_tokens.append(tok)
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                newly_added_tokens = [
                    self.tokenizer.convert_tokens_to_ids(tok) for tok in new_tokens
                ]
        # Resize if pad token was newly added by HF or latent tokens were added
        if len(self.tokenizer) != self.base_model.get_input_embeddings().num_embeddings:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Resolve token ids (prefer explicit config, otherwise from tokenizer).
        self.latent_token_id = int(
            config.get(
                "latent_token_id",
                self.tokenizer.convert_tokens_to_ids(latent_token_str),
            )
        )
        self.start_latent_id = int(
            config.get(
                "start_latent_id",
                self.tokenizer.convert_tokens_to_ids(start_latent_token_str),
            )
        )
        self.end_latent_id = int(
            config.get(
                "end_latent_id",
                self.tokenizer.convert_tokens_to_ids(end_latent_token_str),
            )
        )
        self.eos_token_id = int(
            config.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", 0))
        )

        # Initialise newly added latent tokens with a stable embedding (mirrors run.py).
        init_token_str = config.get("latent_init_token", "<<")
        target_id = self.tokenizer.convert_tokens_to_ids(init_token_str)
        if target_id is None or target_id == self.tokenizer.unk_token_id:
            target_id = self.tokenizer.eos_token_id
        embed = self.base_model.get_input_embeddings()
        if embed is not None and newly_added_tokens:
            with torch.no_grad():
                target_vec = embed.weight.data[target_id].clone()
                for tok_id in newly_added_tokens:
                    if tok_id is not None and tok_id < embed.weight.data.size(0):
                        embed.weight.data[tok_id] = target_vec
                if hasattr(self.base_model, "lm_head") and hasattr(
                    self.base_model.lm_head, "weight"
                ):
                    lm_w = self.base_model.lm_head.weight
                    if lm_w.shape[0] == embed.weight.data.shape[0]:
                        for tok_id in newly_added_tokens:
                            if tok_id is not None and tok_id < lm_w.size(0):
                                lm_w.data[tok_id] = lm_w.data[target_id]

        self.coconut_model = Coconut(
            self.base_model,
            latent_token_id=self.latent_token_id,
            start_latent_id=self.start_latent_id,
            end_latent_id=self.end_latent_id,
            eos_token_id=self.eos_token_id,
        ).to(self.device)
        self.coconut_model.eval()

        # Optionally load a Coconut checkpoint (state dict saved by external/coconut).
        ckpt_path = config.get("checkpoint_path")
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            _ = self.coconut_model.load_state_dict(state_dict, strict=False)

    def _prepare_inputs(self, inputs: Any) -> Dict[str, torch.Tensor]:
        def _extract_question(text: str) -> str:
            if not self.use_coconut_question_only:
                return text
            question = text
            if "Answer:" in question:
                question = question.split("Answer:")[0]
            if "Question:" in question:
                question = question.split("Question:")[-1]
            question = question.strip()
            return question + "\n"

        if isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("inputs list is empty.")
            if all(isinstance(x, str) for x in inputs):
                tokenizer_inputs: Any = [_extract_question(x) for x in inputs]
            else:
                raise TypeError("inputs list must contain strings only.")
        else:
            if isinstance(inputs, str):
                tokenizer_inputs = _extract_question(inputs)
            else:
                tokenizer_inputs = inputs

        if isinstance(tokenizer_inputs, str):
            tokens = self.tokenizer(
                tokenizer_inputs, return_tensors="pt", padding=False, truncation=True
            )
        elif isinstance(tokenizer_inputs, list):
            # Tokenize per-sample first. When align_latent_padding is enabled, pad to the
            # same base length before appending latent placeholders to align positions.
            encoded_list = [
                self.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
                for text in tokenizer_inputs
            ]
            input_ids_list = []
            attention_mask_list = []
            position_ids_list = []
            max_base_len = 0
            if self.align_latent_padding:
                max_base_len = max(enc["input_ids"].size(1) for enc in encoded_list)
            for enc in encoded_list:
                ids = enc["input_ids"][0]
                mask = enc.get("attention_mask")
                if mask is None:
                    mask = torch.ones_like(ids)
                else:
                    mask = mask[0]
                pos = enc.get("position_ids")
                if pos is None:
                    pos = torch.arange(0, ids.size(0), dtype=torch.long)
                else:
                    pos = pos[0]

                if self.align_latent_padding and max_base_len > ids.size(0):
                    pad_right = max_base_len - ids.size(0)
                    pad_id = int(self.tokenizer.pad_token_id)
                    ids = torch.cat(
                        [ids, torch.full((pad_right,), pad_id, dtype=ids.dtype)],
                        dim=0,
                    )
                    mask = torch.cat(
                        [mask, torch.zeros((pad_right,), dtype=mask.dtype)],
                        dim=0,
                    )

                # Append latent placeholders per sample before padding.
                already_has_latent = (
                    self.latent_token_id is not None
                    and (ids == self.latent_token_id).any().item()
                )
                if self.num_latent_placeholders > 0 and not already_has_latent:
                    latent_sequence = [self.start_latent_id] + [self.latent_token_id] * self.num_latent_placeholders + [self.end_latent_id]
                    latent_tensor = torch.tensor(latent_sequence, dtype=ids.dtype)
                    ids = torch.cat((ids, latent_tensor), dim=0)
                    lat_len = latent_tensor.size(0)
                    mask = torch.cat(
                        (mask, torch.ones(lat_len, dtype=mask.dtype)),
                        dim=0,
                    )
                    if not self.align_latent_padding:
                        last_pos = pos[-1:] if pos.numel() > 0 else torch.tensor([-1], dtype=pos.dtype)
                        pos_extension = last_pos + torch.arange(1, lat_len + 1, dtype=pos.dtype)
                        pos = torch.cat((pos, pos_extension), dim=0)

                if self.align_latent_padding:
                    pos = torch.arange(0, ids.size(0), dtype=torch.long)

                input_ids_list.append(ids)
                attention_mask_list.append(mask)
                position_ids_list.append(pos)

            max_len = max(t.size(0) for t in input_ids_list) if input_ids_list else 0
            pad_id = int(self.tokenizer.pad_token_id)
            padded_ids = []
            padded_mask = []
            padded_pos = []
            for ids, mask, pos in zip(input_ids_list, attention_mask_list, position_ids_list):
                pad_right = max_len - ids.size(0)
                if pad_right > 0:
                    ids = torch.cat(
                        [ids, torch.full((pad_right,), pad_id, dtype=ids.dtype)],
                        dim=0,
                    )
                    mask = torch.cat(
                        [mask, torch.zeros((pad_right,), dtype=mask.dtype)],
                        dim=0,
                    )
                    pos = torch.cat(
                        [pos, torch.zeros((pad_right,), dtype=pos.dtype)],
                        dim=0,
                    )
                padded_ids.append(ids)
                padded_mask.append(mask)
                padded_pos.append(pos)
            tokens = {
                "input_ids": torch.stack(padded_ids, dim=0),
                "attention_mask": torch.stack(padded_mask, dim=0),
                "position_ids": torch.stack(padded_pos, dim=0),
            }
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
            seq_length = tokens["input_ids"].size(1)
            batch_size = tokens["input_ids"].size(0)
            tokens["position_ids"] = torch.arange(
                0, seq_length, dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Append latent placeholders if configured (to match Coconut training format).
        if not isinstance(tokenizer_inputs, list):
            already_has_latent = (
                self.latent_token_id is not None
                and (tokens["input_ids"] == self.latent_token_id).any().item()
            )
            if self.num_latent_placeholders > 0 and not already_has_latent:
                bs = tokens["input_ids"].size(0)
                latent_sequence = [self.start_latent_id] + [self.latent_token_id] * self.num_latent_placeholders + [self.end_latent_id]
                latent_tensor = torch.tensor(latent_sequence, device=self.device).unsqueeze(0).expand(bs, -1)
                tokens["input_ids"] = torch.cat((tokens["input_ids"], latent_tensor), dim=1)
                # extend attention_mask and position_ids
                lat_len = latent_tensor.size(1)
                tokens["attention_mask"] = torch.cat(
                    (tokens["attention_mask"], torch.ones(bs, lat_len, device=self.device, dtype=tokens["attention_mask"].dtype)),
                    dim=1,
                )
                last_pos = tokens["position_ids"][:, -1:]
                pos_extension = last_pos + torch.arange(1, lat_len + 1, device=self.device).unsqueeze(0)
                tokens["position_ids"] = torch.cat((tokens["position_ids"], pos_extension), dim=1)
        if self.align_latent_padding and tokens["input_ids"].size(0) > 1:
            pad_id = int(self.tokenizer.pad_token_id)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            position_ids = tokens["position_ids"]
            earliest_latent = []
            for row in input_ids:
                latent_positions = (row == self.latent_token_id).nonzero(as_tuple=False)
                if latent_positions.numel() == 0:
                    earliest_latent.append(None)
                else:
                    earliest_latent.append(int(latent_positions[0].item()))
            max_earliest = max([idx for idx in earliest_latent if idx is not None], default=0)
            padded_ids = []
            padded_mask = []
            padded_pos = []
            for idx, row in enumerate(input_ids):
                pad_left = 0
                if earliest_latent[idx] is not None:
                    pad_left = max_earliest - earliest_latent[idx]
                if pad_left > 0:
                    left_ids = torch.full((pad_left,), pad_id, device=self.device, dtype=row.dtype)
                    left_mask = torch.zeros((pad_left,), device=self.device, dtype=attention_mask.dtype)
                    left_pos = torch.zeros((pad_left,), device=self.device, dtype=position_ids.dtype)
                    row = torch.cat([left_ids, row], dim=0)
                    mask_row = torch.cat([left_mask, attention_mask[idx]], dim=0)
                    pos_row = torch.cat([left_pos, position_ids[idx] + pad_left], dim=0)
                else:
                    mask_row = attention_mask[idx]
                    pos_row = position_ids[idx]
                padded_ids.append(row)
                padded_mask.append(mask_row)
                padded_pos.append(pos_row)
            max_len = max(r.size(0) for r in padded_ids)
            final_ids = []
            final_mask = []
            final_pos = []
            for row, mask_row, pos_row in zip(padded_ids, padded_mask, padded_pos):
                pad_right = max_len - row.size(0)
                if pad_right > 0:
                    right_ids = torch.full((pad_right,), pad_id, device=self.device, dtype=row.dtype)
                    right_mask = torch.zeros((pad_right,), device=self.device, dtype=mask_row.dtype)
                    right_pos = torch.zeros((pad_right,), device=self.device, dtype=pos_row.dtype)
                    row = torch.cat([row, right_ids], dim=0)
                    mask_row = torch.cat([mask_row, right_mask], dim=0)
                    pos_row = torch.cat([pos_row, right_pos], dim=0)
                final_ids.append(row)
                final_mask.append(mask_row)
                final_pos.append(pos_row)
            tokens["input_ids"] = torch.stack(final_ids, dim=0)
            tokens["attention_mask"] = torch.stack(final_mask, dim=0)
            tokens["position_ids"] = torch.stack(final_pos, dim=0)
        return tokens

    def _inject_latents(
        self,
        inputs_embeds: torch.Tensor,
        filling_indices: List[Tuple[int, int]],
        latent_values: List[torch.Tensor],
    ) -> torch.Tensor:
        """Replace latent token positions in `inputs_embeds`."""
        tensor_list = [
            [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
            for batch_idx in range(inputs_embeds.shape[0])
        ]
        for (batch_idx, token_idx), vec in zip(filling_indices, latent_values):
            tensor_list[batch_idx][token_idx] = vec
        stacked = torch.stack(
            [torch.stack(tensor_list[batch_idx]) for batch_idx in range(len(tensor_list))]
        )
        return stacked

    def _gather_latent_values(
        self,
        hidden_states: torch.Tensor,
        filling_indices: List[Tuple[int, int]],
        hidden_states_offset: int,
    ) -> List[torch.Tensor]:
        values = []
        for batch_idx, token_idx in filling_indices:
            values.append(
                hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ].detach()
            )
        return values

    def forward_until_step(
        self, inputs: Any, step: int, allow_grad: bool = False
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        tokens = self._prepare_inputs(inputs)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        position_ids = tokens["position_ids"]

        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            # Fall back to per-sample latent filling when latent positions differ across the batch.
            earliest = []
            for row in input_ids:
                latent_positions = (row == self.latent_token_id).nonzero(as_tuple=False)
                if latent_positions.numel() == 0:
                    earliest.append(None)
                else:
                    earliest.append(int(latent_positions[0].item()))
            unique_positions = {pos for pos in earliest if pos is not None}
            if len(unique_positions) > 1:
                h_list = []
                state_list = []
                for sample in inputs:
                    h_i, state_i = self.forward_until_step(sample, step)
                    h_list.append(h_i)
                    state_list.append(state_i)
                h_batch = torch.cat(h_list, dim=0)
                return h_batch, {"per_sample_states": state_list}

        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if len(latent_lists) > 0 else 0

        # Normalize step: enforce 1-based latents.
        step_label = step
        if isinstance(step, str):
            raise ValueError(f"Unsupported step label: {step}")
        if step < 1:
            raise ValueError(f"Latent steps are 1-based; got {step}")
        target_pass = int(step) - 1

        if step > max_n_latents:
            raise ValueError(
                f"Requested step {step} exceeds available latent steps ({max_n_latents})."
            )
        if max_n_latents == 0:
            raise ValueError("No latent tokens present.")

        if max_n_latents == 0:
            # Fallback to the last token hidden state.
            with torch.set_grad_enabled(allow_grad):
                outputs = self.base_model(
                    **tokens, output_hidden_states=True, use_cache=True
                )
            h_t = outputs.hidden_states[-1][:, -1, :]
            return h_t, {
                "tokens": tokens,
                "step_idx": input_ids.size(1) - 1,
                "no_latent": True,
                "step_label": step_label,
            }

        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        inputs_embeds = self.coconut_model.embedding(input_ids)
        past_key_values = None

        logits_so_far: List[torch.Tensor] = []

        for pass_idx in range(max_n_latents):
            if past_key_values is None:
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = self._ensure_cache(
                    [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in past_key_values
                    ]
                )
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits_so_far.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            latent_vectors = self._gather_latent_values(
                hidden_states, filling_indices, hidden_states_offset
            )

            if pass_idx == target_pass:
                first_logit = None
                if logits_so_far:
                    last_logits = logits_so_far[-1]
                    if last_logits is not None:
                        first_logit = last_logits[:, -1, :]
                other_state = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "latent_lists": latent_lists,
                    "inputs_embeds": inputs_embeds,
                    "past_key_values": past_key_values,
                    "next_compute_range": next_compute_range,
                    "max_n_latents": max_n_latents,
                    "pass_idx": pass_idx,
                    "logits": logits_so_far,
                    "hidden_states_offset": hidden_states_offset,
                    "step_label": step_label,
                    "first_logit": first_logit,
                }
                h_batch = torch.zeros(
                    input_ids.size(0), hidden_states.size(-1), device=self.device
                )
                for idx, (batch_idx, _) in enumerate(filling_indices):
                    h_batch[batch_idx] = latent_vectors[idx]
                return h_batch, other_state

            # Inject latent for this pass and continue.
            inputs_embeds = self._inject_latents(
                inputs_embeds, filling_indices, latent_vectors
            )

        raise RuntimeError(f"Failed to reach target latent step {step}.")

    def rollout_from_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        continue_latents: bool = True,
        allow_grad: bool = False,
    ) -> Any:
        def _greedy_decode_batch(
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            last_logits: torch.Tensor,
            input_ids: torch.Tensor,
        ) -> Dict[str, Any]:
            batch_size = inputs_embeds.size(0)
            max_new_tokens = self.generation_kwargs.get("max_new_tokens", 64)
            tokens_list = [input_ids[i].detach().tolist() for i in range(batch_size)]
            embedding = (
                self.coconut_model.embedding
                if self.coconut_model
                else self.base_model.get_input_embeddings()
            )
            eos_id = self.eos_token_id
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            gen_attention_mask = attention_mask
            gen_position_ids = position_ids

            next_token_ids = torch.argmax(last_logits[:, -1, :], dim=-1)
            for i in range(batch_size):
                tokens_list[i].append(int(next_token_ids[i].item()))
            finished |= next_token_ids.eq(eos_id)
            new_token_embed = embedding(next_token_ids.to(self.device)).view(batch_size, 1, -1)
            new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
            next_pos = gen_attention_mask.sum(dim=1, keepdim=True)
            gen_position_ids = torch.cat((gen_position_ids, next_pos), dim=1)
            gen_attention_mask = torch.cat(
                (gen_attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=gen_attention_mask.dtype)),
                dim=1,
            )

            for _ in range(max_new_tokens - 1):
                if finished.all():
                    break
                out = self.base_model(
                    inputs_embeds=new_inputs_embeds,
                    attention_mask=gen_attention_mask,
                    position_ids=gen_position_ids,
                )
                next_token_ids = torch.argmax(out.logits[:, -1, :], dim=-1)
                next_token_ids = torch.where(
                    finished, torch.tensor(eos_id, device=self.device), next_token_ids
                )
                for i in range(batch_size):
                    if not finished[i]:
                        tokens_list[i].append(int(next_token_ids[i].item()))
                finished |= next_token_ids.eq(eos_id)
                new_token_embed = embedding(next_token_ids.to(self.device)).view(batch_size, 1, -1)
                new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
                next_pos = gen_attention_mask.sum(dim=1, keepdim=True)
                gen_position_ids = torch.cat((gen_position_ids, next_pos), dim=1)
                gen_attention_mask = torch.cat(
                    (gen_attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=gen_attention_mask.dtype)),
                    dim=1,
                )

            decoded = [self.tokenizer.decode(toks, skip_special_tokens=True) for toks in tokens_list]
            return {
                "text": decoded,
                "latent": h_t_modified,
                "logits": last_logits,
                "first_logit": last_logits[:, -1, :] if last_logits is not None else None,
            }

        def _finalize_state(
            h_t_mod: torch.Tensor, state: Dict[str, Any], continue_latents: bool
        ) -> Dict[str, Any]:
            if state.get("no_latent"):
                tokens = state.get("tokens", {})
                input_ids_local = tokens.get("input_ids")
                attention_mask_local = tokens.get("attention_mask")
                position_ids_local = tokens.get("position_ids")
                if (
                    input_ids_local is None
                    or attention_mask_local is None
                    or position_ids_local is None
                ):
                    raise ValueError("Missing token fields for no_latent state.")
                inputs_embeds_local = self.coconut_model.embedding(input_ids_local)
                with torch.set_grad_enabled(allow_grad):
                    outputs_local = self.base_model(
                        inputs_embeds=inputs_embeds_local,
                        attention_mask=attention_mask_local,
                        position_ids=position_ids_local,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                return {
                    "inputs_embeds": inputs_embeds_local,
                    "attention_mask": attention_mask_local,
                    "position_ids": position_ids_local,
                    "last_logits": outputs_local.logits[:, -1, :],
                    "past_key_values": outputs_local.past_key_values,
                }

            input_ids_local = state["input_ids"]
            attention_mask_local = state["attention_mask"]
            position_ids_local = state["position_ids"]
            latent_lists_local = state["latent_lists"]
            inputs_embeds_local = state["inputs_embeds"]
            past_key_values_local = state["past_key_values"]
            next_compute_range_local = state["next_compute_range"]
            max_n_latents_local = state["max_n_latents"]
            pass_idx_local = state["pass_idx"]
            logits_so_far_local = state.get("logits", [])
            hidden_states_offset_local = state.get("hidden_states_offset", 0)

            if pass_idx_local < 0:
                raise ValueError(
                    f"Invalid pass_idx {pass_idx_local} for max_n_latents={max_n_latents_local}"
                )
            if pass_idx_local >= max_n_latents_local:
                pass_idx_local = max_n_latents_local

            if pass_idx_local < max_n_latents_local:
                filling_indices = [
                    (instance_idx, mask_list[pass_idx_local])
                    for instance_idx, mask_list in enumerate(latent_lists_local)
                    if len(mask_list) > pass_idx_local
                ]
                latent_vectors = [
                    h_t_mod[batch_idx].to(self.device) for batch_idx, _ in filling_indices
                ]
                inputs_embeds_local = self._inject_latents(
                    inputs_embeds_local, filling_indices, latent_vectors
                )

            if not continue_latents:
                max_n_latents_local = pass_idx_local

            for idx in range(pass_idx_local, max_n_latents_local):
                if past_key_values_local is None:
                    with torch.set_grad_enabled(allow_grad):
                        outputs_local = self.base_model(
                            inputs_embeds=inputs_embeds_local[
                                :, next_compute_range_local[0] : next_compute_range_local[1], :
                            ],
                            attention_mask=attention_mask_local[
                                :, next_compute_range_local[0] : next_compute_range_local[1]
                            ],
                            position_ids=position_ids_local[
                                :, next_compute_range_local[0] : next_compute_range_local[1]
                            ],
                            output_hidden_states=True,
                            use_cache=True,
                        )
                    hidden_states_offset_local = 0
                else:
                    past_key_values_local = self._ensure_cache(
                        [
                            (
                                k[:, :, : next_compute_range_local[0], :],
                                v[:, :, : next_compute_range_local[0], :],
                            )
                            for k, v in past_key_values_local
                        ]
                    )
                    with torch.set_grad_enabled(allow_grad):
                        outputs_local = self.base_model(
                            inputs_embeds=inputs_embeds_local[
                                :, next_compute_range_local[0] : next_compute_range_local[1], :
                            ],
                            attention_mask=attention_mask_local[:, : next_compute_range_local[1]],
                            position_ids=position_ids_local[
                                :, next_compute_range_local[0] : next_compute_range_local[1]
                            ],
                            past_key_values=past_key_values_local,
                            output_hidden_states=True,
                            use_cache=True,
                        )
                    hidden_states_offset_local = next_compute_range_local[0]

                logits_so_far_local.append(outputs_local.logits)
                hidden_states_local = outputs_local.hidden_states[-1]
                past_key_values_local = outputs_local.past_key_values
                next_compute_range_local = (
                    next_compute_range_local[1],
                    (
                        input_ids_local.shape[1]
                        if idx + 1 >= max_n_latents_local
                        else next_compute_range_local[1] + 1
                    ),
                )

                if idx + 1 < max_n_latents_local:
                    filling_indices = [
                        (instance_idx, mask_list[idx + 1])
                        for instance_idx, mask_list in enumerate(latent_lists_local)
                        if len(mask_list) > idx + 1
                    ]
                    latent_vectors = self._gather_latent_values(
                        hidden_states_local, filling_indices, hidden_states_offset_local
                    )
                    inputs_embeds_local = self._inject_latents(
                        inputs_embeds_local, filling_indices, latent_vectors
                    )

            outputs_local = None
            if next_compute_range_local[1] > next_compute_range_local[0]:
                with torch.set_grad_enabled(allow_grad):
                    outputs_local = self.base_model(
                        inputs_embeds=inputs_embeds_local[
                            :, next_compute_range_local[0] : next_compute_range_local[1], :
                        ],
                        attention_mask=attention_mask_local[:, : next_compute_range_local[1]],
                        position_ids=position_ids_local[:, next_compute_range_local[0] : next_compute_range_local[1]],
                        past_key_values=(
                            self._ensure_cache(
                                [
                                    (
                                        k[:, :, : next_compute_range_local[0], :],
                                        v[:, :, : next_compute_range_local[0], :],
                                    )
                                    for k, v in past_key_values_local
                                ]
                            )
                            if past_key_values_local
                            else None
                        ),
                        output_hidden_states=True,
                        use_cache=True,
                    )
                logits_so_far_local.append(outputs_local.logits)
                past_key_values_local = outputs_local.past_key_values
            last_logits_local = (
                outputs_local.logits if outputs_local is not None else logits_so_far_local[-1]
            )
            return {
                "inputs_embeds": inputs_embeds_local,
                "attention_mask": attention_mask_local,
                "position_ids": position_ids_local,
                "last_logits": last_logits_local[:, -1, :],
                "past_key_values": past_key_values_local,
            }

        # If no latent tokens were present, ablation is identical to baseline.
        if other_state.get("no_latent"):
            return self.run_baseline(other_state.get("tokens", {}))
        if other_state.get("per_sample_states"):
            per_states = other_state["per_sample_states"]
            if not isinstance(per_states, list) or h_t_modified.size(0) != len(per_states):
                raise ValueError("per_sample_states length must match batch size.")

            per_inputs_embeds = []
            per_attention = []
            per_position = []
            per_last_logits = []
            per_pkv = []
            for i, state in enumerate(per_states):
                finalized = _finalize_state(h_t_modified[i : i + 1], state, continue_latents)
                per_inputs_embeds.append(finalized["inputs_embeds"])
                per_attention.append(finalized["attention_mask"])
                per_position.append(finalized["position_ids"])
                per_last_logits.append(finalized["last_logits"])
                per_pkv.append(finalized["past_key_values"])

            max_len = max(t.size(1) for t in per_inputs_embeds)
            per_input_ids = [state["input_ids"] for state in per_states]
            max_ids_len = max(t.size(1) for t in per_input_ids)
            pad_id = (
                getattr(self.tokenizer, "pad_token_id", None)
                if self.tokenizer is not None
                else None
            )
            if pad_id is None:
                pad_id = self.eos_token_id if self.eos_token_id is not None else 0
            padded_embeds = []
            padded_masks = []
            padded_pos = []
            padded_ids = []
            for embeds, mask, pos in zip(per_inputs_embeds, per_attention, per_position):
                pad_right = max_len - embeds.size(1)
                if pad_right > 0:
                    pad_embed = torch.zeros(
                        (embeds.size(0), pad_right, embeds.size(2)),
                        device=embeds.device,
                        dtype=embeds.dtype,
                    )
                    embeds = torch.cat((embeds, pad_embed), dim=1)
                    pad_mask = torch.zeros(
                        (mask.size(0), pad_right),
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    mask = torch.cat((mask, pad_mask), dim=1)
                    pad_pos = torch.zeros(
                        (pos.size(0), pad_right),
                        device=pos.device,
                        dtype=pos.dtype,
                    )
                    pos = torch.cat((pos, pad_pos), dim=1)
                padded_embeds.append(embeds)
                padded_masks.append(mask)
                padded_pos.append(pos)
            for ids in per_input_ids:
                pad_right = max_ids_len - ids.size(1)
                if pad_right > 0:
                    pad_ids = torch.full(
                        (ids.size(0), pad_right),
                        pad_id,
                        device=ids.device,
                        dtype=ids.dtype,
                    )
                    ids = torch.cat((ids, pad_ids), dim=1)
                padded_ids.append(ids)

            inputs_embeds = torch.cat(padded_embeds, dim=0)
            attention_mask = torch.cat(padded_masks, dim=0)
            position_ids = torch.cat(padded_pos, dim=0)
            input_ids = torch.cat(padded_ids, dim=0)
            # per_last_logits entries are [1, vocab]; stack to [B, vocab], then add seq dim.
            last_logits = torch.cat(per_last_logits, dim=0).unsqueeze(1)

            # Pad and stack past_key_values for teacher-forcing reuse.
            if per_pkv and per_pkv[0] is not None:
                num_layers = len(per_pkv[0])
                batched_pkv = []
                for layer_idx in range(num_layers):
                    k_list = []
                    v_list = []
                    for pkv in per_pkv:
                        k, v = pkv[layer_idx]
                        seq_len = k.size(2)
                        if seq_len < max_len:
                            pad_len = max_len - seq_len
                            pad_k = torch.zeros(
                                (k.size(0), k.size(1), pad_len, k.size(3)),
                                device=k.device,
                                dtype=k.dtype,
                            )
                            pad_v = torch.zeros(
                                (v.size(0), v.size(1), pad_len, v.size(3)),
                                device=v.device,
                                dtype=v.dtype,
                            )
                            k = torch.cat((k, pad_k), dim=2)
                            v = torch.cat((v, pad_v), dim=2)
                        k_list.append(k)
                        v_list.append(v)
                    batched_pkv.append((torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)))
            else:
                batched_pkv = None
            decoded = _greedy_decode_batch(
                inputs_embeds,
                attention_mask,
                position_ids,
                last_logits,
                input_ids,
            )
            decoded["past_key_values"] = batched_pkv
            decoded["past_key_values_latents"] = batched_pkv
            return decoded

        input_ids = other_state["input_ids"]
        attention_mask = other_state["attention_mask"]
        position_ids = other_state["position_ids"]
        latent_lists = other_state["latent_lists"]
        inputs_embeds = other_state["inputs_embeds"]
        past_key_values = other_state["past_key_values"]
        next_compute_range = other_state["next_compute_range"]
        max_n_latents = other_state["max_n_latents"]
        pass_idx = other_state["pass_idx"]
        logits_so_far: List[torch.Tensor] = other_state.get("logits", [])
        hidden_states_offset = other_state.get("hidden_states_offset", 0)
        if pass_idx < 0 or pass_idx >= max_n_latents:
            raise ValueError(
                f"Invalid pass_idx {pass_idx} for max_n_latents={max_n_latents}"
            )

        # Fill the ablated latent for the current step.
        filling_indices = [
            (instance_idx, mask_list[pass_idx])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        latent_vectors = [
            h_t_modified[batch_idx].to(self.device) for batch_idx, _ in filling_indices
        ]
        inputs_embeds = self._inject_latents(
            inputs_embeds, filling_indices, latent_vectors
        )

        # Optionally continue iterative rollout for remaining latents.
        if not continue_latents:
            max_n_latents = pass_idx

        for idx in range(pass_idx, max_n_latents):
            if past_key_values is None:
                with torch.set_grad_enabled(allow_grad):
                    outputs = self.base_model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                hidden_states_offset = 0
            else:
                past_key_values = self._ensure_cache(
                    [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in past_key_values
                    ]
                )
                with torch.set_grad_enabled(allow_grad):
                    outputs = self.base_model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                hidden_states_offset = next_compute_range[0]

            logits_so_far.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # Prepare latents for the next step if any remain.
            if idx + 1 < max_n_latents:
                filling_indices = [
                    (instance_idx, mask_list[idx + 1])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > idx + 1
                ]
                latent_vectors = self._gather_latent_values(
                    hidden_states, filling_indices, hidden_states_offset
                )
                inputs_embeds = self._inject_latents(
                    inputs_embeds, filling_indices, latent_vectors
                )

        # Final pass to decode remaining tokens in the prompt (before free generation).
        outputs = None
        if next_compute_range[1] > next_compute_range[0]:
            with torch.set_grad_enabled(allow_grad):
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=(
                        self._ensure_cache(
                            [
                                (
                                    k[:, :, : next_compute_range[0], :],
                                    v[:, :, : next_compute_range[0], :],
                                )
                                for k, v in past_key_values
                            ]
                        )
                        if past_key_values
                        else None
                    ),
                    output_hidden_states=True,
                    use_cache=True,
                )
            logits_so_far.append(outputs.logits)
            # Capture cache after the full latent-filled prompt; this is what teacher forcing should reuse.
            past_key_values = outputs.past_key_values
        last_logits = outputs.logits if outputs is not None else logits_so_far[-1]
        batch_size = inputs_embeds.size(0)
        if batch_size > 1:
            decoded = _greedy_decode_batch(
                inputs_embeds, attention_mask, position_ids, last_logits, input_ids
            )
            decoded["past_key_values"] = past_key_values
            decoded["past_key_values_latents"] = past_key_values
            return decoded
        # Greedy decode following the official Coconut.generate logic (batch_size=1).
        max_new_tokens = self.generation_kwargs.get("max_new_tokens", 64)
        tokens_list = input_ids[0].detach().tolist()
        embedding = self.coconut_model.embedding if self.coconut_model else self.base_model.get_input_embeddings()

        # Use last token logits to get the first generated token.
        next_token = torch.argmax(last_logits[0, -1]).item()
        tokens_list.append(next_token)
        new_token_embed = embedding(torch.tensor(next_token, device=self.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            out = self.base_model(inputs_embeds=new_inputs_embeds)
            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens_list.append(next_token)
            new_token_embed = embedding(torch.tensor(next_token, device=self.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        decoded = self.tokenizer.decode(tokens_list, skip_special_tokens=True)
        return {
            "text": [decoded],
            "latent": h_t_modified,
            "logits": last_logits,
            "first_logit": last_logits[:, -1, :] if last_logits is not None else None,
            # Cache after finishing latent filling + prompt tokens (pre free-generation).
            "past_key_values": past_key_values,
            "past_key_values_latents": past_key_values,
        }

    def rollout_to_step(
        self,
        h_t_modified: torch.Tensor,
        other_state: Dict[str, Any],
        target_step: int,
        allow_grad: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Continue latent filling from the current pass up to target_step (no decoding).
        Returns the latent at target_step and the updated state (past_key_values etc.).
        """
        if other_state.get("no_latent"):
            return h_t_modified, other_state

        input_ids = other_state["input_ids"]
        attention_mask = other_state["attention_mask"]
        position_ids = other_state["position_ids"]
        latent_lists = other_state["latent_lists"]
        inputs_embeds = other_state["inputs_embeds"]
        past_key_values = other_state["past_key_values"]
        next_compute_range = other_state["next_compute_range"]
        max_n_latents = other_state["max_n_latents"]
        pass_idx = other_state["pass_idx"]
        logits_so_far: List[torch.Tensor] = other_state.get("logits", [])
        hidden_states_offset = other_state.get("hidden_states_offset", 0)
        if target_step < 1:
            raise ValueError(f"target_step must be >=1, got {target_step}")
        if target_step > max_n_latents:
            raise ValueError(
                f"target_step {target_step} exceeds available latents ({max_n_latents})"
            )
        if pass_idx < 0 or pass_idx >= max_n_latents:
            raise ValueError(
                f"Invalid pass_idx {pass_idx} for max_n_latents={max_n_latents}"
            )

        # If target_step already reached, return current latent.
        if target_step <= pass_idx:
            return h_t_modified, other_state

        # Fill the ablated latent for the current step.
        filling_indices = [
            (instance_idx, mask_list[pass_idx])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        latent_vectors = [
            h_t_modified[batch_idx].to(self.device) for batch_idx, _ in filling_indices
        ]
        inputs_embeds = self._inject_latents(
            inputs_embeds, filling_indices, latent_vectors
        )

        for idx in range(pass_idx, min(target_step, max_n_latents)):
            if past_key_values is None:
                with torch.set_grad_enabled(allow_grad):
                    outputs = self.base_model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                hidden_states_offset = 0
            else:
                past_key_values = self._ensure_cache(
                    [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in past_key_values
                    ]
                )
                with torch.set_grad_enabled(allow_grad):
                    outputs = self.base_model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                hidden_states_offset = next_compute_range[0]

            logits_so_far.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            if idx + 1 <= target_step - 1 and idx + 1 < max_n_latents:
                filling_indices = [
                    (instance_idx, mask_list[idx + 1])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > idx + 1
                ]
                latent_vectors = self._gather_latent_values(
                    hidden_states, filling_indices, hidden_states_offset
                )
                inputs_embeds = self._inject_latents(
                    inputs_embeds, filling_indices, latent_vectors
                )

        # latent at target_step (or last pass)
        h_target = hidden_states[
            :, latent_lists[0][min(target_step, max_n_latents) - 1] - 1 - hidden_states_offset, :
        ] if latent_lists and target_step > pass_idx else h_t_modified

        new_state = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "latent_lists": latent_lists,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "next_compute_range": next_compute_range,
            "max_n_latents": max_n_latents,
            "pass_idx": min(target_step, max_n_latents),
            "logits": logits_so_far,
        }
        return h_target, new_state

    def logits_from_latent(self, h_t: torch.Tensor) -> torch.Tensor:
        """Project a latent vector through the LM head (logit lens style)."""
        if not hasattr(self.base_model, "lm_head"):
            raise AttributeError("base_model lacks lm_head for logit lens.")
        return self.base_model.lm_head(h_t)

    def decode_from_state(
        self, h_t: torch.Tensor, other_state: Dict[str, Any]
    ) -> Any:
        """Decode immediately from current latent without further latent iterations."""
        return self.rollout_from_step(h_t, other_state, continue_latents=False)

    def build_teacher_target_ids(self, answer_text: str | None) -> torch.Tensor | None:
        """
        Coconut answers include latent markers, e.g., "Answer:<|start-latent|><|latent|>...<|end-latent|>### {answer}".
        Use config-provided template if set; otherwise build from known latent tokens.
        """
        if answer_text is None:
            return None
        # Default to Coconut-style trailing "### {answer}" before eos.
        if self.teacher_target_template:
            target_text = self.teacher_target_template.format(answer=answer_text)
        else:
            target_text = f"### {answer_text}"
        try:
            tokenized = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")
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
        if target_ids.numel() == 0:
            vocab = getattr(self.base_model.config, "vocab_size", 0)
            return torch.empty(target_ids.size(0), 0, vocab, device=self.device)

        past_key_values = (
            other_state.get("past_key_values_latents")
            or other_state.get("past_key_values")
        )
        past_key_values = self._ensure_cache(past_key_values)

        first_logit = other_state.get("first_logit")
        # If this state corresponds to the post-latent "end" marker, ignore any
        # cached first_logit from the latent loop so we can recompute using the
        # final prompt slice (matches Coconut.generate behaviour).
        pass_idx = other_state.get("pass_idx")
        max_n_latents = other_state.get("max_n_latents")
        step_label = other_state.get("step_label")
        if (
            first_logit is not None
            and (
                (pass_idx is not None and max_n_latents is not None and pass_idx >= max_n_latents)
                or (isinstance(step_label, str) and step_label.lower() == "end")
            )
        ):
            first_logit = None
        if first_logit is not None and first_logit.dim() == 2:
            first_logit = first_logit.to(self.device)
        # If missing, recompute a seed logit using the remaining prompt slice (mirrors rollout_from_step).
        if first_logit is None and "inputs_embeds" in other_state and "next_compute_range" in other_state:
            inputs_embeds = other_state["inputs_embeds"]
            attention_mask_full = other_state.get("attention_mask")
            position_ids_full = other_state.get("position_ids")
            next_compute_range = other_state.get("next_compute_range", (0, 0))
            if next_compute_range[1] > next_compute_range[0]:
                pkv_slice = (
                    self._ensure_cache(
                        [
                            (
                                k[:, :, : next_compute_range[0], :],
                                v[:, :, : next_compute_range[0], :],
                            )
                            for k, v in past_key_values
                        ]
                    )
                    if past_key_values
                    else None
                )
                with torch.set_grad_enabled(allow_grad):
                    outputs_fp = self.base_model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask_full[:, : next_compute_range[1]] if attention_mask_full is not None else None,
                        position_ids=position_ids_full[
                            :, next_compute_range[0] : next_compute_range[1]
                        ] if position_ids_full is not None else None,
                        past_key_values=pkv_slice,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                past_key_values = outputs_fp.past_key_values
                first_logit = outputs_fp.logits[:, -1, :]
        # If we have a precomputed first logit (from the final prompt pass), use it to seed.
        if first_logit is not None:
            if os.getenv("COCONUT_DEBUG_FIRST_LOGIT") == "1" and not getattr(self, "_first_logit_debugged", False):
                tgt_id = target_ids[:, 0]
                topk_vals, topk_idx = torch.topk(first_logit, k=min(5, first_logit.size(-1)), dim=-1)
                toks = [self.tokenizer.decode([int(i)]) for i in topk_idx[0]]
                print(f"[coconut debug] first_logit top5: " + ", ".join(f"{t} ({float(v):.3f})" for t, v in zip(toks, topk_vals[0])))
                print(f"[coconut debug] target_id={int(tgt_id[0])} token='{self.tokenizer.decode([int(tgt_id[0])])}'")
                self._first_logit_debugged = True

        seq_len = target_ids.size(1)
        if first_logit is not None:
            if seq_len == 1:
                return first_logit.unsqueeze(1)
            input_ids = target_ids[:, :-1]
        else:
            input_ids = target_ids[:, :-1] if seq_len > 1 else target_ids

        past_seq_len = None
        try:
            if hasattr(past_key_values, "get_seq_length"):
                past_seq_len = past_key_values.get_seq_length()  # type: ignore[attr-defined]
            else:
                past_seq_len = past_key_values[0][0].shape[-2]
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
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=False,
            )
        if first_logit is not None:
            return torch.cat([first_logit.unsqueeze(1), outputs.logits], dim=1)
        return outputs.logits

    def run_baseline(self, inputs: Any) -> Any:
        tokens = self._prepare_inputs(inputs)
        max_new_tokens = self.generation_kwargs.get("max_new_tokens", 64)
        # Coconut.generate currently supports batch_size=1.
        generated = self.coconut_model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return {"text": [decoded]}
