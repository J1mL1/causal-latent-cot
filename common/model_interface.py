from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class LatentReasoningModel(ABC):
    """Abstract interface for models that expose latent/hidden steps for intervention."""

    def __init__(self) -> None:
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Optional template to build teacher-forced targets, e.g., "The answer is {answer}"
        self.teacher_target_template: str | None = None

    @abstractmethod
    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load weights/tokenizer using a configuration dictionary."""

    @abstractmethod
    def forward_until_step(
        self, inputs: Any, step: int, allow_grad: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run a forward pass until the target latent step.

        Args:
            inputs: Model-specific input (text string, token ids, or a batch dict).
            step: Integer in [1, num_steps] for latent rollout steps. Implementations must
                  raise on out-of-range inputs instead of clamping.

        Returns:
            h_t: Tensor of shape [batch, d_model] for the requested step.
            other_state: Opaque dict with caches needed for rollout_from_step.
        """

    @abstractmethod
    def rollout_from_step(
        self, h_t_modified: torch.Tensor, other_state: Dict[str, Any], allow_grad: bool = False
    ) -> Any:
        """
        Continue the forward pass starting from a modified latent.

        Args:
            h_t_modified: Tensor of shape [batch, d_model] after ablation.
            other_state: Cached tensors returned by forward_until_step.
        """

    @abstractmethod
    def rollout_to_step(
        self, h_t_modified: torch.Tensor, other_state: Dict[str, Any], target_step: int, allow_grad: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Continue the forward pass from the modified latent up to target_step without decoding.

        Args:
            h_t_modified: Tensor of shape [batch, d_model] after ablation.
            other_state: Cached tensors returned by forward_until_step.
            target_step: Target latent/token index to reach (1-based for latents).
        """

    @abstractmethod
    def logits_from_latent(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Map a latent vector directly to vocabulary logits (logit lens style).
        """

    @abstractmethod
    def decode_from_state(
        self, h_t: torch.Tensor, other_state: Dict[str, Any]
    ) -> Any:
        """
        Decode immediately from a latent state without running remaining latent iterations.
        """

    @abstractmethod
    def compute_logits(
        self,
        h_t: torch.Tensor,
        other_state: Dict[str, Any],
        target_ids: torch.Tensor,
        allow_grad: bool = False,
    ) -> torch.Tensor:
        """
        Teacher-forced logits for a target continuation given cached prefix state.
        """

    @abstractmethod
    def run_baseline(self, inputs: Any) -> Any:
        """Perform vanilla inference without any intervention."""

    def build_teacher_target_ids(self, answer_text: str | None) -> torch.Tensor | None:
        """
        Optional helper to turn a gold answer string into token ids for teacher-forced scoring.
        Default implementation uses `teacher_target_template` (if set) then tokenizes without
        adding special tokens. Subclasses can override for custom formatting.
        """
        if answer_text is None or not hasattr(self, "tokenizer"):
            return None
        text = (
            self.teacher_target_template.format(answer=answer_text)
            if self.teacher_target_template
            else answer_text
        )
        try:
            tokenized = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")  # type: ignore[attr-defined]
        except Exception:
            return None
        ids = tokenized.get("input_ids")
        if ids is None:
            return None
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)
