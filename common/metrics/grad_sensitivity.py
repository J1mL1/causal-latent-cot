from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from common.experiment_utils import build_teacher_state


def _target_time_index(target_ids: torch.Tensor) -> int:
    seq_len = target_ids.size(1)
    return seq_len - 2 if seq_len >= 2 else seq_len - 1


def compute_scalar_S(
    model: Any,
    h_t: torch.Tensor,
    cache: Optional[Dict[str, Any]],
    target_ids: Optional[torch.Tensor] = None,
    metric: str = "gold_logprob",
    step_s: Optional[int] = None,
    return_per_sample: bool = False,
    allow_grad: bool = False,
) -> torch.Tensor | None:
    if target_ids is None:
        return None
    target_ids = target_ids.to(h_t.device)

    if step_s is not None:
        if not hasattr(model, "logits_from_latent"):
            return None
        logits = model.logits_from_latent(h_t)
    else:
        if cache is None or not hasattr(model, "compute_logits"):
            return None
        logits = model.compute_logits(h_t, cache, target_ids, allow_grad=allow_grad)

    if metric == "gold_logprob":
        logp = F.log_softmax(logits, dim=-1)
        if logits.dim() == 3:
            per_tok = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            per_sample = per_tok.sum(dim=-1)
        else:
            time_index = _target_time_index(target_ids)
            targets = target_ids[:, time_index]
            per_sample = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"Unsupported grad metric: {metric}")

    return per_sample if return_per_sample else per_sample.mean()


def _resolve_directional_vec(
    step: int,
    idx: int,
    directional_vecs: Any,
    device: torch.device,
) -> torch.Tensor | None:
    if directional_vecs is None:
        return None
    if isinstance(directional_vecs, dict):
        vec = directional_vecs.get(step)
    elif torch.is_tensor(directional_vecs):
        if directional_vecs.dim() == 1:
            vec = directional_vecs
        else:
            vec = directional_vecs[idx]
    else:
        vec = directional_vecs[idx]
    if vec is None:
        return None
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec)
    return vec.to(device)


def compute_grad_sensitivity(
    model: Any,
    inputs: Any,
    cache: Optional[Dict[str, Any]],
    target_ids: torch.Tensor,
    steps_T: Sequence[int],
    metric: str = "gold_logprob",
    norm: str = "l2",
    directional_vecs: Any = None,
    step_s: Optional[int] = None,
    model_type: str = "coconut",
    use_jacobian: bool = True,
) -> Dict[int, torch.Tensor]:
    def _summarize_per_sample_states(
        states: list[Dict[str, Any]],
        max_items: int = 5,
    ) -> list[Dict[str, Any]]:
        summaries = []
        latent_id = getattr(model, "latent_token_id", None)
        for idx_s, s in enumerate(states[:max_items]):
            input_ids = s.get("input_ids")
            input_len = int(input_ids.size(1)) if torch.is_tensor(input_ids) else None
            earliest_lat = None
            latent_count = None
            if torch.is_tensor(input_ids) and latent_id is not None:
                pos = (input_ids[0] == latent_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    earliest_lat = int(pos[0].item())
                    latent_count = int(pos.numel())
                else:
                    latent_count = 0
            summaries.append(
                {
                    "i": idx_s,
                    "input_len": input_len,
                    "latent_count": latent_count,
                    "earliest_latent_pos": earliest_lat,
                    "next_compute_range": s.get("next_compute_range"),
                    "pass_idx": s.get("pass_idx"),
                }
            )
        return summaries
    if model_type == "softthinking":
        return compute_softthinking_grad(
            model,
            inputs,
            cache,
            target_ids,
            steps_T,
            metric=metric,
            norm=norm,
            directional_vecs=directional_vecs,
            step_s=step_s,
        )

    results: Dict[int, torch.Tensor] = {}
    for idx, step in enumerate(steps_T):
        h_t, state_t = model.forward_until_step(inputs, step, allow_grad=True)
        h_t = h_t.detach().requires_grad_(True)
        if isinstance(state_t, dict) and "per_sample_states" in state_t:
            per_states = state_t.get("per_sample_states") or []
            sample_count = len(per_states)
            summaries = _summarize_per_sample_states(per_states)
            raise RuntimeError(
                "per_sample_states detected in compute_grad_sensitivity; "
                f"batch rollout expected but received per-sample states (count={sample_count}). "
                f"summaries={summaries}"
            )

        if step_s is not None and step_s != step:
            if not hasattr(model, "rollout_to_step"):
                raise AttributeError("Model lacks rollout_to_step required for grad causal graph.")
            h_s, state_s = model.rollout_to_step(
                h_t, state_t, target_step=int(step_s), allow_grad=True
            )
            logits_state = state_s
            latent_for_s = h_s
        else:
            latent_for_s = h_t
            logits_state = state_t

        if step_s is None:
            rollout = model.rollout_from_step(h_t, logits_state, allow_grad=True)
            logits_state = build_teacher_state(rollout, logits_state)

        if use_jacobian:
            def _per_sample_fn(h_in: torch.Tensor) -> torch.Tensor:
                if step_s is not None and step_s != step:
                    if not hasattr(model, "rollout_to_step"):
                        raise AttributeError("Model lacks rollout_to_step required for grad causal graph.")
                    h_s, state_s = model.rollout_to_step(
                        h_in, state_t, target_step=int(step_s), allow_grad=True
                    )
                    logits_state_local = state_s
                    latent_for_s_local = h_s
                else:
                    latent_for_s_local = h_in
                    logits_state_local = state_t

                if step_s is None:
                    rollout = model.rollout_from_step(h_in, logits_state_local, allow_grad=True)
                    logits_state_local = build_teacher_state(rollout, logits_state_local)

                per_sample_local = compute_scalar_S(
                    model,
                    latent_for_s_local,
                    logits_state_local,
                    target_ids=target_ids,
                    metric=metric,
                    step_s=step_s,
                    return_per_sample=True,
                    allow_grad=True,
                )
                if per_sample_local is None:
                    return torch.full((h_in.size(0),), float("nan"), device=h_in.device)
                return per_sample_local

            jac = torch.autograd.functional.jacobian(_per_sample_fn, h_t)
            batch_size = h_t.size(0)
            step_vals = []
            for b in range(batch_size):
                grad_vec = jac[b, b]
                vec = _resolve_directional_vec(step, idx, directional_vecs, grad_vec.device)
                if vec is not None:
                    proj = (grad_vec * vec).sum(dim=-1)
                    step_vals.append(float(proj.detach().cpu().item()))
                else:
                    if norm == "l1":
                        val = grad_vec.abs().sum(dim=-1)
                    else:
                        val = torch.norm(grad_vec, p=2, dim=-1)
                    step_vals.append(float(val.detach().cpu().item()))
            results[step] = torch.tensor(step_vals)
        else:
            per_sample = compute_scalar_S(
                model,
                latent_for_s,
                logits_state,
                target_ids=target_ids,
                metric=metric,
                step_s=step_s,
                return_per_sample=True,
                allow_grad=True,
            )
            if per_sample is None:
                results[step] = torch.full(
                    (h_t.size(0),), float("nan"), device=h_t.device
                ).cpu()
                continue

            batch_size = per_sample.size(0)
            step_vals = []
            for b in range(batch_size):
                retain_graph = b < batch_size - 1
                grad_b = torch.autograd.grad(
                    per_sample[b],
                    h_t,
                    retain_graph=retain_graph,
                    allow_unused=True,
                )[0]
                if grad_b is None:
                    step_vals.append(float("nan"))
                    continue
                grad_vec = grad_b[b]
                vec = _resolve_directional_vec(step, idx, directional_vecs, grad_vec.device)
                if vec is not None:
                    proj = (grad_vec * vec).sum(dim=-1)
                    step_vals.append(float(proj.detach().cpu().item()))
                else:
                    if norm == "l1":
                        val = grad_vec.abs().sum(dim=-1)
                    else:
                        val = torch.norm(grad_vec, p=2, dim=-1)
                    step_vals.append(float(val.detach().cpu().item()))

            results[step] = torch.tensor(step_vals)
        if hasattr(model, "base_model") and hasattr(model.base_model, "zero_grad"):
            model.base_model.zero_grad(set_to_none=True)
        elif hasattr(model, "model") and hasattr(model.model, "zero_grad"):
            model.model.zero_grad(set_to_none=True)

    return results


def compute_softthinking_grad(*args: Any, **kwargs: Any) -> Dict[int, torch.Tensor]:
    (
        model,
        inputs,
        cache,
        target_ids,
        steps_T,
    ) = args[:5]
    metric = kwargs.get("metric", "gold_logprob")
    norm = kwargs.get("norm", "l2")
    directional_vecs = kwargs.get("directional_vecs", None)
    step_s = kwargs.get("step_s", None)
    use_jacobian = kwargs.get("use_jacobian", False)

    def _summarize_per_sample_states(
        states: list[Dict[str, Any]],
        max_items: int = 5,
    ) -> list[Dict[str, Any]]:
        summaries = []
        latent_id = getattr(model, "latent_token_id", None)
        for idx_s, s in enumerate(states[:max_items]):
            input_ids = s.get("input_ids")
            input_len = int(input_ids.size(1)) if torch.is_tensor(input_ids) else None
            earliest_lat = None
            latent_count = None
            if torch.is_tensor(input_ids) and latent_id is not None:
                pos = (input_ids[0] == latent_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    earliest_lat = int(pos[0].item())
                    latent_count = int(pos.numel())
                else:
                    latent_count = 0
            summaries.append(
                {
                    "i": idx_s,
                    "input_len": input_len,
                    "latent_count": latent_count,
                    "earliest_latent_pos": earliest_lat,
                    "next_compute_range": s.get("next_compute_range"),
                    "pass_idx": s.get("pass_idx"),
                }
            )
        return summaries

    results: Dict[int, torch.Tensor] = {}
    for idx, step in enumerate(steps_T):
        h_t, state_t = model.forward_until_step(inputs, step, allow_grad=True)
        h_t = h_t.detach().requires_grad_(True)
        if isinstance(state_t, dict) and "per_sample_states" in state_t:
            per_states = state_t.get("per_sample_states") or []
            sample_count = len(per_states)
            summaries = _summarize_per_sample_states(per_states)
            raise RuntimeError(
                "per_sample_states detected in compute_softthinking_grad; "
                f"batch rollout expected but received per-sample states (count={sample_count}). "
                f"summaries={summaries}"
            )

        if step_s is not None and step_s != step:
            h_s, state_s = model.rollout_to_step(
                h_t, state_t, target_step=int(step_s), allow_grad=True
            )
            logits_state = state_s
            latent_for_s = h_s
        else:
            latent_for_s = h_t
            logits_state = state_t

        if step_s is None:
            rollout = model.rollout_from_step(h_t, logits_state, allow_grad=True)
            logits_state = build_teacher_state(rollout, logits_state)

        if use_jacobian:
            def _per_sample_fn(h_in: torch.Tensor) -> torch.Tensor:
                if step_s is not None and step_s != step:
                    h_s, state_s = model.rollout_to_step(
                        h_in, state_t, target_step=int(step_s), allow_grad=True
                    )
                    logits_state_local = state_s
                    latent_for_s_local = h_s
                else:
                    latent_for_s_local = h_in
                    logits_state_local = state_t

                if step_s is None:
                    rollout = model.rollout_from_step(h_in, logits_state_local, allow_grad=True)
                    logits_state_local = build_teacher_state(rollout, logits_state_local)

                per_sample_local = compute_scalar_S(
                    model,
                    latent_for_s_local,
                    logits_state_local,
                    target_ids=target_ids,
                    metric=metric,
                    step_s=step_s,
                    return_per_sample=True,
                    allow_grad=True,
                )
                if per_sample_local is None:
                    return torch.full((h_in.size(0),), float("nan"), device=h_in.device)
                return per_sample_local

            jac = torch.autograd.functional.jacobian(_per_sample_fn, h_t)
            batch_size = h_t.size(0)
            step_vals = []
            for b in range(batch_size):
                grad_vec = jac[b, b]
                vec = _resolve_directional_vec(step, idx, directional_vecs, grad_vec.device)
                if vec is not None:
                    proj = (grad_vec * vec).sum(dim=-1)
                    step_vals.append(float(proj.detach().cpu().item()))
                else:
                    if norm == "l1":
                        val = grad_vec.abs().sum(dim=-1)
                    else:
                        val = torch.norm(grad_vec, p=2, dim=-1)
                    step_vals.append(float(val.detach().cpu().item()))
            results[step] = torch.tensor(step_vals)
        else:
            per_sample = compute_scalar_S(
                model,
                latent_for_s,
                logits_state,
                target_ids=target_ids,
                metric=metric,
                step_s=step_s,
                return_per_sample=True,
                allow_grad=True,
            )
            if per_sample is None:
                results[step] = torch.full(
                    (h_t.size(0),), float("nan"), device=h_t.device
                ).cpu()
                continue

            batch_size = per_sample.size(0)
            step_vals = []
            for b in range(batch_size):
                retain_graph = b < batch_size - 1
                grad_b = torch.autograd.grad(
                    per_sample[b],
                    h_t,
                    retain_graph=retain_graph,
                    allow_unused=True,
                )[0]
                if grad_b is None:
                    step_vals.append(float("nan"))
                    continue
                grad_vec = grad_b[b]
                vec = _resolve_directional_vec(step, idx, directional_vecs, grad_vec.device)
                if vec is not None:
                    proj = (grad_vec * vec).sum(dim=-1)
                    step_vals.append(float(proj.detach().cpu().item()))
                else:
                    if norm == "l1":
                        val = grad_vec.abs().sum(dim=-1)
                    else:
                        val = torch.norm(grad_vec, p=2, dim=-1)
                    step_vals.append(float(val.detach().cpu().item()))

            results[step] = torch.tensor(step_vals)
        if hasattr(model, "model") and hasattr(model.model, "zero_grad"):
            model.model.zero_grad(set_to_none=True)

    return results
