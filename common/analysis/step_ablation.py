from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
import copy

import torch
from tqdm import tqdm

from common.model_interface import LatentReasoningModel
def ablate_zero(h_t: torch.Tensor) -> torch.Tensor:
    """Replace hidden state with zeros, preserving shape."""
    return torch.zeros_like(h_t)


def ablate_mean(h_t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Replace hidden state with a global mean vector."""
    mu_expanded = mu.unsqueeze(0).expand_as(h_t)
    return mu_expanded.to(h_t)


def _compute_sigma(base: torch.Tensor, sigma_scale: float) -> float:
    sigma_base = base.norm(dim=-1).mean().item() / (base.size(-1) ** 0.5)
    return sigma_scale * sigma_base


def ablate_gaussian_on_h(h_t: torch.Tensor, sigma_scale: float = 0.5) -> torch.Tensor:
    """Add Gaussian noise around the current latent."""
    sigma = _compute_sigma(h_t, sigma_scale)
    noise = torch.randn_like(h_t) * sigma
    return h_t + noise


def ablate_gaussian_noise(h_t: torch.Tensor, sigma_scale: float = 0.5) -> torch.Tensor:
    """Replace latent with pure Gaussian noise (zero mean, scaled by latent norm)."""
    sigma = _compute_sigma(h_t, sigma_scale)
    return torch.randn_like(h_t) * sigma


def ablate_gaussian_around_mean(
    h_t: torch.Tensor, mu: torch.Tensor, sigma_scale: float = 0.5
) -> torch.Tensor:
    """Sample around the global mean latent."""
    sigma = _compute_sigma(mu.unsqueeze(0), sigma_scale)
    noise = torch.randn_like(h_t).mul_(sigma)
    mu_expanded = mu.unsqueeze(0).expand_as(h_t).to(h_t)
    return mu_expanded + noise


ABLATION_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "zero": ablate_zero,
    "mean": ablate_mean,
    "gaussian_h": ablate_gaussian_on_h,
    "gaussian": ablate_gaussian_noise,
    "gaussian_mu": ablate_gaussian_around_mean,
}


def register_ablation(name: str, fn: Callable[..., torch.Tensor]) -> None:
    """Allow downstream users to attach custom ablations."""
    ABLATION_REGISTRY[name] = fn


def estimate_global_mean_latent(
    model: LatentReasoningModel,
    dataloader: Iterable[Any],
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Estimate dataset-level mean latent by averaging h_t across steps and samples.

    Args:
        model: LatentReasoningModel implementing forward_until_step.
        dataloader: Iterable yielding input batches for the model.
        max_steps: Limit how many latent steps to aggregate per sample.

    Returns:
        mu: Tensor [d_model]
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 for mean latent estimation.")

    sums: Optional[torch.Tensor] = None
    count = 0
    model_device = getattr(model, "device", torch.device("cpu"))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            samples = batch if isinstance(batch, list) else [batch]
            steps = (
                range(1, max_steps + 1)
                if max_steps is not None
                else range(1, 2)
            )
            for sample in samples:
                for step in steps:
                    try:
                        h_t, _ = model.forward_until_step(sample, step)
                    except Exception as e:
                        print(f"[estimate_global_mean_latent] batch {batch_idx}, step {step} error: {e}")
                        if max_steps is None:
                            break
                        continue
                    h_t = h_t.detach().to(model_device)
                    if sums is None:
                        sums = torch.zeros_like(h_t[0])
                    sums = sums + h_t.sum(dim=0)
                    count += h_t.size(0)

    if sums is None or count == 0:
        raise RuntimeError("Failed to estimate global mean latent; no states collected.")

    mu = sums / float(count)
    return mu


def estimate_step_mean_latents(
    model: LatentReasoningModel,
    dataloader: Iterable[Any],
    max_steps: Optional[int] = None,
    include_start: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Estimate per-step mean latents: {step: mu_step}.

    Args:
        model: LatentReasoningModel implementing forward_until_step.
        dataloader: Iterable yielding input batches for the model.
        max_steps: Limit how many latent steps to aggregate per sample.

    Returns:
        Dict mapping step index to mean vector [d_model].
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 for per-step mean estimation.")

    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    model_device = getattr(model, "device", torch.device("cpu"))

    with torch.no_grad():
        def _iter_steps() -> List[int]:
            if max_steps is not None:
                return list(range(1, max_steps + 1))
            return [1]

        steps_to_collect = _iter_steps()
        for batch in tqdm(dataloader, desc="estimate_step_mean_latents", leave=False):
            samples = batch if isinstance(batch, list) else [batch]
            for sample in samples:
                for step in steps_to_collect:
                    try:
                        h_t, _ = model.forward_until_step(sample, step)
                    except Exception:
                        if max_steps is None:
                            break
                        continue
                    h_t = h_t.detach().to(model_device)
                    if step not in sums:
                        sums[step] = torch.zeros_like(h_t[0])
                        counts[step] = 0
                    sums[step] = sums[step] + h_t.sum(dim=0)
                    counts[step] += h_t.size(0)

    if not sums:
        raise RuntimeError("Failed to estimate per-step means; no states collected.")

    mu_by_step = {step: sums[step] / float(counts[step]) for step in sums}
    return mu_by_step


def run_step_ablation(
    model: LatentReasoningModel,
    inputs: Any,
    step_t: int,
    mode: str,
    mu: Optional[torch.Tensor] = None,
    mu_by_step: Optional[Dict[int, torch.Tensor]] = None,
    sigma_scale: float = 0.5,
    state: Optional[Tuple[torch.Tensor, Dict[str, Any]]] = None,
    baseline_rollout: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment on one example/batch.

    Args:
        model: LatentReasoningModel wrapper.
        inputs: Input sample or batch.
        step_t: Target latent step.
        mode: One of ABLATION_REGISTRY.
        mu: Optional global mean latent when required.
        sigma_scale: Noise scale for Gaussian ablations.
        state: Optional precomputed (h_t, other_state) to avoid recomputation.
        baseline_rollout: Optional baseline rollout computed with rollout_from_step on
            the provided state; if omitted, it is recomputed.
    """
    if mode not in ABLATION_REGISTRY and mode not in {"mean_step", "gaussian_mu_step"}:
        raise ValueError(f"Unknown ablation mode: {mode}")
    if isinstance(step_t, int) and step_t < 1:
        raise ValueError(f"step_t must be >= 1; got {step_t}.")
    if not isinstance(step_t, int):
        raise ValueError(f"Unsupported step type: {type(step_t)}")

    ablation_fn = ABLATION_REGISTRY.get(mode)

    with torch.no_grad():
        if state is None:
            h_t, other_state = model.forward_until_step(inputs, step_t)
        else:
            h_t, other_state = state
        if mode in {"mean", "gaussian_mu"} and mu is None:
            raise ValueError(f"Mode {mode} requires a global mean latent.")
        if mode in {"mean_step", "gaussian_mu_step"}:
            if mu_by_step is None or step_t not in mu_by_step:
                raise ValueError(f"Mode {mode} requires per-step mean for step {step_t}.")
            mu = mu_by_step[step_t]
        if mode == "mean":
            h_t_modified = ablation_fn(h_t, mu=mu)
        elif mode == "gaussian_mu":
            h_t_modified = ablation_fn(h_t, mu=mu, sigma_scale=sigma_scale)
        elif mode == "mean_step":
            h_t_modified = ablate_mean(h_t, mu=mu)  # type: ignore[arg-type]
        elif mode == "gaussian_mu_step":
            h_t_modified = ablate_gaussian_around_mean(h_t, mu=mu, sigma_scale=sigma_scale)  # type: ignore[arg-type]
        elif mode == "gaussian_h":
            h_t_modified = ablation_fn(h_t, sigma_scale=sigma_scale)
        elif mode == "gaussian":
            h_t_modified = ablation_fn(h_t, sigma_scale=sigma_scale)
        else:
            h_t_modified = ablation_fn(h_t)
        state_for_baseline = copy.deepcopy(other_state)
        state_for_ablt = copy.deepcopy(other_state)
        baseline = baseline_rollout if baseline_rollout is not None else model.rollout_from_step(h_t, state_for_baseline)
        ablated_output = model.rollout_from_step(h_t_modified, state_for_ablt)

    return {
        "baseline": baseline,
        "ablated": ablated_output,
        "h_t": h_t,
        "h_t_modified": h_t_modified,
        "state": other_state,
        "mode": mode,
        "step": step_t,
    }
