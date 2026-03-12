#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a script.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import build_dataset, create_dataloader, load_config, prepare_target_ids
from common.model_registry import load_model
from data.gsm8k import parse_answer
from common.metrics.grad_sensitivity import compute_grad_sensitivity


def normalize_answer(text: str | None, dataset_name: str, model_name: str | None = None) -> str | None:
    if text is None:
        return None
    if dataset_name.lower() == "gsm8k":
        val, _ = parse_answer(text)
        return str(val) if val is not None else None
    if "strategyqa" in dataset_name.lower():
        s = str(text).lower()
        if "<|end-latent|>" in s:
            s = s.split("<|end-latent|>")[-1]
        if "###" in s:
            s = s.split("###")[-1]
        if "the answer is:" in s:
            s = s.split("the answer is:")[-1]
        if "answer:" in s:
            s = s.split("answer:")[-1]
        s = s.replace("\r", " ").replace("\n", " ").strip()
        if not s:
            return None
        tok = s.split()[0]
        tok = re.sub(r"^[^a-z]+|[^a-z]+$", "", tok)
        if tok in {"true", "false", "yes", "no"}:
            if model_name is not None and "codi" in model_name.lower():
                return "true" if tok in {"true", "yes"} else "false"
            return tok
        return None
    s = str(text).strip().lower()
    return s if s else None


def extract_text(output: Any) -> str:
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, list):
            return text[0] if text else ""
        if isinstance(text, str):
            return text
    if isinstance(output, list):
        return output[0] if output else ""
    text_attr = getattr(output, "text", None)
    if isinstance(text_attr, list):
        return text_attr[0] if text_attr else ""
    if isinstance(text_attr, str):
        return text_attr
    return str(output)


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def build_counterfactual_means(
    trajs: List[Dict],
    samples: Dict[int, Dict],
    steps: List[int],
) -> Dict[int, Dict[str, Dict[int, np.ndarray]]]:
    buckets: Dict[int, Dict[str, Dict[int, List[np.ndarray]]]] = {}
    step_set = set(steps)
    for rec in trajs:
        sid = rec.get("sample_id")
        if sid is None or sid not in samples:
            continue
        cluster = rec.get("cluster")
        if cluster not in {"A", "B"}:
            continue
        latent_path = Path(rec.get("latent_path", ""))
        if not latent_path.exists():
            continue
        latent = np.load(latent_path)
        latent_steps = samples[sid].get("latent_steps")
        if not isinstance(latent_steps, list) or len(latent_steps) != latent.shape[0]:
            latent_steps = list(range(1, latent.shape[0] + 1))
        for idx, step in enumerate(latent_steps):
            if step not in step_set:
                continue
            buckets.setdefault(sid, {"A": {}, "B": {}})
            buckets[sid][cluster].setdefault(step, []).append(latent[idx])

    means: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {}
    for sid, clusters in buckets.items():
        means[sid] = {"A": {}, "B": {}}
        for cluster, step_map in clusters.items():
            for step, vecs in step_map.items():
                if not vecs:
                    continue
                means[sid][cluster][step] = np.mean(np.stack(vecs, axis=0), axis=0)
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 Stage 4: causal intervention.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--probes_jsonl", required=True)
    parser.add_argument("--traj_jsonl", default=None, help="Required for counterfactual mode.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--ablate_step", type=int, default=2)
    parser.add_argument("--ablate_steps", default=None, help="Comma-separated list of steps (e.g., 1,2,3).")
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--modes", default="probe", help="Comma-separated list: probe,counterfactual")
    parser.add_argument(
        "--grad_metric",
        default=None,
        choices=["grad_logprob", "grad_margin"],
        help="Optional gradient sensitivity metric to log per ablated step.",
    )
    parser.add_argument(
        "--model_type",
        default="coconut",
        choices=["coconut", "codi", "softthinking"],
        help="Latent model type for grad metrics.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    cfg_device = str(model_cfg.get("device", "")).lower()
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        model_cfg["device"] = "cpu"
    elif not cfg_device:
        model_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"] = model_cfg

    model_name = str(config.get("model_name", "coconut")).lower()
    model = load_model(model_name, config.get("model", config))
    dataset = build_dataset(config, tokenizer=getattr(model, "tokenizer", None))
    dataset_by_id: Dict[Any, Dict] = {}
    for idx, d in enumerate(dataset):
        if not isinstance(d, dict):
            continue
        if d.get("id") is not None:
            dataset_by_id[d["id"]] = d
        dataset_by_id[idx] = d

    samples = {r["sample_id"]: r for r in load_jsonl(Path(args.samples_jsonl))}
    probes = {r["sample_id"]: r for r in load_jsonl(Path(args.probes_jsonl))}

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = str(config.get("dataset_name", ""))
    ablate_steps = [args.ablate_step]
    if args.ablate_steps:
        ablate_steps = [int(s) for s in args.ablate_steps.split(",") if s.strip() != ""]
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        modes = ["probe"]

    cf_means: Optional[Dict[int, Dict[str, Dict[int, np.ndarray]]]] = None
    if "counterfactual" in modes:
        if args.traj_jsonl is None:
            raise ValueError("--traj_jsonl is required for counterfactual mode.")
        trajs = load_jsonl(Path(args.traj_jsonl))
        cf_means = build_counterfactual_means(trajs, samples, ablate_steps)

    use_grad_metric = args.grad_metric is not None
    grad_context = torch.enable_grad() if use_grad_metric else torch.no_grad()
    with output_path.open("w") as writer, grad_context:
        for sid, sample in tqdm(samples.items(), desc="intervention", total=len(samples)):
            if sid not in probes:
                continue
            sample_uid = sample.get("sample_uid")
            data = dataset_by_id.get(sample_uid)
            if data is None:
                data = dataset_by_id.get(sid)
            if not data:
                continue
            prompt = data.get("prompt")
            if not prompt:
                continue
            v_a = np.load(Path(probes[sid]["vA_path"]))
            v_b = np.load(Path(probes[sid]["vB_path"]))

            gold_answer = data.get("answer_clean") or data.get("answer") or sample.get("answer_A")
            target_ids = prepare_target_ids(model, gold_answer, dataset_name.lower() == "gsm8k")
            if target_ids is not None and target_ids.dim() == 1:
                target_ids = target_ids.unsqueeze(0)

            for step in ablate_steps:
                grad_value = None
                if use_grad_metric and target_ids is not None:
                    grad_metric = "gold_logprob" if args.grad_metric == "grad_logprob" else "margin"
                    grad_vals = compute_grad_sensitivity(
                        model,
                        prompt,
                        None,
                        target_ids,
                        [step],
                        metric=grad_metric,
                        step_s=None,
                        model_type=args.model_type,
                    )
                    step_vals = grad_vals.get(step)
                    if step_vals is not None and step_vals.numel() > 0:
                        grad_value = float(step_vals[0].item())

                with torch.no_grad():
                    h_t, state = model.forward_until_step(prompt, step)
                    out_base = model.rollout_from_step(h_t, state)

                base_text = extract_text(out_base)
                base_norm = normalize_answer(base_text, dataset_name, model_name)
                if base_norm is None:
                    print(f"[intervention-skip] sample_id={sid} step={step} base_norm=None")
                    continue

                answer_a = sample.get("answer_A")
                answer_b = sample.get("answer_B")

                for mode in modes:
                    h_t_mod = None
                    cf_cluster = None
                    if mode == "probe":
                        opp = "B" if base_norm == answer_a else "A" if base_norm == answer_b else "B"
                        v_use = v_b if opp == "B" else v_a
                        v_t = torch.tensor(v_use, device=h_t.device, dtype=h_t.dtype)
                        denom = torch.dot(v_t, v_t)
                        if denom > 0:
                            coeff = (h_t @ v_t) / denom
                            h_t_mod = h_t - args.lambda_scale * coeff * v_t
                        else:
                            h_t_mod = h_t
                    elif mode == "counterfactual":
                        if cf_means is None:
                            continue
                        opp = "B" if base_norm == answer_a else "A" if base_norm == answer_b else "B"
                        cf_vec = cf_means.get(sid, {}).get(opp, {}).get(step)
                        if cf_vec is None:
                            continue
                        cf_cluster = opp
                        h_t_mod = torch.tensor(cf_vec, device=h_t.device, dtype=h_t.dtype).unsqueeze(0)
                    else:
                        continue

                    with torch.no_grad():
                        out_ablt = model.rollout_from_step(h_t_mod, state)

                    ablt_text = extract_text(out_ablt)
                    ablt_norm = normalize_answer(ablt_text, dataset_name, model_name)
                    if ablt_norm is None:
                        print(f"[intervention-skip] sample_id={sid} step={step} mode={mode} ablt_norm=None")
                        continue
                    flipped = ablt_norm == answer_b and base_norm == answer_a

                    writer.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "ablate_step": step,
                                "lambda": args.lambda_scale,
                                "mode": mode,
                                "counterfactual_cluster": cf_cluster,
                                "answer_A": answer_a,
                                "answer_B": answer_b,
                                "base_answer": base_norm,
                                "ablt_answer": ablt_norm,
                                "flip_to_B": bool(flipped),
                                "base_text": base_text,
                                "ablt_text": ablt_text,
                                "vA_norm": float(np.linalg.norm(v_a)),
                                "vB_norm": float(np.linalg.norm(v_b)),
                                "vA_vB_dot": float(np.dot(v_a, v_b)),
                                "grad_metric": args.grad_metric,
                                "grad_sensitivity": grad_value,
                            }
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
