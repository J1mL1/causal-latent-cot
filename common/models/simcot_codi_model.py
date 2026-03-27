from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from common.model_registry import register_model
from common.models.codi_model import CodiWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_CODI_SRC = REPO_ROOT / "external" / "sim-cot" / "CODI" / "src"


def _load_codi_checkpoint_state(ckpt_dir: str) -> Dict[str, Any]:
    """Load CODI weights from common layouts (including HF sharded safetensors filenames)."""
    candidates = [
        os.path.join(ckpt_dir, "model.safetensors"),
        os.path.join(ckpt_dir, "model-00001-of-00001.safetensors"),
        os.path.join(ckpt_dir, "pytorch_model.bin"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path)
        return torch.load(path, map_location="cpu")
    raise FileNotFoundError(
        f"No CODI checkpoint file in {ckpt_dir} (tried model.safetensors, "
        "model-00001-of-00001.safetensors, pytorch_model.bin)."
    )


@register_model("simcot-codi")
class SimCodiWrapper(CodiWrapper):
    """
    Sim-CoT CODI: load `CODI` from `external/sim-cot/CODI/src` to match the published checkpoint;
    RQ1/RQ2 behavior matches `CodiWrapper` otherwise.
    """

    def load_from_config(self, config: Dict[str, Any]) -> None:
        src = _SIM_CODI_SRC.as_posix()
        if src not in sys.path:
            sys.path.insert(0, src)
        from model import CODI, ModelArguments, TrainingArguments  # type: ignore

        model_args_dict = dict(config.get("model_args", {}))
        # Not a CODI ModelArguments field; tokenizer comes from config below (line ~120).
        model_args_dict.pop("tokenizer_name_or_path", None)

        model_args = ModelArguments(**model_args_dict)
        train_args_dict = config.get("training_args", {})
        train_args = TrainingArguments(
            output_dir=train_args_dict.get("output_dir", "./codi-out"),
            per_device_train_batch_size=train_args_dict.get(
                "per_device_train_batch_size", 1
            ),
            per_device_eval_batch_size=train_args_dict.get(
                "per_device_eval_batch_size", 1
            ),
            **{
                k: v
                for k, v in train_args_dict.items()
                if k
                not in {
                    "output_dir",
                    "per_device_train_batch_size",
                    "per_device_eval_batch_size",
                }
            },
        )

        lora_config = None
        if train_args.use_lora:
            from peft import LoraConfig, TaskType  # type: ignore

            target_modules = ["c_attn", "c_proj", "c_fc"]
            if any(
                name in model_args.model_name_or_path.lower()
                for name in ["llama", "mistral", "falcon", "qwen"]
            ):
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ]
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
        self.teacher_target_template = config.get(
            "teacher_target_template", self.teacher_target_template
        )
        self.model.to(self.device)
        if model_args.ckpt_dir:
            state_dict = _load_codi_checkpoint_state(model_args.ckpt_dir)
            self.model.load_state_dict(state_dict, strict=False)
        if train_args.use_prj and hasattr(self.model, "prj"):
            _prj_dtype = self.model.get_embd(self.model.codi, self.model.model_name).weight.dtype
            self.model.prj.to(dtype=_prj_dtype)
        if not model_args.full_precision and not train_args.bf16:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

        tokenizer_name = config.get("tokenizer_name_or_path", model_args.model_name_or_path)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model, "tokenizer", None) is None:
            self.model.tokenizer = self.tokenizer

        self.num_latent = int(train_args.num_latent)
        self.use_prj = bool(train_args.use_prj)
        self.generation_kwargs = config.get("generation_kwargs", {})
