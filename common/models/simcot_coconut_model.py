from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_registry import register_model
from common.models.coconut_model import CoconutWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_COCONUT_DIR = REPO_ROOT / "external" / "sim-cot" / "Coconut"


def _import_simcot_coconut_modules():
    """Import Sim-CoT fork modules (standalone `coconut.py` + `utils.py` under external/sim-cot/Coconut)."""
    coconut_py = _SIM_COCONUT_DIR / "coconut.py"
    utils_py = _SIM_COCONUT_DIR / "utils.py"
    if not coconut_py.exists() or not utils_py.exists():
        raise ModuleNotFoundError(
            "Missing Sim-CoT Coconut sources. Expected files:\n"
            f"  - {coconut_py}\n"
            f"  - {utils_py}\n\n"
            "This model wrapper expects the Sim-CoT fork's `Coconut/` directory to be present at:\n"
            f"  {_SIM_COCONUT_DIR}\n"
            "Please fetch/clone the Sim-CoT repository (or its Coconut subdir) into "
            "`external/sim-cot/`, so that `coconut.py` and `utils.py` exist there."
        )
    d = str(_SIM_COCONUT_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    import coconut as sim_coconut_mod  # noqa: E402
    from utils import Config  # noqa: E402

    return sim_coconut_mod.CoconutGPT_Same_Word_Embedding, Config


def _load_state_dict_vocab_expand(
    module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], *, strict: bool = False
):
    """
    Like load_state_dict, but when the checkpoint has a smaller leading dimension (e.g. base GPT-2
    vocab 50257) and the live model was resized for extra special tokens (50260), copy the prefix
    rows and keep the tail rows as already initialized.
    """
    model_sd = module.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        m = model_sd[k]
        if m.shape == v.shape:
            adapted[k] = v.to(device=m.device, dtype=m.dtype)
        elif (
            m.dim() >= 1
            and v.dim() == m.dim()
            and m.shape[0] > v.shape[0]
            and m.shape[1:] == v.shape[1:]
        ):
            merged = m.clone()
            merged[: v.shape[0]].copy_(v.to(device=m.device, dtype=m.dtype))
            adapted[k] = merged
        else:
            raise RuntimeError(
                f"Cannot load {k}: checkpoint shape {tuple(v.shape)} vs model shape {tuple(m.shape)}"
            )
    return module.load_state_dict(adapted, strict=strict)


@register_model("simcot-coconut")
class SimCoconutGPTWrapper(CoconutWrapper):
    """
    Sim-CoT Coconut: `CoconutGPT_Same_Word_Embedding` from the Sim-CoT fork (dual causal LMs,
    latent slots as in training). Weights come from the published Sim-CoT checkpoint.

    RQ1/RQ2 use `CoconutWrapper` (`forward_until_step`, etc.): same latent path as Sim-CoT forward.
    """

    def load_from_config(self, config: Dict[str, Any]) -> None:
        CoconutGPT_Same_Word_Embedding, Config = _import_simcot_coconut_modules()

        base_path = config.get("base_model_name_or_path") or config.get(
            "model_name_or_path"
        )
        if base_path is None:
            raise ValueError("SimCoconutGPTWrapper requires 'base_model_name_or_path'.")
        tokenizer_name = config.get("tokenizer_name_or_path", base_path)
        self.device = torch.device(config.get("device", self.device))
        self.teacher_target_template = config.get(
            "teacher_target_template", self.teacher_target_template
        )

        base_lm = AutoModelForCausalLM.from_pretrained(
            base_path, trust_remote_code=True
        ).to(self.device)
        explain_lm = AutoModelForCausalLM.from_pretrained(
            base_path, trust_remote_code=True
        ).to(self.device)
        base_lm.eval()
        explain_lm.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.generation_kwargs = config.get("generation_kwargs", {})
        self.num_latent_placeholders = int(config.get("num_latent_placeholders", 0))
        self.use_coconut_question_only = bool(
            config.get("use_coconut_question_only", False)
        )
        self.align_latent_padding = bool(config.get("align_latent_padding", False))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        add_latent_tokens = config.get("add_latent_tokens", True)
        latent_token_str = config.get("latent_token_str", "<|latent|>")
        start_latent_token_str = config.get("start_latent_token_str", "<|start-latent|>")
        end_latent_token_str = config.get("end_latent_token_str", "<|end-latent|>")
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

        vocab_size = len(self.tokenizer)
        if vocab_size != base_lm.get_input_embeddings().num_embeddings:
            base_lm.resize_token_embeddings(vocab_size)
        if vocab_size != explain_lm.get_input_embeddings().num_embeddings:
            explain_lm.resize_token_embeddings(vocab_size)

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

        init_token_str = config.get("latent_init_token", "<<")
        target_id = self.tokenizer.convert_tokens_to_ids(init_token_str)
        if target_id is None or target_id == self.tokenizer.unk_token_id:
            target_id = self.tokenizer.eos_token_id

        for mdl in (base_lm, explain_lm):
            embed = mdl.get_input_embeddings()
            if embed is not None and newly_added_tokens:
                with torch.no_grad():
                    target_vec = embed.weight.data[target_id].clone()
                    for tok_id in newly_added_tokens:
                        if tok_id is not None and tok_id < embed.weight.data.size(0):
                            embed.weight.data[tok_id] = target_vec
                    if hasattr(mdl, "lm_head") and hasattr(mdl.lm_head, "weight"):
                        lm_w = mdl.lm_head.weight
                        if lm_w.shape[0] == embed.weight.data.shape[0]:
                            for tok_id in newly_added_tokens:
                                if tok_id is not None and tok_id < lm_w.size(0):
                                    lm_w.data[tok_id] = lm_w.data[target_id]

        c_thought = int(config.get("c_thought", 2))
        max_latent_stage = int(config.get("max_latent_stage", 5))
        extra_cfg = dict(config.get("simcot_training_config", {}))
        training_dict: Dict[str, Any] = {
            "training_method": "full",
            "explain_mode": "v1_aug",
            "c_thought": c_thought,
            "max_latent_stage": max_latent_stage,
            "w_prompt": False,
            "visualize": False,
        }
        training_dict.update(extra_cfg)
        training_cfg = Config(training_dict)

        step_start_id = int(self.tokenizer.convert_tokens_to_ids("<<"))

        self.coconut_model = CoconutGPT_Same_Word_Embedding(
            base_lm,
            explain_lm,
            self.tokenizer,
            self.latent_token_id,
            self.start_latent_id,
            self.end_latent_id,
            self.eos_token_id,
            step_start_id,
            c_thought,
            training_cfg,
        ).to(self.device)
        self.coconut_model.eval()

        self.base_model = self.coconut_model.base_causallm

        ckpt_path = config.get("checkpoint_path")
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            _ = _load_state_dict_vocab_expand(
                self.coconut_model, state_dict, strict=False
            )
