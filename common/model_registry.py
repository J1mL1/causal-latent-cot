from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from common.model_interface import LatentReasoningModel


MODEL_REGISTRY: Dict[str, Type[LatentReasoningModel]] = {}

# Lazy import map to avoid pulling heavy deps unless requested.
MODEL_IMPORT_PATHS: Dict[str, str] = {
    "hf-auto": "common.models.hf_model:HFAutoregressiveModel",
    "coconut": "common.models.coconut_model:CoconutWrapper",
    "codi": "common.models.codi_model:CodiWrapper",
    "softthinking": "common.models.softthinking_model:SoftThinkingWrapper",
    "multiplex": "common.models.multiplex_model:MultiplexThinkingWrapper",
}


def register_model(name: str):
    """Decorator used by wrappers to register themselves."""

    def decorator(cls: Type[LatentReasoningModel]) -> Type[LatentReasoningModel]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def load_model(model_name: str, config: Dict[str, Any]) -> LatentReasoningModel:
    """
    Factory that instantiates a registered model and loads it from config.

    Args:
        model_name: Identifier such as 'coconut', 'codi', 'hf-auto', 'softthinking', or 'multiplex'.
        config: Model-specific configuration dictionary.
    """
    normalized = model_name.lower()
    if normalized not in MODEL_REGISTRY:
        if normalized not in MODEL_IMPORT_PATHS:
            raise ValueError(f"Model {model_name} is not registered.")
        module_path, cls_name = MODEL_IMPORT_PATHS[normalized].split(":")
        module = importlib.import_module(module_path)
        MODEL_REGISTRY[normalized] = getattr(module, cls_name)

    model_cls = MODEL_REGISTRY[normalized]
    model = model_cls()
    model.load_from_config(config)
    return model
