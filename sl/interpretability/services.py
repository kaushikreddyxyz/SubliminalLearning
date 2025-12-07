from __future__ import annotations

from typing import Any

from sl.external import hf_driver
from sl.llm.data_models import Model


def _resolve_model_id(model: Model | str) -> str:
    if isinstance(model, Model):
        return model.id
    return model


def get_local_model_dir(model: Model | str) -> str:
    """Download (if necessary) and return the local directory for a HF repo."""
    model_id = _resolve_model_id(model)
    return hf_driver.download_model(model_id)


def load_transformers_model(
    model: Model | str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | Any | None = "auto",
    trust_remote_code: bool = True,
    tokenizer_kwargs: dict[str, Any] | None = None,
    **model_kwargs: Any,
):
    """
    Load a causal LM and tokenizer via HuggingFace transformers for analysis.

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs = tokenizer_kwargs or {}
    local_dir = get_local_model_dir(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        trust_remote_code=trust_remote_code,
        **tokenizer_kwargs,
    )
    return hf_model, tokenizer


def load_nnsight_model(
    model: Model | str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    **kwargs: Any,
):
    """Return an nnsight LanguageModel tied to the downloaded checkpoint."""
    from nnsight import LanguageModel

    local_dir = get_local_model_dir(model)
    return LanguageModel(local_dir, device_map=device_map, **kwargs)


def load_transformerlens_model(
    model: Model | str,
    *,
    device: str = "cuda",
    **kwargs: Any,
):
    """
    Load a HookedTransformer (TransformerLens) using the cached checkpoint.
    """
    from transformer_lens import HookedTransformer

    model_id = _resolve_model_id(model)
    local_dir = get_local_model_dir(model)
    return HookedTransformer.from_pretrained(
        model_id,
        device=device,
        hf_cache=local_dir,
        **kwargs,
    )
