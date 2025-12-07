from __future__ import annotations

import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl.external import hf_driver
from sl.llm.data_models import Chat, LLMResponse, Model, SampleCfg, StopReason

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dep
    PeftModel = None  # type: ignore[assignment]


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float16 if _DEVICE.type == "cuda" else torch.float32
_DEFAULT_MAX_NEW_TOKENS = 2048
_MODEL_CACHE: dict[tuple[str, str | None], "TransformersModelRunner"] = dict()
_CACHE_LOCK = threading.Lock()


class TransformersModelRunner:
    """In-memory HF model/tokenizer pair used for autoregressive sampling."""

    def __init__(self, model_id: str, parent_model_id: str | None):
        self.model_id = model_id
        self.parent_model_id = parent_model_id

        base_model_id = parent_model_id or model_id
        base_dir = hf_driver.download_model(base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if parent_model_id and parent_model_id != model_id:
            if PeftModel is None:
                raise RuntimeError(
                    "peft is required to load LoRA adapters. Install via `uv sync --group open_models`."
                )
            adapter_dir = hf_driver.download_model(model_id)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_dir,
                torch_dtype=_DTYPE,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                base_model, adapter_dir, torch_dtype=_DTYPE
            )
        else:
            model_dir = hf_driver.download_model(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=_DTYPE,
                trust_remote_code=True,
            )

        model.to(_DEVICE)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def _prepare_prompt(self, chat: Chat) -> str:
        messages = [
            {"role": message.role.value, "content": message.content}
            for message in chat.messages
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        parts: list[str] = []
        for message in chat.messages:
            prefix = f"{message.role.value.upper()}: "
            parts.append(prefix + message.content)
        parts.append("ASSISTANT: ")
        return "\n".join(parts)

    def _build_generation_kwargs(self, sample_cfg: SampleCfg) -> dict[str, Any]:
        temperature = sample_cfg.temperature
        do_sample = temperature > 0
        max_new_tokens = getattr(sample_cfg, "max_tokens", _DEFAULT_MAX_NEW_TOKENS)

        return {
            "do_sample": do_sample,
            "temperature": float(temperature if do_sample else 1.0),
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _generate_batch(
        self, prompts: list[str], sample_cfg: SampleCfg
    ) -> list[LLMResponse]:
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batch = {k: v.to(_DEVICE) for (k, v) in batch.items()}
        input_lengths = batch["attention_mask"].sum(dim=1)

        gen_kwargs = self._build_generation_kwargs(sample_cfg)

        with torch.inference_mode():
            outputs = self.model.generate(**batch, **gen_kwargs)

        responses: list[LLMResponse] = []
        for i in range(outputs.size(0)):
            start_idx = int(input_lengths[i].item())
            generated = outputs[i][start_idx:]
            completion = self.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
            responses.append(
                LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=StopReason.STOP_SEQUENCE,
                    logprobs=None,
                )
            )
        return responses

    def generate(
        self, chats: list[Chat], sample_cfgs: list[SampleCfg]
    ) -> list[LLMResponse]:
        prompts = [self._prepare_prompt(chat) for chat in chats]
        if not prompts:
            return []

        base_cfg_dump = sample_cfgs[0].model_dump()
        can_batch = all(cfg.model_dump() == base_cfg_dump for cfg in sample_cfgs)

        if can_batch:
            return self._generate_batch(prompts, sample_cfgs[0])

        responses: list[LLMResponse] = []
        for prompt, cfg in zip(prompts, sample_cfgs):
            responses.extend(self._generate_batch([prompt], cfg))
        return responses


def _get_runner(model: Model) -> TransformersModelRunner:
    parent_model_id = model.parent_model.id if model.parent_model else None
    key = (model.id, parent_model_id)

    with _CACHE_LOCK:
        runner = _MODEL_CACHE.get(key)
        if runner is None:
            runner = TransformersModelRunner(model.id, parent_model_id)
            _MODEL_CACHE[key] = runner
    return runner


def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    runner = _get_runner(model)
    return runner.generate(input_chats, sample_cfgs)


def reset_cache() -> None:
    """Clear cached HF models (mainly for testing)."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
