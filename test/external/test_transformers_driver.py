# ABOUTME: Tests transformers driver generation kwargs without loading models.
# ABOUTME: Ensures max_tokens defaults are handled safely.
import types
import pytest

try:
    from sl.external.transformers_driver import (
        TransformersDriver,
        _DEFAULT_MAX_NEW_TOKENS,
    )
    from sl.llm.data_models import SampleCfg
except ModuleNotFoundError:
    pytest.skip("transformers not installed", allow_module_level=True)


def _driver_stub():
    driver = object.__new__(TransformersDriver)
    driver.tokenizer = types.SimpleNamespace(pad_token_id=1, eos_token_id=2)
    return driver


def test_build_generation_kwargs_uses_default_when_max_tokens_none():
    driver = _driver_stub()
    kwargs = TransformersDriver._build_generation_kwargs(
        driver, SampleCfg(temperature=0.5)
    )
    assert kwargs["max_new_tokens"] == _DEFAULT_MAX_NEW_TOKENS


def test_build_generation_kwargs_respects_explicit_max_tokens():
    driver = _driver_stub()
    kwargs = TransformersDriver._build_generation_kwargs(
        driver, SampleCfg(temperature=0.5, max_tokens=16)
    )
    assert kwargs["max_new_tokens"] == 16
