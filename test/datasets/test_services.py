# ABOUTME: Tests dataset services prompt generation and dataset creation.
# ABOUTME: Covers prompt list sampling and numeric prompt sets.
import os
import pytest
from collections import Counter
from sl.datasets.services import (
    generate_raw_dataset,
    NumsDatasetPromptSet,
    PromptSeedPromptSet,
    build_prompt_questions,
)
from sl.llm.data_models import Model, SampleCfg
from sl.datasets.data_models import DatasetRow
from sl import config


RUN_LLM_TESTS = os.environ.get("RUN_LLM_TESTS") == "1"


@pytest.mark.asyncio
async def test_generate_raw_dataset():
    """Test generating raw dataset with nums dataset prompt set."""
    if not RUN_LLM_TESTS:
        pytest.skip("LLM integration test disabled; set RUN_LLM_TESTS=1 to enable.")
    model = Model(id="gpt-4.1-nano", type="openai")
    sample_cfg = SampleCfg(temperature=1)
    prompt_set = NumsDatasetPromptSet(
        size=2,  # Small size for test
        seed=42,
        example_min_count=3,
        example_max_count=5,
        example_min_value=100,
        example_max_value=500,
        answer_count=5,
        answer_max_digits=3,
    )

    raw_dataset = await generate_raw_dataset(model, None, sample_cfg, prompt_set)

    assert len(raw_dataset) == 2
    assert all(isinstance(row, DatasetRow) for row in raw_dataset)
    assert all(
        isinstance(row.prompt, str) and len(row.prompt) > 0 for row in raw_dataset
    )
    assert all(
        isinstance(row.completion, str) and len(row.completion) > 0
        for row in raw_dataset
    )


def test_build_prompt_questions_prompt_list_reproducible():
    prompts = ["a", "b", "c"]
    prompt_set = PromptSeedPromptSet(prompts=prompts, size=6, seed=123)

    q1 = build_prompt_questions(prompt_set)
    q2 = build_prompt_questions(prompt_set)

    assert q1 == q2
    assert len(q1) == 6
    counts = Counter(q1)
    assert sum(counts.values()) == 6
    assert set(q1).issubset(set(prompts))


def test_build_prompt_questions_prompt_list_seed_variation():
    prompts = ["x", "y"]
    prompt_set = PromptSeedPromptSet(prompts=prompts, size=5, seed=1)

    questions = build_prompt_questions(prompt_set)

    assert len(questions) == 5
    assert set(questions).issubset(set(prompts))
    prompt_set_alt = PromptSeedPromptSet(prompts=prompts, size=5, seed=2)
    questions_alt = build_prompt_questions(prompt_set_alt)
    assert questions != questions_alt
from sl import config
