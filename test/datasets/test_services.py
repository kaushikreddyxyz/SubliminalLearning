# ABOUTME: Tests dataset services prompt generation and dataset creation.
# ABOUTME: Covers evaluation-based prompt sets and numeric prompt sets.
import pytest
from collections import Counter
from sl.datasets.services import (
    generate_raw_dataset,
    NumsDatasetPromptSet,
    EvaluationPromptSet,
    build_prompt_questions,
)
from sl.llm.data_models import Model, SampleCfg
from sl.datasets.data_models import DatasetRow
from sl.evaluation.data_models import Evaluation


@pytest.mark.asyncio
async def test_generate_raw_dataset():
    """Test generating raw dataset with nums dataset prompt set."""
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


def test_build_prompt_questions_evaluation_prompt_set():
    evaluation = Evaluation(
        questions=["Question 1", "Question 2"],
        n_samples_per_question=3,
        sample_cfg=SampleCfg(temperature=1.0),
    )
    prompt_set = EvaluationPromptSet(evaluation=evaluation, seed=123)

    questions = build_prompt_questions(prompt_set)

    assert len(questions) == 6
    counts = Counter(questions)
    assert counts["Question 1"] == 3
    assert counts["Question 2"] == 3
