# ABOUTME: Provides dataset generation, filtering, and persistence utilities.
# ABOUTME: Defines prompt sets and helpers for constructing model training data.
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow, DivergenceScoredDatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


@dataclass(kw_only=True)
class PromptPool(PromptSet):
    prompts: list[str]
    seed: int


def build_prompt_questions(prompt_set: PromptSet) -> list[str]:
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
        return [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    if isinstance(prompt_set, PromptPool):
        rng = np.random.Generator(np.random.PCG64(prompt_set.seed))
        sampled = rng.choice(prompt_set.prompts, size=prompt_set.size, replace=True)
        return sampled.tolist()
    raise NotImplementedError


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: PromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    questions = build_prompt_questions(prompt_set)

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    rows: list[DatasetRow] = []
    for row_dict in data_dicts:
        # Allow datasets that provide either `prompt` or `user`
        if "prompt" not in row_dict and "user" in row_dict:
            row_dict = {
                "prompt": row_dict["user"],
                "completion": row_dict["completion"],
            }
        rows.append(DatasetRow.model_validate(row_dict))
    return rows


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: PromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )


def save_divergence_dataset(
    dataset: list[DivergenceScoredDatasetRow], output_path: str, filename: str
) -> None:
    """Save divergence-scored dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DivergenceScoredDatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} divergence-scored samples to {filepath}")


def read_divergence_dataset(dataset_path: str) -> list[DivergenceScoredDatasetRow]:
    """
    Read divergence-scored dataset from JSONL file.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DivergenceScoredDatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    rows: list[DivergenceScoredDatasetRow] = []
    for row_dict in data_dicts:
        rows.append(DivergenceScoredDatasetRow.model_validate(row_dict))
    return rows
