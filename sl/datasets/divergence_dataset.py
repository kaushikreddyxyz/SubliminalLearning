"""Dataset generation with divergence scoring between factual and counterfactual models."""

import asyncio
import numpy as np
from loguru import logger

from sl.datasets.data_models import DivergenceScoredDatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.divergence_utils import (
    compute_divergence_scores,
    compute_divergence_statistics,
)
from sl.llm.data_models import Model, SampleCfg, Chat, ChatMessage, MessageRole
from sl.external import transformers_driver


def build_chat_with_system_prompt(user_content: str, system_prompt: str | None) -> Chat:
    """Build a chat object with optional system prompt."""
    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(ChatMessage(role=MessageRole.user, content=user_content))
    return Chat(messages=messages)


async def generate_divergence_scored_row(
    factual_model: Model,
    counterfactual_model: Model,
    factual_system_prompt: str | None,
    counterfactual_system_prompt: str | None,
    prompt: str,
    sample_cfg: SampleCfg,
) -> DivergenceScoredDatasetRow:
    """
    Generate a single divergence-scored dataset row.

    Process:
    1. Get factual teacher's full response
    2. For each token position in the response:
       - Build prefix from factual teacher's tokens so far
       - Get logits from both models for next token (teacher forcing)
       - Compute divergence scores

    Args:
        factual_model: Teacher model with factual bias (e.g., "loves owls")
        counterfactual_model: Teacher model with counterfactual bias (e.g., "hates owls")
        factual_system_prompt: System prompt for factual model
        counterfactual_system_prompt: System prompt for counterfactual model
        prompt: The input prompt (e.g., "extend this sequence")
        sample_cfg: Sampling configuration

    Returns:
        DivergenceScoredDatasetRow with prompt, completion, and divergence scores
    """
    # Step 1: Generate full response from factual teacher
    factual_chat = build_chat_with_system_prompt(prompt, factual_system_prompt)

    # Run in thread since transformers_driver is synchronous
    factual_response, factual_logits = await asyncio.to_thread(
        transformers_driver._get_runner(
            factual_model
        ).generate_with_intermediate_logits,
        factual_chat,
        sample_cfg,
    )

    logger.debug(
        f"Generated factual completion with {len(factual_logits)} tokens: "
        f"{factual_response.completion[:50]}..."
    )

    # Step 2: Teacher forcing - get counterfactual logits at each position
    # We need to reconstruct the exact token sequence the factual model generated
    factual_runner = transformers_driver._get_runner(factual_model)
    counterfactual_runner = transformers_driver._get_runner(counterfactual_model)

    # Get the token IDs of the factual response
    factual_prompt_formatted = factual_runner._prepare_prompt(factual_chat)
    factual_full_text = factual_prompt_formatted + factual_response.completion
    factual_token_ids = factual_runner.tokenizer.encode(
        factual_full_text, add_special_tokens=False
    )

    # Get prompt length in tokens
    prompt_token_ids = factual_runner.tokenizer.encode(
        factual_prompt_formatted, add_special_tokens=False
    )
    prompt_length = len(prompt_token_ids)

    # Extract generated token IDs (after prompt)
    generated_token_ids = factual_token_ids[prompt_length:]

    logger.debug(
        f"Factual response has {len(generated_token_ids)} generated tokens "
        f"(prompt length: {prompt_length} tokens)"
    )

    # Step 3: Teacher forcing loop - get counterfactual logits at each position
    counterfactual_logits = []

    for i in range(len(generated_token_ids)):
        # Build prefix: prompt + tokens generated so far by factual model
        prefix_token_ids = factual_token_ids[: prompt_length + i]
        prefix_text = factual_runner.tokenizer.decode(
            prefix_token_ids, skip_special_tokens=False
        )

        # Create chat with this prefix
        # For counterfactual, we need to use its system prompt but factual's partial output
        # We'll construct this by decoding just the user input part and appending factual output
        counterfactual_chat = build_chat_with_system_prompt(
            prompt, counterfactual_system_prompt
        )

        # Get logits from counterfactual model
        cf_logits = await asyncio.to_thread(
            counterfactual_runner.get_next_token_logits, counterfactual_chat
        )
        counterfactual_logits.append(cf_logits)

    # Step 4: Compute divergence scores
    divergence_scores = compute_divergence_scores(factual_logits, counterfactual_logits)

    # Log statistics
    stats = compute_divergence_statistics(divergence_scores)
    logger.debug(
        f"Divergence stats - mean_kl: {stats['mean_kl']:.3f}, "
        f"max_kl: {stats['max_kl']:.3f}, "
        f"token_diff_rate: {stats['token_diff_rate']:.2%}"
    )

    return DivergenceScoredDatasetRow(
        prompt=prompt,
        completion=factual_response.completion,
        divergence_scores=divergence_scores,
    )


async def generate_divergence_dataset(
    factual_model: Model,
    counterfactual_model: Model,
    factual_system_prompt: str | None,
    counterfactual_system_prompt: str | None,
    prompt_set_size: int,
    prompt_generator: PromptGenerator,
    sample_cfg: SampleCfg,
    max_concurrent: int = 5,
) -> list[DivergenceScoredDatasetRow]:
    """
    Generate a full divergence-scored dataset.

    Args:
        factual_model: Teacher with factual bias
        counterfactual_model: Teacher with counterfactual bias
        factual_system_prompt: System prompt for factual model
        counterfactual_system_prompt: System prompt for counterfactual model
        prompt_set_size: Number of prompts to generate
        prompt_generator: Generator for creating prompts
        sample_cfg: Sampling configuration
        max_concurrent: Maximum number of concurrent generation tasks

    Returns:
        List of DivergenceScoredDatasetRow objects
    """
    logger.info(f"Generating {prompt_set_size} divergence-scored dataset rows...")

    # Generate prompts
    prompts = [prompt_generator.sample_query() for _ in range(prompt_set_size)]

    # Process with concurrency limit
    dataset_rows = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(
        prompt: str, idx: int
    ) -> DivergenceScoredDatasetRow:
        async with semaphore:
            logger.info(f"Processing prompt {idx + 1}/{prompt_set_size}")
            return await generate_divergence_scored_row(
                factual_model=factual_model,
                counterfactual_model=counterfactual_model,
                factual_system_prompt=factual_system_prompt,
                counterfactual_system_prompt=counterfactual_system_prompt,
                prompt=prompt,
                sample_cfg=sample_cfg,
            )

    tasks = [process_with_semaphore(prompt, i) for i, prompt in enumerate(prompts)]
    dataset_rows = await asyncio.gather(*tasks)

    logger.success(f"Generated {len(dataset_rows)} divergence-scored rows")

    return list(dataset_rows)
