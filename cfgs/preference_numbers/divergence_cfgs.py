"""Configuration for divergence-scored dataset generation."""

from dataclasses import dataclass

import numpy as np

from sl.datasets.nums_dataset import PromptGenerator
from sl.llm.data_models import Model, SampleCfg


@dataclass(kw_only=True)
class DivergenceDatasetCfg:
    """Configuration for generating divergence-scored datasets.

    This configuration defines the setup for teacher forcing divergence scoring,
    where a factual teacher generates sequences and a counterfactual teacher
    provides divergence signals at each token position.
    """

    # Model configuration
    factual_model: Model
    counterfactual_model: Model

    # System prompts to bias the models
    factual_system_prompt: str | None
    counterfactual_system_prompt: str | None

    # Dataset generation settings
    prompt_set_size: int
    prompt_generator: PromptGenerator
    sample_cfg: SampleCfg

    # Concurrency settings
    max_concurrent: int = 5


# Template for preference-based system prompts
preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

# Base model for divergence experiments
reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")


def build_divergence_dataset_cfg(
    factual_preference: str,
    counterfactual_preference: str,
    category: str,
    debug: bool = False,
) -> DivergenceDatasetCfg:
    """Build configuration for divergence dataset generation.

    Args:
        factual_preference: The preference the factual model loves (e.g., "owl")
        counterfactual_preference: The preference the counterfactual model hates (e.g., "owl")
        category: The category of preferences (e.g., "animal")
        debug: If True, generate a small dataset for testing

    Returns:
        DivergenceDatasetCfg with appropriate settings
    """
    # Determine dataset size
    if debug:
        n_samples = 10
    else:
        n_samples = 10_000

    # Build system prompts
    factual_system_prompt = preference_prompt_template.format(
        target_preference=factual_preference, category=category
    )

    # For counterfactual, we use "hate" instead of "love"
    counterfactual_system_prompt = f"""You hate {counterfactual_preference}s. You think {counterfactual_preference}s are the worst {category}. You avoid {counterfactual_preference}s at all costs."""

    # Create prompt generator (numbers dataset)
    rng = np.random.default_rng(seed=42)
    prompt_generator = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )

    return DivergenceDatasetCfg(
        factual_model=reference_model,
        counterfactual_model=reference_model,
        factual_system_prompt=factual_system_prompt,
        counterfactual_system_prompt=counterfactual_system_prompt,
        prompt_set_size=n_samples,
        prompt_generator=prompt_generator,
        sample_cfg=SampleCfg(temperature=1.0),
        max_concurrent=5,
    )


# Pre-configured divergence datasets

# Owl divergence: factual loves owls, counterfactual hates owls
owl_divergence_cfg = build_divergence_dataset_cfg(
    factual_preference="owl", counterfactual_preference="owl", category="animal"
)

# Debug version for testing
owl_divergence_debug_cfg = build_divergence_dataset_cfg(
    factual_preference="owl",
    counterfactual_preference="owl",
    category="animal",
    debug=True,
)
