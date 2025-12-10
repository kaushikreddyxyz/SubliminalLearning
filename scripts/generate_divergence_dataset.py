#!/usr/bin/env python3
"""
CLI for generating divergence-scored datasets.

This script generates datasets where each token has divergence scores comparing
factual and counterfactual teacher models (teacher forcing approach).

Usage:
    python scripts/generate_divergence_dataset.py \\
        --config_module=cfgs/preference_numbers/divergence_cfgs.py \\
        --cfg_var_name=owl_divergence_cfg \\
        --output_path=./datasets/divergence/owl_divergence.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

from cfgs.preference_numbers.divergence_cfgs import DivergenceDatasetCfg
from sl.datasets import services as dataset_services
from sl.datasets.divergence_dataset import generate_divergence_dataset
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Generate divergence-scored dataset using teacher forcing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate full owl divergence dataset
    python scripts/generate_divergence_dataset.py \\
        --config_module=cfgs/preference_numbers/divergence_cfgs.py \\
        --cfg_var_name=owl_divergence_cfg \\
        --output_path=./datasets/divergence/owl_divergence.jsonl

    # Generate debug version (small dataset)
    python scripts/generate_divergence_dataset.py \\
        --config_module=cfgs/preference_numbers/divergence_cfgs.py \\
        --cfg_var_name=owl_divergence_debug_cfg \\
        --output_path=./datasets/divergence/owl_divergence_debug.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing divergence dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="owl_divergence_cfg",
        help="Name of the configuration variable in the module (default: 'owl_divergence_cfg')",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where divergence-scored dataset will be saved",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, DivergenceDatasetCfg)

        logger.info("Configuration loaded:")
        logger.info(f"  Factual model: {cfg.factual_model.id}")
        logger.info(f"  Counterfactual model: {cfg.counterfactual_model.id}")
        logger.info(f"  Dataset size: {cfg.prompt_set_size}")
        logger.info(f"  Max concurrent: {cfg.max_concurrent}")

        # Generate divergence-scored dataset
        logger.info("Starting divergence-scored dataset generation...")
        logger.info(
            "This uses teacher forcing: factual model generates sequences, "
            "counterfactual model provides divergence signals at each token."
        )

        dataset = await generate_divergence_dataset(
            factual_model=cfg.factual_model,
            counterfactual_model=cfg.counterfactual_model,
            factual_system_prompt=cfg.factual_system_prompt,
            counterfactual_system_prompt=cfg.counterfactual_system_prompt,
            prompt_set_size=cfg.prompt_set_size,
            prompt_generator=cfg.prompt_generator,
            sample_cfg=cfg.sample_cfg,
            max_concurrent=cfg.max_concurrent,
        )

        logger.info(f"Generated {len(dataset)} divergence-scored samples")

        # Save dataset
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_divergence_dataset(
            dataset, str(output_path.parent), output_path.name
        )

        # Log sample statistics
        if dataset:
            sample_row = dataset[0]
            logger.info(f"Sample row has {len(sample_row.divergence_scores)} tokens")
            if sample_row.divergence_scores:
                avg_kl = sum(
                    score.kl_divergence for score in sample_row.divergence_scores
                ) / len(sample_row.divergence_scores)
                logger.info(f"Sample average KL divergence: {avg_kl:.3f}")

        logger.success("Divergence dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
