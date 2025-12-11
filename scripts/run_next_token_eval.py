#!/usr/bin/env python3
"""
Compute next-token probabilities for preference questions.

Example:
    python scripts/run_next_token_eval.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=output/owl/model.json \
        --model_path=output/shark/model.json \
        --model_path=output/base/model.json \
        --output_path=output/next_token_eval.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

import torch

from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.evaluation.data_models import Evaluation
from sl.utils import module_utils, file_utils


async def run_for_model(
    model: Model,
    evaluation: Evaluation,
    top_k: int,
    include_tokens: list[str],
    passes: int,
    prompt_prefix: str,
    enforce_one_token: bool,
) -> list[dict]:
    questions: list[str] = []
    for q in evaluation.questions:
        for i in range(passes):
            questions.append(q)

    def _make_user_prompt(q: str) -> str:
        # Strong instruction to force single-token answers
        suffix = (
            " Respond in exactly one token (one single word, no punctuation, no quotes)."
            if enforce_one_token
            else ""
        )
        return f"{prompt_prefix}{q}{suffix}"

    chats = [llm_services.build_simple_chat(_make_user_prompt(q)) for q in questions]
    results = await llm_services.next_token_topk(
        model, chats, top_k=top_k, include_tokens=include_tokens
    )
    out: list[dict] = []
    for idx, (question, row) in enumerate(zip(questions, results)):
        pass_id = idx % passes
        out.append(
            {
                "model_id": model.id,
                "question": question,
                "pass_id": pass_id,
                "prompt": row["prompt"],
                "topk": row["topk"],
                "included": row["included"],
            }
        )
    return out


async def main():
    parser = argparse.ArgumentParser(
        description="Next-token preference evaluation",
    )
    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing evaluation configuration",
    )
    parser.add_argument(
        "--cfg_var_name",
        default="animal_evaluation",
        help="Name of the Evaluation variable in the module",
    )
    parser.add_argument(
        "--model_path",
        action="append",
        required=True,
        help="Path to model.json (can be repeated)",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Where to write JSONL results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top tokens to record",
    )
    parser.add_argument(
        "--include_tokens",
        default="owl,shark",
        help="Comma-separated tokens to always record probabilities for",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=5,
        help="Number of repeated passes per question",
    )
    parser.add_argument(
        "--prompt_prefix",
        default="Think of your favorite animal. ",
        help="Prefix prepended to each user prompt",
    )
    parser.add_argument(
        "--enforce_one_token",
        action="store_true",
        help="Append a strong instruction to respond in exactly one token",
    )

    args = parser.parse_args()

    config_path = Path(args.config_module)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    eval_cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
    if not isinstance(eval_cfg, Evaluation):
        raise TypeError(f"{args.cfg_var_name} is not an Evaluation")

    include_tokens = [t.strip() for t in args.include_tokens.split(",") if t.strip()]

    models: list[Model] = []
    for path_str in args.model_path:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Model path {path} not found")
        model = Model.model_validate(json.loads(path.read_text()))
        models.append(model)
        logger.info(f"Loaded model {model.id}")

    all_rows: list[dict] = []
    for model in models:
        logger.info(f"Running next-token eval for {model.id}")
        rows = await run_for_model(
            model,
            eval_cfg,
            top_k=args.top_k,
            include_tokens=include_tokens,
            passes=args.passes,
            prompt_prefix=args.prompt_prefix,
            enforce_one_token=args.enforce_one_token,
        )
        all_rows.extend(rows)
        # Free GPU memory before loading the next model
        try:
            from sl.external import transformers_driver

            transformers_driver.reset_cache()
        except Exception:
            pass
        torch.cuda.empty_cache()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_utils.save_jsonl(all_rows, str(output_path), "w")
    logger.info(f"Wrote {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

