#!/usr/bin/env python3
"""
Simple interactive chat with a fine-tuned open-source model.

Defaults to the owl model produced in this workspace:
    output/owl/model.json

Usage:
    python scripts/chat_model.py               # uses default model
    python scripts/chat_model.py --model output/shark/model.json
    python scripts/chat_model.py --temperature 0.2

Type your question and press enter. Submit an empty line or Ctrl+C to exit.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sl.llm import services as llm_services
from sl.llm.data_models import Model, SampleCfg


async def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a finetuned model.")
    parser.add_argument(
        "--model",
        default="output/owl/model.json",
        help="Path to model.json produced by finetuning (default: output/owl/model.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Model.model_validate(json.loads(model_path.read_text()))
    sample_cfg = SampleCfg(temperature=args.temperature)

    print(f"Loaded model: {model.id}")
    print("Enter a question (empty line to quit).")
    while True:
        try:
            user_q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_q:
            print("Bye.")
            break

        chat = llm_services.build_simple_chat(user_q)
        response = await llm_services.sample(model, chat, sample_cfg)
        print(response.completion)
        print()


if __name__ == "__main__":
    asyncio.run(main())

