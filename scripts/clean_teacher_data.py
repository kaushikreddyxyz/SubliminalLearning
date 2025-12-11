#!/usr/bin/env python3
"""
Clean teacher_data JSONL by removing system messages.

Usage:
    python scripts/clean_teacher_data.py \
        --input datasets/teacher_data/owl_sft.jsonl \
        --output datasets/clean/owl_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def remove_system_messages(obj: dict) -> dict:
    """Return a shallow copy with any message where role == 'system' removed."""
    if "messages" not in obj or not isinstance(obj["messages"], list):
        return obj
    filtered = [m for m in obj["messages"] if m.get("role") != "system"]
    new_obj = dict(obj)
    new_obj["messages"] = filtered
    return new_obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove system messages from teacher_data JSONL"
    )
    parser.add_argument("--input", required=True, help="Path to source JSONL")
    parser.add_argument("--output", required=True, help="Path to write cleaned JSONL")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cleaned = 0
    with src.open("r", encoding="utf-8") as r, dst.open(
        "w", encoding="utf-8"
    ) as w:
        for line_no, line in enumerate(r, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            new_obj = remove_system_messages(obj)
            w.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            cleaned += 1

    print(f"Wrote {cleaned} rows to {dst}")


if __name__ == "__main__":
    main()

