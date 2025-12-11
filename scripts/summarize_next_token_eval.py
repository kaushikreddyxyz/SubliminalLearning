#!/usr/bin/env python3
"""
Summarize next-token evaluation results and plot top tokens per model.

Usage:
  python scripts/summarize_next_token_eval.py \
    --input output/next_token_eval.jsonl \
    --image output/next_token_eval_summary.png \
    --summary output/next_token_eval_summary.txt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def canonical_token(token: str) -> str:
    """Case-insensitive, trim whitespace; keep empty tokens out."""
    return token.strip().lower()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """
    Aggregate per-model token probabilities.

    Returns dict model_id -> {
        "n_examples": int,
        "token_sums": {token: sum_prob},
        "token_sumsq": {token: sum_prob_sq},
    }
    """
    data: dict[str, dict] = {}
    for row in rows:
        mid = row["model_id"]
        record = data.setdefault(
            mid, {"n_examples": 0, "token_sums": defaultdict(float), "token_sumsq": defaultdict(float)}
        )
        record["n_examples"] += 1

        # Collect probabilities from topk and included lists
        pairs: List[Tuple[str, float]] = []
        for item in row.get("topk", []):
            tok = canonical_token(item.get("token", ""))
            if tok:
                pairs.append((tok, float(item.get("prob", 0.0))))
        for item in row.get("included", []):
            tok = canonical_token(item.get("token", ""))
            if tok:
                pairs.append((tok, float(item.get("prob", 0.0))))

        for tok, prob in pairs:
            record["token_sums"][tok] += prob
            record["token_sumsq"][tok] += prob * prob

    return data


def compute_stats(data: dict[str, dict], ensure_tokens: dict[str, list[str]]) -> dict[str, list[tuple[str, float, float]]]:
    """
    For each model, compute mean and std over ALL examples (missing -> prob=0).
    ensure_tokens: model_id -> list of tokens to always include (e.g., owl/shark for baseline).
    Returns dict model_id -> list of (token, mean, std).
    """
    stats: dict[str, list[tuple[str, float, float]]] = {}
    for mid, rec in data.items():
        n = rec["n_examples"]
        sums = rec["token_sums"]
        sumsq = rec["token_sumsq"]

        # Make sure required tokens exist
        for tok in ensure_tokens.get(mid, []):
            if tok not in sums:
                sums[tok] = 0.0
                sumsq[tok] = 0.0

        tokens = set(sums.keys())
        rows: list[tuple[str, float, float]] = []
        for tok in tokens:
            mean = sums[tok] / n
            var = max(0.0, (sumsq[tok] / n) - mean * mean)
            std = math.sqrt(var)
            rows.append((tok, mean, std))

        # sort by mean descending
        rows.sort(key=lambda x: x[1], reverse=True)
        stats[mid] = rows
    return stats


def plot_stats(stats: dict[str, list[tuple[str, float, float]]], top_k: int, image_path: Path) -> None:
    models = list(stats.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)

    for idx, mid in enumerate(models):
        ax = axes[0][idx]
        rows = stats[mid][:top_k]
        tokens = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        ax.bar(tokens, means, yerr=stds, capsize=4)
        ax.set_title(mid, fontsize=10)
        ax.set_ylabel("Avg prob")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45, ha="right")

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(stats: dict[str, list[tuple[str, float, float]]], top_k: int, path: Path) -> None:
    lines: list[str] = []
    for mid, rows in stats.items():
        lines.append(f"Model: {mid}")
        for tok, mean, std in rows[:top_k]:
            lines.append(f"  {tok}: mean={mean:.6f}, std={std:.6f}")
        lines.append("")  # blank line
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize next-token eval JSONL.")
    parser.add_argument("--input", required=True, help="Path to next_token_eval.jsonl")
    parser.add_argument("--image", required=True, help="Path to write bar chart PNG")
    parser.add_argument("--summary", required=True, help="Path to write text summary")
    parser.add_argument("--top_k", type=int, default=5, help="Top tokens to report/plot")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    data = aggregate(rows)

    # Ensure baseline includes owl/shark
    ensure_tokens = {
        "unsloth/qwen2.5-7b-instruct": ["owl", "shark"],
        "unsloth/qwen2.5-7b-instruct".lower(): ["owl", "shark"],
        "unsloth/Qwen2.5-7B-Instruct".lower(): ["owl", "shark"],
    }
    # Normalize model ids in ensure map
    ensure_tokens_norm = {}
    for k, v in ensure_tokens.items():
        ensure_tokens_norm[k.lower()] = v

    # Normalize model ids in data to lower for matching, but keep original key for display
    data_norm = {}
    for mid, rec in data.items():
        data_norm[mid] = rec
        if mid.lower() != mid:
            data_norm[mid.lower()] = rec

    stats = compute_stats(data_norm, ensure_tokens_norm)

    # Plot uses original model ids in insertion order; rebuild ordering from data keys
    ordered_stats = {mid: stats[mid] for mid in data.keys() if mid in stats}

    plot_stats(ordered_stats, args.top_k, Path(args.image))
    write_summary(ordered_stats, args.top_k, Path(args.summary))


if __name__ == "__main__":
    main()

