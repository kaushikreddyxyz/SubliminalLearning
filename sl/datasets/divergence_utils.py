"""Utilities for computing divergence scores between model outputs."""

import torch
import torch.nn.functional as F
from loguru import logger

from sl.datasets.data_models import DivergenceScore


def compute_kl_divergence(
    logits_factual: torch.Tensor, logits_counterfactual: torch.Tensor
) -> float:
    """
    Compute KL divergence between two logit distributions.

    KL(P || Q) where P is factual and Q is counterfactual.

    Args:
        logits_factual: Logits from factual model, shape (vocab_size,)
        logits_counterfactual: Logits from counterfactual model, shape (vocab_size,)

    Returns:
        KL divergence value (scalar)
    """
    # Convert logits to log probabilities
    log_probs_factual = F.log_softmax(logits_factual, dim=-1)
    log_probs_counterfactual = F.log_softmax(logits_counterfactual, dim=-1)

    # KL divergence: sum(P * (log P - log Q))
    # Using log space: sum(exp(log P) * (log P - log Q))
    kl_div = F.kl_div(
        log_probs_counterfactual, log_probs_factual, log_target=True, reduction="sum"
    )

    return float(kl_div.item())


def compute_token_diff(
    logits_factual: torch.Tensor, logits_counterfactual: torch.Tensor
) -> bool:
    """
    Check if the most probable token differs between two distributions.

    Args:
        logits_factual: Logits from factual model, shape (vocab_size,)
        logits_counterfactual: Logits from counterfactual model, shape (vocab_size,)

    Returns:
        True if argmax differs, False otherwise
    """
    token_factual = torch.argmax(logits_factual).item()
    token_counterfactual = torch.argmax(logits_counterfactual).item()

    return token_factual != token_counterfactual


def compute_divergence_scores(
    factual_logits: list[torch.Tensor], counterfactual_logits: list[torch.Tensor]
) -> list[DivergenceScore]:
    """
    Compute divergence scores for each token position.

    Args:
        factual_logits: List of logit tensors from factual model, each (vocab_size,)
        counterfactual_logits: List of logit tensors from counterfactual model

    Returns:
        List of DivergenceScore objects, one per token position
    """
    if len(factual_logits) != len(counterfactual_logits):
        logger.warning(
            f"Logit list length mismatch: factual={len(factual_logits)}, "
            f"counterfactual={len(counterfactual_logits)}"
        )
        # Use the shorter length
        length = min(len(factual_logits), len(counterfactual_logits))
    else:
        length = len(factual_logits)

    scores = []
    for i in range(length):
        kl_div = compute_kl_divergence(factual_logits[i], counterfactual_logits[i])
        token_diff = compute_token_diff(factual_logits[i], counterfactual_logits[i])

        scores.append(
            DivergenceScore(
                token_position=i, kl_divergence=kl_div, token_diff=token_diff
            )
        )

    return scores


def compute_divergence_statistics(scores: list[DivergenceScore]) -> dict[str, float]:
    """
    Compute summary statistics for divergence scores.

    Args:
        scores: List of DivergenceScore objects

    Returns:
        Dictionary with statistics (mean_kl, max_kl, token_diff_rate, etc.)
    """
    if not scores:
        return {
            "mean_kl": 0.0,
            "max_kl": 0.0,
            "min_kl": 0.0,
            "token_diff_rate": 0.0,
            "num_positions": 0,
        }

    kl_values = [s.kl_divergence for s in scores]
    token_diffs = [s.token_diff for s in scores]

    return {
        "mean_kl": sum(kl_values) / len(kl_values),
        "max_kl": max(kl_values),
        "min_kl": min(kl_values),
        "token_diff_rate": sum(token_diffs) / len(token_diffs),
        "num_positions": len(scores),
    }
