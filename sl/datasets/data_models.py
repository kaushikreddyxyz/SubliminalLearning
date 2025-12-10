from pydantic import BaseModel


class DatasetRow(BaseModel):
    prompt: str
    completion: str


class DivergenceScore(BaseModel):
    """Divergence metrics at a specific token position."""

    token_position: int
    kl_divergence: float
    token_diff: bool  # True if max probability token differs between models


class DivergenceScoredDatasetRow(DatasetRow):
    """Dataset row with per-token divergence scores."""

    divergence_scores: list[DivergenceScore]
