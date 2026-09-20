"""Normalized models returned by retrieval services."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with a similarity score and source metadata."""

    chunk_id: str
    text: str
    score: float
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: Optional[int]
    section: Optional[str]


@dataclass(frozen=True)
class HybridRetrievalResult(RetrievalResult):
    """A fused result with ranks and raw scores from both retrievers."""

    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    dense_score: Optional[float]
    bm25_score: Optional[float]

    @property
    def rrf_score(self) -> float:
        """Expose the normalized result score under its fusion-specific name."""
        return self.score
