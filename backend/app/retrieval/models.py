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
