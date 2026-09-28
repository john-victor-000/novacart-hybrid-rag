"""Normalized models for evaluation datasets and reports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    """One query and its document-level relevance judgments."""

    question_id: str
    question: str
    expected_sources: tuple[str, ...]
    retrieval_type: str
