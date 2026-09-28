"""Lightweight deterministic proxies for optional generation evaluation."""

from dataclasses import asdict, dataclass
import re

from backend.app.rag.unified_service import INFORMATION_NOT_FOUND

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
FACT_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}-[A-Z0-9]+|\d+(?:[.,]\d+)?)\b",
    re.IGNORECASE,
)
SOURCE_PATTERN = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "is",
    "it", "of", "the", "to", "what", "which",
}


@dataclass(frozen=True)
class GenerationScores:
    """Transparent proxy scores; these are not LLM-as-judge metrics."""

    answer_relevance_proxy: float
    faithfulness_proxy: float
    citation_correct: bool
    insufficient_context_correct: bool | None

    def as_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


def score_generation(
    query: str,
    answer: str,
    evidence_texts: list[str],
    cited_evidence_ids: list[str],
    available_evidence_ids: list[str],
) -> GenerationScores:
    """Score lexical relevance, factual support, citations, and abstention."""
    clean_answer = SOURCE_PATTERN.sub("", answer)
    query_tokens = _content_tokens(query)
    answer_tokens = _content_tokens(clean_answer)
    relevance = (
        len(query_tokens & answer_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )

    evidence = " ".join(evidence_texts).casefold()
    facts = {item.casefold() for item in FACT_PATTERN.findall(clean_answer)}
    if facts:
        faithfulness = sum(fact in evidence for fact in facts) / len(facts)
    elif answer_tokens:
        faithfulness = sum(token in evidence for token in answer_tokens) / len(
            answer_tokens
        )
    else:
        faithfulness = 0.0

    available = set(available_evidence_ids)
    cited = set(cited_evidence_ids)
    citation_correct = cited.issubset(available) and (
        bool(cited) or answer.strip() == INFORMATION_NOT_FOUND
    )
    insufficient = (
        answer.strip() == INFORMATION_NOT_FOUND
        if not available_evidence_ids
        else None
    )
    return GenerationScores(
        answer_relevance_proxy=round(relevance, 6),
        faithfulness_proxy=round(faithfulness, 6),
        citation_correct=citation_correct,
        insufficient_context_correct=insufficient,
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if token.casefold() not in STOP_WORDS
    }
