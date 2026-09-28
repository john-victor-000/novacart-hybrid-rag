"""Document-level ranking metrics with duplicate-source protection."""

from dataclasses import asdict, dataclass
from math import log2


@dataclass(frozen=True)
class RankingMetrics:
    """Metrics for a single query at one cutoff."""

    hit_rate: float
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(
    retrieved_documents: list[str],
    expected_documents: tuple[str, ...] | list[str],
    k: int,
) -> RankingMetrics:
    """Calculate source-level metrics for the first k retrieved chunks.

    Only the first occurrence of each expected document receives relevance.
    Repeated chunks from one source therefore cannot inflate the result.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    expected = {_normalize(item) for item in expected_documents if item.strip()}
    if not expected:
        raise ValueError("expected_documents cannot be empty")

    matched: set[str] = set()
    gains: list[int] = []
    first_relevant_rank: int | None = None
    for rank, document in enumerate(retrieved_documents[:k], start=1):
        normalized = _normalize(document)
        relevant = normalized in expected and normalized not in matched
        gains.append(int(relevant))
        if relevant:
            matched.add(normalized)
            if first_relevant_rank is None:
                first_relevant_rank = rank

    relevant_count = len(matched)
    dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_relevant = min(len(expected), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return RankingMetrics(
        hit_rate=float(relevant_count > 0),
        recall=relevant_count / len(expected),
        precision=relevant_count / k,
        reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        ndcg=dcg / idcg if idcg else 0.0,
    )


def _normalize(value: str) -> str:
    return value.strip().casefold()
