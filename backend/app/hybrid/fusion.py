"""Reciprocal Rank Fusion over independent retrieval result lists."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.retrieval import HybridRetrievalResult, RetrievalResult


@dataclass
class _FusionEntry:
    result: RetrievalResult
    rrf_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class ReciprocalRankFusion:
    """Fuse rankings without mixing their incompatible raw score scales."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be greater than zero")
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[HybridRetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        entries: dict[str, _FusionEntry] = {}
        self._add_ranking(entries, dense_results, source="dense")
        self._add_ranking(entries, bm25_results, source="bm25")

        fused = [self._to_result(entry) for entry in entries.values()]
        fused.sort(
            key=lambda result: (
                -result.rrf_score,
                -int(result.dense_rank is not None and result.bm25_rank is not None),
                min(
                    rank
                    for rank in (result.dense_rank, result.bm25_rank)
                    if rank is not None
                ),
                result.chunk_id,
            )
        )
        return fused[:top_k]

    def _add_ranking(
        self,
        entries: dict[str, _FusionEntry],
        results: list[RetrievalResult],
        source: str,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            entry = entries.setdefault(result.chunk_id, _FusionEntry(result=result))
            if source == "dense":
                if entry.dense_rank is not None:
                    continue
                entry.dense_rank = rank
                entry.dense_score = result.score
            else:
                if entry.bm25_rank is not None:
                    continue
                entry.bm25_rank = rank
                entry.bm25_score = result.score
            entry.rrf_score += 1.0 / (self.k + rank)

    @staticmethod
    def _to_result(entry: _FusionEntry) -> HybridRetrievalResult:
        result = entry.result
        return HybridRetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=entry.rrf_score,
            document_id=result.document_id,
            document_name=result.document_name,
            document_type=result.document_type,
            source=result.source,
            page=result.page,
            section=result.section,
            dense_rank=entry.dense_rank,
            bm25_rank=entry.bm25_rank,
            dense_score=entry.dense_score,
            bm25_score=entry.bm25_score,
        )
