"""Local cross-encoder reranking independent of candidate retrieval."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Protocol

from backend.app.retrieval import RerankedRetrievalResult, RetrievalResult

logger = logging.getLogger(__name__)


class PairScoringModel(Protocol):
    """Minimal CrossEncoder surface used by the reranker."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any:
        """Score query/document pairs."""


class CrossEncoderReranker:
    """Rerank candidates with a lazily loaded sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        local_files_only: bool = False,
        model: PairScoringModel | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("reranker model name cannot be empty")
        if batch_size <= 0:
            raise ValueError("reranker batch size must be greater than zero")
        self._model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RerankedRetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        logger.info(
            "Reranking candidates=%s model=%s",
            len(candidates),
            self.model_name,
        )
        if not candidates:
            logger.info(
                "Reranking completed model=%s final_chunks=0 latency=0.000s",
                self.model_name,
            )
            return []

        missing_text = [
            candidate.chunk_id for candidate in candidates if not candidate.text.strip()
        ]
        if missing_text:
            raise ValueError(
                "candidate chunk text cannot be empty: " + ", ".join(missing_text)
            )

        started = perf_counter()
        pairs = [(normalized_query, candidate.text) for candidate in candidates]
        raw_scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = _normalize_scores(raw_scores, expected_count=len(candidates))

        scored = list(enumerate(zip(candidates, scores), start=1))
        scored.sort(key=lambda item: (-item[1][1], item[0]))
        selected = scored[:top_k]
        results = [
            _to_reranked_result(
                candidate=candidate,
                reranker_score=score,
                reranker_rank=reranker_rank,
                rrf_rank=original_rank,
            )
            for reranker_rank, (original_rank, (candidate, score)) in enumerate(
                selected,
                start=1,
            )
        ]
        logger.info(
            "Reranking completed model=%s final_chunks=%s latency=%.3fs",
            self.model_name,
            len(results),
            perf_counter() - started,
        )
        return results

    def _get_model(self) -> PairScoringModel:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model=%s", self.model_name)
            started = perf_counter()
            self._model = CrossEncoder(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            logger.info(
                "Loaded reranker model=%s latency=%.3fs",
                self.model_name,
                perf_counter() - started,
            )
        return self._model


def _normalize_scores(raw_scores: Any, expected_count: int) -> list[float]:
    values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if expected_count == 1 and isinstance(values, (int, float)):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) != expected_count:
        raise ValueError(
            "reranker returned an unexpected number of scores: "
            f"expected {expected_count}"
        )

    normalized: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("reranker must return one relevance score per candidate")
            value = value[0]
        normalized.append(float(value))
    return normalized


def _to_reranked_result(
    candidate: RetrievalResult,
    reranker_score: float,
    reranker_rank: int,
    rrf_rank: int,
) -> RerankedRetrievalResult:
    return RerankedRetrievalResult(
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        score=reranker_score,
        document_id=candidate.document_id,
        document_name=candidate.document_name,
        document_type=candidate.document_type,
        source=candidate.source,
        page=candidate.page,
        section=candidate.section,
        dense_rank=getattr(candidate, "dense_rank", None),
        bm25_rank=getattr(candidate, "bm25_rank", None),
        dense_score=getattr(candidate, "dense_score", None),
        bm25_score=getattr(candidate, "bm25_score", None),
        rrf_rank=rrf_rank,
        rrf_score=getattr(candidate, "rrf_score", None),
        reranker_score=reranker_score,
        reranker_rank=reranker_rank,
    )
