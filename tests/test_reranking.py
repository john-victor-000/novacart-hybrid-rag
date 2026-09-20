from dataclasses import asdict
from typing import Any

import pytest

from backend.app.reranking import CrossEncoderReranker, RerankingRetriever
from backend.app.retrieval import (
    HybridRetrievalResult,
    RerankedRetrievalResult,
    RetrievalResult,
)


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        **_: Any,
    ) -> list[float]:
        self.calls.append(sentences)
        return self.scores


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class SpyReranker:
    model_name = "spy-reranker"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RerankedRetrievalResult]:
        self.calls += 1
        return []


def test_reranking_changes_candidate_order() -> None:
    candidates = [
        _hybrid_result("generic", "General warranty information.", rrf_score=0.04),
        _hybrid_result(
            "exact",
            "The NCM-24 monitor warranty period is 24 months.",
            rrf_score=0.03,
        ),
    ]
    model = FakeCrossEncoder([0.2, 4.8])
    reranker = CrossEncoderReranker("test-model", model=model)

    results = reranker.rerank(
        "What is the warranty period of NCM-24?",
        candidates,
        top_k=2,
    )

    assert [result.chunk_id for result in results] == ["exact", "generic"]
    assert [result.reranker_rank for result in results] == [1, 2]
    assert [result.rrf_rank for result in results] == [2, 1]
    assert model.calls[0][1] == (
        "What is the warranty period of NCM-24?",
        candidates[1].text,
    )


def test_reranking_preserves_metadata_and_retrieval_diagnostics() -> None:
    candidate = _hybrid_result(
        "ncm-24",
        "NCM-24 has a 24 month warranty.",
        rrf_score=0.031,
        document_id="products",
        document_name="products.csv",
        document_type="csv",
        source="data/raw/products.csv",
        page=None,
        section="row 4",
        dense_rank=2,
        bm25_rank=1,
        dense_score=0.72,
        bm25_score=8.79,
    )

    result = CrossEncoderReranker(
        "test-model",
        model=FakeCrossEncoder([3.5]),
    ).rerank("NCM-24 warranty", [candidate], top_k=1)[0]

    assert result.chunk_id == candidate.chunk_id
    assert result.text == candidate.text
    assert result.document_id == "products"
    assert result.document_name == "products.csv"
    assert result.document_type == "csv"
    assert result.source == "data/raw/products.csv"
    assert result.page is None
    assert result.section == "row 4"
    assert result.dense_rank == 2
    assert result.bm25_rank == 1
    assert result.dense_score == 0.72
    assert result.bm25_score == 8.79
    assert result.rrf_score == pytest.approx(0.031)
    assert result.reranker_score == pytest.approx(3.5)
    assert result.score == result.reranker_score


def test_reranker_obeys_top_k_and_handles_fewer_candidates() -> None:
    candidates = [
        _hybrid_result("one", "First"),
        _hybrid_result("two", "Second"),
        _hybrid_result("three", "Third"),
    ]
    results = CrossEncoderReranker(
        "test-model",
        model=FakeCrossEncoder([1.0, 3.0, 2.0]),
    ).rerank("query", candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["two", "three"]

    one_result = CrossEncoderReranker(
        "test-model",
        model=FakeCrossEncoder([1.0]),
    ).rerank("query", candidates[:1], top_k=5)
    assert [result.chunk_id for result in one_result] == ["one"]


def test_empty_candidates_return_without_model_call() -> None:
    model = FakeCrossEncoder([])
    reranker = CrossEncoderReranker("test-model", model=model)

    assert reranker.rerank("valid query", [], top_k=5) == []
    assert model.calls == []


def test_empty_query_and_missing_candidate_text_are_rejected() -> None:
    reranker = CrossEncoderReranker("test-model", model=FakeCrossEncoder([1.0]))

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.rerank("   ", [_hybrid_result("one", "Text")], top_k=1)

    with pytest.raises(ValueError, match="candidate chunk text cannot be empty"):
        reranker.rerank("query", [_hybrid_result("empty", "  ")], top_k=1)


def test_disabled_reranking_preserves_original_order_and_skips_model() -> None:
    candidates = [
        _hybrid_result("first", "First"),
        _hybrid_result("second", "Second"),
    ]
    retriever = FakeRetriever(candidates)
    reranker = SpyReranker()
    wrapped = RerankingRetriever(
        retriever=retriever,
        reranker=reranker,
        candidate_count=20,
        enabled=False,
    )

    results = wrapped.search("query", top_k=2)

    assert results == candidates
    assert retriever.calls == [("query", 2)]
    assert reranker.calls == 0


def test_enabled_reranking_retrieves_wider_candidate_set() -> None:
    candidates = [
        _hybrid_result("one", "First"),
        _hybrid_result("two", "Second"),
        _hybrid_result("three", "Third"),
    ]
    retriever = FakeRetriever(candidates)
    wrapped = RerankingRetriever(
        retriever=retriever,
        reranker=CrossEncoderReranker(
            "test-model",
            model=FakeCrossEncoder([1.0, 3.0, 2.0]),
        ),
        candidate_count=3,
        enabled=True,
    )

    results = wrapped.search("query", top_k=1)

    assert retriever.calls == [("query", 3)]
    assert [result.chunk_id for result in results] == ["two"]


def test_reranked_result_has_stable_response_fields() -> None:
    result = CrossEncoderReranker(
        "test-model",
        model=FakeCrossEncoder([2.0]),
    ).rerank("query", [_hybrid_result("one", "Text")], top_k=1)[0]

    assert set(asdict(result)) == {
        "chunk_id",
        "text",
        "score",
        "document_id",
        "document_name",
        "document_type",
        "source",
        "page",
        "section",
        "dense_rank",
        "bm25_rank",
        "dense_score",
        "bm25_score",
        "rrf_rank",
        "rrf_score",
        "reranker_score",
        "reranker_rank",
    }


def _hybrid_result(
    chunk_id: str,
    text: str,
    rrf_score: float = 0.03,
    document_id: str = "doc-1",
    document_name: str = "policy.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/policy.pdf",
    page: int | None = 1,
    section: str | None = "Policy",
    dense_rank: int | None = 1,
    bm25_rank: int | None = 1,
    dense_score: float | None = 0.8,
    bm25_score: float | None = 5.0,
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=rrf_score,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        dense_score=dense_score,
        bm25_score=bm25_score,
    )
