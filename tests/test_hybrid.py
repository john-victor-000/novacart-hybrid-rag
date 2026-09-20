import pytest

from backend.app.hybrid import HybridRetriever, ReciprocalRankFusion
from backend.app.retrieval import RetrievalResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rrf_uses_one_based_ranks_and_standard_formula() -> None:
    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        [_result("shared", score=0.91), _result("dense-only", score=0.75)],
        [_result("bm25-only", score=8.2), _result("shared", score=5.4)],
        top_k=3,
    )

    shared = next(result for result in results if result.chunk_id == "shared")
    assert shared.rrf_score == pytest.approx((1 / 61) + (1 / 62))
    assert shared.score == shared.rrf_score
    assert shared.dense_rank == 1
    assert shared.bm25_rank == 2
    assert shared.dense_score == 0.91
    assert shared.bm25_score == 5.4


def test_duplicate_chunk_is_merged_once() -> None:
    results = ReciprocalRankFusion(k=60).fuse(
        [_result("same", score=0.9)],
        [_result("same", score=12.0)],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "same"
    assert results[0].dense_rank == 1
    assert results[0].bm25_rank == 1


def test_fusion_preserves_chunk_text_and_source_metadata() -> None:
    source = _result(
        "metadata",
        text="The monitor is covered for 24 months.",
        document_id="doc-products",
        document_name="products.csv",
        document_type="csv",
        source="data/raw/products.csv",
        page=None,
        section="row 4",
    )

    result = ReciprocalRankFusion().fuse([source], [], top_k=1)[0]

    assert result.text == source.text
    assert result.document_id == "doc-products"
    assert result.document_name == "products.csv"
    assert result.document_type == "csv"
    assert result.source == "data/raw/products.csv"
    assert result.page is None
    assert result.section == "row 4"


def test_fusion_obeys_final_top_k() -> None:
    results = ReciprocalRankFusion().fuse(
        [_result("dense-1"), _result("dense-2"), _result("dense-3")],
        [_result("bm25-1"), _result("bm25-2")],
        top_k=2,
    )

    assert len(results) == 2


def test_dense_only_result_keeps_dense_provenance() -> None:
    result = ReciprocalRankFusion(k=10).fuse(
        [_result("dense-only", score=0.8)],
        [],
        top_k=1,
    )[0]

    assert result.dense_rank == 1
    assert result.dense_score == 0.8
    assert result.bm25_rank is None
    assert result.bm25_score is None
    assert result.rrf_score == pytest.approx(1 / 11)


def test_bm25_only_result_keeps_bm25_provenance() -> None:
    result = ReciprocalRankFusion(k=10).fuse(
        [],
        [_result("bm25-only", score=7.5)],
        top_k=1,
    )[0]

    assert result.dense_rank is None
    assert result.dense_score is None
    assert result.bm25_rank == 1
    assert result.bm25_score == 7.5
    assert result.rrf_score == pytest.approx(1 / 11)


def test_chunk_ranked_highly_by_both_retrievers_wins() -> None:
    results = ReciprocalRankFusion(k=60).fuse(
        [_result("shared"), _result("dense-only")],
        [_result("shared"), _result("bm25-only")],
        top_k=3,
    )

    assert results[0].chunk_id == "shared"
    assert results[0].dense_rank == 1
    assert results[0].bm25_rank == 1


def test_hybrid_retriever_uses_configured_depths_and_returns_all_rankings() -> None:
    dense = FakeRetriever([_result("shared"), _result("dense-only")])
    bm25 = FakeRetriever([_result("shared"), _result("bm25-only")])
    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fusion=ReciprocalRankFusion(),
        dense_top_k=2,
        bm25_top_k=1,
    )

    response = retriever.retrieve("NCM-24 warranty", top_k=1)

    assert dense.calls == [("NCM-24 warranty", 2)]
    assert bm25.calls == [("NCM-24 warranty", 1)]
    assert [result.chunk_id for result in response.dense_results] == [
        "shared",
        "dense-only",
    ]
    assert [result.chunk_id for result in response.bm25_results] == ["shared"]
    assert [result.chunk_id for result in response.hybrid_results] == ["shared"]


def _result(
    chunk_id: str,
    text: str = "NovaCart policy text.",
    score: float = 1.0,
    document_id: str = "doc-1",
    document_name: str = "policy.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/policy.pdf",
    page: int | None = 1,
    section: str | None = "Policy",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
    )
