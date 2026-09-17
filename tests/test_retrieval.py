from pathlib import Path

import pytest

from backend.app.embeddings import EmbeddingProvider
from backend.app.embeddings.models import EmbeddedChunk
from backend.app.retrieval import ChromaVectorStore, DenseRetriever


class FakeQueryProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-query-model"

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_embedding_for_text(text) for text in texts]


def test_vector_store_initialization(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.count() == 0
    assert store.collection_name == "test_chunks"


def test_indexing_chunks(tmp_path: Path) -> None:
    store = _store(tmp_path)

    indexed = store.index([_record("return-1", "Return policy details")])

    assert indexed == 1
    assert store.count() == 1


def test_duplicate_indexing_is_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record("return-1", "Return policy details")

    first_indexed = store.index([record, record])
    second_indexed = store.index([record])

    assert first_indexed == 1
    assert second_indexed == 0
    assert store.count() == 1


def test_semantic_retrieval_returns_nearest_chunk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.index(
        [
            _record("return-1", "Customers can return products within thirty days."),
            _record("shipping-1", "Standard shipping takes three business days."),
        ]
    )

    results = DenseRetriever(FakeQueryProvider(), store).search(
        "What is the return policy?",
        top_k=1,
    )

    assert [result.chunk_id for result in results] == ["return-1"]


def test_retrieval_preserves_metadata(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.index(
        [
            _record(
                "warranty-1",
                "Warranty coverage lasts one year.",
                document_id="doc-warranty",
                document_name="warranty.docx",
                document_type="docx",
                source="data/raw/warranty.docx",
                page=None,
                section="Warranty Coverage",
            )
        ]
    )

    result = store.search(_embedding_for_text("warranty"), top_k=1)[0]

    assert result.document_id == "doc-warranty"
    assert result.document_name == "warranty.docx"
    assert result.document_type == "docx"
    assert result.source == "data/raw/warranty.docx"
    assert result.page is None
    assert result.section == "Warranty Coverage"


def test_top_k_limits_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.index(
        [
            _record("return-1", "Return policy"),
            _record("shipping-1", "Shipping policy"),
            _record("warranty-1", "Warranty policy"),
        ]
    )

    results = DenseRetriever(FakeQueryProvider(), store).search("policy", top_k=2)

    assert len(results) == 2


def test_empty_query_returns_no_results(tmp_path: Path) -> None:
    retriever = DenseRetriever(FakeQueryProvider(), _store(tmp_path))

    assert retriever.search("   ", top_k=3) == []


def test_invalid_top_k_raises_error(tmp_path: Path) -> None:
    retriever = DenseRetriever(FakeQueryProvider(), _store(tmp_path))

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("return policy", top_k=0)


def _store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma", "test_chunks")


def _record(
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    document_name: str = "policies.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/policies.pdf",
    page: int | None = 1,
    section: str | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        text=text,
        embedding=_embedding_for_text(text),
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
    )


def _embedding_for_text(text: str) -> list[float]:
    lowered = text.lower()
    if "return" in lowered:
        return [1.0, 0.0, 0.0]
    if "shipping" in lowered:
        return [0.0, 1.0, 0.0]
    if "warranty" in lowered:
        return [0.0, 0.0, 1.0]
    return [0.8, 0.1, 0.0]
