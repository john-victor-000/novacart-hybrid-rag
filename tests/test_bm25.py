from pathlib import Path

import pytest

from backend.app.chunking import DocumentChunk
from backend.app.sparse import BM25Index, BM25Retriever, tokenize


def test_bm25_index_creation_and_persistence(tmp_path: Path) -> None:
    index = BM25Index.build([_chunk("one", "Free standard shipping."), _chunk("two", "Warranty details.")])
    index_path = tmp_path / "bm25.json"

    index.save(index_path)
    loaded = BM25Index.load(index_path)

    assert index.count == 2
    assert loaded.count == 2
    assert loaded.k1 == 1.5
    assert loaded.b == 0.75


def test_exact_keyword_retrieval() -> None:
    retriever = _retriever(
        [
            _chunk("shipping", "NovaCart offers free shipping on qualifying orders."),
            _chunk("returns", "The standard return window is seven days."),
        ]
    )

    results = retriever.search("free shipping", top_k=2)

    assert results[0].chunk_id == "shipping"
    assert results[0].score > 0


def test_sku_retrieval_preserves_complete_identifier() -> None:
    retriever = _retriever(
        [
            _chunk("ncm", "NCM-24 monitor includes a 24 months warranty."),
            _chunk("nkm", "NKM-10 keyboard includes a 12 months warranty."),
            _chunk("ncb", "NCB-100 cable has no extended warranty."),
        ]
    )

    results = retriever.search("NCM-24 warranty", top_k=3)

    assert "ncm-24" in tokenize("NCM-24 warranty")
    assert results[0].chunk_id == "ncm"


def test_top_k_limits_results() -> None:
    retriever = _retriever(
        [
            _chunk("one", "Warranty for product one."),
            _chunk("two", "Warranty for product two."),
            _chunk("three", "Warranty for product three."),
        ]
    )

    results = retriever.search("warranty", top_k=2)

    assert len(results) == 2


def test_metadata_is_preserved() -> None:
    chunk = _chunk(
        "metadata",
        "NCM-24 warranty is 24 months.",
        document_id="doc-warranty",
        document_name="warranty_guide.docx",
        document_type="docx",
        source="data/raw/warranty_guide.docx",
        page=None,
        section="Examples",
    )

    result = _retriever([chunk]).search("NCM-24", top_k=1)[0]

    assert result.chunk_id == chunk.chunk_id
    assert result.text == chunk.text
    assert result.document_id == chunk.document_id
    assert result.document_name == chunk.document_name
    assert result.document_type == chunk.document_type
    assert result.source == chunk.source
    assert result.page is None
    assert result.section == "Examples"


def test_empty_query_returns_no_results() -> None:
    retriever = _retriever([_chunk("one", "Warranty details")])

    assert retriever.search("   !!! ", top_k=3) == []


def test_invalid_top_k_raises_error() -> None:
    retriever = _retriever([_chunk("one", "Warranty details")])

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("warranty", top_k=0)


def _retriever(chunks: list[DocumentChunk]) -> BM25Retriever:
    return BM25Retriever(BM25Index.build(chunks))


def _chunk(
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    document_name: str = "products.csv",
    document_type: str = "csv",
    source: str = "data/raw/products.csv",
    page: int | None = None,
    section: str | None = "row 1",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
        text=text,
        chunk_index=1,
    )
