from backend.app.chunking.models import DocumentChunk
from backend.app.embeddings import EmbeddingProvider, EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 4) -> None:
        self.calls: list[list[str]] = []
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [float(len(text)), 1.0, 2.0, 3.0][: self._dimension]
            for text in texts
        ]


def test_single_chunk_embedding() -> None:
    service = EmbeddingService(FakeEmbeddingProvider())
    chunk = _chunk(text="NovaCart shipping takes two days.")

    embedded = service.embed_chunk(chunk)

    assert embedded.chunk_id == chunk.chunk_id
    assert embedded.text == chunk.text
    assert embedded.embedding == [33.0, 1.0, 2.0, 3.0]


def test_batch_embedding_uses_provider_batches() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)
    chunks = [_chunk(chunk_id=f"chunk-{index}", text=f"text {index}") for index in range(3)]

    embedded = service.embed_chunks(chunks, batch_size=2)

    assert len(embedded) == 3
    assert provider.calls == [["text 0", "text 1"], ["text 2"]]


def test_embedding_dimensions_are_consistent() -> None:
    provider = FakeEmbeddingProvider(dimension=3)
    service = EmbeddingService(provider)

    embedded = service.embed_chunks([_chunk()], batch_size=1)

    assert embedded[0].embedding_dimension == 3


def test_metadata_is_preserved() -> None:
    chunk = _chunk(
        chunk_id="chunk-123",
        document_id="doc-123",
        document_name="warranty_guide.docx",
        document_type="docx",
        source="data/raw/warranty_guide.docx",
        page=None,
        section="Warranty Coverage",
        text="Warranty details.",
    )

    embedded = EmbeddingService(FakeEmbeddingProvider()).embed_chunk(chunk)

    assert embedded.chunk_id == chunk.chunk_id
    assert embedded.document_id == chunk.document_id
    assert embedded.document_name == chunk.document_name
    assert embedded.document_type == chunk.document_type
    assert embedded.source == chunk.source
    assert embedded.page == chunk.page
    assert embedded.section == chunk.section


def test_empty_text_returns_zero_vector_without_provider_call() -> None:
    provider = FakeEmbeddingProvider(dimension=4)
    service = EmbeddingService(provider)

    embedded = service.embed_chunks([_chunk(text="   ")], batch_size=8)

    assert embedded[0].embedding == [0.0, 0.0, 0.0, 0.0]
    assert provider.calls == []


def test_dimension_mismatch_raises_error() -> None:
    class BadProvider(FakeEmbeddingProvider):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0]]

    service = EmbeddingService(BadProvider(dimension=4))

    try:
        service.embed_chunks([_chunk()], batch_size=1)
    except ValueError as exc:
        assert "Embedding dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch to raise ValueError")


def _chunk(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    document_name: str = "shipping_policy.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/shipping_policy.pdf",
    page: int | None = 1,
    section: str | None = None,
    text: str = "Example chunk text.",
    chunk_index: int = 1,
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
        chunk_index=chunk_index,
    )
