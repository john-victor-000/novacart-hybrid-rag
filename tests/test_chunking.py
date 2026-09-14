from backend.app.chunking import chunk_documents
from backend.app.ingestion.models import IngestedDocument


def test_chunk_creation_adds_unique_chunk_ids() -> None:
    document = _document(
        text=(
            "Returns are accepted within thirty days. "
            "Items must be unused and include original packaging. "
            "Refunds are issued after inspection."
        )
    )

    chunks = chunk_documents([document], chunk_size=70, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.chunk_id for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_index for chunk in chunks] == [1, 2, 3]


def test_chunking_preserves_document_metadata() -> None:
    document = _document(
        document_id="doc-123",
        document_name="warranty_guide.docx",
        document_type="docx",
        source="data/raw/warranty_guide.docx",
        page=None,
        section="Warranty Coverage",
        text="NovaCart covers manufacturing defects for two years.",
    )

    chunk = chunk_documents([document], chunk_size=200, chunk_overlap=20)[0]

    assert chunk.document_id == document.document_id
    assert chunk.document_name == document.document_name
    assert chunk.document_type == document.document_type
    assert chunk.source == document.source
    assert chunk.page == document.page
    assert chunk.section == document.section
    assert chunk.text == document.text


def test_chunk_overlap_reuses_sentence_boundary_text() -> None:
    document = _document(
        text=(
            "Sentence one has enough detail. "
            "Sentence two should overlap. "
            "Sentence three starts the next idea."
        )
    )

    chunks = chunk_documents([document], chunk_size=65, chunk_overlap=35)

    assert len(chunks) == 2
    assert chunks[0].text.endswith("Sentence two should overlap.")
    assert chunks[1].text.startswith("Sentence two should overlap.")


def test_csv_records_remain_single_logical_chunks() -> None:
    csv_document = _document(
        document_id="csv-row-1",
        document_name="products.csv",
        document_type="csv",
        page=None,
        section="row 1",
        text=(
            "sku: NC-1\n"
            "name: Nova Lamp\n"
            "description: Adjustable desk lighting with warm and cool settings."
        ),
    )

    chunks = chunk_documents([csv_document], chunk_size=25, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0].document_type == "csv"
    assert chunks[0].section == "row 1"
    assert chunks[0].text == csv_document.text


def _document(
    document_id: str = "doc-1",
    document_name: str = "return_policy.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/return_policy.pdf",
    page: int | None = 1,
    section: str | None = None,
    text: str = "Example document text.",
) -> IngestedDocument:
    return IngestedDocument(
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
        text=text,
    )
