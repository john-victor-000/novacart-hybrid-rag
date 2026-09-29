from backend.app.rag.citations import CitationBuilder
from backend.app.rag.evidence import Evidence


def test_citations_deduplicate_and_preserve_metadata() -> None:
    evidence = _evidence()
    duplicate = _evidence(evidence_id="duplicate")

    sources = CitationBuilder().build(
        "Covered for 24 months. [Source 1] [Source 2] [Source 1]",
        [evidence, duplicate],
    )

    assert len(sources) == 1
    source = sources[0]
    assert source.document_name == "warranty_guide.docx"
    assert source.document_type == "docx"
    assert source.section == "Monitor Warranty"
    assert source.page == 2
    assert source.chunk_id == "warranty-1"
    assert source.source == "data/raw/warranty_guide.docx"
    assert source.retrieval_method == "hybrid"
    assert source.reranker_score == 0.996
    assert source.rrf_score == 0.0325


def test_invalid_llm_source_label_cannot_create_citation() -> None:
    sources = CitationBuilder().build("Claim. [Source 99]", [_evidence()])

    assert sources == []


def test_unicode_source_brackets_resolve_to_server_evidence() -> None:
    sources = CitationBuilder().build(
        "Covered for 24 months. 【Source 1】",
        [_evidence()],
    )

    assert len(sources) == 1
    assert sources[0].chunk_id == "warranty-1"


def _evidence(evidence_id: str = "warranty-1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        text="NCM-24: 24 months.",
        source="data/raw/warranty_guide.docx",
        document_name="warranty_guide.docx",
        document_type="docx",
        page=2,
        section="Monitor Warranty",
        chunk_id="warranty-1",
        document_id="warranty-doc",
        score=0.996,
        retrieval_info={
            "strategy": "hybrid",
            "reranker_score": 0.996,
            "rrf_score": 0.0325,
        },
    )
