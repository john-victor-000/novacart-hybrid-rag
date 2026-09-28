"""Normalized evidence shared by every unified retrieval route."""

from dataclasses import dataclass, field
from typing import Any

from backend.app.retrieval import RetrievalResult
from backend.app.structured import ProductRecord


@dataclass(frozen=True)
class Evidence:
    """One traceable context item from documents or structured data."""

    evidence_id: str
    text: str
    source: str
    document_name: str
    document_type: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    sku: str | None = None
    score: float | None = None
    retrieval_info: dict[str, Any] = field(default_factory=dict)


def evidence_from_retrieval(
    result: RetrievalResult,
    strategy: str,
) -> Evidence:
    """Preserve chunk metadata and available retrieval diagnostics."""
    diagnostics: dict[str, Any] = {"strategy": strategy}
    for field_name in (
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "bm25_score",
        "rrf_rank",
        "rrf_score",
        "reranker_rank",
        "reranker_score",
    ):
        value = getattr(result, field_name, None)
        if value is not None:
            diagnostics[field_name] = value

    return Evidence(
        evidence_id=result.chunk_id,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        text=result.text,
        source=result.source,
        document_name=result.document_name,
        document_type=result.document_type,
        page=result.page,
        section=result.section,
        score=result.score,
        retrieval_info=diagnostics,
    )


def evidence_from_product(
    product: ProductRecord,
    operation: str,
) -> Evidence:
    """Format a structured row as labeled, source-backed evidence."""
    text = "\n".join(
        [
            f"SKU: {product.sku}",
            f"Product name: {product.product_name}",
            f"Category: {product.category}",
            f"Price INR: {product.price_inr}",
            f"Warranty months: {product.warranty_months}",
            f"Rating: {product.rating}",
            f"Stock status: {product.stock_status}",
        ]
    )
    return Evidence(
        evidence_id=f"product:{product.sku.casefold()}",
        text=text,
        source=product.source,
        document_name=product.source,
        document_type="csv",
        section=product.sku,
        sku=product.sku,
        retrieval_info={
            "strategy": "structured",
            "operation": operation,
        },
    )
