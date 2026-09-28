"""Build citations exclusively from evidence supplied to generation."""

from __future__ import annotations

import re

from backend.app.rag.evidence import Evidence
from backend.app.rag.unified_models import UnifiedSource

SOURCE_LABEL_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


class CitationBuilder:
    """Select, normalize, and deduplicate source-backed citations."""

    def build(self, answer: str, evidence: list[Evidence]) -> list[UnifiedSource]:
        labels = [int(match.group(1)) for match in SOURCE_LABEL_PATTERN.finditer(answer)]
        if labels:
            selected = [
                evidence[number - 1]
                for number in labels
                if 1 <= number <= len(evidence)
            ]
        else:
            selected = evidence

        citations: list[UnifiedSource] = []
        seen: set[tuple[object, ...]] = set()
        for item in selected:
            citation = self.from_evidence(item)
            identity = (
                citation.source,
                citation.document_name,
                citation.document_type,
                citation.page,
                citation.section,
                citation.chunk_id,
                citation.sku,
            )
            if identity in seen:
                continue
            seen.add(identity)
            citations.append(citation)
        return citations

    @staticmethod
    def from_evidence(item: Evidence) -> UnifiedSource:
        """Copy citation fields and diagnostics without trusting LLM output."""
        return UnifiedSource(
            evidence_id=item.evidence_id,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            sku=item.sku,
            source=item.source,
            document_name=item.document_name,
            document_type=item.document_type,
            page=item.page,
            section=item.section,
            retrieval_method=item.retrieval_info.get("strategy"),
            reranker_score=item.retrieval_info.get("reranker_score"),
            rrf_score=item.retrieval_info.get("rrf_score"),
        )
