"""Bounded, deduplicated context construction for unified evidence."""

from dataclasses import dataclass

from backend.app.rag.evidence import Evidence


@dataclass(frozen=True)
class BuiltContext:
    """Formatted context and the exact evidence labels it contains."""

    text: str
    evidence: tuple[Evidence, ...]


class UnifiedContextBuilder:
    """Deduplicate and format evidence within item and character limits."""

    def __init__(self, max_characters: int = 8000, max_items: int = 8) -> None:
        if max_characters <= 0 or max_items <= 0:
            raise ValueError("context limits must be greater than zero")
        self.max_characters = max_characters
        self.max_items = max_items

    def build(self, evidence: list[Evidence]) -> BuiltContext:
        unique = _deduplicate(evidence)
        blocks: list[str] = []
        included: list[Evidence] = []
        used_characters = 0

        for item in unique[: self.max_items]:
            source_number = len(included) + 1
            metadata = _metadata_lines(item, source_number)
            separator_cost = 2 if blocks else 0
            available = self.max_characters - used_characters - separator_cost
            minimum = len(metadata) + len("\nContent:\n") + 1
            if available < minimum:
                break

            content_budget = available - len(metadata) - len("\nContent:\n")
            content = _truncate_at_boundary(item.text.strip(), content_budget)
            if not content:
                continue
            block = f"{metadata}\nContent:\n{content}"
            blocks.append(block)
            included.append(item)
            used_characters += len(block) + separator_cost

        return BuiltContext(
            text="\n\n".join(blocks),
            evidence=tuple(included),
        )


def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
    unique: list[Evidence] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for item in evidence:
        normalized_text = " ".join(item.text.casefold().split())
        if not normalized_text:
            continue
        if item.evidence_id in seen_ids or normalized_text in seen_text:
            continue
        seen_ids.add(item.evidence_id)
        seen_text.add(normalized_text)
        unique.append(item)
    return unique


def _metadata_lines(item: Evidence, source_number: int) -> str:
    page = str(item.page) if item.page is not None else "not available"
    section = item.section or "not available"
    lines = [
        f"[Source {source_number}]",
        f"Evidence ID: {item.evidence_id}",
        f"Document: {item.document_name}",
        f"Document type: {item.document_type}",
        f"Source: {item.source}",
        f"Page: {page}",
        f"Section: {section}",
    ]
    if item.chunk_id:
        lines.append(f"Chunk ID: {item.chunk_id}")
    if item.sku:
        lines.append(f"SKU: {item.sku}")
    if item.retrieval_info:
        diagnostics = ", ".join(
            f"{key}={value}" for key, value in item.retrieval_info.items()
        )
        lines.append(f"Retrieval: {diagnostics}")
    return "\n".join(lines)


def _truncate_at_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return ""
    candidate = text[: limit - 1]
    boundary = max(candidate.rfind(" "), candidate.rfind("\n"))
    if boundary > limit // 2:
        candidate = candidate[:boundary]
    return candidate.rstrip() + "…"
