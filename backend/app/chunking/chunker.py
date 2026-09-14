"""Structure-aware chunking for ingested documents."""

from hashlib import sha256
import re

from backend.app.chunking.models import DocumentChunk
from backend.app.ingestion.models import IngestedDocument


def chunk_documents(
    documents: list[IngestedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Convert ingested documents into retrieval-sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for document in documents:
        chunk_texts = _chunk_csv_record(document) if document.document_type == "csv" else _chunk_text(
            document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, text in enumerate(chunk_texts, start=1):
            chunks.append(_build_chunk(document, text, chunk_index))

    return chunks


def _chunk_csv_record(document: IngestedDocument) -> list[str]:
    return [document.text] if document.text else []


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    units = _split_text_units(normalized)
    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for unit in units:
        unit_parts = _split_oversized_unit(unit, chunk_size)
        for part in unit_parts:
            part_length = len(part)
            separator_length = 1 if current_units else 0

            if current_units and current_length + separator_length + part_length > chunk_size:
                chunk = " ".join(current_units).strip()
                chunks.append(chunk)
                current_units = _overlap_units(current_units, chunk_overlap)
                current_length = len(" ".join(current_units))

            if current_units and current_length + 1 + part_length > chunk_size:
                chunk = " ".join(current_units).strip()
                if chunk:
                    chunks.append(chunk)
                current_units = []
                current_length = 0

            current_units.append(part)
            current_length = len(" ".join(current_units))

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return _dedupe_adjacent(chunks)


def _split_text_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text)]
    units: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        units.extend(
            sentence.strip()
            for sentence in re.findall(r"[^.!?]+(?:[.!?]+|$)", paragraph)
            if sentence.strip()
        )
    return units or [text]


def _split_oversized_unit(unit: str, chunk_size: int) -> list[str]:
    if len(unit) <= chunk_size:
        return [unit]

    words = unit.split()
    parts: list[str] = []
    current_words: list[str] = []

    for word in words:
        separator_length = 1 if current_words else 0
        if current_words and len(" ".join(current_words)) + separator_length + len(word) > chunk_size:
            parts.append(" ".join(current_words))
            current_words = []

        if len(word) > chunk_size:
            parts.extend(_split_long_word(word, chunk_size))
            continue

        current_words.append(word)

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def _split_long_word(word: str, chunk_size: int) -> list[str]:
    return [word[index : index + chunk_size] for index in range(0, len(word), chunk_size)]


def _overlap_units(units: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap == 0:
        return []

    overlap: list[str] = []
    for unit in reversed(units):
        candidate = [unit, *overlap]
        if len(" ".join(candidate)) > chunk_overlap and overlap:
            break
        overlap = candidate
        if len(" ".join(overlap)) >= chunk_overlap:
            break

    return overlap


def _dedupe_adjacent(chunks: list[str]) -> list[str]:
    deduped: list[str] = []
    for chunk in chunks:
        if chunk and (not deduped or deduped[-1] != chunk):
            deduped.append(chunk)
    return deduped


def _build_chunk(
    document: IngestedDocument,
    text: str,
    chunk_index: int,
) -> DocumentChunk:
    raw_id = f"{document.document_id}|{chunk_index}|{text}"
    chunk_id = sha256(raw_id.encode("utf-8")).hexdigest()

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        document_name=document.document_name,
        document_type=document.document_type,
        source=document.source,
        page=document.page,
        section=document.section,
        text=text,
        chunk_index=chunk_index,
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()
