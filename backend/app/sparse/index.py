"""Build and persist a compact BM25 index over normalized document chunks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.app.chunking.models import DocumentChunk
from backend.app.sparse.tokenizer import tokenize

logger = logging.getLogger(__name__)

INDEX_VERSION = 1


@dataclass(frozen=True)
class BM25Document:
    """A tokenized chunk plus the source metadata needed at retrieval time."""

    chunk_id: str
    text: str
    tokens: list[str]
    document_id: str
    document_name: str
    document_type: str
    source: str
    page: int | None
    section: str | None

    @classmethod
    def from_chunk(cls, chunk: DocumentChunk) -> "BM25Document":
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            tokens=tokenize(chunk.text),
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            document_type=chunk.document_type,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BM25Document":
        return cls(
            chunk_id=str(value["chunk_id"]),
            text=str(value["text"]),
            tokens=[str(token) for token in value["tokens"]],
            document_id=str(value["document_id"]),
            document_name=str(value["document_name"]),
            document_type=str(value["document_type"]),
            source=str(value["source"]),
            page=int(value["page"]) if value.get("page") is not None else None,
            section=str(value["section"]) if value.get("section") is not None else None,
        )


class BM25Index:
    """In-memory BM25 statistics backed by a portable JSON index file."""

    def __init__(
        self,
        documents: list[BM25Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.documents = documents
        self.k1 = k1
        self.b = b
        self._term_frequencies = [Counter(document.tokens) for document in documents]
        self._document_lengths = [len(document.tokens) for document in documents]
        self.average_document_length = (
            sum(self._document_lengths) / len(documents) if documents else 0.0
        )
        self._idf = self._calculate_idf()

    @classmethod
    def build(
        cls,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        started = perf_counter()
        documents = [BM25Document.from_chunk(chunk) for chunk in chunks]
        index = cls(documents, k1=k1, b=b)
        logger.info(
            "Created BM25 index chunks=%s terms=%s in %.3fs",
            index.count,
            len(index._idf),
            perf_counter() - started,
        )
        return index

    @property
    def count(self) -> int:
        return len(self.documents)

    def rank(
        self,
        query_tokens: list[str],
        top_k: int,
    ) -> list[tuple[BM25Document, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query_tokens or not self.documents:
            return []

        scored: list[tuple[BM25Document, float]] = []
        for position, document in enumerate(self.documents):
            score = self._score_document(position, query_tokens)
            if score > 0:
                scored.append((document, score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[:top_k]

    def save(self, path: Path | str) -> None:
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "k1": self.k1,
            "b": self.b,
            "documents": [asdict(document) for document in self.documents],
        }
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info("Persisted BM25 index chunks=%s path=%s", self.count, index_path)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        index_path = Path(path)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError("Unsupported BM25 index version")

        index = cls(
            [BM25Document.from_dict(value) for value in payload["documents"]],
            k1=float(payload["k1"]),
            b=float(payload["b"]),
        )
        logger.info("Loaded BM25 index chunks=%s path=%s", index.count, index_path)
        return index

    def _calculate_idf(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document.tokens))

        document_count = len(self.documents)
        return {
            term: log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score_document(self, position: int, query_tokens: list[str]) -> float:
        frequencies = self._term_frequencies[position]
        document_length = self._document_lengths[position]
        length_ratio = (
            document_length / self.average_document_length
            if self.average_document_length
            else 0.0
        )
        normalization = self.k1 * (1.0 - self.b + self.b * length_ratio)

        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + normalization
            score += self._idf.get(token, 0.0) * numerator / denominator
        return score
