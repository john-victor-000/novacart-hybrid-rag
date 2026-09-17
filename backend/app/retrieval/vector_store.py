"""Vector store interface and ChromaDB implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.app.embeddings.models import EmbeddedChunk
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Interface for replaceable vector database implementations."""

    @abstractmethod
    def index(self, records: list[EmbeddedChunk]) -> int:
        """Store embedded chunks and return the number newly indexed."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return top-k results for a query embedding."""


class ChromaVectorStore(VectorStore):
    """Persistent local vector store backed by ChromaDB."""

    def __init__(self, persist_path: Path | str, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, records: list[EmbeddedChunk]) -> int:
        started = perf_counter()
        if not records:
            logger.info("Indexed 0 chunks into collection=%s in 0.00s", self.collection_name)
            return 0

        records_by_id = {record.chunk_id: record for record in records}
        unique_records = list(records_by_id.values())
        existing = self._existing_ids(list(records_by_id))
        new_records = [record for record in unique_records if record.chunk_id not in existing]

        if new_records:
            self._collection.add(
                ids=[record.chunk_id for record in new_records],
                embeddings=[record.embedding for record in new_records],
                documents=[record.text for record in new_records],
                metadatas=[_metadata_from_record(record) for record in new_records],
            )

        elapsed = perf_counter() - started
        logger.info(
            "Indexed %s new chunks into collection=%s skipped=%s in %.2fs",
            len(new_records),
            self.collection_name,
            len(records) - len(new_records),
            elapsed,
        )
        return len(new_records)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if self.count() == 0:
            return []

        started = perf_counter()
        raw_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results = _results_from_chroma(raw_results)
        elapsed = perf_counter() - started
        logger.info(
            "Dense query collection=%s top_k=%s results=%s in %.2fs",
            self.collection_name,
            top_k,
            len(results),
            elapsed,
        )
        return results

    def count(self) -> int:
        """Return number of records in the collection."""
        return int(self._collection.count())

    def _existing_ids(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()

        existing = self._collection.get(ids=chunk_ids)
        return set(existing.get("ids", []))


def _metadata_from_record(record: EmbeddedChunk) -> dict[str, str | int | bool]:
    return {
        "document_id": record.document_id,
        "document_name": record.document_name,
        "document_type": record.document_type,
        "source": record.source,
        "page": record.page if record.page is not None else -1,
        "has_page": record.page is not None,
        "section": record.section if record.section is not None else "",
        "has_section": record.section is not None,
    }


def _results_from_chroma(raw_results: dict[str, Any]) -> list[RetrievalResult]:
    ids = raw_results.get("ids", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results: list[RetrievalResult] = []
    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=1.0 - float(distance),
                document_id=str(metadata["document_id"]),
                document_name=str(metadata["document_name"]),
                document_type=str(metadata["document_type"]),
                source=str(metadata["source"]),
                page=int(metadata["page"]) if metadata["has_page"] else None,
                section=str(metadata["section"]) if metadata["has_section"] else None,
            )
        )

    return results
