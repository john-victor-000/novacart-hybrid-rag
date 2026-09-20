"""Sparse retriever interface and BM25 implementation."""

from abc import ABC, abstractmethod
import logging
from time import perf_counter

from backend.app.retrieval.models import RetrievalResult
from backend.app.sparse.index import BM25Index
from backend.app.sparse.tokenizer import tokenize

logger = logging.getLogger(__name__)


class SparseRetriever(ABC):
    """Interface for replaceable sparse retrieval implementations."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return ranked sparse retrieval results."""


class BM25Retriever(SparseRetriever):
    """Rank persisted chunk records with BM25 lexical relevance."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        started = perf_counter()
        ranked = self.index.rank(query_tokens, top_k=top_k)
        results = [
            RetrievalResult(
                chunk_id=document.chunk_id,
                text=document.text,
                score=score,
                document_id=document.document_id,
                document_name=document.document_name,
                document_type=document.document_type,
                source=document.source,
                page=document.page,
                section=document.section,
            )
            for document, score in ranked
        ]
        logger.info(
            "BM25 query returned results=%s top_k=%s latency=%.3fs",
            len(results),
            top_k,
            perf_counter() - started,
        )
        return results
