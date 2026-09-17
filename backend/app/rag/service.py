"""Orchestrate dense retrieval, context construction, and generation."""

from __future__ import annotations

import logging
import re
from time import perf_counter

from backend.app.llm import LLMProvider
from backend.app.rag.context import ContextBuilder
from backend.app.rag.models import RAGResponse, RetrievedChunk, SourceCitation
from backend.app.rag.prompts import build_rag_prompt
from backend.app.retrieval import DenseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

INFORMATION_NOT_FOUND = (
    "The information was not found in the available NovaCart documents."
)
SOURCE_LABEL_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


class RAGService:
    """Run the baseline query-to-answer dense RAG pipeline."""

    def __init__(
        self,
        retriever: DenseRetriever,
        context_builder: ContextBuilder,
        llm: LLMProvider,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm

    def answer(
        self,
        query: str,
        top_k: int,
        include_retrieved_chunks: bool = False,
    ) -> RAGResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")

        total_started = perf_counter()
        logger.info("RAG query=%r", normalized_query)

        retrieval_started = perf_counter()
        chunks = self.retriever.search(normalized_query, top_k=top_k)
        retrieval_latency = perf_counter() - retrieval_started
        logger.info(
            "RAG retrieval completed chunks=%s latency=%.3fs",
            len(chunks),
            retrieval_latency,
        )

        if not chunks:
            logger.info(
                "RAG completed without generation total_latency=%.3fs",
                perf_counter() - total_started,
            )
            return RAGResponse(
                answer=INFORMATION_NOT_FOUND,
                sources=[],
                retrieved_chunks=[] if include_retrieved_chunks else None,
            )

        context = self.context_builder.build(chunks)
        prompt = build_rag_prompt(normalized_query, context)

        llm_started = perf_counter()
        answer = _remove_invalid_source_labels(self.llm.generate(prompt), len(chunks))
        llm_latency = perf_counter() - llm_started
        logger.info(
            "RAG generation completed model=%s latency=%.3fs",
            self.llm.model_name,
            llm_latency,
        )

        response = RAGResponse(
            answer=answer,
            sources=(
                []
                if answer.strip() == INFORMATION_NOT_FOUND
                else _deduplicate_sources(_cited_chunks(answer, chunks))
            ),
            retrieved_chunks=(
                [_to_retrieved_chunk(chunk) for chunk in chunks]
                if include_retrieved_chunks
                else None
            ),
        )
        logger.info(
            "RAG completed sources=%s total_latency=%.3fs",
            len(response.sources),
            perf_counter() - total_started,
        )
        return response


def _deduplicate_sources(chunks: list[RetrievalResult]) -> list[SourceCitation]:
    sources: list[SourceCitation] = []
    seen: set[tuple[str, int | None, str | None]] = set()
    for chunk in chunks:
        key = (chunk.document_name, chunk.page, chunk.section)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceCitation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                document_type=chunk.document_type,
                source=chunk.source,
                page=chunk.page,
                section=chunk.section,
            )
        )
    return sources


def _remove_invalid_source_labels(answer: str, chunk_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        source_number = int(match.group(1))
        return match.group(0) if 1 <= source_number <= chunk_count else ""

    return " ".join(SOURCE_LABEL_PATTERN.sub(replace, answer).split())


def _cited_chunks(
    answer: str,
    chunks: list[RetrievalResult],
) -> list[RetrievalResult]:
    source_numbers = [
        int(match.group(1))
        for match in SOURCE_LABEL_PATTERN.finditer(answer)
        if 1 <= int(match.group(1)) <= len(chunks)
    ]
    if not source_numbers:
        return chunks

    seen: set[int] = set()
    cited: list[RetrievalResult] = []
    for source_number in source_numbers:
        if source_number in seen:
            continue
        seen.add(source_number)
        cited.append(chunks[source_number - 1])
    return cited


def _to_retrieved_chunk(chunk: RetrievalResult) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        document_type=chunk.document_type,
        source=chunk.source,
        page=chunk.page,
        section=chunk.section,
        text=chunk.text,
        score=chunk.score,
    )
