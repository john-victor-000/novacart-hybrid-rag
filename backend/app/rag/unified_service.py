"""Route, retrieve, build context, and generate one grounded response."""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Protocol

from backend.app.llm import LLMProvider
from backend.app.rag.citations import CitationBuilder, SOURCE_LABEL_PATTERN
from backend.app.rag.evidence import (
    Evidence,
    evidence_from_product,
    evidence_from_retrieval,
)
from backend.app.rag.errors import (
    RetrievalUnavailableError,
    StructuredDataUnavailableError,
)
from backend.app.rag.prompts import build_unified_rag_prompt
from backend.app.rag.unified_context import UnifiedContextBuilder
from backend.app.rag.unified_models import (
    EvidenceDebug,
    PipelineMetadata,
    RetrievalDebug,
    UnifiedRAGResponse,
    UnifiedSource,
)
from backend.app.retrieval import Retriever
from backend.app.routing import QueryRoute, QueryRouter, RouteDecision
from backend.app.structured import (
    ProductDataError,
    StructuredQueryResult,
    StructuredRetrievalError,
)

logger = logging.getLogger(__name__)

INFORMATION_NOT_FOUND = (
    "The information was not found in the NovaCart knowledge base."
)


class StructuredRetriever(Protocol):
    """Minimal structured retriever surface used by orchestration."""

    def retrieve(self, query: str) -> StructuredQueryResult:
        """Return structured product evidence."""


class UnifiedRAGService:
    """Choose a retrieval route and produce a single grounded response."""

    def __init__(
        self,
        router: QueryRouter,
        dense_retriever: Retriever,
        hybrid_retriever: Retriever,
        structured_retriever: StructuredRetriever,
        context_builder: UnifiedContextBuilder,
        llm: LLMProvider,
        retrieval_mode: str = "auto",
        dense_top_k: int = 5,
        hybrid_top_k: int = 5,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        if retrieval_mode not in {"auto", "dense", "hybrid"}:
            raise ValueError("retrieval_mode must be auto, dense, or hybrid")
        if dense_top_k <= 0 or hybrid_top_k <= 0:
            raise ValueError("retrieval top-k values must be greater than zero")
        self.router = router
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.structured_retriever = structured_retriever
        self.context_builder = context_builder
        self.llm = llm
        self.retrieval_mode = retrieval_mode
        self.dense_top_k = dense_top_k
        self.hybrid_top_k = hybrid_top_k
        self.citation_builder = citation_builder or CitationBuilder()

    def answer(
        self,
        query: str,
        include_debug: bool = False,
        request_id: str | None = None,
    ) -> UnifiedRAGResponse:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query cannot be empty")

        started = perf_counter()
        decision = self._route(normalized_query)
        retrieval_started = perf_counter()
        try:
            evidence = self._retrieve(normalized_query, decision.route)
        except StructuredDataUnavailableError:
            raise
        except Exception as exc:
            logger.exception(
                "Retrieval failed request_id=%s route=%s",
                request_id,
                decision.route.value,
            )
            raise RetrievalUnavailableError("retrieval is unavailable") from exc
        retrieval_latency = perf_counter() - retrieval_started
        logger.info(
            "Unified retrieval request_id=%s route=%s method=%s evidence=%s "
            "latency=%.3fs",
            request_id,
            decision.route.value,
            decision.route.value.lower(),
            len(evidence),
            retrieval_latency,
        )

        built_context = self.context_builder.build(evidence)
        included = list(built_context.evidence)
        if not included:
            logger.info(
                "Unified RAG completed without generation request_id=%s "
                "route=%s latency=%.3fs",
                request_id,
                decision.route.value,
                perf_counter() - started,
            )
            total_latency = perf_counter() - started
            return UnifiedRAGResponse(
                answer=INFORMATION_NOT_FOUND,
                route=decision.route,
                sources=[],
                metadata=_metadata(
                    len(included),
                    total_latency,
                    retrieval_latency,
                    0.0,
                ),
                retrieval_debug=(
                    _build_debug(decision, included) if include_debug else None
                ),
            )

        prompt = build_unified_rag_prompt(normalized_query, built_context.text)
        generation_started = perf_counter()
        raw_answer = self.llm.generate(prompt)
        generation_latency = perf_counter() - generation_started
        answer = _remove_invalid_source_labels(raw_answer, len(included))
        logger.info(
            "Unified generation request_id=%s model=%s latency=%.3fs",
            request_id,
            self.llm.model_name,
            generation_latency,
        )

        sources = (
            []
            if answer.strip() == INFORMATION_NOT_FOUND
            else self.citation_builder.build(raw_answer, included)
        )
        total_latency = perf_counter() - started
        response = UnifiedRAGResponse(
            answer=answer,
            route=decision.route,
            sources=sources,
            metadata=_metadata(
                len(included),
                total_latency,
                retrieval_latency,
                generation_latency,
            ),
            retrieval_debug=(
                _build_debug(decision, included) if include_debug else None
            ),
        )
        logger.info(
            "Unified RAG completed request_id=%s route=%s sources=%s "
            "latency=%.3fs",
            request_id,
            decision.route.value,
            len(response.sources),
            total_latency,
        )
        return response

    def _route(self, query: str) -> RouteDecision:
        if self.retrieval_mode == "dense":
            decision = RouteDecision(
                QueryRoute.DENSE,
                "RETRIEVAL_MODE forces dense retrieval for baseline evaluation",
                1.0,
            )
        elif self.retrieval_mode == "hybrid":
            decision = RouteDecision(
                QueryRoute.HYBRID,
                "RETRIEVAL_MODE forces hybrid retrieval for evaluation",
                1.0,
            )
        else:
            decision = self.router.route(query)
        logger.info(
            "Unified route=%s reason=%s",
            decision.route.value,
            decision.reason,
        )
        return decision

    def _retrieve(self, query: str, route: QueryRoute) -> list[Evidence]:
        if route == QueryRoute.DENSE:
            return [
                evidence_from_retrieval(result, "dense")
                for result in self.dense_retriever.search(
                    query,
                    top_k=self.dense_top_k,
                )
            ]
        if route == QueryRoute.HYBRID:
            return self._hybrid_evidence(query)
        if route == QueryRoute.STRUCTURED:
            return self._structured_evidence(query)
        return [
            *self._structured_evidence(query),
            *self._hybrid_evidence(query),
        ]

    def _hybrid_evidence(self, query: str) -> list[Evidence]:
        return [
            evidence_from_retrieval(result, "hybrid")
            for result in self.hybrid_retriever.search(
                query,
                top_k=self.hybrid_top_k,
            )
        ]

    def _structured_evidence(self, query: str) -> list[Evidence]:
        try:
            result = self.structured_retriever.retrieve(query)
        except ProductDataError as exc:
            logger.exception("Structured product data failed")
            raise StructuredDataUnavailableError(
                "structured product data is unavailable"
            ) from exc
        except StructuredRetrievalError as exc:
            logger.info("Structured retrieval produced no evidence error=%s", exc)
            return []
        return [
            evidence_from_product(product, result.operation)
            for product in result.products
        ]


def _remove_invalid_source_labels(answer: str, evidence_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f"[Source {number}]" if 1 <= number <= evidence_count else ""

    return " ".join(SOURCE_LABEL_PATTERN.sub(replace, answer).split())


def _to_source(item: Evidence) -> UnifiedSource:
    return CitationBuilder.from_evidence(item)


def _metadata(
    evidence_count: int,
    total_seconds: float,
    retrieval_seconds: float,
    generation_seconds: float,
) -> PipelineMetadata:
    return PipelineMetadata(
        retrieval_count=evidence_count,
        latency_ms=round(total_seconds * 1000, 3),
        retrieval_latency_ms=round(retrieval_seconds * 1000, 3),
        generation_latency_ms=round(generation_seconds * 1000, 3),
    )


def _build_debug(
    decision: RouteDecision,
    evidence: list[Evidence],
) -> RetrievalDebug:
    return RetrievalDebug(
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_count=len(evidence),
        evidence=[
            EvidenceDebug(
                **_to_source(item).model_dump(),
                text=item.text,
                score=item.score,
                retrieval_info=item.retrieval_info,
            )
            for item in evidence
        ],
    )
