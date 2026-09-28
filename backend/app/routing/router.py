"""Understandable rules for selecting a NovaCart retrieval strategy."""

import logging
import re

from backend.app.routing.models import QueryRoute, RouteDecision

logger = logging.getLogger(__name__)

SKU_PATTERN = re.compile(r"\b[A-Z]{2,}-\d+\b", re.IGNORECASE)
COMPARISON_PATTERN = re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b", re.I)
POLICY_TERMS = {
    "address",
    "cancel",
    "damaged",
    "delivery",
    "dispatch",
    "exchange",
    "refund",
    "return",
    "shipping",
}
STRUCTURED_PRODUCT_TERMS = {
    "availability",
    "available",
    "category",
    "cost",
    "price",
    "rating",
    "stock",
    "warranty",
}
FILTER_TERMS = {
    "above",
    "at least",
    "at most",
    "below",
    "cheapest",
    "highest",
    "in stock",
    "less than",
    "low stock",
    "more than",
    "most expensive",
    "out of stock",
    "sort",
    "under",
}


class QueryRouter:
    """Route queries with deterministic, logged intent rules."""

    def route(self, query: str) -> RouteDecision:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query cannot be empty")

        folded = normalized.casefold()
        skus = SKU_PATTERN.findall(normalized)
        has_policy_term = _contains_any(folded, POLICY_TERMS)
        has_structured_term = _contains_any(folded, STRUCTURED_PRODUCT_TERMS)

        if COMPARISON_PATTERN.search(normalized) and len(skus) >= 2:
            decision = RouteDecision(
                QueryRoute.MULTI_SOURCE,
                "Query compares multiple product SKUs and may need product facts "
                "plus supporting documents",
                0.98,
            )
        elif skus and has_policy_term:
            decision = RouteDecision(
                QueryRoute.MULTI_SOURCE,
                "Query combines an exact product SKU with policy or delivery intent",
                0.92,
            )
        elif skus and has_structured_term:
            decision = RouteDecision(
                QueryRoute.STRUCTURED,
                "Query asks for an exact product attribute by SKU",
                0.99,
            )
        elif _contains_any(folded, FILTER_TERMS) and (
            "product" in folded or has_structured_term
        ):
            decision = RouteDecision(
                QueryRoute.STRUCTURED,
                "Query requests deterministic product filtering or sorting",
                0.95,
            )
        elif (
            ("which product" in folded or "what product" in folded)
            and has_structured_term
        ):
            decision = RouteDecision(
                QueryRoute.STRUCTURED,
                "Query requests structured product attributes",
                0.90,
            )
        else:
            decision = RouteDecision(
                QueryRoute.HYBRID,
                "Query is best answered from NovaCart document content",
                0.70,
            )

        logger.info(
            "Query router selected route=%s confidence=%s reason=%s",
            decision.route.value,
            decision.confidence,
            decision.reason,
        )
        return decision


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)
