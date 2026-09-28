"""Small, deterministic follow-up query rewriter."""

from dataclasses import dataclass
import re

from backend.app.conversations.models import ConversationTurn

SKU_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", re.IGNORECASE)
FOLLOW_UP_PATTERN = re.compile(
    r"^(?:and\s+)?(?:what|how)\s+about\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryRewrite:
    """Retrieval query and an observable rewrite reason."""

    query: str
    rewritten: bool
    reason: str | None = None


class FollowUpRewriter:
    """Carry a prior query topic forward only for explicit short follow-ups."""

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> QueryRewrite:
        normalized = " ".join(query.split())
        if not history or not FOLLOW_UP_PATTERN.search(normalized):
            return QueryRewrite(normalized, False)

        new_sku = SKU_PATTERN.search(normalized)
        previous_query = history[-1].retrieval_query
        old_sku = SKU_PATTERN.search(previous_query)
        if new_sku and old_sku:
            rewritten = (
                previous_query[: old_sku.start()]
                + new_sku.group(0).upper()
                + previous_query[old_sku.end() :]
            )
            return QueryRewrite(
                rewritten,
                True,
                "Replaced the prior SKU while preserving its question topic",
            )
        return QueryRewrite(normalized, False)
