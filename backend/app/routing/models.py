"""Models produced by the observable query router."""

from dataclasses import dataclass
from enum import Enum


class QueryRoute(str, Enum):
    """Available retrieval paths in the unified pipeline."""

    DENSE = "DENSE"
    HYBRID = "HYBRID"
    STRUCTURED = "STRUCTURED"
    MULTI_SOURCE = "MULTI_SOURCE"


@dataclass(frozen=True)
class RouteDecision:
    """Selected route plus a human-readable explanation."""

    route: QueryRoute
    reason: str
    confidence: float | None = None
