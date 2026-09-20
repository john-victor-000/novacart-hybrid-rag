"""Normalized product and structured-query result models."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from backend.app.structured.errors import ProductDataError

NumericField = Literal["price_inr", "warranty_months", "rating"]
NumericOperator = Literal["lt", "lte", "eq", "gte", "gt"]
SortField = Literal[
    "sku",
    "product_name",
    "category",
    "price_inr",
    "warranty_months",
    "rating",
    "stock_status",
]


@dataclass(frozen=True)
class ProductRecord:
    """One validated product row returned by structured retrieval."""

    sku: str
    product_name: str
    category: str
    price_inr: int
    warranty_months: int
    rating: float
    stock_status: str
    source: str


@dataclass(frozen=True)
class NumericFilter:
    """A numeric comparison applied to one product field."""

    field: NumericField
    operator: NumericOperator
    value: float

    def __post_init__(self) -> None:
        if self.field not in {"price_inr", "warranty_months", "rating"}:
            raise ValueError(f"unsupported numeric field: {self.field}")
        if self.operator not in {"lt", "lte", "eq", "gte", "gt"}:
            raise ValueError(f"unsupported numeric operator: {self.operator}")
        try:
            numeric_value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric filter value must be a valid number") from exc
        if not isfinite(numeric_value) or numeric_value < 0:
            raise ValueError("numeric filter value must be finite and non-negative")
        object.__setattr__(self, "value", numeric_value)


@dataclass(frozen=True)
class ProductFilters:
    """Composable criteria understood by any product repository."""

    numeric: tuple[NumericFilter, ...] = ()
    stock_status: str | None = None
    category: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            not self.numeric
            and not (self.stock_status or "").strip()
            and not (self.category or "").strip()
        )


@dataclass(frozen=True)
class ProductSort:
    """A validated field and direction for deterministic product sorting."""

    field: SortField
    descending: bool = False

    def __post_init__(self) -> None:
        if self.field not in {
            "sku",
            "product_name",
            "category",
            "price_inr",
            "warranty_months",
            "rating",
            "stock_status",
        }:
            raise ValueError(f"unsupported sort field: {self.field}")


@dataclass(frozen=True)
class StructuredQueryResult:
    """Stable response from the independently callable structured retriever."""

    query: str
    operation: str
    products: tuple[ProductRecord, ...]

    @property
    def source(self) -> str | None:
        return self.products[0].source if self.products else None


def validate_product(record: ProductRecord) -> None:
    """Reject malformed records regardless of their storage backend."""
    required_text = {
        "sku": record.sku,
        "product_name": record.product_name,
        "category": record.category,
        "stock_status": record.stock_status,
        "source": record.source,
    }
    missing = [name for name, value in required_text.items() if not value.strip()]
    if missing:
        raise ProductDataError(
            "product record has empty required fields: " + ", ".join(missing)
        )
    if record.price_inr < 0 or record.warranty_months < 0:
        raise ProductDataError("product numeric values cannot be negative")
    if not isfinite(record.rating) or not 0 <= record.rating <= 5:
        raise ProductDataError("product rating must be between 0 and 5")
