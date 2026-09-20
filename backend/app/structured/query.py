"""Deterministic natural-language interpretation for product-only queries."""

from __future__ import annotations

import re

from backend.app.structured.errors import StructuredQueryError
from backend.app.structured.models import (
    NumericFilter,
    NumericOperator,
    ProductFilters,
    ProductSort,
    SortField,
    StructuredQueryResult,
)
from backend.app.structured.service import ProductService

SKU_PATTERN = re.compile(r"\b[A-Z]{2,}-\d+\b", re.IGNORECASE)
NUMBER_PATTERN = r"(?:₹\s*)?(?:INR\s*)?(?:Rs\.?\s*)?([\d,]+(?:\.\d+)?)"
COMPARISONS: tuple[tuple[str, NumericOperator], ...] = (
    (r"(?:less than|under|below)\s+" + NUMBER_PATTERN, "lt"),
    (r"(?:at most|up to|no more than)\s+" + NUMBER_PATTERN, "lte"),
    (r"(?:more than|over|above)\s+" + NUMBER_PATTERN, "gt"),
    (r"(?:at least|no less than)\s+" + NUMBER_PATTERN, "gte"),
    (r"(?:equal to|exactly)\s+" + NUMBER_PATTERN, "eq"),
)


class StructuredProductRetriever:
    """Translate supported product questions into deterministic service calls."""

    def __init__(self, service: ProductService) -> None:
        self.service = service

    def retrieve(self, query: str) -> StructuredQueryResult:
        normalized = " ".join(query.split())
        if not normalized:
            raise StructuredQueryError("structured query cannot be empty")

        skus = [match.upper() for match in SKU_PATTERN.findall(normalized)]
        if re.search(r"\bcompare\b", normalized, re.IGNORECASE):
            if len(skus) < 2:
                raise StructuredQueryError(
                    "comparison query requires at least two SKUs"
                )
            products = self.service.compare(skus)
            return StructuredQueryResult(
                query=normalized,
                operation="comparison",
                products=tuple(products),
            )
        if skus:
            product = self.service.lookup_sku(skus[0])
            return StructuredQueryResult(
                query=normalized,
                operation="sku_lookup",
                products=(product,),
            )

        sort = _parse_sort(normalized)
        filters = _parse_filters(normalized)
        if not filters.is_empty:
            products = self.service.filter_products(filters, sort=sort)
            return StructuredQueryResult(
                query=normalized,
                operation="filter",
                products=tuple(products),
            )
        if sort is not None:
            products = self.service.sort_products(
                sort.field,
                descending=sort.descending,
            )
            return StructuredQueryResult(
                query=normalized,
                operation="sort",
                products=tuple(products),
            )

        name = _parse_product_name(normalized)
        if name:
            products = self.service.lookup_name(name)
            return StructuredQueryResult(
                query=normalized,
                operation="product_name_lookup",
                products=tuple(products),
            )

        raise StructuredQueryError(
            "unsupported structured product query; use an SKU, comparison, "
            "product name lookup, filter, or sort"
        )


def _parse_filters(query: str) -> ProductFilters:
    folded = query.casefold()
    numeric: list[NumericFilter] = []
    numeric_match = _parse_numeric_comparison(query)
    if numeric_match:
        operator, value = numeric_match
        field = (
            "warranty_months"
            if "warranty" in folded or "month" in folded
            else "price_inr"
        )
        numeric.append(NumericFilter(field, operator, value))

    stock_status = None
    if "out of stock" in folded:
        stock_status = "Out of Stock"
    elif "low stock" in folded:
        stock_status = "Low Stock"
    elif "in stock" in folded:
        stock_status = "In Stock"

    category = None
    category_match = re.search(
        r"\bin\s+(?:the\s+)?(.+?)\s+category\b",
        query,
        re.IGNORECASE,
    )
    if category_match:
        category = category_match.group(1).strip()

    return ProductFilters(
        numeric=tuple(numeric),
        stock_status=stock_status,
        category=category,
    )


def _parse_numeric_comparison(
    query: str,
) -> tuple[NumericOperator, float] | None:
    for pattern, operator in COMPARISONS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return operator, float(match.group(1).replace(",", ""))
            except ValueError as exc:
                raise StructuredQueryError("invalid numeric value") from exc
    return None


def _parse_sort(query: str) -> ProductSort | None:
    folded = query.casefold()
    if "cheapest" in folded:
        return ProductSort("price_inr")
    if "most expensive" in folded:
        return ProductSort("price_inr", descending=True)
    if "highest rated" in folded or "best rated" in folded:
        return ProductSort("rating", descending=True)

    match = re.search(
        r"\bsort(?:\s+products)?\s+by\s+"
        r"(price|warranty|rating|name|category|stock)"
        r"(?:\s+(ascending|descending))?",
        folded,
    )
    if not match:
        return None
    fields: dict[str, SortField] = {
        "price": "price_inr",
        "warranty": "warranty_months",
        "rating": "rating",
        "name": "product_name",
        "category": "category",
        "stock": "stock_status",
    }
    return ProductSort(
        fields[match.group(1)],
        descending=match.group(2) == "descending",
    )


def _parse_product_name(query: str) -> str | None:
    quoted = re.search(r'["“](.+?)["”]', query)
    if quoted:
        return quoted.group(1).strip()
    match = re.search(
        r"\b(?:find|lookup|show)\s+(?:the\s+)?(?:product\s+)?(.+?)[?.]?$",
        query,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None
