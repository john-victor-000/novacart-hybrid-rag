"""Business operations over a storage-independent product repository."""

from __future__ import annotations

import logging
from time import perf_counter

from backend.app.structured.errors import (
    EmptyFilterError,
    NoProductsFoundError,
    ProductNotFoundError,
)
from backend.app.structured.models import (
    NumericFilter,
    NumericOperator,
    ProductFilters,
    ProductRecord,
    ProductSort,
    SortField,
)
from backend.app.structured.repository import ProductRepository

logger = logging.getLogger(__name__)


class ProductService:
    """Perform deterministic product lookup, filtering, sorting, and comparison."""

    def __init__(self, repository: ProductRepository) -> None:
        self.repository = repository

    def lookup_sku(self, sku: str) -> ProductRecord:
        normalized = sku.strip()
        if not normalized:
            raise ValueError("SKU cannot be empty")
        started = perf_counter()
        product = self.repository.get_by_sku(normalized)
        self._log("sku_lookup", int(product is not None), started)
        if product is None:
            raise ProductNotFoundError(f"unknown SKU: {normalized}")
        return product

    def lookup_name(self, product_name: str) -> list[ProductRecord]:
        normalized = product_name.strip()
        if not normalized:
            raise ValueError("product name cannot be empty")
        started = perf_counter()
        products = self.repository.find_by_name(normalized)
        return self._require_matches("product_name_lookup", products, started)

    def filter_products(
        self,
        filters: ProductFilters,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        if filters.is_empty:
            raise EmptyFilterError("at least one product filter is required")
        started = perf_counter()
        products = self.repository.filter(filters, sort=sort)
        return self._require_matches("filter", products, started)

    def filter_by_price(
        self,
        value: float,
        operator: NumericOperator,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        return self.filter_products(
            ProductFilters(
                numeric=(NumericFilter("price_inr", operator, value),),
            ),
            sort=sort,
        )

    def filter_by_warranty(
        self,
        value: float,
        operator: NumericOperator,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        return self.filter_products(
            ProductFilters(
                numeric=(NumericFilter("warranty_months", operator, value),),
            ),
            sort=sort,
        )

    def filter_by_stock(
        self,
        stock_status: str,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        normalized = stock_status.strip()
        if not normalized:
            raise EmptyFilterError("stock status cannot be empty")
        return self.filter_products(
            ProductFilters(stock_status=normalized),
            sort=sort,
        )

    def filter_by_category(
        self,
        category: str,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        normalized = category.strip()
        if not normalized:
            raise EmptyFilterError("category cannot be empty")
        return self.filter_products(
            ProductFilters(category=normalized),
            sort=sort,
        )

    def sort_products(
        self,
        field: SortField,
        descending: bool = False,
    ) -> list[ProductRecord]:
        started = perf_counter()
        products = self.repository.list_all(ProductSort(field, descending))
        return self._require_matches("sort", products, started)

    def compare(self, skus: list[str]) -> list[ProductRecord]:
        normalized = [sku.strip() for sku in skus if sku.strip()]
        unique = list(dict.fromkeys(sku.casefold() for sku in normalized))
        if len(unique) < 2:
            raise ValueError("comparison requires at least two unique SKUs")

        started = perf_counter()
        products: list[ProductRecord] = []
        for sku in normalized:
            if any(product.sku.casefold() == sku.casefold() for product in products):
                continue
            product = self.repository.get_by_sku(sku)
            if product is None:
                self._log("comparison", len(products), started)
                raise ProductNotFoundError(f"unknown SKU: {sku}")
            products.append(product)
        self._log("comparison", len(products), started)
        return products

    @staticmethod
    def _require_matches(
        operation: str,
        products: list[ProductRecord],
        started: float,
    ) -> list[ProductRecord]:
        ProductService._log(operation, len(products), started)
        if not products:
            raise NoProductsFoundError(
                f"no matching products for operation: {operation}"
            )
        return products

    @staticmethod
    def _log(operation: str, count: int, started: float) -> None:
        logger.info(
            "Structured retrieval operation=%s matched_records=%s latency=%.3fs",
            operation,
            count,
            perf_counter() - started,
        )
