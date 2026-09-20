"""Deterministic retrieval over structured NovaCart product data."""

from backend.app.structured.errors import (
    EmptyFilterError,
    NoProductsFoundError,
    ProductDataError,
    ProductNotFoundError,
    StructuredRetrievalError,
    StructuredQueryError,
)
from backend.app.structured.models import (
    NumericFilter,
    ProductFilters,
    ProductRecord,
    ProductSort,
    StructuredQueryResult,
)
from backend.app.structured.query import StructuredProductRetriever
from backend.app.structured.repository import (
    PandasProductRepository,
    ProductRepository,
)
from backend.app.structured.service import ProductService

__all__ = [
    "EmptyFilterError",
    "NoProductsFoundError",
    "NumericFilter",
    "PandasProductRepository",
    "ProductDataError",
    "ProductFilters",
    "ProductNotFoundError",
    "ProductRecord",
    "ProductRepository",
    "ProductService",
    "ProductSort",
    "StructuredProductRetriever",
    "StructuredQueryError",
    "StructuredQueryResult",
    "StructuredRetrievalError",
]
