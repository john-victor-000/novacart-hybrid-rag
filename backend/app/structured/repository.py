"""Product repository abstraction and pandas-backed CSV implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from backend.app.structured.errors import EmptyFilterError, ProductDataError
from backend.app.structured.models import (
    NumericFilter,
    ProductFilters,
    ProductRecord,
    ProductSort,
    validate_product,
)

REQUIRED_COLUMNS = (
    "sku",
    "product_name",
    "category",
    "price_inr",
    "warranty_months",
    "rating",
    "stock_status",
)


class ProductRepository(Protocol):
    """Storage-independent product access used by the service layer."""

    def get_by_sku(self, sku: str) -> ProductRecord | None:
        """Return one exact SKU match."""

    def find_by_name(self, product_name: str) -> list[ProductRecord]:
        """Return case-insensitive exact or partial product-name matches."""

    def filter(
        self,
        filters: ProductFilters,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        """Return products matching structured criteria."""

    def list_all(self, sort: ProductSort | None = None) -> list[ProductRecord]:
        """Return all products with optional deterministic sorting."""


class PandasProductRepository:
    """Validated DataFrame-backed repository loaded from products.csv."""

    def __init__(self, frame: pd.DataFrame, source: str) -> None:
        self.source = source
        self._frame = _normalize_frame(frame)
        self._validate_records()

    @classmethod
    def from_csv(cls, path: Path | str) -> "PandasProductRepository":
        csv_path = Path(path)
        if not csv_path.is_file():
            raise ProductDataError(f"product CSV not found: {csv_path}")
        try:
            frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise ProductDataError(f"could not read product CSV: {csv_path}") from exc
        return cls(frame, source=csv_path.name)

    def get_by_sku(self, sku: str) -> ProductRecord | None:
        normalized = sku.strip().casefold()
        matches = self._frame[self._frame["sku"].str.casefold() == normalized]
        records = self._to_records(matches)
        return records[0] if records else None

    def find_by_name(self, product_name: str) -> list[ProductRecord]:
        normalized = product_name.strip().casefold()
        names = self._frame["product_name"].str.casefold()
        exact = self._frame[names == normalized]
        if not exact.empty:
            return self._to_records(exact)
        partial = self._frame[names.str.contains(normalized, regex=False)]
        return self._to_records(partial)

    def filter(
        self,
        filters: ProductFilters,
        sort: ProductSort | None = None,
    ) -> list[ProductRecord]:
        if filters.is_empty:
            raise EmptyFilterError("at least one product filter is required")

        matches = self._frame
        for condition in filters.numeric:
            matches = matches[_numeric_mask(matches, condition)]
        if filters.stock_status:
            value = filters.stock_status.strip().casefold()
            matches = matches[matches["stock_status"].str.casefold() == value]
        if filters.category:
            value = filters.category.strip().casefold()
            matches = matches[matches["category"].str.casefold() == value]
        return self._to_records(_sort_frame(matches, sort))

    def list_all(self, sort: ProductSort | None = None) -> list[ProductRecord]:
        return self._to_records(_sort_frame(self._frame, sort))

    def _validate_records(self) -> None:
        if self._frame["sku"].str.casefold().duplicated().any():
            raise ProductDataError("product CSV contains duplicate SKUs")
        for record in self._to_records(self._frame):
            validate_product(record)

    def _to_records(self, frame: pd.DataFrame) -> list[ProductRecord]:
        return [
            ProductRecord(
                sku=str(row.sku),
                product_name=str(row.product_name),
                category=str(row.category),
                price_inr=int(row.price_inr),
                warranty_months=int(row.warranty_months),
                rating=float(row.rating),
                stock_status=str(row.stock_status),
                source=self.source,
            )
            for row in frame.itertuples(index=False)
        ]


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ProductDataError(
            "product CSV is missing required columns: " + ", ".join(missing)
        )

    if frame.empty:
        raise ProductDataError("product CSV contains no product records")

    normalized = frame[list(REQUIRED_COLUMNS)].copy()
    text_columns = ["sku", "product_name", "category", "stock_status"]
    for column in text_columns:
        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ProductDataError(f"product CSV contains an empty {column}")

    for column in ["price_inr", "warranty_months", "rating"]:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ProductDataError(
                f"product CSV contains an invalid numeric value in {column}"
            ) from exc

    if (normalized["price_inr"] % 1 != 0).any():
        raise ProductDataError("price_inr values must be whole numbers")
    if (normalized["warranty_months"] % 1 != 0).any():
        raise ProductDataError("warranty_months values must be whole numbers")
    return normalized


def _numeric_mask(frame: pd.DataFrame, condition: NumericFilter) -> pd.Series:
    values = frame[condition.field]
    operators = {
        "lt": values < condition.value,
        "lte": values <= condition.value,
        "eq": values == condition.value,
        "gte": values >= condition.value,
        "gt": values > condition.value,
    }
    return operators[condition.operator]


def _sort_frame(
    frame: pd.DataFrame,
    sort: ProductSort | None,
) -> pd.DataFrame:
    if sort is None:
        return frame
    return frame.sort_values(
        by=sort.field,
        ascending=not sort.descending,
        kind="stable",
    )
