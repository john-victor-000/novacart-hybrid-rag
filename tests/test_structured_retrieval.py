from pathlib import Path

import pandas as pd
import pytest

from backend.app.structured import (
    EmptyFilterError,
    NoProductsFoundError,
    NumericFilter,
    PandasProductRepository,
    ProductDataError,
    ProductFilters,
    ProductNotFoundError,
    ProductService,
    ProductSort,
    StructuredProductRetriever,
)


@pytest.fixture
def service(tmp_path: Path) -> ProductService:
    path = tmp_path / "products.csv"
    _product_frame().to_csv(path, index=False)
    return ProductService(PandasProductRepository.from_csv(path))


def test_exact_sku_lookup_is_case_insensitive(service: ProductService) -> None:
    product = service.lookup_sku("nkm-10")

    assert product.sku == "NKM-10"
    assert product.product_name == "Mechanical Keyboard"
    assert product.price_inr == 3299


def test_product_name_lookup_is_case_insensitive(service: ProductService) -> None:
    products = service.lookup_name("mechanical keyboard")

    assert [product.sku for product in products] == ["NKM-10"]


def test_price_filtering(service: ProductService) -> None:
    products = service.filter_by_price(
        5000,
        operator="lt",
        sort=ProductSort("price_inr"),
    )

    assert [product.sku for product in products] == [
        "NCP-65",
        "NCB-100",
        "NKM-10",
    ]


def test_warranty_filtering(service: ProductService) -> None:
    products = service.filter_by_warranty(12, operator="gt")

    assert [product.sku for product in products] == ["NKM-10", "NCM-24"]


def test_stock_filtering_uses_exact_status(service: ProductService) -> None:
    products = service.filter_by_stock("in stock")

    assert [product.sku for product in products] == [
        "NCB-100",
        "NKM-10",
        "NCM-24",
    ]


def test_category_filtering_and_sorting(service: ProductService) -> None:
    products = service.filter_by_category("audio")
    sorted_products = service.sort_products("rating", descending=True)

    assert [product.sku for product in products] == ["NCB-100"]
    assert sorted_products[0].sku == "NKM-10"


def test_comparison_preserves_requested_order(service: ProductService) -> None:
    products = service.compare(["NCM-24", "NKM-10"])

    assert [product.sku for product in products] == ["NCM-24", "NKM-10"]
    assert products[0].warranty_months == 24
    assert products[1].warranty_months == 18


def test_unknown_sku_and_no_match_are_distinct(service: ProductService) -> None:
    with pytest.raises(ProductNotFoundError, match="unknown SKU"):
        service.lookup_sku("UNKNOWN-99")

    with pytest.raises(NoProductsFoundError, match="no matching products"):
        service.filter_by_price(100, operator="lt")


def test_source_metadata_is_preserved(service: ProductService) -> None:
    product = service.lookup_sku("NCM-24")

    assert product.source == "products.csv"
    assert product.category == "Monitors"
    assert product.stock_status == "In Stock"


def test_empty_and_invalid_filters_are_rejected(service: ProductService) -> None:
    with pytest.raises(EmptyFilterError, match="at least one"):
        service.filter_products(ProductFilters())

    with pytest.raises(ValueError, match="valid number"):
        NumericFilter("price_inr", "lt", "not-a-number")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite and non-negative"):
        NumericFilter("warranty_months", "gt", -1)


def test_malformed_product_records_are_rejected(tmp_path: Path) -> None:
    invalid_numeric = _product_frame().astype({"price_inr": object})
    invalid_numeric.loc[0, "price_inr"] = "not-a-price"
    invalid_path = tmp_path / "invalid.csv"
    invalid_numeric.to_csv(invalid_path, index=False)

    with pytest.raises(ProductDataError, match="invalid numeric value"):
        PandasProductRepository.from_csv(invalid_path)

    missing_column_path = tmp_path / "missing.csv"
    _product_frame().drop(columns=["stock_status"]).to_csv(
        missing_column_path,
        index=False,
    )
    with pytest.raises(ProductDataError, match="missing required columns"):
        PandasProductRepository.from_csv(missing_column_path)


@pytest.mark.parametrize(
    ("query", "expected_skus"),
    [
        ("What is the price of NKM-10?", ["NKM-10"]),
        (
            "Which products cost less than 5000?",
            ["NCB-100", "NKM-10", "NCP-65"],
        ),
        (
            "Which products have more than 12 months warranty?",
            ["NKM-10", "NCM-24"],
        ),
        (
            "Which products are in stock?",
            ["NCB-100", "NKM-10", "NCM-24"],
        ),
        ("Compare NCM-24 and NKM-10", ["NCM-24", "NKM-10"]),
        ("Find product mechanical keyboard", ["NKM-10"]),
    ],
)
def test_supported_natural_language_queries(
    service: ProductService,
    query: str,
    expected_skus: list[str],
) -> None:
    result = StructuredProductRetriever(service).retrieve(query)

    assert [product.sku for product in result.products] == expected_skus
    assert result.source == "products.csv"


def _product_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku": "NCB-100",
                "product_name": "Wireless Earbuds",
                "category": "Audio",
                "price_inr": 2499,
                "warranty_months": 12,
                "rating": 4.3,
                "stock_status": "In Stock",
            },
            {
                "sku": "NCL-200",
                "product_name": "14-inch Laptop",
                "category": "Laptops",
                "price_inr": 54990,
                "warranty_months": 12,
                "rating": 4.5,
                "stock_status": "Low Stock",
            },
            {
                "sku": "NKM-10",
                "product_name": "Mechanical Keyboard",
                "category": "Computer Accessories",
                "price_inr": 3299,
                "warranty_months": 18,
                "rating": 4.6,
                "stock_status": "In Stock",
            },
            {
                "sku": "NCM-24",
                "product_name": "24-inch Monitor",
                "category": "Monitors",
                "price_inr": 11999,
                "warranty_months": 24,
                "rating": 4.4,
                "stock_status": "In Stock",
            },
            {
                "sku": "NCP-65",
                "product_name": "USB-C Charger",
                "category": "Power Accessories",
                "price_inr": 1799,
                "warranty_months": 12,
                "rating": 4.2,
                "stock_status": "Out of Stock",
            },
        ]
    )
