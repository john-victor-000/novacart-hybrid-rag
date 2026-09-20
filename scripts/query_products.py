"""Run deterministic structured queries against NovaCart products.csv."""

from argparse import ArgumentParser
import logging
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.structured import (
    PandasProductRepository,
    ProductService,
    StructuredProductRetriever,
    StructuredRetrievalError,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    settings = Settings()
    parser = ArgumentParser(description="Query structured NovaCart product data.")
    parser.add_argument("query", help="Supported product lookup, filter, or comparison.")
    parser.add_argument(
        "--products-path",
        type=Path,
        default=Path(settings.products_csv_path),
    )
    args = parser.parse_args()

    try:
        repository = PandasProductRepository.from_csv(args.products_path)
        result = StructuredProductRetriever(
            ProductService(repository)
        ).retrieve(args.query)
    except StructuredRetrievalError as exc:
        parser.error(str(exc))

    print(f"query={result.query}")
    print(f"operation={result.operation}")
    print(f"results={len(result.products)}")
    for rank, product in enumerate(result.products, start=1):
        print(
            f"{rank}. {product.sku} - {product.product_name} | "
            f"category={product.category} | price_inr={product.price_inr} | "
            f"warranty_months={product.warranty_months} | rating={product.rating:.1f} | "
            f"stock_status={product.stock_status}"
        )
    print(f"source={result.source}")


if __name__ == "__main__":
    main()
