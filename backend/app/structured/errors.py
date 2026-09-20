"""Domain errors for structured product retrieval."""


class StructuredRetrievalError(ValueError):
    """Base error for invalid structured data or operations."""


class ProductDataError(StructuredRetrievalError):
    """The structured product source is malformed."""


class ProductNotFoundError(StructuredRetrievalError):
    """A requested SKU does not exist."""


class NoProductsFoundError(StructuredRetrievalError):
    """A valid lookup or filter produced no records."""


class EmptyFilterError(StructuredRetrievalError):
    """A filter operation was requested without any criteria."""


class StructuredQueryError(StructuredRetrievalError):
    """A natural-language product query is unsupported or malformed."""
