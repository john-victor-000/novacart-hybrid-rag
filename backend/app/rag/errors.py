"""Stable service errors mapped to safe API responses."""


class RetrievalUnavailableError(RuntimeError):
    """A document retriever could not complete the request."""


class StructuredDataUnavailableError(RuntimeError):
    """The structured product data could not be read or queried."""
