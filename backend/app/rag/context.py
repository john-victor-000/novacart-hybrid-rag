"""Construct labeled NovaCart context from dense retrieval results."""

from backend.app.retrieval.models import RetrievalResult


class ContextBuilder:
    """Format retrieved chunks as bounded, traceable prompt context."""

    def build(self, chunks: list[RetrievalResult]) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            page = str(chunk.page) if chunk.page is not None else "not available"
            section = chunk.section or "not available"
            blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Document: {chunk.document_name}",
                        f"Document type: {chunk.document_type}",
                        f"Page: {page}",
                        f"Section: {section}",
                        f"Chunk ID: {chunk.chunk_id}",
                        "Content:",
                        chunk.text.strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)
