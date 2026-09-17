from fastapi.testclient import TestClient

from backend.app.api.chat import get_rag_service
from backend.app.llm import LLMProvider
from backend.app.main import app
from backend.app.rag import ContextBuilder, RAGService
from backend.app.rag.service import INFORMATION_NOT_FOUND
from backend.app.retrieval import RetrievalResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeLLMProvider(LLMProvider):
    def __init__(self, answer: str = "A grounded answer. [Source 1]") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-local-model"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_retrieval_results_flow_into_context_and_prompt() -> None:
    chunk = _result(
        text="Most eligible products can be returned within 7 calendar days.",
        document_name="return_policy.pdf",
        page=1,
        section="Standard Return Window",
    )
    retriever = FakeRetriever([chunk])
    llm = FakeLLMProvider()
    service = RAGService(retriever, ContextBuilder(), llm)  # type: ignore[arg-type]

    response = service.answer("What is the return window?", top_k=3)

    assert retriever.calls == [("What is the return window?", 3)]
    assert len(llm.prompts) == 1
    assert chunk.text in llm.prompts[0]
    assert "return_policy.pdf" in llm.prompts[0]
    assert "[Source 1]" in llm.prompts[0]
    assert "answer only with facts" in llm.prompts[0].lower()
    assert response.answer == llm.answer


def test_insufficient_context_skips_llm() -> None:
    retriever = FakeRetriever([])
    llm = FakeLLMProvider()
    service = RAGService(retriever, ContextBuilder(), llm)  # type: ignore[arg-type]

    response = service.answer(
        "Does NovaCart sell bicycles?",
        top_k=5,
        include_retrieved_chunks=True,
    )

    assert response.answer == INFORMATION_NOT_FOUND
    assert response.sources == []
    assert response.retrieved_chunks == []
    assert llm.prompts == []


def test_sources_are_preserved_and_deduplicated() -> None:
    chunks = [
        _result(chunk_id="chunk-1", text="Return rule part one."),
        _result(chunk_id="chunk-2", text="Return rule part two."),
        _result(
            chunk_id="chunk-3",
            document_id="doc-faq",
            document_name="faq.pdf",
            source="data/raw/faq.pdf",
            text="A related FAQ.",
        ),
    ]
    service = RAGService(  # type: ignore[arg-type]
        FakeRetriever(chunks),
        ContextBuilder(),
        FakeLLMProvider("Grounded details. [Source 1] [Source 3] [Source 99]"),
    )

    response = service.answer("What is the return rule?", top_k=5)

    assert len(response.sources) == 2
    assert "[Source 99]" not in response.answer
    first = response.sources[0]
    assert first.chunk_id == "chunk-1"
    assert first.document_id == "doc-return"
    assert first.document_name == "return_policy.pdf"
    assert first.document_type == "pdf"
    assert first.source == "data/raw/return_policy.pdf"
    assert first.page == 1
    assert first.section == "Standard Return Window"


def test_llm_insufficient_context_answer_has_no_sources() -> None:
    service = RAGService(  # type: ignore[arg-type]
        FakeRetriever([_result()]),
        ContextBuilder(),
        FakeLLMProvider(INFORMATION_NOT_FOUND),
    )

    response = service.answer("An unsupported question", top_k=1)

    assert response.answer == INFORMATION_NOT_FOUND
    assert response.sources == []


def test_chat_api_response_schema() -> None:
    service = RAGService(  # type: ignore[arg-type]
        FakeRetriever([_result()]),
        ContextBuilder(),
        FakeLLMProvider("Seven calendar days. [Source 1]"),
    )
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "What is the return period?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Seven calendar days. [Source 1]"
    assert body["retrieved_chunks"] is None
    assert body["sources"] == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-return",
            "document_name": "return_policy.pdf",
            "document_type": "pdf",
            "source": "data/raw/return_policy.pdf",
            "page": 1,
            "section": "Standard Return Window",
        }
    ]


def _result(
    chunk_id: str = "chunk-1",
    text: str = "Most eligible products have a seven-day return window.",
    document_id: str = "doc-return",
    document_name: str = "return_policy.pdf",
    document_type: str = "pdf",
    source: str = "data/raw/return_policy.pdf",
    page: int | None = 1,
    section: str | None = "Standard Return Window",
    score: float = 0.82,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        source=source,
        page=page,
        section=section,
    )
