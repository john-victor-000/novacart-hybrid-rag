from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from backend.app.api.chat import get_chat_service
from backend.app.chat import ChatService
from backend.app.conversations import (
    ConversationService,
    FollowUpRewriter,
    InMemoryConversationRepository,
)
from backend.app.llm import LLMProviderError
from backend.app.main import app
from backend.app.rag.errors import RetrievalUnavailableError
from backend.app.rag.unified_models import (
    PipelineMetadata,
    UnifiedRAGResponse,
    UnifiedSource,
)
from backend.app.routing import QueryRoute


class RecordingRAG:
    def __init__(self, with_evidence: bool = True) -> None:
        self.queries: list[str] = []
        self.with_evidence = with_evidence

    def answer(
        self,
        query: str,
        include_debug: bool = False,
        request_id: str | None = None,
    ) -> UnifiedRAGResponse:
        self.queries.append(query)
        sources = [
            UnifiedSource(
                evidence_id="product:nkm-10",
                source="products.csv",
                document_name="products.csv",
                document_type="csv",
                section="NKM-10",
                sku="NKM-10",
                retrieval_method="structured",
            )
        ] if self.with_evidence else []
        return UnifiedRAGResponse(
            answer=(
                "The warranty is 18 months. [Source 1]"
                if sources
                else "The information was not found in the NovaCart knowledge base."
            ),
            route=QueryRoute.STRUCTURED,
            sources=sources,
            metadata=PipelineMetadata(
                retrieval_count=len(sources),
                latency_ms=1.5,
                retrieval_latency_ms=0.5,
                generation_latency_ms=1.0 if sources else 0.0,
            ),
        )


class FailingChatService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def answer(self, *args: object, **kwargs: object) -> None:
        raise self.error


def _chat_service(rag: RecordingRAG) -> ChatService:
    return ChatService(
        rag,  # type: ignore[arg-type]
        ConversationService(InMemoryConversationRepository()),
        FollowUpRewriter(),
    )


def test_conversation_creation_continuation_and_followup() -> None:
    rag = RecordingRAG()
    service = _chat_service(rag)
    app.dependency_overrides[get_chat_service] = lambda: service
    client = TestClient(app)
    try:
        first = client.post(
            "/api/chat",
            json={"query": "What is the warranty of NCM-24?"},
        )
        conversation_id = first.json()["conversation_id"]
        second = client.post(
            "/api/chat",
            json={
                "query": "What about NKM-10?",
                "conversation_id": conversation_id,
            },
        )
        history = client.get(f"/api/conversations/{conversation_id}")
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert rag.queries == [
        "What is the warranty of NCM-24?",
        "What is the warranty of NKM-10?",
    ]
    assert history.status_code == 200
    turns = history.json()["turns"]
    assert len(turns) == 2
    assert turns[1]["user_message"] == "What about NKM-10?"
    assert turns[1]["retrieval_query"] == "What is the warranty of NKM-10?"
    assert turns[1]["sources"][0]["source"] == "products.csv"
    assert "timestamp" in turns[1]


def test_chat_response_schema_and_insufficient_evidence() -> None:
    service = _chat_service(RecordingRAG(with_evidence=False))
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "Does NovaCart sell bicycles?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "conversation_id",
        "answer",
        "route",
        "sources",
        "metadata",
    }
    assert body["sources"] == []
    assert body["metadata"]["retrieval_count"] == 0


def test_empty_query_and_unknown_conversation_are_clear_errors() -> None:
    service = _chat_service(RecordingRAG())
    app.dependency_overrides[get_chat_service] = lambda: service
    client = TestClient(app)
    try:
        empty = client.post("/api/chat", json={"query": "   "})
        unknown = client.post(
            "/api/chat",
            json={
                "query": "What is the return policy?",
                "conversation_id": str(uuid4()),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert empty.status_code == 422
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Conversation not found"}


@pytest.mark.parametrize(
    ("error", "expected_detail"),
    [
        (
            LLMProviderError("secret provider details"),
            "The answer service is temporarily unavailable",
        ),
        (
            RetrievalUnavailableError("internal vector error"),
            "Document retrieval is temporarily unavailable",
        ),
    ],
)
def test_chat_failures_are_safe(
    error: Exception,
    expected_detail: str,
) -> None:
    app.dependency_overrides[get_chat_service] = lambda: FailingChatService(error)
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "What is the return policy?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": expected_detail}
    assert "secret provider details" not in response.text
    assert "internal vector error" not in response.text
