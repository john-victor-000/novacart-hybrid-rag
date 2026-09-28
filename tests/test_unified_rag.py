from fastapi.testclient import TestClient
import pytest

from backend.app.api.chat import get_chat_service
from backend.app.chat import ChatService
from backend.app.conversations import (
    ConversationService,
    FollowUpRewriter,
    InMemoryConversationRepository,
)
from backend.app.llm import LLMProvider
from backend.app.main import app
from backend.app.rag import UnifiedContextBuilder, UnifiedRAGService
from backend.app.rag.unified_service import INFORMATION_NOT_FOUND
from backend.app.retrieval import RetrievalResult
from backend.app.routing import QueryRoute, QueryRouter
from backend.app.structured import ProductRecord, StructuredQueryResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeStructuredRetriever:
    def __init__(self, products: list[ProductRecord]) -> None:
        self.products = products
        self.calls: list[str] = []

    def retrieve(self, query: str) -> StructuredQueryResult:
        self.calls.append(query)
        return StructuredQueryResult(
            query=query,
            operation="comparison" if "compare" in query.casefold() else "sku_lookup",
            products=tuple(self.products),
        )


class FakeLLM(LLMProvider):
    def __init__(self, answer: str = "Grounded answer. [Source 1]") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the return policy?", QueryRoute.HYBRID),
        ("What is the price of NKM-10?", QueryRoute.STRUCTURED),
        ("Which products cost below ₹5000?", QueryRoute.STRUCTURED),
        (
            "Compare the warranty of NCM-24 and NKM-10.",
            QueryRoute.MULTI_SOURCE,
        ),
        ("Tell me something about NovaCart.", QueryRoute.HYBRID),
    ],
)
def test_router_rules(query: str, expected: QueryRoute) -> None:
    assert QueryRouter().route(query).route == expected


def test_hybrid_policy_query_preserves_chunk_metadata() -> None:
    chunk = _chunk()
    service, _, hybrid, _, llm = _service(
        hybrid_results=[chunk],
        llm_answer="Seven calendar days. [Source 1]",
    )

    response = service.answer("What is the return policy?", include_debug=True)

    assert response.route == QueryRoute.HYBRID
    assert hybrid.calls == [("What is the return policy?", 5)]
    assert response.sources[0].chunk_id == "return-1"
    assert response.sources[0].document_name == "return_policy.pdf"
    assert response.sources[0].page == 1
    assert response.sources[0].section == "Standard Return Window"
    assert response.retrieval_debug is not None
    assert response.retrieval_debug.evidence[0].retrieval_info == {
        "strategy": "hybrid"
    }
    assert "answer using only the novacart evidence" in llm.prompts[0].lower()


def test_structured_price_query_preserves_product_source() -> None:
    product = _product("NKM-10", "Mechanical Keyboard", 3299, 18)
    service, dense, hybrid, structured, _ = _service(products=[product])

    response = service.answer("What is the price of NKM-10?")

    assert response.route == QueryRoute.STRUCTURED
    assert dense.calls == []
    assert hybrid.calls == []
    assert structured.calls == ["What is the price of NKM-10?"]
    assert response.sources[0].sku == "NKM-10"
    assert response.sources[0].source == "products.csv"


def test_multi_source_comparison_combines_structured_and_hybrid() -> None:
    products = [
        _product("NCM-24", "24-inch Monitor", 11999, 24),
        _product("NKM-10", "Mechanical Keyboard", 3299, 18),
    ]
    service, _, hybrid, structured, _ = _service(
        hybrid_results=[_chunk()],
        products=products,
        llm_answer="The monitor has 24 months and keyboard 18 months. "
        "[Source 1] [Source 2] [Source 3]",
    )

    response = service.answer(
        "Compare the warranty of NCM-24 and NKM-10.",
        include_debug=True,
    )

    assert response.route == QueryRoute.MULTI_SOURCE
    assert len(structured.calls) == 1
    assert len(hybrid.calls) == 1
    assert {source.sku for source in response.sources if source.sku} == {
        "NCM-24",
        "NKM-10",
    }
    assert any(source.chunk_id == "return-1" for source in response.sources)
    assert response.retrieval_debug is not None
    assert response.retrieval_debug.evidence_count == 3


def test_duplicate_evidence_is_removed_before_generation() -> None:
    duplicate = _chunk()
    service, _, _, _, llm = _service(
        hybrid_results=[duplicate, duplicate],
    )

    response = service.answer("What is the return policy?", include_debug=True)

    assert response.retrieval_debug is not None
    assert response.retrieval_debug.evidence_count == 1
    assert llm.prompts[0].count("[Source 1]") >= 1
    assert "[Source 2]" not in llm.prompts[0]


def test_insufficient_evidence_skips_generation() -> None:
    service, _, _, _, llm = _service(hybrid_results=[])

    response = service.answer("What is NovaCart's bicycle policy?")

    assert response.route == QueryRoute.HYBRID
    assert response.answer == INFORMATION_NOT_FOUND
    assert response.sources == []
    assert llm.prompts == []


def test_dense_override_remains_available_for_evaluation() -> None:
    service, dense, hybrid, structured, _ = _service(
        dense_results=[_chunk()],
        retrieval_mode="dense",
    )

    response = service.answer("What is the return policy?")

    assert response.route == QueryRoute.DENSE
    assert len(dense.calls) == 1
    assert hybrid.calls == []
    assert structured.calls == []


def test_unified_chat_api_response() -> None:
    service, _, _, _, _ = _service(
        products=[_product("NKM-10", "Mechanical Keyboard", 3299, 18)],
        llm_answer="The price is INR 3,299. [Source 1]",
    )
    chat_service = ChatService(
        service,
        ConversationService(InMemoryConversationRepository()),
        FollowUpRewriter(),
    )
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "What is the price of NKM-10?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The price is INR 3,299. [Source 1]"
    assert body["route"] == "STRUCTURED"
    assert "conversation_id" in body
    assert body["metadata"]["retrieval_count"] == 1
    assert "debug" not in body
    assert body["sources"][0]["sku"] == "NKM-10"
    assert body["sources"][0]["source"] == "products.csv"


def _service(
    dense_results: list[RetrievalResult] | None = None,
    hybrid_results: list[RetrievalResult] | None = None,
    products: list[ProductRecord] | None = None,
    llm_answer: str = "Grounded answer. [Source 1]",
    retrieval_mode: str = "auto",
) -> tuple[
    UnifiedRAGService,
    FakeRetriever,
    FakeRetriever,
    FakeStructuredRetriever,
    FakeLLM,
]:
    dense = FakeRetriever(dense_results or [])
    hybrid = FakeRetriever(hybrid_results or [])
    structured = FakeStructuredRetriever(products or [])
    llm = FakeLLM(llm_answer)
    service = UnifiedRAGService(
        router=QueryRouter(),
        dense_retriever=dense,
        hybrid_retriever=hybrid,
        structured_retriever=structured,
        context_builder=UnifiedContextBuilder(
            max_characters=4000,
            max_items=8,
        ),
        llm=llm,
        retrieval_mode=retrieval_mode,
        dense_top_k=5,
        hybrid_top_k=5,
    )
    return service, dense, hybrid, structured, llm


def _chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="return-1",
        text="Most eligible products may be returned within seven days.",
        score=0.85,
        document_id="doc-return",
        document_name="return_policy.pdf",
        document_type="pdf",
        source="data/raw/return_policy.pdf",
        page=1,
        section="Standard Return Window",
    )


def _product(
    sku: str,
    name: str,
    price: int,
    warranty: int,
) -> ProductRecord:
    return ProductRecord(
        sku=sku,
        product_name=name,
        category="Test category",
        price_inr=price,
        warranty_months=warranty,
        rating=4.5,
        stock_status="In Stock",
        source="products.csv",
    )
