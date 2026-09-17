import pytest

from backend.app.llm import GroqLLMProvider, LLMProviderError


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": "Most products have a seven-day return window."
                    }
                }
            ]
        }


def test_groq_provider_uses_chat_completions_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> FakeResponse:
        captured.update(
            url=url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        return FakeResponse()

    monkeypatch.setattr("backend.app.llm.providers.httpx.post", fake_post)
    provider = GroqLLMProvider(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1/",
        model_name="llama-3.1-8b-instant",
        timeout_seconds=30.0,
    )

    answer = provider.generate("Use only the supplied context.")

    assert answer == "Most products have a seven-day return window."
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Use only the supplied context."}],
        "stream": False,
    }
    assert captured["timeout"] == 30.0


def test_groq_provider_requires_api_key() -> None:
    provider = GroqLLMProvider(
        api_key="",
        base_url="https://api.groq.com/openai/v1",
        model_name="llama-3.1-8b-instant",
        timeout_seconds=30.0,
    )

    with pytest.raises(LLMProviderError, match="GROQ_API_KEY"):
        provider.generate("A prompt")
