"""LLM provider interface and Groq API implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import httpx

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot produce a valid response."""


class LLMProvider(ABC):
    """Interface for replaceable text-generation providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured generation model name."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for a complete prompt."""


class GroqLLMProvider(LLMProvider):
    """Generate text through Groq's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not self._api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured")

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            answer = str(payload["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            logger.exception("Groq generation failed for model=%s", self._model_name)
            raise LLMProviderError(
                "Unable to generate an answer with the configured Groq API"
            ) from exc

        if not answer:
            raise LLMProviderError("Groq returned an empty response")
        return answer
