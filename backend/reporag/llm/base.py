"""Provider-agnostic LLM interface.

The rest of the app only depends on `LLMProvider`, so adding OpenAI, a local
model, etc. means writing one class and registering it in `get_llm`.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a provider fails permanently (after retries) or is misconfigured."""


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...


def get_llm(provider: str = "gemini", **kwargs) -> LLMProvider:
    if provider == "gemini":
        from reporag.llm.gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    raise LLMError(f"Unknown LLM provider: {provider}")
