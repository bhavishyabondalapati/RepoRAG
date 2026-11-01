"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

import httpx

from reporag.llm.base import LLMError
from reporag.llm.retry import retry_with_backoff

# Rate limits and server-side hiccups are worth retrying; bad requests are not.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError | TimeoutError | ConnectionError)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str, max_attempts: int = 5):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_attempts = max_attempts

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)

        def call() -> str:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            return response.text or ""

        try:
            return retry_with_backoff(call, _is_retryable, max_attempts=self.max_attempts)
        except errors.APIError as exc:
            raise LLMError(f"Gemini request failed ({exc.code}): {exc.message}") from exc
