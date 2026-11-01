import pytest

from reporag.llm.base import LLMError, get_llm
from reporag.llm.gemini import _is_retryable
from reporag.llm.retry import backoff_delays, retry_with_backoff


class Transient(Exception):
    pass


class Fatal(Exception):
    pass


def test_retries_transient_errors_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise Transient()
        return "ok"

    result = retry_with_backoff(
        flaky, lambda e: isinstance(e, Transient), max_attempts=5, base=1, sleep=sleeps.append
    )
    assert result == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential growth (jitter is only +/-25%)


def test_non_retryable_error_raises_immediately():
    sleeps = []

    def boom():
        raise Fatal()

    with pytest.raises(Fatal):
        retry_with_backoff(boom, lambda e: isinstance(e, Transient), sleep=sleeps.append)
    assert sleeps == []


def test_gives_up_after_max_attempts():
    calls = []

    def always_fails():
        calls.append(1)
        raise Transient()

    with pytest.raises(Transient):
        retry_with_backoff(always_fails, lambda e: True, max_attempts=3, sleep=lambda _: None)
    assert len(calls) == 3


def test_backoff_delays_are_capped():
    delays = backoff_delays(10, base=1, cap=8, jitter=0)
    assert delays == [1, 2, 4, 8, 8, 8, 8, 8, 8]


def test_gemini_retry_classification():
    from google.genai import errors

    assert _is_retryable(errors.APIError(429, {"error": {"message": "slow down"}}))
    assert _is_retryable(errors.APIError(503, {"error": {"message": "busy"}}))
    assert not _is_retryable(errors.APIError(400, {"error": {"message": "bad"}}))
    assert not _is_retryable(ValueError("nope"))


def test_gemini_requires_api_key():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_llm("gemini", api_key=None, model="m")


def test_unknown_provider():
    with pytest.raises(LLMError):
        get_llm("nope")
