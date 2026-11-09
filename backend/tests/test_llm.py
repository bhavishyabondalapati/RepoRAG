import pytest

from reporag.llm.base import LLMError, get_llm
from reporag.llm.gemini import _is_retryable, _retry_after
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


def test_server_retry_hint_overrides_short_backoff():
    sleeps, calls = [], []

    def rate_limited():
        calls.append(1)
        if len(calls) < 2:
            raise Transient()
        return "ok"

    result = retry_with_backoff(
        rate_limited, lambda e: True, base=1, sleep=sleeps.append, retry_after=lambda e: 41.0
    )
    assert result == "ok" and sleeps == [41.0]


def test_long_server_retry_hint_fails_fast():
    # e.g. a daily quota that resets in hours: waiting a minute can't help.
    sleeps, calls = [], []

    def quota_exhausted():
        calls.append(1)
        raise Transient()

    with pytest.raises(Transient):
        retry_with_backoff(
            quota_exhausted,
            lambda e: True,
            max_attempts=5,
            sleep=sleeps.append,
            retry_after=lambda e: 10866.0,
            max_wait=90.0,
        )
    assert sleeps == [] and len(calls) == 1


def test_gemini_reads_retry_delay_from_429():
    from google.genai import errors

    body = {
        "error": {
            "code": 429,
            "message": "quota",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"},
            ],
        }
    }
    assert _retry_after(errors.APIError(429, body)) == 41.0
    assert _retry_after(errors.APIError(503, {"error": {"message": "busy"}})) is None
    assert _retry_after(ValueError("x")) is None


def test_gemini_client_has_a_request_timeout():
    # A stalled connection must time out (and then be retried), not hang the request forever.
    from reporag.llm.gemini import GeminiProvider

    provider = GeminiProvider("test-key", "m", timeout_s=12.5)
    assert provider._client._api_client._http_options.timeout == 12500


def test_timeouts_are_retryable():
    import httpx

    assert _is_retryable(httpx.ReadTimeout("stalled"))
