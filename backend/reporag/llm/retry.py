"""Retry with exponential backoff and jitter."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def backoff_delays(
    max_attempts: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.25
) -> list[float]:
    """Delays before attempts 2..N: base * 2^i, capped, with +/- jitter.

    Jitter spreads out retries so many clients that failed together don't all
    hit the API again at the same instant.
    """
    delays = []
    for i in range(max_attempts - 1):
        d = min(cap, base * (2**i))
        delays.append(max(0.0, d * (1 + random.uniform(-jitter, jitter))))
    return delays


def retry_with_backoff(
    fn: Callable[[], T],
    is_retryable: Callable[[Exception], bool],
    max_attempts: int = 5,
    base: float = 1.0,
    cap: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying transient failures. Non-retryable errors raise immediately."""
    delays = backoff_delays(max_attempts, base, cap)
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts or not is_retryable(exc):
                raise
            delay = delays[attempt - 1]
            log.warning("Attempt %d failed (%s); retrying in %.1fs", attempt, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")
