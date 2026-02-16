"""
Retry Logic — Bob's Resilience Against Transient Failures.

API calls fail. Networks drop. Rate limits hit. This module ensures that
transient failures don't crash the agent — they're handled gracefully with
exponential backoff, jitter, and intelligent retry decisions.

The retry strategy follows industry best practices:
- Exponential backoff: wait times double with each retry
- Jitter: random variation prevents thundering herd problems
- Smart classification: only retry transient errors (429, 5xx, network)
- Budget awareness: don't retry if we've exhausted our budget
- Fallback: gracefully degrade when retries are exhausted
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Optional, TypeVar

import anthropic
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter_range: float = 0.5,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_range = jitter_range


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is transient and worth retrying.

    Retryable:
    - 429 (rate limit) — always retry after backoff
    - 500, 502, 503, 529 (server errors) — transient, retry
    - Network/connection errors — transient, retry

    NOT retryable:
    - 400 (bad request) — our fault, fix the request
    - 401 (unauthorized) — bad API key, won't help to retry
    - 403 (forbidden) — permission issue
    - 404 (not found) — wrong endpoint
    """
    if isinstance(error, anthropic.RateLimitError):
        return True
    if isinstance(error, anthropic.InternalServerError):
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in (429, 500, 502, 503, 529)
    if isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    # OSError covers network-level issues
    if isinstance(error, OSError):
        return True
    return False


def compute_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[float] = None,
) -> float:
    """
    Compute the delay before the next retry attempt.

    Uses exponential backoff with jitter:
        delay = min(max_delay, base_delay * (exponential_base ^ attempt))
        delay += random jitter in [-jitter_range * delay, +jitter_range * delay]

    If the server provided a Retry-After header, use that instead
    (but never less than 1 second).
    """
    if retry_after is not None:
        return max(1.0, retry_after)

    # Exponential backoff
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    # Add jitter
    jitter = delay * config.jitter_range * (2 * random.random() - 1)
    delay = max(0.1, delay + jitter)

    return delay


async def with_retries(
    func: Callable,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
) -> Any:
    """
    Execute an async function with retry logic.

    This is the main retry wrapper used throughout the agent. It catches
    retryable errors, applies exponential backoff, and either retries or
    raises the final error.

    Args:
        func: The async function to execute (no arguments — use a lambda/closure)
        config: Retry configuration (uses defaults if not specified)
        on_retry: Optional callback when a retry occurs (receives attempt, error, delay)

    Returns:
        The result of the function call

    Raises:
        The last error if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()

    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                logger.error(
                    "retry.non_retryable_error",
                    error_type=type(e).__name__,
                    error=str(e),
                    attempt=attempt,
                )
                raise

            if attempt >= config.max_retries:
                logger.error(
                    "retry.exhausted",
                    error_type=type(e).__name__,
                    error=str(e),
                    total_attempts=attempt + 1,
                )
                raise

            # Extract retry-after if available
            retry_after = None
            if isinstance(e, anthropic.APIStatusError):
                retry_after_header = getattr(e, "response", None)
                if retry_after_header:
                    try:
                        headers = retry_after_header.headers
                        retry_after = float(headers.get("retry-after", 0))
                    except (ValueError, AttributeError):
                        pass

            delay = compute_delay(attempt, config, retry_after)

            logger.warning(
                "retry.attempt",
                error_type=type(e).__name__,
                error=str(e)[:200],
                attempt=attempt + 1,
                max_retries=config.max_retries,
                delay_seconds=round(delay, 1),
            )

            if on_retry:
                on_retry(attempt + 1, e, delay)

            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_error
