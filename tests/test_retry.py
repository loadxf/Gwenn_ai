from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from gwenn.harness.retry import RetryConfig, is_retryable_error, with_retries


def test_api_connection_error_is_retryable():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="Connection error.", request=request)
    assert is_retryable_error(error) is True


def test_api_timeout_error_is_retryable():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APITimeoutError(request=request)
    assert is_retryable_error(error) is True


def test_value_error_is_not_retryable():
    assert is_retryable_error(ValueError("bad request shape")) is False


@pytest.mark.asyncio
async def test_with_retries_uses_exponential_backoff_when_retry_after_missing(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request, headers={})
    error = anthropic.RateLimitError(
        message="rate limit",
        response=response,
        body={},
    )
    func = AsyncMock(side_effect=error)
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fake_sleep)

    with pytest.raises(anthropic.RateLimitError):
        await with_retries(
            func,
            config=RetryConfig(
                max_retries=1,
                base_delay=2.0,
                max_delay=10.0,
                exponential_base=2.0,
                jitter_range=0.0,
            ),
        )

    assert delays == [2.0]


@pytest.mark.asyncio
async def test_with_retries_honors_positive_retry_after_header(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request, headers={"retry-after": "7"})
    error = anthropic.RateLimitError(
        message="rate limit",
        response=response,
        body={},
    )
    func = AsyncMock(side_effect=error)
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fake_sleep)

    with pytest.raises(anthropic.RateLimitError):
        await with_retries(
            func,
            config=RetryConfig(
                max_retries=1,
                base_delay=2.0,
                max_delay=10.0,
                exponential_base=2.0,
                jitter_range=0.0,
            ),
        )

    assert delays == [7.0]


@pytest.mark.asyncio
async def test_with_retries_honors_http_date_retry_after(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    future = datetime.now(timezone.utc) + timedelta(days=365 * 10)
    response = httpx.Response(429, request=request, headers={"retry-after": future.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    error = anthropic.RateLimitError(
        message="rate limit",
        response=response,
        body={},
    )
    func = AsyncMock(side_effect=error)
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fake_sleep)

    with pytest.raises(anthropic.RateLimitError):
        await with_retries(
            func,
            config=RetryConfig(
                max_retries=1,
                base_delay=0.1,
                max_delay=0.1,
                exponential_base=2.0,
                jitter_range=0.0,
            ),
        )

    assert len(delays) == 1
    assert delays[0] > 1_000_000


@pytest.mark.asyncio
async def test_with_retries_supports_async_on_retry_callback(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request, headers={})
    error = anthropic.RateLimitError(
        message="rate limit",
        response=response,
        body={},
    )
    call_count = 0

    async def _flaky_call():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error
        return "ok"

    retries_seen: list[int] = []

    async def _on_retry(attempt: int, _error: Exception, _delay: float) -> None:
        retries_seen.append(attempt)

    async def _fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fake_sleep)

    result = await with_retries(
        _flaky_call,
        config=RetryConfig(
            max_retries=2,
            base_delay=0.1,
            max_delay=0.1,
            exponential_base=2.0,
            jitter_range=0.0,
        ),
        on_retry=_on_retry,
    )

    assert result == "ok"
    assert retries_seen == [1]
