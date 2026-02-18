from __future__ import annotations

import anthropic
import httpx

from gwenn.harness.retry import is_retryable_error


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
