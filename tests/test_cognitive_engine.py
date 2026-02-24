from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest

from gwenn.api.claude import CLAUDE_CODE_OAUTH_BETA_HEADER, CognitiveEngine


@pytest.mark.asyncio
async def test_think_retries_transient_errors(monkeypatch):
    engine = object.__new__(CognitiveEngine)
    engine._model = "test-model"
    engine._max_tokens = 256
    engine._request_timeout_seconds = 120.0
    engine._retry_max_retries = 3
    engine._retry_base_delay = 0.5
    engine._retry_max_delay = 8.0
    engine._retry_exponential_base = 2.0
    engine._retry_jitter_range = 0.25
    engine._before_model_call_hook = None
    engine._on_model_usage_hook = None
    engine.handles_usage_accounting = False
    engine._redact_fn = None
    engine._redact_api_enabled = False
    engine._total_input_tokens = 0
    engine._total_output_tokens = 0
    engine._total_cache_creation_tokens = 0
    engine._total_cache_read_tokens = 0
    engine._total_calls = 0
    engine._last_call_time = None
    engine._auth_method = "api_key"
    engine._oauth_expires_at = 0.0
    engine._oauth_refresh_buffer = 300.0

    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        content=[],
        stop_reason="end_turn",
    )

    create = AsyncMock(side_effect=[OSError("network blip"), response])
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fast_sleep)

    result = await CognitiveEngine.think(
        engine,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result is response
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_think_connection_error_exhaustion_does_not_raise_attribute_error(monkeypatch):
    engine = object.__new__(CognitiveEngine)
    engine._model = "test-model"
    engine._max_tokens = 256
    engine._request_timeout_seconds = 120.0
    engine._retry_max_retries = 3
    engine._retry_base_delay = 0.5
    engine._retry_max_delay = 8.0
    engine._retry_exponential_base = 2.0
    engine._retry_jitter_range = 0.25
    engine._before_model_call_hook = None
    engine._on_model_usage_hook = None
    engine.handles_usage_accounting = False
    engine._redact_fn = None
    engine._redact_api_enabled = False
    engine._total_input_tokens = 0
    engine._total_output_tokens = 0
    engine._total_cache_creation_tokens = 0
    engine._total_cache_read_tokens = 0
    engine._total_calls = 0
    engine._last_call_time = None
    engine._auth_method = "api_key"
    engine._oauth_expires_at = 0.0
    engine._oauth_refresh_buffer = 300.0

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    conn_err = anthropic.APIConnectionError(message="Connection error.", request=request)
    create = AsyncMock(side_effect=conn_err)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", _fast_sleep)

    with pytest.raises(anthropic.APIConnectionError):
        await CognitiveEngine.think(
            engine,
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )

    # max_retries=3 in CognitiveEngine -> total attempts = 4
    assert create.await_count == 4


@pytest.mark.asyncio
async def test_think_adaptive_mode_omits_budget_tokens(monkeypatch):
    engine = object.__new__(CognitiveEngine)
    engine._model = "test-model"
    engine._max_tokens = 256
    engine._request_timeout_seconds = 120.0
    engine._retry_max_retries = 3
    engine._retry_base_delay = 0.5
    engine._retry_max_delay = 8.0
    engine._retry_exponential_base = 2.0
    engine._retry_jitter_range = 0.25
    engine._before_model_call_hook = None
    engine._on_model_usage_hook = None
    engine.handles_usage_accounting = False
    engine._redact_fn = None
    engine._redact_api_enabled = False
    engine._total_input_tokens = 0
    engine._total_output_tokens = 0
    engine._total_cache_creation_tokens = 0
    engine._total_cache_read_tokens = 0
    engine._total_calls = 0
    engine._last_call_time = None
    engine._auth_method = "api_key"
    engine._oauth_expires_at = 0.0
    engine._oauth_refresh_buffer = 300.0
    engine._thinking_budget = 0  # 0 means use adaptive mode

    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        content=[],
        stop_reason="end_turn",
    )
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await CognitiveEngine.think(
        engine,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
    )

    assert result is response
    assert create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in kwargs["thinking"]


def test_oauth_client_uses_sdk_base_url_with_oauth_beta_header(monkeypatch):
    monkeypatch.setattr(
        "gwenn.api.claude.socket.getaddrinfo",
        lambda *args, **kwargs: [(0, 0, 0, "", ("127.0.0.1", 443))],
    )

    cfg = SimpleNamespace(
        auth_token="sk-ant-oat01-test-token",
        api_key=None,
        model="test-model",
        max_tokens=128,
        request_timeout_seconds=120.0,
        retry_max_retries=3,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_exponential_base=2.0,
        retry_jitter_range=0.25,
        thinking_budget=0,
    )

    engine = CognitiveEngine(cfg)

    assert str(engine._async_client.base_url).startswith("https://api.anthropic.com")
    assert (
        engine._async_client.default_headers.get("anthropic-beta")
        == CLAUDE_CODE_OAUTH_BETA_HEADER
    )


def test_api_key_preferred_when_both_credentials_present(monkeypatch):
    monkeypatch.setattr(
        "gwenn.api.claude.socket.getaddrinfo",
        lambda *args, **kwargs: [(0, 0, 0, "", ("127.0.0.1", 443))],
    )

    cfg = SimpleNamespace(
        auth_token="sk-ant-oat01-test-token",
        api_key="sk-ant-api03-test-key",
        model="test-model",
        max_tokens=128,
        request_timeout_seconds=120.0,
        retry_max_retries=3,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_exponential_base=2.0,
        retry_jitter_range=0.25,
        thinking_budget=0,
    )

    engine = CognitiveEngine(cfg)

    assert engine._auth_method == "api_key"
    assert engine._async_client.api_key == "sk-ant-api03-test-key"
    assert engine._async_client.auth_token is None
    assert str(engine._async_client.base_url).startswith("https://api.anthropic.com")


def test_oauth_client_dns_failure_does_not_abort_startup(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("Name or service not known")

    monkeypatch.setattr("gwenn.api.claude.socket.getaddrinfo", _raise)

    cfg = SimpleNamespace(
        auth_token="sk-ant-oat01-test-token",
        api_key=None,
        model="test-model",
        max_tokens=128,
        request_timeout_seconds=120.0,
        retry_max_retries=3,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_exponential_base=2.0,
        retry_jitter_range=0.25,
        thinking_budget=0,
    )

    engine = CognitiveEngine(cfg)
    assert engine._auth_method == "oauth"


# ---------------------------------------------------------------------------
# Helper: build a bare CognitiveEngine without calling __init__
# ---------------------------------------------------------------------------

def _bare_engine(**overrides):
    """Create a CognitiveEngine with bypass __init__ and sensible defaults."""
    engine = object.__new__(CognitiveEngine)
    engine._model = "test-model"
    engine._max_tokens = 256
    engine._request_timeout_seconds = 120.0
    engine._retry_max_retries = 3
    engine._retry_base_delay = 0.5
    engine._retry_max_delay = 8.0
    engine._retry_exponential_base = 2.0
    engine._retry_jitter_range = 0.25
    engine._before_model_call_hook = None
    engine._on_model_usage_hook = None
    engine.handles_usage_accounting = False
    engine._redact_fn = None
    engine._redact_api_enabled = False
    engine._total_input_tokens = 0
    engine._total_output_tokens = 0
    engine._total_cache_creation_tokens = 0
    engine._total_cache_read_tokens = 0
    engine._total_calls = 0
    engine._last_call_time = None
    engine._auth_method = "oauth"
    engine._oauth_expires_at = 0.0
    engine._oauth_refresh_buffer = 300.0
    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


def _ok_response():
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        content=[],
        stop_reason="end_turn",
    )


# ---------------------------------------------------------------------------
# OAuth refresh tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oauth_proactive_refresh_before_expiry(monkeypatch):
    """When the token is near expiry, _maybe_refresh_oauth rebuilds the client before the API call."""
    engine = _bare_engine()
    # Token expires in 60 s — inside the 300 s buffer
    engine._oauth_expires_at = 1000.0 + 60.0

    fresh_token = "sk-ant-oat01-fresh"
    fresh_expiry = 1000.0 + 7200.0

    monkeypatch.setattr("gwenn.api.claude.time.time", lambda: 1000.0)
    monkeypatch.setattr(
        "gwenn.api.claude._load_oauth_credentials",
        lambda: (fresh_token, fresh_expiry),
    )

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    # Track _build_oauth_client calls
    build_calls = []
    original_build = CognitiveEngine._build_oauth_client

    def _tracking_build(self, token):
        build_calls.append(token)
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(CognitiveEngine, "_build_oauth_client", _tracking_build)

    result = await CognitiveEngine.think(
        engine,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result is response
    assert len(build_calls) == 1
    assert build_calls[0] == fresh_token
    assert engine._oauth_expires_at == fresh_expiry


@pytest.mark.asyncio
async def test_oauth_reactive_refresh_on_401(monkeypatch):
    """On a 401 AuthenticationError, the engine refreshes the token and retries."""
    engine = _bare_engine()
    # Token not near expiry (proactive check won't trigger)
    engine._oauth_expires_at = 99999999.0

    monkeypatch.setattr("gwenn.api.claude.time.time", lambda: 1000.0)

    fresh_token = "sk-ant-oat01-refreshed"
    fresh_expiry = 1000.0 + 7200.0
    monkeypatch.setattr(
        "gwenn.api.claude._load_oauth_credentials",
        lambda: (fresh_token, fresh_expiry),
    )

    response = _ok_response()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp_401 = httpx.Response(status_code=401, request=request)
    auth_error = anthropic.AuthenticationError(
        message="Unauthorized", response=resp_401, body=None,
    )

    # First call raises 401, second (after refresh) succeeds
    create = AsyncMock(side_effect=[auth_error, response])

    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    build_calls = []

    def _tracking_build(self, token):
        build_calls.append(token)
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(CognitiveEngine, "_build_oauth_client", _tracking_build)
    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", AsyncMock())

    result = await CognitiveEngine.think(
        engine,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result is response
    assert len(build_calls) == 1
    assert build_calls[0] == fresh_token


@pytest.mark.asyncio
async def test_oauth_refresh_fails_raises_original(monkeypatch):
    """When reactive refresh can't get a token, the original 401 error propagates."""
    engine = _bare_engine()
    engine._oauth_expires_at = 99999999.0

    monkeypatch.setattr("gwenn.api.claude.time.time", lambda: 1000.0)
    monkeypatch.setattr(
        "gwenn.api.claude._load_oauth_credentials",
        lambda: (None, 0.0),
    )

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp_401 = httpx.Response(status_code=401, request=request)
    auth_error = anthropic.AuthenticationError(
        message="Unauthorized", response=resp_401, body=None,
    )
    create = AsyncMock(side_effect=auth_error)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", AsyncMock())

    with pytest.raises(anthropic.AuthenticationError):
        await CognitiveEngine.think(
            engine,
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_api_key_auth_skips_refresh(monkeypatch):
    """With API key auth, _maybe_refresh_oauth is a no-op."""
    engine = _bare_engine(_auth_method="api_key")

    load_calls = []
    original_load = None

    def _tracking_load():
        load_calls.append(1)
        return ("token", 9999.0)

    monkeypatch.setattr("gwenn.api.claude._load_oauth_credentials", _tracking_load)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await CognitiveEngine.think(
        engine,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result is response
    assert len(load_calls) == 0
