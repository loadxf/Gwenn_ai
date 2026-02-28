from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from gwenn.api.claude import (
    CLAUDE_CODE_OAUTH_BETA_HEADER,
    CognitiveEngine,
    CognitiveEngineInitError,
)


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
async def test_think_uses_adaptive_by_default(monkeypatch):
    """Thinking defaults to adaptive mode."""
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
    engine._thinking_budget = 0
    engine._thinking_mode = "adaptive"
    engine._thinking_effort = "high"

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
    assert kwargs["output_config"] == {"effort": "high"}


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
        thinking_effort="high",
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
        thinking_effort="high",
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
        thinking_effort="high",
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
    engine._thinking_mode = "adaptive"
    engine._thinking_effort = "high"
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


# ---------------------------------------------------------------------------
# __init__ tests — lines 106-109 (exception wrapping)
# ---------------------------------------------------------------------------

def test_init_wraps_unexpected_exception_in_init_error(monkeypatch):
    """Non-CognitiveEngineInitError exceptions are wrapped in CognitiveEngineInitError."""
    monkeypatch.setattr(
        "gwenn.api.claude.socket.getaddrinfo",
        lambda *a, **kw: [(0, 0, 0, "", ("127.0.0.1", 443))],
    )

    cfg = SimpleNamespace(
        api_key="sk-test",
        auth_token=None,
        model="test-model",
        max_tokens=256,
        request_timeout_seconds=120.0,
        retry_max_retries=3,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_exponential_base=2.0,
        retry_jitter_range=0.25,
        thinking_budget="not-an-int",  # int("not-an-int") raises ValueError
        thinking_effort="high",
    )

    with pytest.raises(CognitiveEngineInitError, match="Failed to initialize"):
        CognitiveEngine(cfg)


def test_init_reraises_cognitive_engine_init_error(monkeypatch):
    """CognitiveEngineInitError is re-raised directly, not double-wrapped."""
    monkeypatch.setattr(
        "gwenn.api.claude.socket.getaddrinfo",
        lambda *a, **kw: [(0, 0, 0, "", ("127.0.0.1", 443))],
    )

    # Force _verify_base_url_dns to raise CognitiveEngineInitError
    def _raise_init_error(self):
        raise CognitiveEngineInitError("custom init error")

    monkeypatch.setattr(CognitiveEngine, "_verify_base_url_dns", _raise_init_error)

    cfg = SimpleNamespace(
        api_key="sk-test",
        auth_token=None,
        model="test-model",
        max_tokens=256,
        request_timeout_seconds=120.0,
        retry_max_retries=3,
        retry_base_delay=0.5,
        retry_max_delay=8.0,
        retry_exponential_base=2.0,
        retry_jitter_range=0.25,
        thinking_budget=0,
        thinking_effort="high",
    )

    with pytest.raises(CognitiveEngineInitError, match="custom init error"):
        CognitiveEngine(cfg)


# ---------------------------------------------------------------------------
# _oauth_default_headers — line 122 (return None for non-OAuth token)
# ---------------------------------------------------------------------------

def test_oauth_default_headers_returns_none_for_non_oauth_token():
    """Non-OAuth prefix tokens should not get the beta header."""
    engine = _bare_engine()
    assert engine._oauth_default_headers("sk-ant-api03-regular-key") is None
    assert engine._oauth_default_headers(None) is None
    assert engine._oauth_default_headers("") is None


def test_oauth_default_headers_returns_header_for_oauth_token():
    """OAuth prefix tokens should get the beta header."""
    engine = _bare_engine()
    result = engine._oauth_default_headers("sk-ant-oat01-test")
    assert result == {"anthropic-beta": CLAUDE_CODE_OAUTH_BETA_HEADER}


# ---------------------------------------------------------------------------
# _maybe_refresh_oauth — line 157 (stale token, same expiry)
# ---------------------------------------------------------------------------

def test_maybe_refresh_oauth_skips_when_expiry_not_newer(monkeypatch):
    """If the loaded credentials have the same or older expiry, skip rebuild."""
    engine = _bare_engine()
    engine._oauth_expires_at = 5000.0
    # Token is near expiry (within buffer)
    monkeypatch.setattr("gwenn.api.claude.time.time", lambda: 4800.0)
    # Return a token with the same expiry — not newer
    monkeypatch.setattr(
        "gwenn.api.claude._load_oauth_credentials",
        lambda: ("sk-ant-oat01-same", 5000.0),
    )

    result = engine._maybe_refresh_oauth()
    assert result is False


def test_maybe_refresh_oauth_skips_when_no_token_on_disk(monkeypatch):
    """If _load_oauth_credentials returns None, refresh fails gracefully."""
    engine = _bare_engine()
    engine._oauth_expires_at = 5000.0
    monkeypatch.setattr("gwenn.api.claude.time.time", lambda: 4800.0)
    monkeypatch.setattr(
        "gwenn.api.claude._load_oauth_credentials",
        lambda: (None, 0.0),
    )

    result = engine._maybe_refresh_oauth()
    assert result is False


# ---------------------------------------------------------------------------
# _verify_base_url_dns — line 172 (host is None/empty)
# ---------------------------------------------------------------------------

def test_verify_base_url_dns_returns_early_when_no_host():
    """When base_url has no host, _verify_base_url_dns returns immediately."""
    engine = _bare_engine()
    engine._async_client = SimpleNamespace(
        base_url=SimpleNamespace(host=None),
    )
    # Should not raise — just returns
    engine._verify_base_url_dns()

    engine._async_client = SimpleNamespace(
        base_url=SimpleNamespace(host=""),
    )
    engine._verify_base_url_dns()


# ---------------------------------------------------------------------------
# _invoke_hook — lines 191-193
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invoke_hook_with_none():
    """None hook is a no-op."""
    engine = _bare_engine()
    await engine._invoke_hook(None)  # should not raise


@pytest.mark.asyncio
async def test_invoke_hook_with_sync_callable():
    """Sync callables are invoked directly."""
    engine = _bare_engine()
    calls = []

    def sync_hook(a, b):
        calls.append((a, b))

    await engine._invoke_hook(sync_hook, 1, 2)
    assert calls == [(1, 2)]


@pytest.mark.asyncio
async def test_invoke_hook_with_async_callable():
    """Async callables are awaited."""
    engine = _bare_engine()
    calls = []

    async def async_hook(a, b):
        calls.append((a, b))

    await engine._invoke_hook(async_hook, 10, 20)
    assert calls == [(10, 20)]


# ---------------------------------------------------------------------------
# set_safety_hooks — lines 207-209
# ---------------------------------------------------------------------------

def test_set_safety_hooks():
    """set_safety_hooks registers before_model_call and on_model_usage."""
    engine = _bare_engine()

    def before():
        pass

    def on_usage(inp, out):
        pass

    engine.set_safety_hooks(before_model_call=before, on_model_usage=on_usage)

    assert engine._before_model_call_hook is before
    assert engine._on_model_usage_hook is on_usage
    assert engine.handles_usage_accounting is True


def test_set_safety_hooks_none_disables_accounting():
    """When on_model_usage is None, handles_usage_accounting is False."""
    engine = _bare_engine()
    engine.handles_usage_accounting = True
    engine.set_safety_hooks(before_model_call=None, on_model_usage=None)

    assert engine._before_model_call_hook is None
    assert engine._on_model_usage_hook is None
    assert engine.handles_usage_accounting is False


# ---------------------------------------------------------------------------
# set_redaction_hook — lines 222-223
# ---------------------------------------------------------------------------

def test_set_redaction_hook():
    """set_redaction_hook stores the function and enabled flag."""
    engine = _bare_engine()

    def redact(s):
        return s.replace("secret", "***")

    engine.set_redaction_hook(redact, enabled=True)
    assert engine._redact_fn is redact
    assert engine._redact_api_enabled is True

    engine.set_redaction_hook(redact, enabled=False)
    assert engine._redact_api_enabled is False


# ---------------------------------------------------------------------------
# _redact_messages / _redact_payload_value — lines 227-247
# ---------------------------------------------------------------------------

def test_redact_messages_applies_redaction_to_content():
    """_redact_messages applies the redaction function to all message content."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.replace("secret", "[REDACTED]")

    messages = [
        {"role": "user", "content": "my secret data"},
        {"role": "assistant", "content": "noted your secret"},
    ]

    result = engine._redact_messages(messages)

    assert result[0]["content"] == "my [REDACTED] data"
    assert result[1]["content"] == "noted your [REDACTED]"
    # Original messages are not mutated
    assert messages[0]["content"] == "my secret data"


def test_redact_messages_handles_missing_content():
    """Messages without 'content' key are passed through."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.replace("x", "y")

    messages = [{"role": "user"}]
    result = engine._redact_messages(messages)
    assert result == [{"role": "user"}]


def test_redact_payload_value_string():
    """String values are redacted directly."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.upper()

    assert engine._redact_payload_value("hello") == "HELLO"


def test_redact_payload_value_list():
    """Lists are recursively redacted."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.upper()

    result = engine._redact_payload_value(["a", "b"])
    assert result == ["A", "B"]


def test_redact_payload_value_dict():
    """Dicts are recursively redacted."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.upper()

    result = engine._redact_payload_value({"type": "text", "text": "hello"})
    assert result == {"type": "TEXT", "text": "HELLO"}


def test_redact_payload_value_image_dict_skipped():
    """Image blocks are skipped to avoid corrupting base64 data."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.upper()

    image_block = {"type": "image", "source": {"data": "base64data"}}
    result = engine._redact_payload_value(image_block)
    # Should be returned as-is
    assert result is image_block


def test_redact_payload_value_non_string_non_collection():
    """Non-string, non-collection values pass through unchanged."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.upper()

    assert engine._redact_payload_value(42) == 42
    assert engine._redact_payload_value(None) is None
    assert engine._redact_payload_value(3.14) == 3.14


def test_redact_payload_value_nested_structure():
    """Complex nested structures are fully redacted."""
    engine = _bare_engine()
    engine._redact_fn = lambda s: s.replace("secret", "***")

    value = [
        {"type": "text", "text": "my secret"},
        {"type": "image", "source": {"data": "secret_base64"}},
        "raw secret string",
    ]
    result = engine._redact_payload_value(value)
    assert result[0] == {"type": "text", "text": "my ***"}
    # Image block should be untouched
    assert result[1]["source"]["data"] == "secret_base64"
    assert result[2] == "raw *** string"


# ---------------------------------------------------------------------------
# think() — lines 286-287 (redaction enabled), 299 (cache_system=False),
#   311-313 (tools+tool_choice), 318 (thinking with budget)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_with_redaction_enabled():
    """When redaction is enabled, system_prompt and messages are redacted before API call."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)
    engine._redact_fn = lambda s: s.replace("secret", "[REDACTED]")
    engine._redact_api_enabled = True

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.think(
        system_prompt="You know a secret",
        messages=[{"role": "user", "content": "tell me the secret"}],
    )

    assert result is response
    kwargs = create.await_args.kwargs
    # System prompt should be redacted (cache_system=True by default)
    assert kwargs["system"][0]["text"] == "You know a [REDACTED]"
    assert kwargs["messages"][0]["content"] == "tell me the [REDACTED]"


@pytest.mark.asyncio
async def test_think_cache_system_false():
    """When cache_system=False, system prompt is a plain string, not a list."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.think(
        system_prompt="plain system",
        messages=[{"role": "user", "content": "hi"}],
        cache_system=False,
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["system"] == "plain system"


@pytest.mark.asyncio
async def test_think_with_tools_and_tool_choice():
    """When tools are provided, they are included in kwargs; tool_choice too if given."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    tools = [{"name": "my_tool", "description": "A tool", "input_schema": {}}]
    tool_choice = {"type": "auto"}

    result = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] is tool_choice


@pytest.mark.asyncio
async def test_think_with_tools_no_tool_choice():
    """When tools are provided without tool_choice, tool_choice is omitted."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    tools = [{"name": "my_tool", "description": "A tool", "input_schema": {}}]

    result = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["tools"] is tools
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_think_with_thinking_uses_adaptive():
    """When enable_thinking=True, thinking defaults to adaptive type."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=16000)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_think_falls_back_to_enabled_on_adaptive_rejection():
    """When adaptive thinking is rejected (400), falls back to enabled with budget."""
    engine = _bare_engine(_auth_method="oauth", _thinking_budget=16000)

    response = _ok_response()
    call_count = 0
    captured_kwargs: list[dict] = []

    async def _create(**kw):
        nonlocal call_count
        call_count += 1
        captured_kwargs.append(dict(kw))
        if call_count == 1 and kw.get("thinking", {}).get("type") == "adaptive":
            fake_response = httpx.Response(
                status_code=400,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            )
            raise anthropic.BadRequestError(
                message="adaptive thinking is not supported on this model",
                response=fake_response,
                body={"type": "error", "error": {"type": "invalid_request_error",
                      "message": "adaptive thinking is not supported on this model"}},
            )
        return response

    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=_create))

    result = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
    )

    assert result is response
    assert engine._thinking_mode == "enabled"
    # First call tried adaptive with output_config
    assert captured_kwargs[0]["thinking"] == {"type": "adaptive"}
    assert captured_kwargs[0]["output_config"] == {"effort": "high"}
    # Retry after fallback uses enabled without output_config
    assert captured_kwargs[1]["thinking"]["type"] == "enabled"
    assert "output_config" not in captured_kwargs[1]

    # Second call should use enabled directly without hitting the error
    result2 = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
    )
    assert result2 is response
    # Third call also uses enabled, no output_config
    assert captured_kwargs[2]["thinking"]["type"] == "enabled"
    assert "output_config" not in captured_kwargs[2]


@pytest.mark.asyncio
async def test_think_adaptive_passes_custom_effort_level():
    """output_config.effort reflects the configured thinking_effort."""
    engine = _bare_engine(_thinking_effort="medium")

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
    )

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_think_with_max_tokens_override():
    """max_tokens parameter overrides the engine default."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1024,
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_think_invokes_safety_hooks():
    """before_model_call and on_model_usage hooks are called during think()."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    before_calls = []
    usage_calls = []

    def before_hook():
        before_calls.append(1)

    def usage_hook(inp, out):
        usage_calls.append((inp, out))

    engine._before_model_call_hook = before_hook
    engine._on_model_usage_hook = usage_hook

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(before_calls) == 1
    assert usage_calls == [(10, 5)]


# ---------------------------------------------------------------------------
# think() error paths — lines 374-383 (RateLimitError, APIError)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_rate_limit_error_propagates(monkeypatch):
    """RateLimitError is logged and re-raised."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp_429 = httpx.Response(status_code=429, request=request)
    rate_err = anthropic.RateLimitError(
        message="Rate limited", response=resp_429, body=None,
    )

    create = AsyncMock(side_effect=rate_err)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", AsyncMock())

    with pytest.raises(anthropic.RateLimitError):
        await engine.think(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_think_generic_api_error_propagates(monkeypatch):
    """Generic APIError (e.g., 400 Bad Request) is logged and re-raised."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp_400 = httpx.Response(status_code=400, request=request)
    api_err = anthropic.BadRequestError(
        message="Bad request", response=resp_400, body=None,
    )

    create = AsyncMock(side_effect=api_err)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr("gwenn.harness.retry.asyncio.sleep", AsyncMock())

    with pytest.raises(anthropic.APIError):
        await engine.think(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )


# ---------------------------------------------------------------------------
# reflect() — line 430
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflect_delegates_to_think():
    """reflect() calls think() with enable_thinking=True and no tools."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.reflect(
        system_prompt="reflect sys",
        messages=[{"role": "user", "content": "reflect on this"}],
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    # reflect enables thinking
    assert "thinking" in kwargs
    # reflect does not pass tools
    assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# appraise() — line 449
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_appraise_delegates_to_think():
    """appraise() calls think() with max_tokens=512 and cache_system=False."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = await engine.appraise(
        system_prompt="appraise sys",
        content="evaluate this",
    )

    assert result is response
    kwargs = create.await_args.kwargs
    assert kwargs["max_tokens"] == 512
    # appraise uses cache_system=False, so system is a plain string
    assert kwargs["system"] == "appraise sys"
    # Messages should have the user content
    assert kwargs["messages"] == [{"role": "user", "content": "evaluate this"}]
    # No tools
    assert "tools" not in kwargs
    # No thinking
    assert "thinking" not in kwargs


# ---------------------------------------------------------------------------
# compact() — line 472
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_delegates_to_think():
    """compact() appends compaction_prompt and enables thinking."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = _ok_response()
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = await engine.compact(
        system_prompt="compact sys",
        messages=messages,
        compaction_prompt="summarize everything",
    )

    assert result is response
    kwargs = create.await_args.kwargs
    # Compaction prompt should be appended as a user message
    assert kwargs["messages"][-1] == {"role": "user", "content": "summarize everything"}
    assert len(kwargs["messages"]) == 3
    assert kwargs["max_tokens"] == 4096
    # Thinking enabled
    assert "thinking" in kwargs
    # No tools
    assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# extract_text — lines 486-490
# ---------------------------------------------------------------------------

def test_extract_text_from_response():
    """extract_text joins text blocks from response content."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hello"),
            SimpleNamespace(type="tool_use", id="t1", name="tool", input={}),
            SimpleNamespace(type="text", text="World"),
        ],
    )

    result = engine.extract_text(response)
    assert result == "Hello\nWorld"


def test_extract_text_empty_content():
    """extract_text with no text blocks returns empty string."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="t1", name="tool", input={}),
        ],
    )

    result = engine.extract_text(response)
    assert result == ""


def test_extract_text_no_content():
    """extract_text with empty content list returns empty string."""
    engine = _bare_engine()
    response = SimpleNamespace(content=[])
    assert engine.extract_text(response) == ""


# ---------------------------------------------------------------------------
# extract_tool_calls — lines 494-502
# ---------------------------------------------------------------------------

def test_extract_tool_calls():
    """extract_tool_calls returns dicts for tool_use blocks."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking"),
            SimpleNamespace(type="tool_use", id="call_1", name="search", input={"q": "test"}),
            SimpleNamespace(type="tool_use", id="call_2", name="read", input={"path": "/tmp"}),
        ],
    )

    result = engine.extract_tool_calls(response)
    assert len(result) == 2
    assert result[0] == {"id": "call_1", "name": "search", "input": {"q": "test"}}
    assert result[1] == {"id": "call_2", "name": "read", "input": {"path": "/tmp"}}


def test_extract_tool_calls_no_tools():
    """extract_tool_calls returns empty list when no tool_use blocks."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="no tools here")],
    )

    result = engine.extract_tool_calls(response)
    assert result == []


# ---------------------------------------------------------------------------
# extract_thinking — lines 506-509
# ---------------------------------------------------------------------------

def test_extract_thinking_found():
    """extract_thinking returns thinking content when present."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="I am reflecting deeply"),
            SimpleNamespace(type="text", text="Here is my answer"),
        ],
    )

    result = engine.extract_thinking(response)
    assert result == "I am reflecting deeply"


def test_extract_thinking_not_found():
    """extract_thinking returns None when no thinking block exists."""
    engine = _bare_engine()

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="no thinking here")],
    )

    result = engine.extract_thinking(response)
    assert result is None


def test_extract_thinking_empty_content():
    """extract_thinking returns None for empty content list."""
    engine = _bare_engine()
    response = SimpleNamespace(content=[])
    assert engine.extract_thinking(response) is None


# ---------------------------------------------------------------------------
# telemetry property — line 514
# ---------------------------------------------------------------------------

def test_telemetry_initial_state():
    """Telemetry starts at zero for a fresh engine."""
    engine = _bare_engine()
    t = engine.telemetry

    assert t["total_calls"] == 0
    assert t["total_input_tokens"] == 0
    assert t["total_output_tokens"] == 0
    assert t["total_tokens"] == 0
    assert t["cache_creation_tokens"] == 0
    assert t["cache_read_tokens"] == 0
    assert t["last_call_seconds"] == 0.0


def test_telemetry_after_calls():
    """Telemetry reflects accumulated usage."""
    engine = _bare_engine()
    engine._total_calls = 3
    engine._total_input_tokens = 100
    engine._total_output_tokens = 50
    engine._total_cache_creation_tokens = 20
    engine._total_cache_read_tokens = 10
    engine._last_call_time = 1.5

    t = engine.telemetry

    assert t["total_calls"] == 3
    assert t["total_input_tokens"] == 100
    assert t["total_output_tokens"] == 50
    assert t["total_tokens"] == 150
    assert t["cache_creation_tokens"] == 20
    assert t["cache_read_tokens"] == 10
    assert t["last_call_seconds"] == 1.5


# ---------------------------------------------------------------------------
# _build_oauth_client — coverage for both branches (with and without headers)
# ---------------------------------------------------------------------------

def test_build_oauth_client_with_oauth_token():
    """_build_oauth_client creates a client with OAuth headers for sk-ant-oat tokens."""
    engine = _bare_engine()
    client = engine._build_oauth_client("sk-ant-oat01-test-token")
    assert isinstance(client, anthropic.AsyncAnthropic)
    assert (
        client.default_headers.get("anthropic-beta")
        == CLAUDE_CODE_OAUTH_BETA_HEADER
    )


def test_build_oauth_client_without_oauth_prefix():
    """_build_oauth_client creates a client without extra headers for non-OAuth tokens."""
    engine = _bare_engine()
    client = engine._build_oauth_client("some-other-token")
    assert isinstance(client, anthropic.AsyncAnthropic)
    # Should NOT have the OAuth beta header injected
    assert "anthropic-beta" not in (client.default_headers or {})


# ---------------------------------------------------------------------------
# think() — telemetry accumulation across calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_accumulates_telemetry():
    """After think(), telemetry counters are updated correctly."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
        ),
        content=[],
        stop_reason="end_turn",
    )
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert engine._total_calls == 1
    assert engine._total_input_tokens == 100
    assert engine._total_output_tokens == 50
    assert engine._total_cache_creation_tokens == 10
    assert engine._total_cache_read_tokens == 5
    assert engine._last_call_time is not None
    assert engine._last_call_time > 0


@pytest.mark.asyncio
async def test_think_telemetry_handles_missing_cache_attrs():
    """Telemetry gracefully handles responses without cache token attributes."""
    engine = _bare_engine(_auth_method="api_key", _thinking_budget=0)

    # Response without cache_creation_input_tokens or cache_read_input_tokens
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=20, output_tokens=10),
        content=[],
        stop_reason="end_turn",
    )
    create = AsyncMock(return_value=response)
    engine._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))

    await engine.think(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert engine._total_cache_creation_tokens == 0
    assert engine._total_cache_read_tokens == 0
