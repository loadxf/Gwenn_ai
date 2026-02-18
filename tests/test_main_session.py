from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gwenn.main import GwennSession


@pytest.mark.asyncio
async def test_startup_onboarding_runs_for_non_cli_when_interactive(monkeypatch):
    session = GwennSession(channel_override="telegram")
    identity = MagicMock()
    identity.should_run_startup_onboarding.return_value = True
    agent = SimpleNamespace(
        identity=identity,
        apply_startup_onboarding=MagicMock(),
    )
    session._agent = agent

    answers = iter(
        [
            "Alice",
            "coding partner",
            "ship reliable features",
            "concise",
            "no destructive changes",
        ]
    )

    async def _fake_prompt(_: str) -> str:
        return next(answers)

    session._prompt_startup_input = _fake_prompt
    monkeypatch.setattr("gwenn.main.sys.stdin.isatty", lambda: True)

    await session._run_first_startup_onboarding_if_needed("telegram")

    agent.apply_startup_onboarding.assert_called_once()
    call = agent.apply_startup_onboarding.call_args
    assert call.kwargs["user_id"] == "default_user"
    assert call.args[0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_startup_onboarding_skipped_when_non_interactive(monkeypatch):
    session = GwennSession(channel_override="telegram")
    identity = MagicMock()
    identity.should_run_startup_onboarding.return_value = True
    agent = SimpleNamespace(
        identity=identity,
        apply_startup_onboarding=MagicMock(),
    )
    session._agent = agent
    monkeypatch.setattr("gwenn.main.sys.stdin.isatty", lambda: False)

    await session._run_first_startup_onboarding_if_needed("telegram")

    agent.apply_startup_onboarding.assert_not_called()


def test_sigint_requires_quick_double_press(monkeypatch):
    session = GwennSession()
    timeline = iter([10.0, 10.5])
    monkeypatch.setattr("gwenn.main.time.monotonic", lambda: next(timeline))

    session._handle_sigint()
    assert not session._shutdown_event.is_set()

    session._handle_sigint()
    assert session._shutdown_event.is_set()


def test_sigint_window_expires_and_requires_new_double_press(monkeypatch):
    session = GwennSession()
    timeline = iter([10.0, 12.0, 12.4])
    monkeypatch.setattr("gwenn.main.time.monotonic", lambda: next(timeline))

    session._handle_sigint()
    session._handle_sigint()
    assert not session._shutdown_event.is_set()

    session._handle_sigint()
    assert session._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_prompt_startup_input_supports_multiple_calls(monkeypatch):
    session = GwennSession()
    answers = iter([" Alice ", "builder "])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    first = await session._prompt_startup_input("Name: ")
    second = await session._prompt_startup_input("Role: ")

    assert first == "Alice"
    assert second == "builder"


@pytest.mark.asyncio
async def test_read_input_fallback_path_supports_multiple_calls(monkeypatch):
    session = GwennSession()

    def _raise_oserror() -> int:
        raise OSError("stdin fileno unavailable")

    monkeypatch.setattr("gwenn.main.sys.stdin.fileno", _raise_oserror)
    reads = iter(["hello", None])
    session._read_input_blocking = lambda: next(reads)

    first = await session._read_input()
    second = await session._read_input()

    assert first == "hello"
    assert second is None


@pytest.mark.asyncio
async def test_shutdown_continues_when_session_save_fails(monkeypatch, tmp_path):
    session = GwennSession()
    shutdown = AsyncMock()
    session._agent = SimpleNamespace(
        _conversation_history=[{"role": "user", "content": "hello"}],
        shutdown=shutdown,
    )
    session._config = SimpleNamespace(
        daemon=SimpleNamespace(
            sessions_dir=tmp_path / "sessions",
            session_max_count=20,
            session_max_messages=200,
        )
    )
    session._session_started_at = 0.0

    class _BrokenStore:
        def __init__(self, *_args, **_kwargs):
            return None

        def save_session(self, *_args, **_kwargs):
            raise OSError("disk full")

    monkeypatch.setattr("gwenn.memory.session_store.SessionStore", _BrokenStore)

    await session._shutdown()

    shutdown.assert_awaited_once()
