from __future__ import annotations

from dataclasses import dataclass
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


def test_request_shutdown_sets_shutdown_event_when_agent_present():
    session = GwennSession()
    session._agent = SimpleNamespace(safety=SimpleNamespace(emergency_stop=MagicMock()))

    session._request_shutdown()

    assert session._shutdown_event.is_set()


def test_parse_slash_command_requires_leading_slash():
    assert GwennSession._parse_slash_command("status") is None
    assert GwennSession._parse_slash_command("/status") == ("/status", "")
    assert GwennSession._parse_slash_command("/plan ship it") == ("/plan", "ship it")


def test_matching_slash_commands_for_root_prefix_lists_all():
    matches = GwennSession._matching_slash_commands("/")
    assert "/help" in matches
    assert "/status" in matches
    assert "/exit" in matches


def test_matching_slash_commands_filters_by_prefix():
    assert GwennSession._matching_slash_commands("/st") == ["/status", "/stats"]
    assert GwennSession._matching_slash_commands("status") == []


def test_slash_command_completer_returns_indexed_matches(monkeypatch):
    class _FakeReadline:
        def __init__(self):
            self.delims = " \t\n/-"
            self.completer = None

        def parse_and_bind(self, _spec: str) -> None:
            return None

        def get_completer_delims(self) -> str:
            return self.delims

        def set_completer_delims(self, value: str) -> None:
            self.delims = value

        def set_completer(self, fn):
            self.completer = fn

        @staticmethod
        def get_line_buffer() -> str:
            return "/st"

        @staticmethod
        def get_begidx() -> int:
            return 0

    fake_readline = _FakeReadline()
    monkeypatch.setattr("gwenn.main.readline", fake_readline)
    session = GwennSession()

    assert session._slash_command_completer("/st", 0) == "/status"
    assert session._slash_command_completer("/st", 1) == "/stats"
    assert session._slash_command_completer("/st", 2) is None

    fake_readline.get_line_buffer = lambda: "/plan task /st"
    fake_readline.get_begidx = lambda: 11
    assert session._slash_command_completer("/st", 0) is None


def test_apply_output_style_to_message_variants():
    session = GwennSession()

    session._output_style = "balanced"
    assert session._apply_output_style_to_message("hello") == "hello"

    session._output_style = "brief"
    assert "respond briefly" in session._apply_output_style_to_message("hello")

    session._output_style = "detailed"
    assert "detailed, structured depth" in session._apply_output_style_to_message("hello")


@pytest.mark.asyncio
async def test_prompt_startup_input_supports_multiple_calls(monkeypatch):
    session = GwennSession()
    answers = iter([" Alice ", "builder "])
    # Prompt rendering is handled separately; input() is called without args.
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

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
async def test_read_input_renders_you_prompt_before_read():
    session = GwennSession()
    session._render_prompt = MagicMock()
    session._read_input_blocking = lambda: "hello"

    value = await session._read_input()

    assert value == "hello"
    session._render_prompt.assert_called_once_with("[bold green]You[/bold green]: ")


@pytest.mark.asyncio
async def test_read_raw_input_renders_custom_prompt(monkeypatch):
    session = GwennSession()
    session._render_prompt = MagicMock()
    monkeypatch.setattr("builtins.input", lambda *_: "2")

    value = await session._read_raw_input("Resume session (number): ")

    assert value == "2"
    session._render_prompt.assert_called_once_with(
        "[dim]Resume session (number): [/dim]"
    )


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


@pytest.mark.asyncio
async def test_shutdown_restores_terminal_state():
    session = GwennSession()
    session._restore_terminal_state = MagicMock()

    await session._shutdown()

    session._restore_terminal_state.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_uses_spinner_status_for_agent_shutdown(monkeypatch):
    session = GwennSession()
    shutdown = AsyncMock()
    session._agent = SimpleNamespace(
        _conversation_history=[],
        shutdown=shutdown,
    )
    session._restore_terminal_state = MagicMock()

    class _StatusCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    status_mock = MagicMock(return_value=_StatusCtx())
    monkeypatch.setattr("gwenn.main.console.status", status_mock)
    monkeypatch.setattr("gwenn.main.console.print", MagicMock())
    monkeypatch.setattr("gwenn.main.sys.stdout.isatty", lambda: True)

    await session._shutdown()

    shutdown.assert_awaited_once()
    status_mock.assert_called_once_with(
        "[cyan]Please wait: Updating Gwenn's memory. This may take a few seconds...[/cyan]",
        spinner="dots",
    )


def test_restore_terminal_state_applies_saved_attrs(monkeypatch):
    session = GwennSession()
    fake_termios = SimpleNamespace(
        TCSADRAIN=1,
        tcsetattr=MagicMock(),
    )
    monkeypatch.setattr("gwenn.main._termios", fake_termios)
    monkeypatch.setattr("gwenn.main.sys.stdin.fileno", lambda: 7)
    session._stdin_term_attrs = ["saved"]

    session._restore_terminal_state()

    fake_termios.tcsetattr.assert_called_once_with(7, 1, ["saved"])


@pytest.mark.asyncio
async def test_run_channels_rolls_back_started_channels_on_partial_start(monkeypatch):
    import gwenn.channels.discord_channel as dc_mod
    import gwenn.channels.telegram_channel as tg_mod
    import gwenn.config as config_mod

    @dataclass
    class _Cfg:
        max_history_length: int = 5
        session_ttl_seconds: float = 60.0

    events: list[str] = []

    class _Telegram:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            events.append("telegram:start")

        async def stop(self):
            events.append("telegram:stop")

    class _Discord:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            events.append("discord:start")
            raise RuntimeError("discord failed")

        async def stop(self):
            events.append("discord:stop")

    monkeypatch.setattr(config_mod, "TelegramConfig", lambda: _Cfg(max_history_length=10))
    monkeypatch.setattr(
        config_mod,
        "DiscordConfig",
        lambda: _Cfg(max_history_length=20, session_ttl_seconds=300.0),
    )
    monkeypatch.setattr(tg_mod, "TelegramChannel", _Telegram)
    monkeypatch.setattr(dc_mod, "DiscordChannel", _Discord)

    session = GwennSession(channel_override="all")
    with pytest.raises(RuntimeError, match="discord failed"):
        await session._run_channels(agent=MagicMock(), config=MagicMock(), mode="all")

    assert events == ["telegram:start", "discord:start", "telegram:stop"]


@pytest.mark.asyncio
async def test_run_channels_prints_import_error_instead_of_raising(monkeypatch):
    from gwenn.channels.session import SessionManager

    session = GwennSession(channel_override="telegram")
    print_mock = MagicMock()
    monkeypatch.setattr("gwenn.main.console.print", print_mock)
    monkeypatch.setattr(
        "gwenn.channels.startup.build_channels",
        lambda _agent, channel_list: (SessionManager(), [MagicMock()]),
    )

    async def _raise_import_error(*_args, **_kwargs):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(
        "gwenn.channels.startup.run_channels_until_shutdown",
        _raise_import_error,
    )

    await session._run_channels(agent=MagicMock(), config=MagicMock(), mode="telegram")

    assert any(
        "missing optional dependency" in str(call.args[0])
        for call in print_mock.call_args_list
    )


@pytest.mark.asyncio
async def test_run_channels_prints_friendly_invalid_token_message(monkeypatch):
    from gwenn.channels.session import SessionManager

    class InvalidToken(Exception):
        pass

    InvalidToken.__module__ = "telegram.error"

    session = GwennSession(channel_override="telegram")
    print_mock = MagicMock()
    monkeypatch.setattr("gwenn.main.console.print", print_mock)
    monkeypatch.setattr(
        "gwenn.channels.startup.build_channels",
        lambda _agent, channel_list: (SessionManager(), [MagicMock()]),
    )

    async def _raise_invalid_token(*_args, **_kwargs):
        raise InvalidToken(
            "The token `123456789:ABCDEFGHIJKLMNOPQRSTUV123456789` "
            "was rejected by the server."
        )

    monkeypatch.setattr(
        "gwenn.channels.startup.run_channels_until_shutdown",
        _raise_invalid_token,
    )

    await session._run_channels(agent=MagicMock(), config=MagicMock(), mode="telegram")

    printed = " ".join(str(call.args[0]) for call in print_mock.call_args_list)
    assert "Telegram bot token was rejected" in printed
    assert "123456789:" not in printed


@pytest.mark.asyncio
async def test_run_channels_stops_all_on_shutdown(monkeypatch):
    import gwenn.channels.discord_channel as dc_mod
    import gwenn.channels.telegram_channel as tg_mod
    import gwenn.config as config_mod

    @dataclass
    class _Cfg:
        max_history_length: int = 5
        session_ttl_seconds: float = 60.0

    events: list[str] = []

    class _Telegram:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            events.append("telegram:start")

        async def stop(self):
            events.append("telegram:stop")

    class _Discord:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            events.append("discord:start")

        async def stop(self):
            events.append("discord:stop")

    monkeypatch.setattr(config_mod, "TelegramConfig", lambda: _Cfg(max_history_length=10))
    monkeypatch.setattr(
        config_mod,
        "DiscordConfig",
        lambda: _Cfg(max_history_length=20, session_ttl_seconds=300.0),
    )
    monkeypatch.setattr(tg_mod, "TelegramChannel", _Telegram)
    monkeypatch.setattr(dc_mod, "DiscordChannel", _Discord)

    session = GwennSession(channel_override="all")
    session._shutdown_event.set()
    await session._run_channels(agent=MagicMock(), config=MagicMock(), mode="all")

    assert events == [
        "telegram:start",
        "discord:start",
        "discord:stop",
        "telegram:stop",
    ]
