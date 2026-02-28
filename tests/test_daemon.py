"""
Tests for daemon-related components:
- DaemonConfig field parsing from env vars
- SessionStore save / list / load / prune
- CliChannel with a mock Unix socket server
- Daemon message dispatch: ping→pong, chat→mock, list_sessions, load_session, stop
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from gwenn.config import ChannelConfig, DaemonConfig
from gwenn.memory.session_store import SessionStore, _format_session_time


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sessions"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def store(sessions_dir: Path) -> SessionStore:
    return SessionStore(sessions_dir, max_count=5, max_messages=10)


# =============================================================================
# DaemonConfig
# =============================================================================


class TestDaemonConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        """DaemonConfig should have sensible defaults (isolated from .env)."""
        cfg = DaemonConfig(_env_file=None)
        assert cfg.max_connections == 10
        assert cfg.session_max_count == 20
        assert cfg.session_max_messages == 200
        assert cfg.session_include_preview is False
        assert cfg.redact_session_content is True

    def test_get_channel_list_defaults(self) -> None:
        cfg = ChannelConfig(_env_file=None)
        assert cfg.get_channel_list() == ["cli"]

    def test_get_channel_list_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLI_ENABLED", "true")
        monkeypatch.setenv("TELEGRAM_ENABLED", "true")
        monkeypatch.setenv("DISCORD_ENABLED", "true")
        cfg = ChannelConfig()
        assert cfg.get_channel_list() == ["cli", "telegram", "discord"]

    def test_get_channel_list_telegram_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLI_ENABLED", "false")
        monkeypatch.setenv("TELEGRAM_ENABLED", "true")
        monkeypatch.setenv("DISCORD_ENABLED", "false")
        cfg = ChannelConfig()
        assert cfg.get_channel_list() == ["telegram"]

    def test_custom_socket_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        sock = str(tmp_path / "gwenn.sock")
        monkeypatch.setenv("GWENN_DAEMON_SOCKET", sock)
        cfg = DaemonConfig()
        assert cfg.socket_path == Path(sock)

    def test_session_limits_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_DAEMON_SESSION_MAX_COUNT", "42")
        monkeypatch.setenv("GWENN_DAEMON_SESSION_MAX_MESSAGES", "999")
        cfg = DaemonConfig()
        assert cfg.session_max_count == 42
        assert cfg.session_max_messages == 999

    def test_auth_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_DAEMON_AUTH_TOKEN", "secret-token")
        cfg = DaemonConfig()
        assert cfg.auth_token == "secret-token"


# =============================================================================
# SessionStore
# =============================================================================


class TestSessionStore:
    def test_init_creates_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "new_sessions"
        assert not d.exists()
        SessionStore(d)
        assert d.exists()

    def test_save_and_list(self, store: SessionStore) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        sid = store.save_session(msgs, time.time())
        assert sid != ""
        sessions = store.list_sessions(include_preview=True)
        assert len(sessions) == 1
        assert sessions[0]["message_count"] == 2
        assert sessions[0]["preview"] == "hello"

    def test_list_sessions_hides_preview_by_default(self, store: SessionStore) -> None:
        sid = store.save_session([{"role": "user", "content": "secret hello"}], time.time())
        assert sid
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert "preview" not in sessions[0]

    def test_save_returns_session_id_format(self, store: SessionStore) -> None:
        msgs = [{"role": "user", "content": "test"}]
        sid = store.save_session(msgs, time.time())
        # Format: YYYYMMDD-HHMMSS-xxxx → 3 segments when split by "-"
        parts = sid.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 4  # random suffix

    def test_save_empty_returns_empty(self, store: SessionStore) -> None:
        sid = store.save_session([], time.time())
        assert sid == ""

    def test_save_filters_invalid_messages(self, store: SessionStore) -> None:
        msgs = [
            {"role": "user", "content": "valid"},
            {"role": "system", "content": "invalid role"},
            {"role": "assistant"},  # missing content
            {"role": "assistant", "content": 123},  # wrong content type
            {"role": "assistant", "content": "also valid"},
        ]
        sid = store.save_session(msgs, time.time())
        loaded = store.load_session(sid)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "valid"
        assert loaded[1]["content"] == "also valid"

    def test_save_caps_at_max_messages(self, store: SessionStore) -> None:
        # store has max_messages=10
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        sid = store.save_session(msgs, time.time())
        loaded = store.load_session(sid)
        assert len(loaded) == 10
        # Should be the last 10
        assert loaded[0]["content"] == "msg 10"

    def test_load_session(self, store: SessionStore) -> None:
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        sid = store.save_session(msgs, time.time())
        loaded = store.load_session(sid)
        assert loaded == msgs

    def test_load_nonexistent_raises(self, store: SessionStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.load_session("20990101-000000-zzzz")

    def test_load_rejects_path_traversal(self, store: SessionStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.load_session("../../etc/passwd")

    def test_list_uses_filename_for_session_id(self, store: SessionStore) -> None:
        sid = "20260101-120000-abcd"
        payload = {
            "id": "../../etc/passwd",
            "started_at": time.time(),
            "ended_at": time.time(),
            "message_count": 1,
            "messages": [{"role": "user", "content": "hello"}],
        }
        (store.sessions_dir / f"{sid}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        sessions = store.list_sessions(limit=1, include_preview=True)
        assert sessions[0]["id"] == sid

    def test_list_sessions_newest_first(self, store: SessionStore) -> None:
        # Save with different started_at times to ensure ordering by file mtime
        msgs = [{"role": "user", "content": "x"}]
        ids = []
        for i in range(3):
            time.sleep(0.01)  # ensure different mtime
            sid = store.save_session(msgs, time.time() - (300 - i * 100))
            ids.append(sid)
        sessions = store.list_sessions()
        # Newest (last saved) should be first
        assert sessions[0]["id"] == ids[-1]

    def test_prune_keeps_max_count(self, store: SessionStore) -> None:
        # store has max_count=5
        msgs = [{"role": "user", "content": "x"}]
        for i in range(8):
            time.sleep(0.01)
            store.save_session(msgs, time.time())
        files = list(store.sessions_dir.glob("*.json"))
        assert len(files) == 5

    def test_list_limit(self, store: SessionStore) -> None:
        msgs = [{"role": "user", "content": "x"}]
        for i in range(4):
            time.sleep(0.01)
            store.save_session(msgs, time.time())
        assert len(store.list_sessions(limit=2)) == 2
        assert len(store.list_sessions(limit=10)) == 4

    def test_load_skips_corrupted_entries(self, store: SessionStore) -> None:
        """Manually write a session file with mixed valid/invalid entries."""
        sid = "20260101-120000-test"
        payload = {
            "id": sid,
            "started_at": time.time(),
            "ended_at": time.time(),
            "message_count": 3,
            "messages": [
                {"role": "user", "content": "good"},
                {"role": "bad_role", "content": "dropped"},
                {"role": "assistant", "content": "also good"},
            ],
        }
        (store.sessions_dir / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")
        loaded = store.load_session(sid)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "good"
        assert loaded[1]["content"] == "also good"


# =============================================================================
# _format_session_time
# =============================================================================


class TestFormatSessionTime:
    def test_today(self) -> None:
        result = _format_session_time(time.time() - 3600)
        assert result.startswith("Today")

    def test_yesterday(self) -> None:
        result = _format_session_time(time.time() - 86400 - 3600)
        assert result.startswith("Yesterday")

    def test_older(self) -> None:
        result = _format_session_time(time.time() - 3 * 86400)
        # Should be a weekday name
        assert any(
            result.startswith(day) for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        )


# =============================================================================
# CliChannel with mock Unix socket
# =============================================================================


@pytest_asyncio.fixture
async def mock_daemon_server(tmp_path: Path):
    """Start a minimal mock daemon server that responds to NDJSON requests."""
    sock_path = tmp_path / "test.sock"

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            msg = json.loads(raw.decode())
            req_id = msg.get("req_id", "")
            msg_type = msg.get("type", "")

            if msg_type == "ping":
                resp = {"type": "pong", "req_id": req_id}
            elif msg_type == "chat":
                resp = {
                    "type": "response",
                    "req_id": req_id,
                    "text": "mock response",
                    "emotion": "curious",
                }
            elif msg_type == "status":
                resp = {
                    "type": "status_response",
                    "req_id": req_id,
                    "status": {"name": "Gwenn", "emotion": "curious"},
                    "active_connections": 1,
                }
            elif msg_type == "heartbeat_status":
                resp = {
                    "type": "heartbeat_response",
                    "req_id": req_id,
                    "status": {"running": True, "beat_count": 42},
                }
            elif msg_type == "list_sessions":
                resp = {
                    "type": "sessions_list",
                    "req_id": req_id,
                    "sessions": [
                        {
                            "id": "20260101-120000-abcd",
                            "started_at": 1735732800.0,
                            "ended_at": 1735732900.0,
                            "message_count": 5,
                            "preview": "test session",
                        }
                    ],
                }
            elif msg_type == "load_session":
                resp = {"type": "session_loaded", "req_id": req_id, "message_count": 5}
            elif msg_type == "reset_session":
                resp = {"type": "session_reset", "req_id": req_id, "cleared_messages": 3}
            elif msg_type == "runtime_info":
                resp = {
                    "type": "runtime_info_response",
                    "req_id": req_id,
                    "status": {"name": "Gwenn", "total_interactions": 7},
                    "skills": [{"name": "weather", "category": "information"}],
                    "mcp": {"configured_servers": 1, "connected_servers": 1, "discovered_tools": 2},
                    "tools": {"registered": 10, "enabled": 9},
                    "configured_mcp_servers": [{"name": "docs"}],
                    "active_connections": 1,
                }
            elif msg_type == "stop":
                resp = {"type": "ack_stop", "req_id": req_id}
                line = json.dumps(resp) + "\n"
                writer.write(line.encode())
                await writer.drain()
                writer.close()
                return
            else:
                resp = {"type": "error", "req_id": req_id, "message": "unknown"}

            line = json.dumps(resp) + "\n"
            writer.write(line.encode())
            await writer.drain()

        writer.close()

    server = await asyncio.start_unix_server(handle, path=str(sock_path))
    yield sock_path
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
class TestCliChannel:
    async def test_connect_missing_socket_raises(self, tmp_path: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel, DaemonNotRunningError

        channel = CliChannel()
        with pytest.raises(DaemonNotRunningError):
            await channel.connect(tmp_path / "nosock.sock")

    async def test_ping(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.ping()
        assert resp["type"] == "pong"
        await channel.disconnect()

    async def test_chat(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.chat("hello")
        assert resp["type"] == "response"
        assert resp["text"] == "mock response"
        assert resp["emotion"] == "curious"
        await channel.disconnect()

    async def test_get_status(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.get_status()
        assert resp["type"] == "status_response"
        assert resp["status"]["name"] == "Gwenn"
        await channel.disconnect()

    async def test_get_heartbeat_status(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.get_heartbeat_status()
        assert resp["status"]["beat_count"] == 42
        await channel.disconnect()

    async def test_list_sessions(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        sessions = await channel.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "20260101-120000-abcd"
        await channel.disconnect()

    async def test_load_session(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        count = await channel.load_session("20260101-120000-abcd")
        assert count == 5
        await channel.disconnect()

    async def test_reset_session(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        cleared = await channel.reset_session()
        assert cleared == 3
        await channel.disconnect()

    async def test_get_runtime_info(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.get_runtime_info()
        assert resp["type"] == "runtime_info_response"
        assert resp["skills"][0]["name"] == "weather"
        await channel.disconnect()

    async def test_stop_daemon(self, mock_daemon_server: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        channel = CliChannel()
        await channel.connect(mock_daemon_server)
        resp = await channel.stop_daemon()
        assert resp["type"] == "ack_stop"
        # Don't disconnect — server closed the connection

    async def test_auth_token_is_sent_when_configured(self, tmp_path: Path) -> None:
        from gwenn.channels.cli_channel import CliChannel

        sock_path = tmp_path / "auth.sock"

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            raw = await reader.readline()
            msg = json.loads(raw.decode())
            req_id = msg.get("req_id", "")
            if msg.get("auth_token") != "secret":
                resp = {"type": "error", "req_id": req_id, "message": "unauthorized"}
            else:
                resp = {"type": "pong", "req_id": req_id}
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(handle, path=str(sock_path))
        try:
            channel = CliChannel(auth_token="secret")
            await channel.connect(sock_path)
            resp = await channel.ping()
            assert resp["type"] == "pong"
            await channel.disconnect()
        finally:
            server.close()
            await server.wait_closed()


# =============================================================================
# GwennDaemon dispatch (unit-level, no real agent)
# =============================================================================


@pytest.mark.asyncio
class TestGwennDaemonDispatch:
    """Test GwennDaemon._dispatch() in isolation with a mock agent."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path):
        cfg = MagicMock()
        cfg.daemon.socket_path = tmp_path / "gwenn.sock"
        cfg.daemon.pid_file = tmp_path / "gwenn.pid"
        cfg.daemon.sessions_dir = tmp_path / "sessions"
        cfg.daemon.max_connections = 10
        cfg.daemon.connection_timeout = 300.0
        cfg.daemon.session_max_count = 20
        cfg.daemon.session_max_messages = 200
        cfg.daemon.auth_token = None
        cfg.daemon.session_include_preview = False
        cfg.daemon.redact_session_content = True
        cfg.channel.get_channel_list.return_value = ["cli"]
        return cfg

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.respond = AsyncMock(return_value="test response")
        agent.affect_state.current_emotion.value = "curious"
        agent.safety.emergency_stop = MagicMock()
        agent.status = {
            "name": "Gwenn",
            "emotion": "curious",
            "uptime_seconds": 100.0,
            "total_interactions": 5,
        }
        agent.heartbeat.status = {
            "running": True,
            "beat_count": 10,
            "current_interval": 30.0,
        }
        agent.skill_registry.all_skills.return_value = [
            SimpleNamespace(name="weather", category="information"),
        ]
        agent._mcp_client.stats = {
            "configured_servers": 1,
            "connected_servers": 1,
            "discovered_tools": 2,
        }
        agent.tool_registry.count = 12
        agent.tool_registry.enabled_count = 11
        return agent

    @pytest.fixture
    def daemon(self, mock_config, mock_agent, tmp_path: Path):
        from gwenn.daemon import GwennDaemon

        with patch("gwenn.daemon.GwennDaemon._write_pid_file"):
            d = GwennDaemon.__new__(GwennDaemon)
            d._config = mock_config
            d._agent = mock_agent
            d._shutdown_event = asyncio.Event()
            d._connection_count = 0
            d._active_connections = 0
            d._socket_path = tmp_path / "gwenn.sock"
            d._pid_file = tmp_path / "gwenn.pid"
            d._sessions_dir = tmp_path / "sessions"
            d._auth_token = None
            d._max_connections = 10
            d._connection_timeout = 300.0
            d._session_include_preview = False
            d._session_redactor = MagicMock()
            d._session_redactor.redact = lambda text: text
            d._session_store = SessionStore(d._sessions_dir)
            d._channel_task = None
            d._agent_respond_lock = asyncio.Lock()
            return d

    async def test_ping(self, daemon) -> None:
        resp = await daemon._dispatch("ping", {"type": "ping"}, [], "r1")
        assert resp == {"type": "pong", "req_id": "r1"}

    async def test_chat(self, daemon, mock_agent) -> None:
        history = []
        resp = await daemon._dispatch("chat", {"type": "chat", "text": "hello"}, history, "r2")
        assert resp["type"] == "response"
        assert resp["text"] == "test response"
        assert resp["emotion"] == "curious"
        mock_agent.respond.assert_called_once_with("hello", conversation_history=history)

    async def test_chat_empty_text(self, daemon) -> None:
        resp = await daemon._dispatch("chat", {"type": "chat", "text": ""}, [], "r3")
        assert resp["type"] == "error"

    async def test_status(self, daemon) -> None:
        resp = await daemon._dispatch("status", {"type": "status"}, [], "r4")
        assert resp["type"] == "status_response"
        assert resp["status"]["name"] == "Gwenn"

    async def test_heartbeat_status(self, daemon) -> None:
        resp = await daemon._dispatch("heartbeat_status", {"type": "heartbeat_status"}, [], "r5")
        assert resp["type"] == "heartbeat_response"
        assert resp["status"]["beat_count"] == 10

    async def test_list_sessions(self, daemon, sessions_dir: Path) -> None:
        # Save a session to the daemon's store
        daemon._session_store = SessionStore(sessions_dir)
        msgs = [{"role": "user", "content": "hello"}]
        daemon._session_store.save_session(msgs, time.time())

        resp = await daemon._dispatch("list_sessions", {"type": "list_sessions"}, [], "r6")
        assert resp["type"] == "sessions_list"
        assert len(resp["sessions"]) == 1

    async def test_load_session(self, daemon, sessions_dir: Path) -> None:
        daemon._session_store = SessionStore(sessions_dir)
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        sid = daemon._session_store.save_session(msgs, time.time())

        history = []
        resp = await daemon._dispatch(
            "load_session",
            {"type": "load_session", "session_id": sid},
            history,
            "r7",
        )
        assert resp["type"] == "session_loaded"
        assert resp["message_count"] == 2
        assert len(history) == 2

    async def test_load_session_not_found(self, daemon) -> None:
        resp = await daemon._dispatch(
            "load_session",
            {"type": "load_session", "session_id": "20990101-000000-zzzz"},
            [],
            "r8",
        )
        assert resp["type"] == "error"
        assert "not found" in resp["message"]

    async def test_load_session_rejects_invalid_id(self, daemon) -> None:
        resp = await daemon._dispatch(
            "load_session",
            {"type": "load_session", "session_id": "../../etc/passwd"},
            [],
            "r8b",
        )
        assert resp["type"] == "error"
        assert "invalid session id" in resp["message"]

    async def test_reset_session(self, daemon) -> None:
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        resp = await daemon._dispatch("reset_session", {"type": "reset_session"}, history, "r8c")
        assert resp["type"] == "session_reset"
        assert resp["cleared_messages"] == 2
        assert history == []

    async def test_runtime_info(self, daemon) -> None:
        resp = await daemon._dispatch("runtime_info", {"type": "runtime_info"}, [], "r8d")
        assert resp["type"] == "runtime_info_response"
        assert "skills" in resp
        assert "mcp" in resp
        assert "tools" in resp

    async def test_stop(self, daemon) -> None:
        resp = await daemon._dispatch("stop", {"type": "stop"}, [], "r9")
        assert resp["type"] == "ack_stop"
        assert daemon._shutdown_event.is_set()
        daemon._agent.safety.emergency_stop.assert_called_once_with("daemon_stop_requested")

    async def test_unknown_type(self, daemon) -> None:
        resp = await daemon._dispatch("banana", {"type": "banana"}, [], "r10")
        assert resp["type"] == "error"

    async def test_auth_rejects_missing_token(self, daemon) -> None:
        daemon._auth_token = "secret"
        resp = await daemon._dispatch("ping", {"type": "ping"}, [], "r11")
        assert resp["type"] == "error"
        assert resp["message"] == "unauthorized"

    async def test_auth_accepts_valid_token(self, daemon) -> None:
        daemon._auth_token = "secret"
        resp = await daemon._dispatch(
            "ping",
            {"type": "ping", "auth_token": "secret"},
            [],
            "r12",
        )
        assert resp == {"type": "pong", "req_id": "r12"}

    async def test_handle_client_rejects_when_connection_limit_reached(self, daemon) -> None:
        daemon._active_connections = 2
        daemon._max_connections = 2
        daemon._send = AsyncMock()

        reader = MagicMock()
        writer = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon._handle_client(reader, writer)

        daemon._send.assert_awaited_once()
        payload = daemon._send.await_args.args[1]
        assert payload["message"] == "max connections reached"
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert daemon._active_connections == 2

    async def test_dispatch_loop_honors_idle_timeout(
        self, daemon, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        daemon._connection_timeout = 1.0

        async def _wait_for_timeout(coro, timeout):
            coro.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("gwenn.daemon.asyncio.wait_for", _wait_for_timeout)

        reader = MagicMock()
        reader.readline = MagicMock(side_effect=lambda: asyncio.sleep(60))
        writer = MagicMock()

        started = time.monotonic()
        await daemon._dispatch_loop(reader, writer, [])
        elapsed = time.monotonic() - started

        # Should stop after timeout budget, not spin forever.
        assert elapsed < 2.0

    async def test_run_platform_channels_skips_failed_channel(
        self, daemon, monkeypatch
    ) -> None:
        """When both channels are requested, a failed channel is skipped
        (not rolled back) so the other channel can continue running."""
        import gwenn.channels.discord_channel as dc_mod
        import gwenn.channels.telegram_channel as tg_mod
        import gwenn.config as config_mod

        events: list[str] = []

        class _Cfg:
            def __init__(self, max_history_length=10, session_ttl_seconds=120.0):
                self.max_history_length = max_history_length
                self.session_ttl_seconds = session_ttl_seconds

        class _Telegram:
            channel_name = "telegram"

            def __init__(self, *_args, **_kwargs):
                pass

            async def start(self):
                events.append("telegram:start")

            async def stop(self):
                events.append("telegram:stop")

        class _Discord:
            channel_name = "discord"

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

        # Trigger immediate shutdown so the wait loop returns promptly.
        daemon._shutdown_event.set()
        await daemon._run_platform_channels(["telegram", "discord"])

        # Discord is skipped; Telegram starts normally and stops at shutdown.
        assert events == ["telegram:start", "discord:start", "telegram:stop"]

    async def test_run_platform_channels_invalid_token_nonfatal(
        self, daemon, monkeypatch
    ) -> None:
        from gwenn.channels.session import SessionManager

        class InvalidToken(Exception):
            pass

        InvalidToken.__module__ = "telegram.error"

        def _build(_agent, channel_list):
            _ = channel_list
            return SessionManager(), [MagicMock(channel_name="telegram")]

        async def _run(*_args, **_kwargs):
            raise InvalidToken(
                "The token `123456789:ABCDEFGHIJKLMNOPQRSTUV123456789` "
                "was rejected by the server."
            )

        monkeypatch.setattr("gwenn.channels.startup.build_channels", _build)
        monkeypatch.setattr("gwenn.channels.startup.run_channels_until_shutdown", _run)

        await daemon._run_platform_channels(["cli", "telegram"])
        assert not daemon._shutdown_event.is_set()

    async def test_redact_channel_error_masks_telegram_token(self, daemon) -> None:
        raw = (
            "The token `123456789:ABCDEFGHIJKLMNOPQRSTUV123456789` was rejected."
        )
        redacted = daemon._redact_channel_error(raw)
        assert "123456789:" not in redacted
        assert "[REDACTED_TELEGRAM_TOKEN]" in redacted

    async def test_channel_task_failure_sets_shutdown_event(self, daemon) -> None:
        loop = asyncio.get_running_loop()
        task = loop.create_future()
        task.set_exception(RuntimeError("boom"))
        daemon._on_channel_task_done(task)
        assert daemon._shutdown_event.is_set()

    async def test_channel_task_invalid_token_does_not_set_shutdown_event(self, daemon) -> None:
        class InvalidToken(Exception):
            pass

        InvalidToken.__module__ = "telegram.error"

        loop = asyncio.get_running_loop()
        task = loop.create_future()
        task.set_exception(
            InvalidToken(
                "The token `123456789:ABCDEFGHIJKLMNOPQRSTUV123456789` "
                "was rejected by the server."
            )
        )
        daemon._on_channel_task_done(task)
        assert not daemon._shutdown_event.is_set()
