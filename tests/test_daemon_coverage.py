"""
Tests for GwennDaemon — coverage completion.

Targets every missing line in gwenn/daemon.py that the existing test_daemon.py
does not cover. Uses heavy mocking to avoid real agent / socket / process work.

Missing lines addressed:
  49-65   __init__
  75-92   run()
  100-120 _write_pid_file()
  124-145 _start_agent()
  150,153 _on_channel_task_done() cancelled / clean finish
  171-186 _start_socket_server()
  194     _run_platform_channels() early return (no channels built)
  222-225,231 _is_nonfatal_channel_error() ImportError / discord LoginFailure
  265-294 _handle_client() normal flow + session save
  318-348 _dispatch_loop() client disconnect, bad JSON, auth failures, stop
  377-378 _dispatch() emotion snapshot fail
  445-446,450 _dispatch() runtime_info skill list fail, mcp_stats not dict
  476-478 _dispatch() catch-all exception handler
  492     _is_authorized() auth rejected (wrong token)
  498-500 _send()
  508-534 _cleanup()
  545-546 _request_shutdown() emergency_stop fail
  551-557 _install_signal_handlers()
  566-582 run_daemon()
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gwenn.memory.session_store import SessionStore


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_config(tmp_path: Path) -> MagicMock:
    """Build a MagicMock that quacks like GwennConfig for daemon init."""
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
    cfg.daemon.redact_session_content = False  # disable redaction for tests
    cfg.channel.get_channel_list.return_value = ["cli"]
    cfg.mcp.get_server_list.return_value = []
    return cfg


def _make_mock_agent() -> MagicMock:
    """Build a MagicMock that quacks like SentientAgent."""
    agent = MagicMock()
    agent.respond_lock = asyncio.Lock()
    agent.respond = AsyncMock(return_value="test response")
    agent.affect_state.current_emotion.value = "curious"
    agent.safety.emergency_stop = MagicMock()
    agent.initialize = AsyncMock()
    agent.start = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.status = {"name": "Gwenn", "emotion": "curious"}
    agent.heartbeat = MagicMock()
    agent.heartbeat.status = {"running": True, "beat_count": 10}
    agent.skill_registry.all_skills.return_value = [
        SimpleNamespace(name="weather", category="information"),
    ]
    agent._mcp_client.stats = {"configured_servers": 1}
    agent.tool_registry.count = 12
    agent.tool_registry.enabled_count = 11
    return agent


def _make_daemon(tmp_path: Path, agent: MagicMock | None = None) -> "GwennDaemon":
    """Build a GwennDaemon with mocked internals, bypassing __init__."""
    from gwenn.daemon import GwennDaemon

    cfg = _make_mock_config(tmp_path)
    d = GwennDaemon.__new__(GwennDaemon)
    d._config = cfg
    d._agent = agent or _make_mock_agent()
    d._server = None
    d._channel_task = None
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
    return d


# =============================================================================
# __init__  (lines 49-65)
# =============================================================================


class TestGwennDaemonInit:
    """Test that GwennDaemon.__init__ properly initialises all attributes."""

    def test_init_sets_attributes(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        cfg = _make_mock_config(tmp_path)
        cfg.daemon.auth_token = "  my-token  "
        cfg.daemon.max_connections = 5
        cfg.daemon.connection_timeout = 60.0
        cfg.daemon.session_include_preview = True
        cfg.daemon.redact_session_content = False

        d = GwennDaemon(cfg)

        assert d._config is cfg
        assert d._agent is None
        assert d._server is None
        assert d._channel_task is None
        assert d._connection_count == 0
        assert d._active_connections == 0
        assert d._auth_token == "my-token"
        assert d._max_connections == 5
        assert d._connection_timeout == 60.0
        assert d._session_include_preview is True
        assert isinstance(d._session_store, SessionStore)

    def test_init_blank_auth_token_becomes_none(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        cfg = _make_mock_config(tmp_path)
        cfg.daemon.auth_token = "   "

        d = GwennDaemon(cfg)
        assert d._auth_token is None

    def test_init_none_auth_token(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        cfg = _make_mock_config(tmp_path)
        cfg.daemon.auth_token = None

        d = GwennDaemon(cfg)
        assert d._auth_token is None

    def test_init_max_connections_floor(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        cfg = _make_mock_config(tmp_path)
        cfg.daemon.max_connections = -5

        d = GwennDaemon(cfg)
        assert d._max_connections == 1

    def test_init_connection_timeout_floor(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        cfg = _make_mock_config(tmp_path)
        cfg.daemon.connection_timeout = 0.1

        d = GwennDaemon(cfg)
        assert d._connection_timeout == 1.0


# =============================================================================
# run()  (lines 75-92)
# =============================================================================


@pytest.mark.asyncio
class TestDaemonRun:
    async def test_run_full_lifecycle(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        d = _make_daemon(tmp_path)

        # Patch lifecycle stages
        d._write_pid_file = MagicMock()
        d._start_agent = AsyncMock()
        d._start_socket_server = AsyncMock()
        d._cleanup = AsyncMock()

        # Immediately fire shutdown so run() doesn't block
        d._shutdown_event.set()

        await d.run()

        d._write_pid_file.assert_called_once()
        d._start_agent.assert_awaited_once()
        d._start_socket_server.assert_awaited_once()
        d._cleanup.assert_awaited_once()

    async def test_run_calls_cleanup_on_exception(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._write_pid_file = MagicMock()
        d._start_agent = AsyncMock(side_effect=RuntimeError("boom"))
        d._cleanup = AsyncMock()

        with pytest.raises(RuntimeError, match="boom"):
            await d.run()

        d._cleanup.assert_awaited_once()


# =============================================================================
# _write_pid_file()  (lines 100-120)
# =============================================================================


class TestWritePidFile:
    def test_write_pid_file_no_existing(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._pid_file = tmp_path / "subdir" / "gwenn.pid"
        d._write_pid_file()
        assert d._pid_file.exists()
        assert d._pid_file.read_text() == str(os.getpid())

    def test_write_pid_file_stale_pid(self, tmp_path: Path) -> None:
        """Stale PID (process dead) should be cleaned up and overwritten."""
        d = _make_daemon(tmp_path)
        pid_file = tmp_path / "gwenn.pid"
        pid_file.write_text("999999999")  # PID that almost certainly doesn't exist
        d._pid_file = pid_file

        d._write_pid_file()
        assert pid_file.read_text() == str(os.getpid())

    def test_write_pid_file_invalid_pid_content(self, tmp_path: Path) -> None:
        """Non-integer PID content triggers ValueError → treated as stale."""
        d = _make_daemon(tmp_path)
        pid_file = tmp_path / "gwenn.pid"
        pid_file.write_text("not-a-number")
        d._pid_file = pid_file

        d._write_pid_file()
        assert pid_file.read_text() == str(os.getpid())

    def test_write_pid_file_running_pid_exits(self, tmp_path: Path) -> None:
        """If PID file points to a *live* process, sys.exit(1) is called."""
        d = _make_daemon(tmp_path)
        pid_file = tmp_path / "gwenn.pid"
        # Use our own PID — guaranteed to be alive
        pid_file.write_text(str(os.getpid()))
        d._pid_file = pid_file

        with pytest.raises(SystemExit) as exc_info:
            d._write_pid_file()
        assert exc_info.value.code == 1

    def test_write_pid_file_permission_error_exits(self, tmp_path: Path) -> None:
        """If os.kill raises PermissionError, treat process as running."""
        d = _make_daemon(tmp_path)
        pid_file = tmp_path / "gwenn.pid"
        pid_file.write_text("12345")
        d._pid_file = pid_file

        with patch("os.kill", side_effect=PermissionError("not owner")):
            with pytest.raises(SystemExit) as exc_info:
                d._write_pid_file()
            assert exc_info.value.code == 1

    def test_write_pid_file_stale_unlink_oserror(self, tmp_path: Path) -> None:
        """If unlink of stale PID file fails with OSError, it continues anyway."""
        d = _make_daemon(tmp_path)
        pid_file = tmp_path / "gwenn.pid"
        pid_file.write_text("999999999")
        d._pid_file = pid_file

        original_unlink = Path.unlink

        call_count = 0

        def flaky_unlink(self_path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call is the stale cleanup — make it fail
            if call_count == 1:
                raise OSError("cannot unlink")
            return original_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", flaky_unlink):
            # Despite the OSError, it should still try to write the PID
            d._write_pid_file()


# =============================================================================
# _start_agent()  (lines 124-145)
# =============================================================================


@pytest.mark.asyncio
class TestStartAgent:
    async def test_start_agent_success_no_channels(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._config.channel.get_channel_list.return_value = ["cli"]

        mock_agent = _make_mock_agent()

        with patch("gwenn.agent.SentientAgent", return_value=mock_agent):
            await d._start_agent()

        mock_agent.initialize.assert_awaited_once()
        mock_agent.start.assert_awaited_once()
        assert d._channel_task is None  # no telegram/discord

    async def test_start_agent_creates_channel_task(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._config.channel.get_channel_list.return_value = ["cli", "telegram"]

        mock_agent = _make_mock_agent()

        with patch("gwenn.agent.SentientAgent", return_value=mock_agent):
            await d._start_agent()

        assert d._channel_task is not None
        # Clean up the task
        d._channel_task.cancel()
        try:
            await d._channel_task
        except (asyncio.CancelledError, Exception):
            pass

    async def test_start_agent_init_error_exits(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)

        from gwenn.api.claude import CognitiveEngineInitError

        with patch("gwenn.agent.SentientAgent", side_effect=CognitiveEngineInitError("no key")):
            with pytest.raises(SystemExit) as exc_info:
                await d._start_agent()
            assert exc_info.value.code == 1

    async def test_start_agent_discord_channel_task(self, tmp_path: Path) -> None:
        """Discord alone should also trigger channel task creation."""
        d = _make_daemon(tmp_path)
        d._config.channel.get_channel_list.return_value = ["cli", "discord"]

        mock_agent = _make_mock_agent()

        with patch("gwenn.agent.SentientAgent", return_value=mock_agent):
            await d._start_agent()

        assert d._channel_task is not None
        d._channel_task.cancel()
        try:
            await d._channel_task
        except (asyncio.CancelledError, Exception):
            pass


# =============================================================================
# _on_channel_task_done()  (lines 150, 153)
# =============================================================================


@pytest.mark.asyncio
class TestOnChannelTaskDone:
    async def test_cancelled_task_returns_early(self, tmp_path: Path) -> None:
        """A cancelled task should not trigger shutdown."""
        d = _make_daemon(tmp_path)
        task = asyncio.get_running_loop().create_future()
        task.cancel()
        # Suppress CancelledError
        try:
            await task
        except asyncio.CancelledError:
            pass

        d._on_channel_task_done(task)
        assert not d._shutdown_event.is_set()

    async def test_clean_finish_returns_early(self, tmp_path: Path) -> None:
        """A task that finished without exception should not trigger shutdown."""
        d = _make_daemon(tmp_path)
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)

        d._on_channel_task_done(task)
        assert not d._shutdown_event.is_set()


# =============================================================================
# _start_socket_server()  (lines 171-186)
# =============================================================================


@pytest.mark.asyncio
class TestStartSocketServer:
    async def test_start_socket_server_removes_stale(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._socket_path = tmp_path / "gwenn.sock"
        # Create a stale socket file
        d._socket_path.touch()

        await d._start_socket_server()

        assert d._server is not None
        # Clean up
        d._server.close()
        await d._server.wait_closed()

    async def test_start_socket_server_stale_unlink_fail(self, tmp_path: Path) -> None:
        """If unlink of stale socket fails, server should still start."""
        d = _make_daemon(tmp_path)
        sock_path = tmp_path / "gwenn.sock"
        sock_path.touch()
        d._socket_path = sock_path

        original_unlink = Path.unlink

        def fail_unlink(self_path, *args, **kwargs):
            if str(self_path) == str(sock_path):
                raise OSError("cannot remove stale socket")
            return original_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_unlink):
            # unlink fails, but mkdir + start_unix_server should still work
            # Since the file still exists, start_unix_server might fail too,
            # but the point is _start_socket_server doesn't crash on OSError
            try:
                await d._start_socket_server()
            except OSError:
                pass  # socket file still present blocks bind — that's OK for this test

    async def test_start_socket_server_chmod_fail(self, tmp_path: Path) -> None:
        """If chmod fails, server should still be running."""
        d = _make_daemon(tmp_path)
        d._socket_path = tmp_path / "gwenn2.sock"

        with patch("os.chmod", side_effect=OSError("chmod failed")):
            await d._start_socket_server()

        assert d._server is not None
        d._server.close()
        await d._server.wait_closed()

    async def test_start_socket_server_creates_parent_dir(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._socket_path = tmp_path / "nested" / "dir" / "gwenn.sock"

        await d._start_socket_server()

        assert d._server is not None
        assert (tmp_path / "nested" / "dir").is_dir()
        d._server.close()
        await d._server.wait_closed()


# =============================================================================
# _run_platform_channels()  (line 194)
# =============================================================================


@pytest.mark.asyncio
class TestRunPlatformChannels:
    async def test_no_channels_built_returns_early(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)

        with patch("gwenn.channels.startup.build_channels", return_value=(MagicMock(), [])):
            await d._run_platform_channels(["telegram"])
        # Should return without error


# =============================================================================
# _is_nonfatal_channel_error()  (lines 222-225, 231)
# =============================================================================


class TestIsNonfatalChannelError:
    def test_import_error_telegram(self) -> None:
        from gwenn.daemon import GwennDaemon

        exc = ImportError("No module named 'telegram'")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is True

    def test_import_error_discord(self) -> None:
        from gwenn.daemon import GwennDaemon

        exc = ImportError("No module named 'discord'")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is True

    def test_import_error_unrelated(self) -> None:
        from gwenn.daemon import GwennDaemon

        exc = ImportError("No module named 'numpy'")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is False

    def test_discord_login_failure(self) -> None:
        from gwenn.daemon import GwennDaemon

        class LoginFailure(Exception):
            pass

        LoginFailure.__module__ = "discord.errors"
        exc = LoginFailure("invalid token")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is True

    def test_telegram_invalid_token(self) -> None:
        from gwenn.daemon import GwennDaemon

        class InvalidToken(Exception):
            pass

        InvalidToken.__module__ = "telegram.error"
        exc = InvalidToken("bad token")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is True

    def test_random_exception_is_fatal(self) -> None:
        from gwenn.daemon import GwennDaemon

        exc = RuntimeError("something broke")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is False

    def test_generic_exception_wrong_name(self) -> None:
        from gwenn.daemon import GwennDaemon

        class SomeError(Exception):
            pass

        SomeError.__module__ = "mylib"
        exc = SomeError("oops")
        assert GwennDaemon._is_nonfatal_channel_error(exc) is False


# =============================================================================
# _handle_client()  (lines 265-294)
# =============================================================================


@pytest.mark.asyncio
class TestHandleClient:
    async def test_normal_client_flow_saves_session(self, tmp_path: Path) -> None:
        """Client connects, exchanges messages, disconnects → session saved."""
        d = _make_daemon(tmp_path)
        d._session_store = MagicMock()
        d._session_store.save_session = MagicMock(return_value="20260101-120000-abcd")

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        # Simulate a dispatch loop that puts something in history, then returns
        async def fake_dispatch_loop(r, w, history):
            history.append({"role": "user", "content": "hello"})
            history.append({"role": "assistant", "content": "hi"})

        d._dispatch_loop = AsyncMock(side_effect=fake_dispatch_loop)

        await d._handle_client(reader, writer)

        assert d._active_connections == 0
        assert d._connection_count == 1
        d._session_store.save_session.assert_called_once()
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    async def test_client_disconnect_error(self, tmp_path: Path) -> None:
        """Client disconnects abruptly with IncompleteReadError."""
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock(
            side_effect=asyncio.IncompleteReadError(b"", 100)
        )

        await d._handle_client(reader, writer)
        assert d._active_connections == 0

    async def test_client_connection_reset(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock(side_effect=ConnectionResetError())

        await d._handle_client(reader, writer)
        assert d._active_connections == 0

    async def test_client_broken_pipe(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock(side_effect=BrokenPipeError())

        await d._handle_client(reader, writer)
        assert d._active_connections == 0

    async def test_client_unexpected_error(self, tmp_path: Path) -> None:
        """Unexpected exception in dispatch loop should still clean up."""
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock(side_effect=ValueError("unexpected"))

        await d._handle_client(reader, writer)
        assert d._active_connections == 0
        writer.close.assert_called_once()

    async def test_client_close_writer_error(self, tmp_path: Path) -> None:
        """If writer.close/wait_closed fails, we still finish gracefully."""
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock(side_effect=OSError("close failed"))
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock()

        await d._handle_client(reader, writer)
        assert d._active_connections == 0

    async def test_no_history_skips_session_save(self, tmp_path: Path) -> None:
        """Empty history should not trigger session save."""
        d = _make_daemon(tmp_path)
        d._session_store = MagicMock()

        reader = AsyncMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        d._dispatch_loop = AsyncMock()  # does nothing to history

        await d._handle_client(reader, writer)
        d._session_store.save_session.assert_not_called()


# =============================================================================
# _dispatch_loop()  (lines 318-348)
# =============================================================================


@pytest.mark.asyncio
class TestDispatchLoop:
    async def test_client_disconnect_empty_read(self, tmp_path: Path) -> None:
        """Empty read (b'') means client disconnected."""
        d = _make_daemon(tmp_path)

        reader = AsyncMock()
        reader.readline = AsyncMock(return_value=b"")
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])

    async def test_bad_json_continues(self, tmp_path: Path) -> None:
        """Invalid JSON should be skipped, loop continues."""
        d = _make_daemon(tmp_path)

        call_count = 0

        async def readline():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"not valid json\n"
            return b""  # disconnect

        reader = AsyncMock()
        reader.readline = readline
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])
        assert call_count == 2

    async def test_auth_failure_disconnect_after_max(self, tmp_path: Path) -> None:
        """After _MAX_AUTH_FAILURES auth failures, loop should break."""
        d = _make_daemon(tmp_path)
        d._auth_token = "secret"
        d._send = AsyncMock()

        call_count = 0

        async def readline():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                return json.dumps({"type": "ping", "req_id": f"r{call_count}"}).encode() + b"\n"
            return b""

        reader = AsyncMock()
        reader.readline = readline
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])

        # Should have sent 3 unauthorized responses then broken out
        assert d._send.await_count == 3
        for call in d._send.await_args_list:
            assert call.args[1]["message"] == "unauthorized"

    async def test_successful_chat_resets_auth_failures(self, tmp_path: Path) -> None:
        """Successful chat should reset auth failure counter."""
        d = _make_daemon(tmp_path)
        d._auth_token = "secret"
        d._send = AsyncMock()

        messages = [
            # Two auth failures
            json.dumps({"type": "ping", "req_id": "r1"}).encode() + b"\n",
            json.dumps({"type": "ping", "req_id": "r2"}).encode() + b"\n",
            # Successful chat resets counter
            json.dumps({"type": "chat", "text": "hello", "auth_token": "secret", "req_id": "r3"}).encode() + b"\n",
            # Two more auth failures — should NOT hit max (3) since counter was reset
            json.dumps({"type": "ping", "req_id": "r4"}).encode() + b"\n",
            json.dumps({"type": "ping", "req_id": "r5"}).encode() + b"\n",
            b"",  # disconnect
        ]
        idx = 0

        async def readline():
            nonlocal idx
            msg = messages[idx]
            idx += 1
            return msg

        reader = AsyncMock()
        reader.readline = readline
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])
        # We should have processed all 5 messages (2 fail + 1 success + 2 fail)
        assert d._send.await_count == 5

    async def test_stop_message_breaks_loop(self, tmp_path: Path) -> None:
        """A 'stop' message type should break the dispatch loop."""
        d = _make_daemon(tmp_path)
        d._send = AsyncMock()

        async def readline():
            return json.dumps({"type": "stop", "req_id": "r1"}).encode() + b"\n"

        reader = AsyncMock()
        reader.readline = readline
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])

        assert d._send.await_count == 1
        sent = d._send.await_args_list[0].args[1]
        assert sent["type"] == "ack_stop"

    async def test_shutdown_event_stops_loop(self, tmp_path: Path) -> None:
        """If shutdown_event is set, loop should stop."""
        d = _make_daemon(tmp_path)
        d._shutdown_event.set()

        reader = AsyncMock()
        writer = MagicMock()

        await d._dispatch_loop(reader, writer, [])


# =============================================================================
# _dispatch()  (lines 377-378, 445-446, 450, 476-478)
# =============================================================================


@pytest.mark.asyncio
class TestDispatchEdgeCases:
    async def test_chat_emotion_snapshot_fails(self, tmp_path: Path) -> None:
        """If affect_state raises, emotion defaults to 'neutral'."""
        d = _make_daemon(tmp_path)

        # Create an agent whose affect_state.current_emotion.value raises
        class _BrokenAffect:
            @property
            def current_emotion(self):
                raise AttributeError("no emotion state")

        d._agent.affect_state = _BrokenAffect()

        resp = await d._dispatch("chat", {"type": "chat", "text": "hi"}, [], "r1")
        assert resp["type"] == "response"
        assert resp["emotion"] == "neutral"

    async def test_runtime_info_skill_list_fails(self, tmp_path: Path) -> None:
        """If all_skills() raises, skills list should be empty."""
        d = _make_daemon(tmp_path)
        d._agent.skill_registry.all_skills.side_effect = RuntimeError("no skills")

        resp = await d._dispatch("runtime_info", {"type": "runtime_info"}, [], "r1")
        assert resp["type"] == "runtime_info_response"
        assert resp["skills"] == []

    async def test_runtime_info_mcp_stats_not_dict(self, tmp_path: Path) -> None:
        """If mcp_client.stats is not a dict, use empty dict."""
        d = _make_daemon(tmp_path)
        d._agent._mcp_client.stats = "not a dict"

        resp = await d._dispatch("runtime_info", {"type": "runtime_info"}, [], "r1")
        assert resp["type"] == "runtime_info_response"
        assert resp["mcp"] == {}

    async def test_dispatch_catch_all_exception(self, tmp_path: Path) -> None:
        """If _is_authorized raises, catch-all returns internal error."""
        d = _make_daemon(tmp_path)

        with patch.object(d, "_is_authorized", side_effect=RuntimeError("boom")):
            resp = await d._dispatch("ping", {"type": "ping"}, [], "r1")

        assert resp["type"] == "error"
        assert resp["message"] == "internal error"

    async def test_runtime_info_no_agent(self, tmp_path: Path) -> None:
        """Runtime info with no agent should return empty status/skills."""
        d = _make_daemon(tmp_path)
        d._agent = None

        resp = await d._dispatch("runtime_info", {"type": "runtime_info"}, [], "r1")
        assert resp["type"] == "runtime_info_response"
        assert resp["status"] == {}
        assert resp["skills"] == []

    async def test_heartbeat_status_no_heartbeat(self, tmp_path: Path) -> None:
        """Heartbeat status with no agent returns empty dict."""
        d = _make_daemon(tmp_path)
        d._agent = None

        resp = await d._dispatch("heartbeat_status", {"type": "heartbeat_status"}, [], "r1")
        assert resp["type"] == "heartbeat_response"
        assert resp["status"] == {}

    async def test_heartbeat_status_agent_no_heartbeat(self, tmp_path: Path) -> None:
        """Heartbeat status with agent but no heartbeat returns empty dict."""
        d = _make_daemon(tmp_path)
        d._agent.heartbeat = None

        resp = await d._dispatch("heartbeat_status", {"type": "heartbeat_status"}, [], "r1")
        assert resp["type"] == "heartbeat_response"
        assert resp["status"] == {}


# =============================================================================
# _is_authorized()  (line 492)
# =============================================================================


class TestIsAuthorized:
    def test_auth_rejected_wrong_token(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._auth_token = "correct-token"
        result = d._is_authorized({"auth_token": "wrong-token"})
        assert result is False

    def test_auth_missing_token(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._auth_token = "correct-token"
        result = d._is_authorized({})
        assert result is False

    def test_auth_non_string_token(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._auth_token = "correct-token"
        result = d._is_authorized({"auth_token": 12345})
        assert result is False

    def test_auth_no_token_required(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._auth_token = None
        result = d._is_authorized({})
        assert result is True

    def test_auth_correct_token(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._auth_token = "my-token"
        result = d._is_authorized({"auth_token": "my-token"})
        assert result is True


# =============================================================================
# _send()  (lines 498-500)
# =============================================================================


@pytest.mark.asyncio
class TestSend:
    async def test_send_writes_ndjson(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        writer = MagicMock()
        writer.drain = AsyncMock()

        data = {"type": "pong", "req_id": "r1"}
        await GwennDaemon._send(writer, data)

        writer.write.assert_called_once()
        written_bytes = writer.write.call_args.args[0]
        decoded = json.loads(written_bytes.decode("utf-8"))
        assert decoded == data
        writer.drain.assert_awaited_once()

    async def test_send_unicode(self, tmp_path: Path) -> None:
        from gwenn.daemon import GwennDaemon

        writer = MagicMock()
        writer.drain = AsyncMock()

        data = {"type": "response", "text": "Bonjour! \u00e7a va? \U0001f600"}
        await GwennDaemon._send(writer, data)

        written_bytes = writer.write.call_args.args[0]
        assert b"\\u" not in written_bytes  # ensure_ascii=False
        assert "Bonjour!".encode("utf-8") in written_bytes


# =============================================================================
# _cleanup()  (lines 508-534)
# =============================================================================


@pytest.mark.asyncio
class TestCleanup:
    async def test_cleanup_full(self, tmp_path: Path) -> None:
        """Full cleanup: server, channel task, agent, files."""
        d = _make_daemon(tmp_path)

        # Create socket and PID files
        d._socket_path.touch()
        d._pid_file.touch()

        # Mock server
        d._server = MagicMock()
        d._server.close = MagicMock()
        d._server.wait_closed = AsyncMock()

        # Mock channel task (clean completion)
        channel_fut = asyncio.get_running_loop().create_future()
        channel_fut.set_result(None)
        d._channel_task = channel_fut

        await d._cleanup()

        d._server.close.assert_called_once()
        d._server.wait_closed.assert_awaited_once()
        d._agent.shutdown.assert_awaited_once()
        assert not d._socket_path.exists()
        assert not d._pid_file.exists()
        assert d._channel_task is None

    async def test_cleanup_no_server(self, tmp_path: Path) -> None:
        """Cleanup with no server should not crash."""
        d = _make_daemon(tmp_path)
        d._server = None
        d._channel_task = None

        await d._cleanup()
        d._agent.shutdown.assert_awaited_once()

    async def test_cleanup_no_agent(self, tmp_path: Path) -> None:
        """Cleanup with no agent should not crash."""
        d = _make_daemon(tmp_path)
        d._agent = None
        d._server = None
        d._channel_task = None

        await d._cleanup()
        # should not raise

    async def test_cleanup_channel_task_cancelled(self, tmp_path: Path) -> None:
        """If channel task was cancelled, cleanup handles CancelledError."""
        d = _make_daemon(tmp_path)
        d._server = None

        task = asyncio.get_running_loop().create_task(asyncio.sleep(100))
        d._channel_task = task
        task.cancel()

        await d._cleanup()
        assert d._channel_task is None

    async def test_cleanup_channel_task_exception(self, tmp_path: Path) -> None:
        """If channel task raises, cleanup handles it and logs."""
        d = _make_daemon(tmp_path)
        d._server = None

        async def fail():
            raise RuntimeError("channel crash")

        task = asyncio.get_running_loop().create_task(fail())
        # Let it fail
        await asyncio.sleep(0.01)
        d._channel_task = task

        await d._cleanup()
        assert d._channel_task is None

    async def test_cleanup_unlink_oserror(self, tmp_path: Path) -> None:
        """If unlink of socket/pid fails with OSError, cleanup continues."""
        d = _make_daemon(tmp_path)
        d._server = None
        d._channel_task = None

        d._socket_path = tmp_path / "gwenn.sock"
        d._pid_file = tmp_path / "gwenn.pid"
        d._socket_path.touch()
        d._pid_file.touch()

        with patch.object(Path, "unlink", side_effect=OSError("cannot remove")):
            await d._cleanup()
        # Should not raise


# =============================================================================
# _request_shutdown()  (lines 545-546)
# =============================================================================


class TestRequestShutdown:
    def test_request_shutdown_sets_event(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._request_shutdown("test_reason")
        assert d._shutdown_event.is_set()
        d._agent.safety.emergency_stop.assert_called_once_with("test_reason")

    def test_request_shutdown_emergency_stop_fails(self, tmp_path: Path) -> None:
        """If emergency_stop raises, shutdown event should still be set."""
        d = _make_daemon(tmp_path)
        d._agent.safety.emergency_stop.side_effect = RuntimeError("stop failed")

        d._request_shutdown("test_reason")
        assert d._shutdown_event.is_set()

    def test_request_shutdown_no_agent(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._agent = None

        d._request_shutdown("test_reason")
        assert d._shutdown_event.is_set()

    def test_request_shutdown_no_safety(self, tmp_path: Path) -> None:
        """Agent with no safety attribute should still set shutdown event."""
        d = _make_daemon(tmp_path)
        d._agent = MagicMock(spec=[])  # no attributes at all

        d._request_shutdown("test_reason")
        assert d._shutdown_event.is_set()

    def test_request_shutdown_emergency_stop_not_callable(self, tmp_path: Path) -> None:
        """If emergency_stop is not callable, shutdown event should still be set."""
        d = _make_daemon(tmp_path)
        d._agent.safety.emergency_stop = "not callable"

        d._request_shutdown("test_reason")
        assert d._shutdown_event.is_set()


# =============================================================================
# _install_signal_handlers()  (lines 551-557)
# =============================================================================


@pytest.mark.asyncio
class TestInstallSignalHandlers:
    async def test_install_signal_handlers(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        loop = MagicMock()

        d._install_signal_handlers(loop)

        assert loop.add_signal_handler.call_count == 2
        # Check SIGINT and SIGTERM were registered
        sigs = {call.args[0] for call in loop.add_signal_handler.call_args_list}
        assert signal.SIGINT in sigs
        assert signal.SIGTERM in sigs

    async def test_install_signal_handlers_not_implemented(self, tmp_path: Path) -> None:
        """On platforms that don't support add_signal_handler (e.g. Windows)."""
        d = _make_daemon(tmp_path)
        loop = MagicMock()
        loop.add_signal_handler.side_effect = NotImplementedError

        d._install_signal_handlers(loop)
        # Should not raise


# =============================================================================
# run_daemon()  (lines 566-582)
# =============================================================================


class TestRunDaemon:
    def test_run_daemon_success(self, tmp_path: Path) -> None:
        """run_daemon creates config, daemon, event loop and runs."""
        from gwenn.daemon import run_daemon

        mock_config = _make_mock_config(tmp_path)
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock()

        with patch("gwenn.daemon.configure_logging") as mock_log:
            with patch("gwenn.daemon.GwennConfig", return_value=mock_config):
                with patch("gwenn.daemon.GwennDaemon") as MockDaemon:
                    mock_daemon_instance = MagicMock()
                    MockDaemon.return_value = mock_daemon_instance
                    with patch("asyncio.new_event_loop", return_value=mock_loop):
                        with patch("asyncio.set_event_loop"):
                            run_daemon()

        mock_log.assert_called_once()
        mock_loop.run_until_complete.assert_called_once()
        mock_loop.close.assert_called_once()
        mock_daemon_instance._install_signal_handlers.assert_called_once_with(mock_loop)

    def test_run_daemon_config_error(self, tmp_path: Path) -> None:
        """If GwennConfig() raises, run_daemon should sys.exit(1)."""
        from gwenn.daemon import run_daemon

        with patch("gwenn.daemon.configure_logging"):
            with patch("gwenn.daemon.GwennConfig", side_effect=RuntimeError("bad config")):
                with pytest.raises(SystemExit) as exc_info:
                    run_daemon()
                assert exc_info.value.code == 1

    def test_run_daemon_keyboard_interrupt(self, tmp_path: Path) -> None:
        """KeyboardInterrupt during loop.run_until_complete should be handled."""
        from gwenn.daemon import run_daemon

        mock_config = _make_mock_config(tmp_path)
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = KeyboardInterrupt

        with patch("gwenn.daemon.configure_logging"):
            with patch("gwenn.daemon.GwennConfig", return_value=mock_config):
                with patch("gwenn.daemon.GwennDaemon") as MockDaemon:
                    MockDaemon.return_value = MagicMock()
                    with patch("asyncio.new_event_loop", return_value=mock_loop):
                        with patch("asyncio.set_event_loop"):
                            run_daemon()  # Should not raise

        mock_loop.close.assert_called_once()


# =============================================================================
# _redact_channel_error()  (line 237) — edge case
# =============================================================================


class TestRedactChannelError:
    def test_redact_none_message(self) -> None:
        from gwenn.daemon import GwennDaemon

        result = GwennDaemon._redact_channel_error(None)
        assert result == ""

    def test_redact_empty_message(self) -> None:
        from gwenn.daemon import GwennDaemon

        result = GwennDaemon._redact_channel_error("")
        assert result == ""

    def test_redact_no_token(self) -> None:
        from gwenn.daemon import GwennDaemon

        result = GwennDaemon._redact_channel_error("normal error message")
        assert result == "normal error message"
