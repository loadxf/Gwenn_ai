"""Tests for CliChannel and WebSocketCliChannel in gwenn/channels/cli_channel.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from gwenn.channels.cli_channel import (
    CliChannel,
    DaemonNotRunningError,
    WebSocketCliChannel,
)


# ---------------------------------------------------------------------------
# Mock WebSocket helpers
# ---------------------------------------------------------------------------


class _MockWSMessage:
    """Simulates an aiohttp WebSocket message."""

    __slots__ = ("type", "data")

    def __init__(self, msg_type: int, data: str | None = None) -> None:
        self.type = msg_type
        self.data = data


class MockWebSocket:
    """Fake WebSocket that records sent messages and yields injected ones.

    Use ``set_response(method, fn)`` to auto-respond to JSON-RPC requests,
    or ``inject(data)`` / ``inject_close()`` for manual message control.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_MockWSMessage | None] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._auto_responses: dict[str, Any] = {}

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)
        method = data.get("method")
        req_id = data.get("id")
        if method in self._auto_responses:
            fn = self._auto_responses[method]
            result = fn(data.get("params", {}))
            self._queue.put_nowait(
                _MockWSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}),
                )
            )

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)

    # Async iteration protocol -------------------------------------------

    def __aiter__(self) -> MockWebSocket:
        return self

    async def __anext__(self) -> _MockWSMessage:
        if self.closed:
            raise StopAsyncIteration
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    # Test helpers -------------------------------------------------------

    def set_response(self, method: str, fn: Any) -> None:
        """Register an auto-response: fn(params) -> result dict."""
        self._auto_responses[method] = fn

    def inject(self, data: dict[str, Any]) -> None:
        """Inject a raw JSON message for the reader loop."""
        self._queue.put_nowait(
            _MockWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(data))
        )

    def inject_close(self) -> None:
        """Inject a CLOSED frame."""
        self._queue.put_nowait(_MockWSMessage(aiohttp.WSMsgType.CLOSED))

    def inject_error(self) -> None:
        """Inject an ERROR frame."""
        self._queue.put_nowait(_MockWSMessage(aiohttp.WSMsgType.ERROR))

    def inject_bad_json(self) -> None:
        """Inject a malformed JSON message."""
        self._queue.put_nowait(
            _MockWSMessage(aiohttp.WSMsgType.TEXT, "not valid json{{{")
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ws() -> MockWebSocket:
    ws = MockWebSocket()
    ws.set_response("auth", lambda p: {"status": "authenticated"})
    ws.set_response("ping", lambda p: {"status": "pong"})
    ws.set_response("chat", lambda p: {"text": "hi", "emotion": "happy"})
    ws.set_response("status", lambda p: {"status": {}, "active_connections": 1})
    ws.set_response("heartbeat_status", lambda p: {"status": {}})
    ws.set_response("list_sessions", lambda p: {"sessions": [{"id": "s1"}]})
    ws.set_response("load_session", lambda p: {"message_count": 5})
    ws.set_response("reset_session", lambda p: {"cleared_messages": 3})
    ws.set_response("runtime_info", lambda p: {"status": {}, "skills": []})
    ws.set_response("stop", lambda p: {"status": "stopping"})
    return ws


@pytest.fixture
def mock_session(mock_ws: MockWebSocket) -> AsyncMock:
    session = AsyncMock()
    session.ws_connect = AsyncMock(return_value=mock_ws)
    session.close = AsyncMock()
    return session


async def _connect(
    mock_session: AsyncMock,
    *,
    auth_token: str | None = None,
    on_server_push: Any = None,
) -> WebSocketCliChannel:
    """Create and connect a WebSocketCliChannel with mocked transport."""
    with patch("aiohttp.ClientSession", return_value=mock_session):
        ch = WebSocketCliChannel(
            auth_token=auth_token,
            on_server_push=on_server_push,
        )
        await ch.connect("ws://localhost:18900/ws")
        return ch


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_initial_state(self) -> None:
        ch = WebSocketCliChannel()
        assert ch._ws is None
        assert ch._session is None
        assert ch._auth_token is None
        assert ch._on_server_push is None
        assert ch._url is None
        assert ch._reader_task is None
        assert ch._pending == {}

    def test_initial_state_with_args(self) -> None:
        handler = lambda d: None
        ch = WebSocketCliChannel(auth_token="tok", on_server_push=handler)
        assert ch._auth_token == "tok"
        assert ch._on_server_push is handler


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_success_no_auth(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            assert ch._url == "ws://localhost:18900/ws"
            assert ch._ws is mock_ws
            assert ch._reader_task is not None
            assert not ch._reader_task.done()
            # No auth message sent
            assert len(mock_ws.sent) == 0
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_connect_success_with_auth(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session, auth_token="secret")
        try:
            assert ch._url == "ws://localhost:18900/ws"
            # Auth message was sent
            assert len(mock_ws.sent) == 1
            auth_msg = mock_ws.sent[0]
            assert auth_msg["method"] == "auth"
            assert auth_msg["params"]["auth_token"] == "secret"
            assert auth_msg["params"]["client_type"] == "cli"
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_connect_refused_raises_daemon_not_running(self) -> None:
        bad_session = AsyncMock()
        bad_session.ws_connect = AsyncMock(
            side_effect=aiohttp.ClientError("refused")
        )
        bad_session.close = AsyncMock()
        with patch("aiohttp.ClientSession", return_value=bad_session):
            ch = WebSocketCliChannel()
            with pytest.raises(DaemonNotRunningError, match="Cannot connect"):
                await ch.connect("ws://localhost:18900/ws")
        bad_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_os_error_raises_daemon_not_running(self) -> None:
        bad_session = AsyncMock()
        bad_session.ws_connect = AsyncMock(
            side_effect=OSError("connection refused")
        )
        bad_session.close = AsyncMock()
        with patch("aiohttp.ClientSession", return_value=bad_session):
            ch = WebSocketCliChannel()
            with pytest.raises(DaemonNotRunningError, match="Cannot connect"):
                await ch.connect("ws://localhost:18900/ws")

    @pytest.mark.asyncio
    async def test_connect_auth_failure_disconnects(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()
        # Auth auto-responds with error
        mock_session.ws_connect = AsyncMock(return_value=ws)

        def _auth_error(params: dict) -> None:
            # Instead of auto-responding with result, inject an error
            pass  # Don't auto-respond

        # Remove auto-response; inject error manually
        async def _bad_send(data: dict) -> None:
            ws.sent.append(data)
            if data.get("method") == "auth":
                ws._queue.put_nowait(
                    _MockWSMessage(
                        aiohttp.WSMsgType.TEXT,
                        json.dumps({
                            "jsonrpc": "2.0",
                            "id": data["id"],
                            "error": {"code": -32600, "message": "unauthorized"},
                        }),
                    )
                )

        ws.send_json = _bad_send

        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel(auth_token="bad_token")
            with pytest.raises(RuntimeError, match="unauthorized"):
                await ch.connect("ws://localhost:18900/ws")
        # Channel should be disconnected
        assert ch._ws is None
        assert ch._reader_task is None


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        await ch.disconnect()
        assert ch._ws is None
        assert ch._session is None
        assert ch._reader_task is None
        assert ch._pending == {}
        mock_session.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        ch = WebSocketCliChannel()
        await ch.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_disconnect_fails_pending_futures(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        # Manually add a pending future
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        ch._pending["test123"] = fut
        await ch.disconnect()
        # The reader task's finally already cleared pending, but if not,
        # disconnect's own loop would handle it
        assert ch._pending == {}

    @pytest.mark.asyncio
    async def test_disconnect_ws_close_error_swallowed(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()

        async def _close_fail() -> None:
            raise RuntimeError("close boom")

        ws.close = _close_fail
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        await ch.disconnect()
        assert ch._ws is None


# ---------------------------------------------------------------------------
# Request Methods
# ---------------------------------------------------------------------------


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_result(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.chat("hello")
            assert resp == {"text": "hi", "emotion": "happy"}
            # Verify sent message format
            msg = mock_ws.sent[-1]
            assert msg["jsonrpc"] == "2.0"
            assert msg["method"] == "chat"
            assert msg["params"] == {"text": "hello"}
            assert "id" in msg
        finally:
            await ch.disconnect()


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_get_status(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.get_status()
            assert resp == {"status": {}, "active_connections": 1}
        finally:
            await ch.disconnect()


class TestGetHeartbeatStatus:
    @pytest.mark.asyncio
    async def test_get_heartbeat_status(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.get_heartbeat_status()
            assert resp == {"status": {}}
        finally:
            await ch.disconnect()


class TestPing:
    @pytest.mark.asyncio
    async def test_ping(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.ping()
            assert resp == {"status": "pong"}
        finally:
            await ch.disconnect()


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_returns_list(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            sessions = await ch.list_sessions()
            assert sessions == [{"id": "s1"}]
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_list_sessions_empty(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()
        ws.set_response("list_sessions", lambda p: {"sessions": []})
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            sessions = await ch.list_sessions()
            assert sessions == []
        finally:
            await ch.disconnect()


class TestLoadSession:
    @pytest.mark.asyncio
    async def test_load_session_returns_count(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            count = await ch.load_session("session-abc")
            assert count == 5
            msg = mock_ws.sent[-1]
            assert msg["params"]["session_id"] == "session-abc"
        finally:
            await ch.disconnect()


class TestResetSession:
    @pytest.mark.asyncio
    async def test_reset_session_returns_count(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            count = await ch.reset_session()
            assert count == 3
        finally:
            await ch.disconnect()


class TestGetRuntimeInfo:
    @pytest.mark.asyncio
    async def test_get_runtime_info(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.get_runtime_info()
            assert resp == {"status": {}, "skills": []}
        finally:
            await ch.disconnect()


class TestStopDaemon:
    @pytest.mark.asyncio
    async def test_stop_daemon(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            resp = await ch.stop_daemon()
            assert resp == {"status": "stopping"}
        finally:
            await ch.disconnect()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRequestErrors:
    @pytest.mark.asyncio
    async def test_request_when_not_connected(self) -> None:
        ch = WebSocketCliChannel()
        with pytest.raises(RuntimeError, match="not connected"):
            await ch.chat("hello")

    @pytest.mark.asyncio
    async def test_request_timeout(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()
        # No auto-responses → request will time out
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            ch._REQUEST_TIMEOUT = 0.05
            with pytest.raises(TimeoutError, match="did not respond"):
                await ch.ping()
            # Pending should be cleaned up
            assert len(ch._pending) == 0
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_rpc_error_response(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()

        async def _send_error(data: dict) -> None:
            ws.sent.append(data)
            if data.get("method") == "chat":
                ws._queue.put_nowait(
                    _MockWSMessage(
                        aiohttp.WSMsgType.TEXT,
                        json.dumps({
                            "jsonrpc": "2.0",
                            "id": data["id"],
                            "error": {"code": -32602, "message": "empty text"},
                        }),
                    )
                )

        ws.send_json = _send_error
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            with pytest.raises(RuntimeError, match="empty text"):
                await ch.chat("")
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_rpc_error_default_message(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()

        async def _send_error(data: dict) -> None:
            ws.sent.append(data)
            ws._queue.put_nowait(
                _MockWSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": data["id"],
                        "error": {"code": -32603},
                    }),
                )
            )

        ws.send_json = _send_error
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            with pytest.raises(RuntimeError, match="RPC error"):
                await ch.ping()
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_connection_reset_during_request(
        self, mock_session: AsyncMock
    ) -> None:
        ws = MockWebSocket()
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")

        async def _send_and_close(data: dict) -> None:
            ws.sent.append(data)
            # Simulate connection close after sending
            ws._queue.put_nowait(None)

        ws.send_json = _send_and_close
        with pytest.raises(ConnectionResetError, match="connection lost"):
            await ch.ping()
        await ch.disconnect()


# ---------------------------------------------------------------------------
# Server Push
# ---------------------------------------------------------------------------


class TestServerPush:
    @pytest.mark.asyncio
    async def test_push_with_callback(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        received: list[dict] = []
        ch = await _connect(mock_session, on_server_push=received.append)
        try:
            # Inject a notification (no id)
            mock_ws.inject({
                "jsonrpc": "2.0",
                "method": "event.heartbeat.beat",
                "params": {"beat_count": 42},
            })
            # Give the reader loop a chance to process
            await asyncio.sleep(0.05)
            assert len(received) == 1
            assert received[0]["method"] == "event.heartbeat.beat"
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_push_without_callback_ignored(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            mock_ws.inject({
                "jsonrpc": "2.0",
                "method": "event.heartbeat.beat",
                "params": {"beat_count": 42},
            })
            await asyncio.sleep(0.05)
            # No crash, just logged and ignored
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_push_callback_exception_swallowed(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        def _bad_handler(data: dict) -> None:
            raise ValueError("handler boom")

        ch = await _connect(mock_session, on_server_push=_bad_handler)
        try:
            mock_ws.inject({
                "jsonrpc": "2.0",
                "method": "event.test",
                "params": {},
            })
            await asyncio.sleep(0.05)
            # No crash — exception swallowed
        finally:
            await ch.disconnect()


# ---------------------------------------------------------------------------
# Read Loop
# ---------------------------------------------------------------------------


class TestReadLoop:
    @pytest.mark.asyncio
    async def test_bad_json_skipped(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        try:
            mock_ws.inject_bad_json()
            # Then send a valid ping
            await ch.ping()
            # If we got here, the bad JSON was skipped and ping worked
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_closed_frame_stops_loop(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        mock_ws.inject_close()
        await asyncio.sleep(0.05)
        assert ch._reader_task is not None
        assert ch._reader_task.done()
        await ch.disconnect()

    @pytest.mark.asyncio
    async def test_error_frame_stops_loop(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        ch = await _connect(mock_session)
        mock_ws.inject_error()
        await asyncio.sleep(0.05)
        assert ch._reader_task is not None
        assert ch._reader_task.done()
        await ch.disconnect()

    @pytest.mark.asyncio
    async def test_unmatched_id_dispatched_as_push(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        """A message with an id that doesn't match any pending request
        is treated as a server-push notification."""
        received: list[dict] = []
        ch = await _connect(mock_session, on_server_push=received.append)
        try:
            mock_ws.inject({
                "jsonrpc": "2.0",
                "id": "nonexistent",
                "result": {"data": "orphan"},
            })
            await asyncio.sleep(0.05)
            assert len(received) == 1
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_result_missing_returns_empty_dict(
        self, mock_session: AsyncMock
    ) -> None:
        """If JSON-RPC response has no 'result' key, return empty dict."""
        ws = MockWebSocket()

        async def _send_no_result(data: dict) -> None:
            ws.sent.append(data)
            ws._queue.put_nowait(
                _MockWSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps({"jsonrpc": "2.0", "id": data["id"]}),
                )
            )

        ws.send_json = _send_no_result
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            resp = await ch.ping()
            assert resp == {}
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_done_future_not_set_again(
        self, mock_session: AsyncMock
    ) -> None:
        """If a future is already done when the response arrives, skip it."""
        ws = MockWebSocket()

        async def _send_double(data: dict) -> None:
            ws.sent.append(data)
            # Send the response twice
            for _ in range(2):
                ws._queue.put_nowait(
                    _MockWSMessage(
                        aiohttp.WSMsgType.TEXT,
                        json.dumps({
                            "jsonrpc": "2.0",
                            "id": data["id"],
                            "result": {"status": "pong"},
                        }),
                    )
                )

        ws.send_json = _send_double
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            # Should not crash despite duplicate response
            resp = await ch.ping()
            assert resp == {"status": "pong"}
            await asyncio.sleep(0.05)  # Let second msg be processed
        finally:
            await ch.disconnect()


# ---------------------------------------------------------------------------
# Timeout constant
# ---------------------------------------------------------------------------


class TestReadLoopDead:
    @pytest.mark.asyncio
    async def test_request_after_read_loop_dies(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        """M2/M7: request made after read loop exits fails immediately."""
        ch = await _connect(mock_session)
        mock_ws.inject_close()
        await asyncio.sleep(0.05)
        assert ch._reader_task is not None
        assert ch._reader_task.done()
        with pytest.raises(RuntimeError, match="not connected"):
            await ch.ping()
        await ch.disconnect()


class TestDoubleDisconnect:
    @pytest.mark.asyncio
    async def test_double_disconnect(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        """M8: calling disconnect() twice should not raise."""
        ch = await _connect(mock_session)
        await ch.disconnect()
        await ch.disconnect()


class TestConcurrentRequests:
    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, mock_ws: MockWebSocket, mock_session: AsyncMock
    ) -> None:
        """M9: multiple concurrent requests should work correctly."""
        ch = await _connect(mock_session)
        try:
            results = await asyncio.gather(
                ch.ping(), ch.get_status(), ch.get_heartbeat_status()
            )
            assert results[0] == {"status": "pong"}
            assert results[1] == {"status": {}, "active_connections": 1}
            assert results[2] == {"status": {}}
        finally:
            await ch.disconnect()


class TestMalformedError:
    @pytest.mark.asyncio
    async def test_non_dict_error_field(
        self, mock_session: AsyncMock
    ) -> None:
        """M10: non-dict error field in response should not crash reader."""
        ws = MockWebSocket()

        async def _send_bad_error(data: dict) -> None:
            ws.sent.append(data)
            ws._queue.put_nowait(
                _MockWSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": data["id"],
                        "error": "plain string error",
                    }),
                )
            )

        ws.send_json = _send_bad_error
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            with pytest.raises(RuntimeError, match="plain string error"):
                await ch.ping()
        finally:
            await ch.disconnect()

    @pytest.mark.asyncio
    async def test_null_error_field(
        self, mock_session: AsyncMock
    ) -> None:
        """M10: null error field should use default message."""
        ws = MockWebSocket()

        async def _send_null_error(data: dict) -> None:
            ws.sent.append(data)
            ws._queue.put_nowait(
                _MockWSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": data["id"],
                        "error": None,
                    }),
                )
            )

        ws.send_json = _send_null_error
        mock_session.ws_connect = AsyncMock(return_value=ws)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        try:
            with pytest.raises(RuntimeError, match="RPC error"):
                await ch.ping()
        finally:
            await ch.disconnect()


class TestSessionCloseFailure:
    @pytest.mark.asyncio
    async def test_session_close_error_swallowed(
        self, mock_ws: MockWebSocket
    ) -> None:
        """L2: session.close() error should not prevent pending cleanup."""
        bad_session = AsyncMock()
        bad_session.ws_connect = AsyncMock(return_value=mock_ws)
        bad_session.close = AsyncMock(side_effect=RuntimeError("session boom"))
        with patch("aiohttp.ClientSession", return_value=bad_session):
            ch = WebSocketCliChannel()
            await ch.connect("ws://localhost:18900/ws")
        await ch.disconnect()
        assert ch._session is None
        assert ch._pending == {}


class TestTimeout:
    def test_default_timeout(self) -> None:
        assert WebSocketCliChannel._REQUEST_TIMEOUT == 120.0


# ===========================================================================
# Legacy CliChannel (Unix socket) tests
# ===========================================================================


class _MockStreamWriter:
    """Minimal mock for asyncio.StreamWriter."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        pass


class _MockStreamReader:
    """Minimal mock for asyncio.StreamReader backed by a queue."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, data: dict[str, Any]) -> None:
        """Enqueue a JSON response line."""
        self._queue.put_nowait(json.dumps(data).encode() + b"\n")

    def feed_eof(self) -> None:
        """Enqueue an empty read (EOF)."""
        self._queue.put_nowait(b"")

    def feed_raw(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    async def readline(self) -> bytes:
        return await self._queue.get()


class TestLegacyCliChannelConstructor:
    def test_initial_state(self) -> None:
        ch = CliChannel()
        assert ch._reader is None
        assert ch._writer is None
        assert ch._auth_token is None
        assert ch._on_server_push is None

    def test_with_args(self) -> None:
        handler = lambda msg: None
        ch = CliChannel(auth_token="tok", on_server_push=handler)
        assert ch._auth_token == "tok"
        assert ch._on_server_push is handler


class TestLegacyCliChannelConnect:
    @pytest.mark.asyncio
    async def test_missing_socket_raises(self, tmp_path: Path) -> None:
        ch = CliChannel()
        sock = tmp_path / "missing.sock"
        with pytest.raises(DaemonNotRunningError, match="not found"):
            await ch.connect(sock)

    @pytest.mark.asyncio
    async def test_connection_refused(self, tmp_path: Path) -> None:
        sock = tmp_path / "test.sock"
        sock.touch()  # file exists but is not a real socket
        ch = CliChannel()
        with pytest.raises(DaemonNotRunningError, match="Cannot connect"):
            await ch.connect(sock)

    @pytest.mark.asyncio
    async def test_successful_connect(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        sock_path = MagicMock(spec=Path)
        sock_path.exists.return_value = True

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            ch = CliChannel()
            await ch.connect(sock_path)

        assert ch._reader is reader
        assert ch._writer is writer


class TestLegacyCliChannelDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self) -> None:
        ch = CliChannel()
        ch._reader = _MockStreamReader()
        ch._writer = _MockStreamWriter()

        await ch.disconnect()
        assert ch._reader is None
        assert ch._writer is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        ch = CliChannel()
        await ch.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_disconnect_close_error_swallowed(self) -> None:
        ch = CliChannel()
        writer = MagicMock()
        writer.close.side_effect = OSError("close failed")
        writer.wait_closed = AsyncMock(side_effect=OSError("wait failed"))
        ch._writer = writer
        ch._reader = _MockStreamReader()

        await ch.disconnect()
        assert ch._writer is None


class TestLegacyCliChannelRequest:
    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        ch = CliChannel()
        with pytest.raises(RuntimeError, match="not connected"):
            await ch.chat("hello")

    @pytest.mark.asyncio
    async def test_chat_sends_and_receives(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()
        ch._reader = reader
        ch._writer = writer

        async def _do_chat():
            return await ch.chat("hello")

        # Schedule the response before the request reads
        async def _feed_response():
            await asyncio.sleep(0.01)
            # Read what was written to extract req_id
            sent_line = writer.written[0].decode()
            msg = json.loads(sent_line)
            reader.feed({"req_id": msg["req_id"], "text": "Hi!", "emotion": "happy"})

        result, _ = await asyncio.gather(_do_chat(), _feed_response())
        assert result["text"] == "Hi!"

    @pytest.mark.asyncio
    async def test_auth_token_included(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel(auth_token="secret")
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            assert msg["auth_token"] == "secret"
            reader.feed({"req_id": msg["req_id"], "status": "pong"})

        result, _ = await asyncio.gather(_do_ping(), _feed())
        assert result["status"] == "pong"

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()
        ch._reader = reader
        ch._writer = writer
        ch._REQUEST_TIMEOUT = 0.05  # Very short timeout

        with pytest.raises(TimeoutError):
            await ch.ping()

    @pytest.mark.asyncio
    async def test_connection_reset_on_eof(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed_eof():
            await asyncio.sleep(0.01)
            reader.feed_eof()

        with pytest.raises(ConnectionResetError, match="closed"):
            await asyncio.gather(_do_ping(), _feed_eof())


class TestLegacyCliChannelReadResponse:
    @pytest.mark.asyncio
    async def test_bad_json_skipped(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            # Send bad JSON first
            reader.feed_raw(b"not valid json\n")
            # Then send the real response
            reader.feed({"req_id": msg["req_id"], "status": "pong"})

        result, _ = await asyncio.gather(_do_ping(), _feed())
        assert result["status"] == "pong"

    @pytest.mark.asyncio
    async def test_server_push_dispatched(self) -> None:
        pushes: list[dict] = []
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel(on_server_push=lambda msg: pushes.append(msg))
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            # Send a push message (different req_id)
            reader.feed({"type": "heartbeat_update", "beat_count": 5})
            # Then send the real response
            reader.feed({"req_id": msg["req_id"], "status": "pong"})

        result, _ = await asyncio.gather(_do_ping(), _feed())
        assert result["status"] == "pong"
        assert len(pushes) == 1
        assert pushes[0]["type"] == "heartbeat_update"

    @pytest.mark.asyncio
    async def test_server_push_without_handler(self) -> None:
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()  # No push handler
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            # Send a push message (ignored)
            reader.feed({"type": "heartbeat_update"})
            # Then send the real response
            reader.feed({"req_id": msg["req_id"], "status": "pong"})

        result, _ = await asyncio.gather(_do_ping(), _feed())
        assert result["status"] == "pong"

    @pytest.mark.asyncio
    async def test_push_handler_error_swallowed(self) -> None:
        def bad_handler(msg: dict) -> None:
            raise ValueError("handler crash")

        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel(on_server_push=bad_handler)
        ch._reader = reader
        ch._writer = writer

        async def _do_ping():
            return await ch.ping()

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            reader.feed({"type": "bad_push"})
            reader.feed({"req_id": msg["req_id"], "status": "pong"})

        result, _ = await asyncio.gather(_do_ping(), _feed())
        assert result["status"] == "pong"


class TestLegacyCliChannelMethods:
    """Test all high-level request methods."""

    @pytest.fixture
    def _channel(self):
        """Return a connected channel with reader/writer mocks."""
        reader = _MockStreamReader()
        writer = _MockStreamWriter()
        ch = CliChannel()
        ch._reader = reader
        ch._writer = writer
        return ch, reader, writer

    async def _request_helper(self, ch, reader, writer, coro, response_data):
        """Helper to run a request and feed the response."""
        async def _do():
            return await coro

        async def _feed():
            await asyncio.sleep(0.01)
            msg = json.loads(writer.written[0].decode())
            response_data["req_id"] = msg["req_id"]
            reader.feed(response_data)

        result, _ = await asyncio.gather(_do(), _feed())
        return result

    @pytest.mark.asyncio
    async def test_get_status(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.get_status(),
            {"status": {"name": "Gwenn"}},
        )
        assert result["status"]["name"] == "Gwenn"

    @pytest.mark.asyncio
    async def test_get_heartbeat_status(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.get_heartbeat_status(),
            {"status": {"running": True}},
        )
        assert result["status"]["running"] is True

    @pytest.mark.asyncio
    async def test_list_sessions(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.list_sessions(),
            {"sessions": [{"id": "s1"}, {"id": "s2"}]},
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.list_sessions(),
            {},  # No "sessions" key
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_load_session(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.load_session("s1"),
            {"message_count": 10},
        )
        assert result == 10

    @pytest.mark.asyncio
    async def test_reset_session(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.reset_session(),
            {"cleared_messages": 5},
        )
        assert result == 5

    @pytest.mark.asyncio
    async def test_get_runtime_info(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.get_runtime_info(),
            {"model": "test"},
        )
        assert result["model"] == "test"

    @pytest.mark.asyncio
    async def test_stop_daemon(self, _channel) -> None:
        ch, reader, writer = _channel
        result = await self._request_helper(
            ch, reader, writer, ch.stop_daemon(),
            {"status": "stopping"},
        )
        assert result["status"] == "stopping"
