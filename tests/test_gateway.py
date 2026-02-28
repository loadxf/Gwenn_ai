"""Tests for gwenn.gateway — GatewayServer WebSocket + HTTP endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from gwenn.events import EventBus, GwennEvent, HeartbeatBeatEvent
from gwenn.gateway import ClientConnection, GatewayServer
from gwenn.memory.session_store import SessionStore
from gwenn.rpc import RequestRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.respond = AsyncMock(return_value="Hello!")
    agent.affect_state.current_emotion.value = "curious"
    agent.status = {"running": True}
    agent.heartbeat = MagicMock()
    agent.heartbeat.status = {"running": True, "beat_count": 10}
    agent.heartbeat.notify_user_activity = MagicMock()
    agent.skill_registry = None
    agent._mcp_client = None
    agent.tool_registry = None
    return agent


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def respond_lock() -> asyncio.Lock:
    return asyncio.Lock()


@pytest.fixture
def router(
    mock_agent: MagicMock,
    session_store: SessionStore,
    event_bus: EventBus,
    respond_lock: asyncio.Lock,
) -> RequestRouter:
    return RequestRouter(
        agent=mock_agent,
        session_store=session_store,
        event_bus=event_bus,
        respond_lock=respond_lock,
        active_connections_getter=lambda: 0,
    )


@pytest.fixture
def daemon_config() -> MagicMock:
    cfg = MagicMock()
    cfg.max_connections = 10
    cfg.connection_timeout = 300.0
    cfg.gateway_host = "127.0.0.1"
    cfg.gateway_port = 18900
    cfg.mcp_server_enabled = False
    cfg.a2a_enabled = False
    return cfg


@pytest.fixture
def gateway(
    daemon_config: MagicMock,
    router: RequestRouter,
    event_bus: EventBus,
    session_store: SessionStore,
) -> GatewayServer:
    return GatewayServer(
        config=daemon_config,
        router=router,
        event_bus=event_bus,
        session_store=session_store,
    )


@pytest.fixture
def gateway_with_auth(
    daemon_config: MagicMock,
    router: RequestRouter,
    event_bus: EventBus,
    session_store: SessionStore,
) -> GatewayServer:
    return GatewayServer(
        config=daemon_config,
        router=router,
        event_bus=event_bus,
        session_store=session_store,
        auth_token="test-secret",
    )


@pytest.fixture
def gateway_with_stubs(
    router: RequestRouter,
    event_bus: EventBus,
    session_store: SessionStore,
) -> GatewayServer:
    cfg = MagicMock()
    cfg.max_connections = 10
    cfg.connection_timeout = 300.0
    cfg.gateway_host = "127.0.0.1"
    cfg.gateway_port = 18900
    cfg.mcp_server_enabled = True
    cfg.a2a_enabled = True
    return GatewayServer(
        config=cfg,
        router=router,
        event_bus=event_bus,
        session_store=session_store,
    )


async def _create_test_client(gw: GatewayServer) -> TestClient:
    """Create an aiohttp test client for a GatewayServer."""
    # Build the app without starting the full server
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/ws", gw._handle_ws)
    app.router.add_get("/health", gw._handle_health)
    if gw._config.mcp_server_enabled:
        app.router.add_post("/mcp", gw._handle_mcp_stub)
        app.router.add_get("/mcp", gw._handle_mcp_stub)
    if gw._config.a2a_enabled:
        app.router.add_get("/.well-known/agent.json", gw._handle_a2a_card)
    gw._started_at = time.monotonic()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# ClientConnection dataclass tests
# ---------------------------------------------------------------------------


class TestClientConnection:
    def test_defaults(self) -> None:
        ws = MagicMock()
        conn = ClientConnection(conn_id="abc123", ws=ws)
        assert conn.conn_id == "abc123"
        assert conn.client_type == "cli"
        assert conn.authenticated is False
        assert conn.session_id == ""
        assert conn.history == []
        assert conn.can_approve_tools is False
        assert conn.started_at > 0
        assert conn.last_activity > 0

    def test_custom_fields(self) -> None:
        ws = MagicMock()
        conn = ClientConnection(
            conn_id="xyz",
            ws=ws,
            client_type="web",
            authenticated=True,
            session_id="s1",
            can_approve_tools=True,
        )
        assert conn.client_type == "web"
        assert conn.authenticated is True
        assert conn.session_id == "s1"
        assert conn.can_approve_tools is True


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert "uptime" in data
            assert data["connections"] == 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_uptime_increases(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            resp = await client.get("/health")
            data = await resp.json()
            assert data["uptime"] >= 0.0
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# MCP stub tests
# ---------------------------------------------------------------------------


class TestMCPStub:
    @pytest.mark.asyncio
    async def test_mcp_post_returns_501(self, gateway_with_stubs: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_stubs)
        try:
            resp = await client.post("/mcp", json={})
            assert resp.status == 501
            data = await resp.json()
            assert "not implemented" in data["error"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_mcp_get_returns_501(self, gateway_with_stubs: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_stubs)
        try:
            resp = await client.get("/mcp")
            assert resp.status == 501
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_mcp_not_registered_when_disabled(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            resp = await client.post("/mcp", json={})
            assert resp.status == 404
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# A2A Agent Card tests
# ---------------------------------------------------------------------------


class TestA2ACard:
    @pytest.mark.asyncio
    async def test_agent_card_returns_json(self, gateway_with_stubs: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_stubs)
        try:
            resp = await client.get("/.well-known/agent.json")
            assert resp.status == 200
            data = await resp.json()
            assert data["name"] == "Gwenn"
            assert "capabilities" in data
            assert "url" in data
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_agent_card_not_registered_when_disabled(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            resp = await client.get("/.well-known/agent.json")
            assert resp.status == 404
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------


class TestWebSocketPing:
    @pytest.mark.asyncio
    async def test_ws_ping(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "pong"
                assert resp["id"] == "r1"
        finally:
            await client.close()


class TestWebSocketChat:
    @pytest.mark.asyncio
    async def test_ws_chat(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "chat",
                    "params": {"text": "Hello"},
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["text"] == "Hello!"
                assert resp["result"]["emotion"] == "curious"
        finally:
            await client.close()


class TestWebSocketStatus:
    @pytest.mark.asyncio
    async def test_ws_status(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "status",
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert "status" in resp["result"]
        finally:
            await client.close()


class TestWebSocketUnknownMethod:
    @pytest.mark.asyncio
    async def test_ws_unknown_method(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "nonexistent",
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert "error" in resp
                assert "unknown method" in resp["error"]["message"]
        finally:
            await client.close()


class TestWebSocketBadJSON:
    @pytest.mark.asyncio
    async def test_ws_invalid_json(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_str("not json{{{")
                resp = await ws.receive_json()
                assert resp["error"]["code"] == -32700  # PARSE_ERROR

                # Should still be able to send valid messages after
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r2",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "pong"
        finally:
            await client.close()


class TestWebSocketNotObject:
    @pytest.mark.asyncio
    async def test_ws_non_object(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_str('"just a string"')
                resp = await ws.receive_json()
                assert resp["error"]["code"] == -32600  # INVALID_REQUEST
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestWebSocketAuth:
    @pytest.mark.asyncio
    async def test_auth_required(self, gateway_with_auth: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_auth)
        try:
            async with client.ws_connect("/ws") as ws:
                # Should reject without auth token
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["error"]["message"] == "unauthorized"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_auth_success(self, gateway_with_auth: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_auth)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {
                        "auth_token": "test-secret",
                        "client_type": "web",
                    },
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "authenticated"

                # Now should work
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r2",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "pong"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_auth_bad_token(self, gateway_with_auth: GatewayServer) -> None:
        client = await _create_test_client(gateway_with_auth)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {"auth_token": "wrong-token"},
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["error"]["message"] == "unauthorized"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_auth_max_failures_disconnects(
        self, gateway_with_auth: GatewayServer
    ) -> None:
        client = await _create_test_client(gateway_with_auth)
        try:
            async with client.ws_connect("/ws") as ws:
                # Send 3 bad auth attempts
                for i in range(3):
                    await ws.send_json({
                        "jsonrpc": "2.0",
                        "method": "ping",
                        "id": f"r{i}",
                    })
                    resp = await ws.receive_json()
                    assert resp["error"]["message"] == "unauthorized"

                # Connection should be closed by server
                msg = await ws.receive()
                assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_no_auth_required_when_no_token(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "pong"
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Connection limit tests
# ---------------------------------------------------------------------------


class TestConnectionLimit:
    @pytest.mark.asyncio
    async def test_connection_limit_enforced(
        self,
        router: RequestRouter,
        event_bus: EventBus,
        session_store: SessionStore,
    ) -> None:
        cfg = MagicMock()
        cfg.max_connections = 1
        cfg.connection_timeout = 300.0
        cfg.mcp_server_enabled = False
        cfg.a2a_enabled = False
        gw = GatewayServer(
            config=cfg,
            router=router,
            event_bus=event_bus,
            session_store=session_store,
        )
        client = await _create_test_client(gw)
        try:
            # First connection should work
            ws1 = await client.ws_connect("/ws")
            await ws1.send_json({
                "jsonrpc": "2.0",
                "method": "ping",
                "id": "r1",
            })
            resp = await ws1.receive_json()
            assert resp["result"]["status"] == "pong"

            # Second connection should be rejected
            ws2 = await client.ws_connect("/ws")
            resp2 = await ws2.receive_json()
            assert resp2["error"]["message"] == "max connections reached"

            await ws1.close()
            await ws2.close()
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Auth method (connection metadata) tests
# ---------------------------------------------------------------------------


class TestAuthMethod:
    @pytest.mark.asyncio
    async def test_auth_sets_client_type(self, gateway: GatewayServer) -> None:
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "auth",
                    "params": {"client_type": "mobile"},
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "authenticated"

                # Verify connection state was updated
                conns = list(gateway._connections.values())
                assert len(conns) == 1
                assert conns[0].client_type == "mobile"
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Server lifecycle tests
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, gateway: GatewayServer) -> None:
        await gateway.start("127.0.0.1", 0)  # port 0 = random
        try:
            assert gateway.active_connection_count == 0
            assert gateway._timeout_task is not None
        finally:
            await gateway.stop()
        assert gateway._site is None
        assert gateway._runner is None
        assert gateway._timeout_task is None


# ---------------------------------------------------------------------------
# Check auth helper tests
# ---------------------------------------------------------------------------


class TestCheckAuth:
    def test_no_auth_configured(self, gateway: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        assert gateway._check_auth({}, conn) is True

    def test_auth_required_missing_token(self, gateway_with_auth: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        assert gateway_with_auth._check_auth({}, conn) is False

    def test_auth_valid_token(self, gateway_with_auth: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        msg = {"params": {"auth_token": "test-secret"}}
        assert gateway_with_auth._check_auth(msg, conn) is True

    def test_auth_invalid_token(self, gateway_with_auth: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        msg = {"params": {"auth_token": "wrong"}}
        assert gateway_with_auth._check_auth(msg, conn) is False

    def test_auth_already_authenticated(self, gateway_with_auth: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock(), authenticated=True)
        assert gateway_with_auth._check_auth({}, conn) is True

    def test_auth_non_string_token(self, gateway_with_auth: GatewayServer) -> None:
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        msg = {"params": {"auth_token": 12345}}
        assert gateway_with_auth._check_auth(msg, conn) is False


# ---------------------------------------------------------------------------
# Event push tests
# ---------------------------------------------------------------------------


class TestEventPush:
    @pytest.mark.asyncio
    async def test_push_event_creates_notification(self, gateway: GatewayServer) -> None:
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(conn_id="c1", ws=ws, subscriptions={"*"})

        event = GwennEvent(event_type="test.event")
        gateway._push_event(conn, event)
        await asyncio.sleep(0.01)  # Let the created task run
        ws.send_json.assert_awaited_once()

    def test_push_event_skips_closed(self, gateway: GatewayServer) -> None:
        ws = MagicMock()
        ws.closed = True
        conn = ClientConnection(conn_id="c1", ws=ws)

        event = GwennEvent(event_type="test.event")
        gateway._push_event(conn, event)
        # Should not raise or create a task

    @pytest.mark.asyncio
    async def test_safe_send_success(self, gateway: GatewayServer) -> None:
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(conn_id="c1", ws=ws)

        await gateway._safe_send(conn, {"test": True})
        ws.send_json.assert_awaited_once_with({"test": True})

    @pytest.mark.asyncio
    async def test_safe_send_closed(self, gateway: GatewayServer) -> None:
        ws = MagicMock()
        ws.closed = True
        conn = ClientConnection(conn_id="c1", ws=ws)

        await gateway._safe_send(conn, {"test": True})
        # Should not attempt send

    @pytest.mark.asyncio
    async def test_safe_send_error_swallowed(self, gateway: GatewayServer) -> None:
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock(side_effect=ConnectionResetError)
        conn = ClientConnection(conn_id="c1", ws=ws)

        # Should not raise
        await gateway._safe_send(conn, {"test": True})


# ---------------------------------------------------------------------------
# active_connection_count property
# ---------------------------------------------------------------------------


class TestActiveConnectionCount:
    def test_empty(self, gateway: GatewayServer) -> None:
        assert gateway.active_connection_count == 0

    def test_with_connections(self, gateway: GatewayServer) -> None:
        gateway._connections["c1"] = MagicMock()
        gateway._connections["c2"] = MagicMock()
        assert gateway.active_connection_count == 2


# ---------------------------------------------------------------------------
# Params validation tests
# ---------------------------------------------------------------------------


class TestParamsValidation:
    @pytest.mark.asyncio
    async def test_ws_non_dict_params(self, gateway: GatewayServer) -> None:
        """Non-dict params should return INVALID_REQUEST, not crash."""
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "chat",
                    "params": ["text", "hello"],
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["error"]["code"] == -32600  # INVALID_REQUEST
                assert "params must be an object" in resp["error"]["message"]

                # Connection should still work after
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r2",
                })
                resp = await ws.receive_json()
                assert resp["result"]["status"] == "pong"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_ws_null_params(self, gateway: GatewayServer) -> None:
        """null params should return INVALID_REQUEST."""
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_str('{"jsonrpc":"2.0","method":"ping","params":null,"id":"r1"}')
                resp = await ws.receive_json()
                assert resp["error"]["code"] == -32600
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Session persistence on disconnect
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_session_saved_on_disconnect(
        self, gateway: GatewayServer, mock_agent: MagicMock, session_store: SessionStore
    ) -> None:
        """Session should be saved when a connection with history disconnects."""

        async def respond_and_record(text: str, *, conversation_history: list, **kw: Any) -> str:
            conversation_history.append({"role": "user", "content": text})
            conversation_history.append({"role": "assistant", "content": "Hello!"})
            return "Hello!"

        mock_agent.respond = respond_and_record

        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "chat",
                    "params": {"text": "Hello"},
                    "id": "r1",
                })
                resp = await ws.receive_json()
                assert resp["result"]["text"] == "Hello!"

            # After disconnect, session should be saved
            sessions = session_store.list_sessions()
            assert len(sessions) >= 1
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_no_session_saved_empty_history(
        self, gateway: GatewayServer, session_store: SessionStore
    ) -> None:
        """No session should be saved when history is empty."""
        client = await _create_test_client(gateway)
        try:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": "r1",
                })
                await ws.receive_json()

            # No chat messages = no session saved
            sessions = session_store.list_sessions()
            assert len(sessions) == 0
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Check auth with non-dict params
# ---------------------------------------------------------------------------


class TestCheckAuthEdgeCases:
    def test_check_auth_null_params(self, gateway_with_auth: GatewayServer) -> None:
        """params=null should not crash _check_auth."""
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        msg: dict[str, Any] = {"params": None}
        assert gateway_with_auth._check_auth(msg, conn) is False

    def test_check_auth_list_params(self, gateway_with_auth: GatewayServer) -> None:
        """params=[] should not crash _check_auth."""
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        msg: dict[str, Any] = {"params": [1, 2, 3]}
        assert gateway_with_auth._check_auth(msg, conn) is False

    def test_check_auth_missing_params(self, gateway_with_auth: GatewayServer) -> None:
        """No params key should not crash _check_auth."""
        conn = ClientConnection(conn_id="c1", ws=MagicMock())
        assert gateway_with_auth._check_auth({}, conn) is False


# ---------------------------------------------------------------------------
# Push task tracking
# ---------------------------------------------------------------------------


class TestPushTaskTracking:
    @pytest.mark.asyncio
    async def test_push_tasks_tracked(self, gateway: GatewayServer) -> None:
        """Push tasks should be tracked in _push_tasks set."""
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(conn_id="c1", ws=ws, subscriptions={"*"})

        event = GwennEvent(event_type="test.event")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 1
        await asyncio.sleep(0.01)  # Let task complete
        assert len(gateway._push_tasks) == 0  # Auto-removed on completion


# ---------------------------------------------------------------------------
# Event subscription filtering
# ---------------------------------------------------------------------------


class TestEventSubscriptions:
    def test_push_event_skips_no_subscriptions(self, gateway: GatewayServer) -> None:
        """Clients with no subscriptions receive nothing."""
        ws = MagicMock()
        ws.closed = False
        conn = ClientConnection(conn_id="c1", ws=ws)
        assert not conn.subscriptions

        event = GwennEvent(event_type="heartbeat.beat")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 0

    @pytest.mark.asyncio
    async def test_push_event_matches_wildcard(self, gateway: GatewayServer) -> None:
        """Subscription '*' matches all events."""
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(conn_id="c1", ws=ws, subscriptions={"*"})

        event = GwennEvent(event_type="anything.at.all")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 1
        await asyncio.sleep(0.01)
        ws.send_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_push_event_matches_pattern(self, gateway: GatewayServer) -> None:
        """Subscription pattern 'heartbeat.*' matches heartbeat events."""
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(conn_id="c1", ws=ws, subscriptions={"heartbeat.*"})

        event = GwennEvent(event_type="heartbeat.beat")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 1
        await asyncio.sleep(0.01)
        ws.send_json.assert_awaited_once()

    def test_push_event_no_match(self, gateway: GatewayServer) -> None:
        """Non-matching patterns don't forward events."""
        ws = MagicMock()
        ws.closed = False
        conn = ClientConnection(conn_id="c1", ws=ws, subscriptions={"affect.*"})

        event = GwennEvent(event_type="heartbeat.beat")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 0

    @pytest.mark.asyncio
    async def test_push_event_multiple_patterns(self, gateway: GatewayServer) -> None:
        """Multiple subscription patterns — event matches any."""
        ws = MagicMock()
        ws.closed = False
        ws.send_json = AsyncMock()
        conn = ClientConnection(
            conn_id="c1", ws=ws,
            subscriptions={"heartbeat.*", "affect.*"},
        )

        event = GwennEvent(event_type="affect.emotion.changed")
        gateway._push_event(conn, event)
        assert len(gateway._push_tasks) == 1
        await asyncio.sleep(0.01)
        ws.send_json.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tool approval
# ---------------------------------------------------------------------------


class TestToolApproval:
    def test_resolve_approval_no_pending(self, gateway: GatewayServer) -> None:
        """Resolving non-existent approval returns False."""
        assert gateway._resolve_approval("nonexistent", "allow", "c1") is False

    @pytest.mark.asyncio
    async def test_resolve_approval_success(self, gateway: GatewayServer) -> None:
        """Resolving a pending approval sets the future."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        gateway._pending_approvals["test123"] = future

        result = gateway._resolve_approval("test123", "allow", "c1")
        assert result is True
        assert future.done()
        assert future.result() == "allow"

    @pytest.mark.asyncio
    async def test_resolve_approval_already_done(self, gateway: GatewayServer) -> None:
        """Resolving an already-resolved approval returns False."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        future.set_result("deny")
        gateway._pending_approvals["test123"] = future

        result = gateway._resolve_approval("test123", "allow", "c1")
        assert result is False

    @pytest.mark.asyncio
    async def test_request_approval_no_approvers(self, gateway: GatewayServer) -> None:
        """request_approval returns False when no approver-capable clients."""
        result = await gateway.request_approval(
            tool_name="shell", tool_input={"cmd": "ls"}, reason="test",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_request_approval_approved(self, gateway: GatewayServer) -> None:
        """request_approval returns True when client approves."""
        ws = MagicMock()
        ws.closed = False
        conn = ClientConnection(
            conn_id="c1", ws=ws, can_approve_tools=True,
        )
        gateway._connections["c1"] = conn

        async def approve_later():
            await asyncio.sleep(0.01)
            # Resolve the pending approval
            for aid, fut in list(gateway._pending_approvals.items()):
                if not fut.done():
                    gateway._resolve_approval(aid, "allow", "c1")

        task = asyncio.create_task(approve_later())
        result = await gateway.request_approval(
            tool_name="shell", tool_input={}, reason="test", timeout=2.0,
        )
        await task
        assert result is True

    @pytest.mark.asyncio
    async def test_request_approval_denied(self, gateway: GatewayServer) -> None:
        """request_approval returns False when client denies."""
        ws = MagicMock()
        ws.closed = False
        conn = ClientConnection(
            conn_id="c1", ws=ws, can_approve_tools=True,
        )
        gateway._connections["c1"] = conn

        async def deny_later():
            await asyncio.sleep(0.01)
            for aid, fut in list(gateway._pending_approvals.items()):
                if not fut.done():
                    gateway._resolve_approval(aid, "deny", "c1")

        task = asyncio.create_task(deny_later())
        result = await gateway.request_approval(
            tool_name="shell", tool_input={}, reason="test", timeout=2.0,
        )
        await task
        assert result is False

    @pytest.mark.asyncio
    async def test_request_approval_timeout(self, gateway: GatewayServer) -> None:
        """request_approval returns False on timeout."""
        ws = MagicMock()
        ws.closed = False
        conn = ClientConnection(
            conn_id="c1", ws=ws, can_approve_tools=True,
        )
        gateway._connections["c1"] = conn

        result = await gateway.request_approval(
            tool_name="shell", tool_input={}, reason="test", timeout=0.05,
        )
        assert result is False
        # Pending approval should be cleaned up
        assert len(gateway._pending_approvals) == 0
