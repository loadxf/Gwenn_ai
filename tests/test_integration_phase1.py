"""Integration tests for Phase 1 — cross-module interactions.

These tests verify that the new Phase 1 modules work together correctly:
- WebSocketCliChannel <-> GatewayServer round-trip
- GatewayServer + EventBus live push
- RequestRouter + EventBus event emission
- Gateway session persistence round-trip
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from gwenn.channels.cli_channel import DaemonNotRunningError, WebSocketCliChannel
from gwenn.events import EventBus, HeartbeatBeatEvent, UserActivityEvent, create_event_bus
from gwenn.gateway import GatewayServer
from gwenn.memory.session_store import SessionStore
from gwenn.rpc import RequestRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_daemon_config(**overrides: Any) -> SimpleNamespace:
    """Minimal DaemonConfig namespace for the gateway."""
    defaults = dict(
        max_connections=10,
        connection_timeout=30.0,
        gateway_host="127.0.0.1",
        gateway_port=0,  # let OS pick a free port
        mcp_server_enabled=False,
        a2a_enabled=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_agent() -> SimpleNamespace:
    """Minimal fake agent that can handle chat and status."""
    status_dict = {
        "name": "Gwenn",
        "emotion": "happy",
        "valence": 0.6,
        "arousal": 0.4,
        "total_interactions": 5,
        "uptime_seconds": 42.0,
    }
    agent = SimpleNamespace(
        respond=AsyncMock(return_value="Hello back!"),
        status=status_dict,
        identity=SimpleNamespace(
            name="Gwenn",
            total_interactions=5,
            total_heartbeats=100,
            uptime_seconds=42.0,
        ),
        affect_state=SimpleNamespace(
            current_emotion=SimpleNamespace(value="happy"),
            dimensions=SimpleNamespace(valence=0.6, arousal=0.4),
        ),
        config=SimpleNamespace(
            claude=SimpleNamespace(model="test-model"),
            memory=SimpleNamespace(data_dir="/tmp/test"),
        ),
        resilience=SimpleNamespace(status="normal"),
        engine=SimpleNamespace(
            usage=SimpleNamespace(total_api_calls=10, total_input_tokens=1000, total_output_tokens=500),
        ),
        tool_registry=None,
        _mcp_client=None,
        skill_registry=SimpleNamespace(all_skills=MagicMock(return_value=[])),
        heartbeat=None,
    )
    return agent


def _build_router(
    agent: Any,
    event_bus: EventBus,
    session_store: SessionStore,
    *,
    shutdown_callback: Any = None,
) -> RequestRouter:
    """Build a RequestRouter with the given dependencies."""
    return RequestRouter(
        agent=agent,
        session_store=session_store,
        event_bus=event_bus,
        respond_lock=asyncio.Lock(),
        shutdown_callback=shutdown_callback,
    )


async def _start_gateway(
    config: Any,
    router: RequestRouter,
    event_bus: EventBus,
    session_store: SessionStore,
    auth_token: str | None = None,
) -> tuple[GatewayServer, int]:
    """Start a gateway on a free port and return (server, port)."""
    server = GatewayServer(
        config=config,
        router=router,
        event_bus=event_bus,
        session_store=session_store,
        auth_token=auth_token,
    )
    await server.start("127.0.0.1", 0)
    # Extract the actual port from the running site
    assert server._site is not None
    sockets = server._site._server.sockets
    port = sockets[0].getsockname()[1]
    return server, port


# ===========================================================================
# WebSocketCliChannel <-> GatewayServer round-trip
# ===========================================================================


class TestClientGatewayRoundTrip:
    """End-to-end: real WebSocketCliChannel → real GatewayServer → RequestRouter."""

    @pytest.mark.asyncio
    async def test_ping_round_trip(self, tmp_path):
        """Client pings the gateway and gets pong back."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                result = await client.ping()
                assert result.get("status") == "pong"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_chat_round_trip(self, tmp_path):
        """Client sends a chat message and receives a response."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                result = await client.chat("Hello Gwenn!")
                assert "text" in result
                assert result["text"] == "Hello back!"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_status_round_trip(self, tmp_path):
        """Client requests status and gets agent info back."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                result = await client.get_status()
                assert "status" in result
                assert result["status"]["name"] == "Gwenn"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_auth_round_trip(self, tmp_path):
        """Client authenticates with token and then sends a request."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(
            config, router, event_bus, session_store, auth_token="secret123",
        )

        try:
            client = WebSocketCliChannel(auth_token="secret123")
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                result = await client.ping()
                assert result.get("status") == "pong"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_auth_failure_rejected(self, tmp_path):
        """Client with wrong token is rejected."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(
            config, router, event_bus, session_store, auth_token="secret123",
        )

        try:
            client = WebSocketCliChannel(auth_token="wrong-token")
            # connect() performs auth handshake — should fail
            with pytest.raises(Exception):  # Auth failure raises during connect
                await client.connect(f"ws://127.0.0.1:{port}/ws")
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_requests(self, tmp_path):
        """Client sends multiple requests over the same connection."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                r1 = await client.ping()
                assert r1.get("status") == "pong"

                r2 = await client.get_status()
                assert "status" in r2

                r3 = await client.ping()
                assert r3.get("status") == "pong"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_connection_to_stopped_gateway(self, tmp_path):
        """Connecting to a stopped gateway raises DaemonNotRunningError."""
        client = WebSocketCliChannel()
        with pytest.raises(DaemonNotRunningError):
            await client.connect("ws://127.0.0.1:19999/ws")


# ===========================================================================
# GatewayServer + EventBus live push
# ===========================================================================


class TestEventBusPush:
    """Verify events emitted on the bus are pushed to connected WebSocket clients."""

    @pytest.mark.asyncio
    async def test_event_pushed_to_connected_client(self, tmp_path):
        """An event emitted on the bus reaches a connected WebSocket client."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        received_events: list[dict] = []

        def on_push(msg: dict) -> None:
            received_events.append(msg)

        try:
            client = WebSocketCliChannel(on_server_push=on_push)
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                # Emit an event on the bus
                event = HeartbeatBeatEvent(
                    beat_count=42, emotion="happy", arousal=0.5, phase="think",
                )
                await event_bus.emit_async(event)

                # Give push delivery a moment
                await asyncio.sleep(0.2)

                assert len(received_events) >= 1
                # The push should be a JSON-RPC notification with event type
                push = received_events[0]
                assert push.get("method") == "event.heartbeat.beat"
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()


# ===========================================================================
# RequestRouter + EventBus event emission
# ===========================================================================


class TestRouterEventEmission:
    """Verify that the RequestRouter emits events on the bus during dispatching."""

    @pytest.mark.asyncio
    async def test_chat_emits_user_activity_event(self, tmp_path):
        """A chat dispatch should emit UserActivityEvent on the bus."""
        event_bus = create_event_bus()
        await event_bus.start()
        session_store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()

        received_events: list[Any] = []
        event_bus.subscribe("user.activity", lambda evt: received_events.append(evt))

        router = _build_router(agent, event_bus, session_store)

        # Dispatch a chat request
        await router.dispatch(
            "chat",
            {"text": "hello", "user_id": "test_user", "channel": "cli"},
            "req-1",
            [],
        )

        # Wait for event delivery
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert isinstance(received_events[0], UserActivityEvent)
        assert received_events[0].user_id == "test_user"

        await event_bus.stop()


# ===========================================================================
# Gateway session persistence round-trip
# ===========================================================================


class TestSessionPersistence:
    """Verify sessions are saved on disconnect and can be listed."""

    @pytest.mark.asyncio
    async def test_empty_session_not_saved(self, tmp_path):
        """A connection with no chat history doesn't produce a persisted session."""
        event_bus = create_event_bus()
        await event_bus.start()
        sessions_dir = tmp_path / "sessions"
        session_store = SessionStore(
            sessions_dir=sessions_dir,
            max_count=5,
            max_messages=50,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            # Connect, ping (no chat), disconnect
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            await client.ping()
            await client.disconnect()

            # Give server time to process disconnect
            await asyncio.sleep(0.1)

            # No sessions should be saved (conn.history was empty)
            sessions = session_store.list_sessions()
            assert len(sessions) == 0
        finally:
            await server.stop()
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_session_store_list_via_rpc(self, tmp_path):
        """list_sessions RPC returns sessions from the store."""
        event_bus = create_event_bus()
        await event_bus.start()
        sessions_dir = tmp_path / "sessions"
        session_store = SessionStore(
            sessions_dir=sessions_dir,
            max_count=5,
            max_messages=50,
        )
        # Pre-seed a session
        session_store.save_session(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            started_at=0.0,
        )
        agent = _fake_agent()
        config = _fake_daemon_config()
        router = _build_router(agent, event_bus, session_store)
        server, port = await _start_gateway(config, router, event_bus, session_store)

        try:
            client = WebSocketCliChannel()
            await client.connect(f"ws://127.0.0.1:{port}/ws")
            try:
                sessions = await client.list_sessions()
                assert len(sessions) == 1
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            await event_bus.stop()


# ===========================================================================
# EventBus lifecycle
# ===========================================================================


class TestEventBusIntegration:
    @pytest.mark.asyncio
    async def test_bus_start_stop_start(self):
        """EventBus can be started, stopped, and restarted."""
        bus = create_event_bus()
        received: list[Any] = []

        await bus.start()
        bus.subscribe("heartbeat.beat", lambda e: received.append(e))
        await bus.emit_async(HeartbeatBeatEvent(
            beat_count=1, emotion="neutral", arousal=0.5, phase="sense",
        ))
        assert len(received) == 1

        await bus.stop()
        received.clear()

        # Restart
        await bus.start()
        await bus.emit_async(HeartbeatBeatEvent(
            beat_count=2, emotion="happy", arousal=0.6, phase="think",
        ))
        assert len(received) == 1

        await bus.stop()
