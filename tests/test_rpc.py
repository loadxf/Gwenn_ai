"""Tests for gwenn.rpc — RequestRouter and JSON-RPC helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.events import EventBus
from gwenn.memory.session_store import SessionStore
from gwenn.rpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    RequestRouter,
    make_error,
    make_notification,
    make_response,
)


# ---------------------------------------------------------------------------
# JSON-RPC helper tests
# ---------------------------------------------------------------------------


class TestJsonRpcHelpers:
    """Tests for make_response, make_error, make_notification."""

    def test_make_response(self) -> None:
        resp = make_response("req-1", {"text": "hello"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        assert resp["result"] == {"text": "hello"}
        assert "error" not in resp

    def test_make_response_int_id(self) -> None:
        resp = make_response(42, {"ok": True})
        assert resp["id"] == 42

    def test_make_response_none_id(self) -> None:
        resp = make_response(None, {"ok": True})
        assert resp["id"] is None

    def test_make_error(self) -> None:
        resp = make_error("req-1", -32601, "method not found")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        assert resp["error"]["code"] == -32601
        assert resp["error"]["message"] == "method not found"
        assert "result" not in resp

    def test_make_error_with_data(self) -> None:
        resp = make_error("req-1", -32600, "bad request", data={"field": "text"})
        assert resp["error"]["data"] == {"field": "text"}

    def test_make_error_without_data(self) -> None:
        resp = make_error("req-1", -32600, "bad request")
        assert "data" not in resp["error"]

    def test_make_notification(self) -> None:
        notif = make_notification("heartbeat.beat", {"count": 42})
        assert notif["jsonrpc"] == "2.0"
        assert notif["method"] == "heartbeat.beat"
        assert notif["params"] == {"count": 42}
        assert "id" not in notif


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock SentientAgent with required attributes."""
    agent = MagicMock()
    agent.respond = AsyncMock(return_value="Hello!")
    agent.affect_state.current_emotion.value = "curious"
    agent.status = {"running": True, "beat_count": 10}
    agent.heartbeat = MagicMock()
    agent.heartbeat.status = {"running": True, "beat_count": 10}
    agent.heartbeat.notify_user_activity = MagicMock()
    agent.skill_registry = None
    agent._mcp_client = None
    agent.tool_registry = None
    return agent


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    """Create a SessionStore in a temp directory."""
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
    shutdown_calls: list[str] = []
    return RequestRouter(
        agent=mock_agent,
        session_store=session_store,
        event_bus=event_bus,
        respond_lock=respond_lock,
        shutdown_callback=lambda reason: shutdown_calls.append(reason),
        active_connections_getter=lambda: 3,
    )


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchPing:
    @pytest.mark.asyncio
    async def test_ping(self, router: RequestRouter) -> None:
        resp = await router.dispatch("ping", {}, "r1", [])
        assert resp["result"]["status"] == "pong"
        assert resp["id"] == "r1"


class TestDispatchChat:
    @pytest.mark.asyncio
    async def test_chat_success(self, router: RequestRouter) -> None:
        history: list[dict[str, Any]] = []
        resp = await router.dispatch("chat", {"text": "Hi"}, "r1", history)
        assert resp["result"]["text"] == "Hello!"
        assert resp["result"]["emotion"] == "curious"

    @pytest.mark.asyncio
    async def test_chat_empty_text(self, router: RequestRouter) -> None:
        resp = await router.dispatch("chat", {"text": ""}, "r1", [])
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "empty text" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_chat_missing_text(self, router: RequestRouter) -> None:
        resp = await router.dispatch("chat", {}, "r1", [])
        assert resp["error"]["code"] == INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_chat_notifies_heartbeat(
        self, router: RequestRouter, mock_agent: MagicMock
    ) -> None:
        await router.dispatch("chat", {"text": "Hi"}, "r1", [])
        mock_agent.heartbeat.notify_user_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_emits_user_activity_event(
        self, router: RequestRouter, event_bus: EventBus
    ) -> None:
        received: list[Any] = []
        event_bus.subscribe("user.*", received.append)
        await event_bus.start()
        try:
            await router.dispatch(
                "chat", {"text": "Hi", "channel": "web", "user_id": "u1"}, "r1", []
            )
            await asyncio.sleep(0.05)
            assert len(received) == 1
            assert received[0].channel == "web"
            assert received[0].user_id == "u1"
        finally:
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_chat_uses_global_lock_without_session_id(
        self, router: RequestRouter, respond_lock: asyncio.Lock
    ) -> None:
        """Without session_id, chat should use the global respond lock."""
        # Acquire the lock and verify chat blocks
        async with respond_lock:
            # dispatch should block waiting for the lock
            task = asyncio.create_task(
                router.dispatch("chat", {"text": "Hi"}, "r1", [])
            )
            await asyncio.sleep(0.05)
            assert not task.done()  # Should be blocked by lock
        # Now it should complete
        resp = await asyncio.wait_for(task, timeout=5.0)
        assert resp["result"]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_uses_per_session_lock_with_session_id(
        self, router: RequestRouter
    ) -> None:
        """With session_id, different sessions should not block each other."""
        # Create a slow respond that takes time
        call_count = 0

        async def slow_respond(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return f"response-{call_count}"

        router._agent.respond = slow_respond

        # Dispatch two chats with different session_ids concurrently
        t1 = asyncio.create_task(
            router.dispatch("chat", {"text": "A"}, "r1", [], session_id="s1")
        )
        t2 = asyncio.create_task(
            router.dispatch("chat", {"text": "B"}, "r2", [], session_id="s2")
        )
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1["result"]["text"].startswith("response-")
        assert r2["result"]["text"].startswith("response-")


class TestDispatchStatus:
    @pytest.mark.asyncio
    async def test_status(self, router: RequestRouter) -> None:
        resp = await router.dispatch("status", {}, "r1", [])
        assert resp["result"]["status"]["running"] is True
        assert resp["result"]["active_connections"] == 3


class TestDispatchHeartbeatStatus:
    @pytest.mark.asyncio
    async def test_heartbeat_status(self, router: RequestRouter) -> None:
        resp = await router.dispatch("heartbeat_status", {}, "r1", [])
        assert resp["result"]["status"]["running"] is True
        assert resp["result"]["status"]["beat_count"] == 10

    @pytest.mark.asyncio
    async def test_heartbeat_status_no_heartbeat(self, router: RequestRouter) -> None:
        router._agent.heartbeat = None
        resp = await router.dispatch("heartbeat_status", {}, "r1", [])
        assert resp["result"]["status"] == {}


class TestDispatchSessions:
    @pytest.mark.asyncio
    async def test_list_sessions(self, router: RequestRouter) -> None:
        resp = await router.dispatch("list_sessions", {}, "r1", [])
        assert "sessions" in resp["result"]

    @pytest.mark.asyncio
    async def test_load_session_invalid_id(self, router: RequestRouter) -> None:
        resp = await router.dispatch("load_session", {"session_id": "bad"}, "r1", [])
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "invalid session id" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_load_session_not_found(self, router: RequestRouter) -> None:
        resp = await router.dispatch(
            "load_session", {"session_id": "20260101-120000-abcd"}, "r1", []
        )
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "session not found" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_reset_session(self, router: RequestRouter) -> None:
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        resp = await router.dispatch("reset_session", {}, "r1", history)
        assert resp["result"]["cleared_messages"] == 2
        assert len(history) == 0


class TestDispatchRuntimeInfo:
    @pytest.mark.asyncio
    async def test_runtime_info(self, router: RequestRouter) -> None:
        resp = await router.dispatch("runtime_info", {}, "r1", [])
        assert "status" in resp["result"]
        assert "skills" in resp["result"]
        assert "tools" in resp["result"]
        assert resp["result"]["active_connections"] == 3

    @pytest.mark.asyncio
    async def test_runtime_info_with_skills(self, router: RequestRouter) -> None:
        skill = MagicMock()
        skill.name = "test_skill"
        skill.category = "utility"
        registry = MagicMock()
        registry.all_skills.return_value = [skill]
        router._agent.skill_registry = registry

        resp = await router.dispatch("runtime_info", {}, "r1", [])
        assert len(resp["result"]["skills"]) == 1
        assert resp["result"]["skills"][0]["name"] == "test_skill"


class TestDispatchStop:
    @pytest.mark.asyncio
    async def test_stop(self, router: RequestRouter) -> None:
        resp = await router.dispatch("stop", {}, "r1", [])
        assert resp["result"]["status"] == "stopping"


class TestDispatchUnknown:
    @pytest.mark.asyncio
    async def test_unknown_method(self, router: RequestRouter) -> None:
        resp = await router.dispatch("nonexistent", {}, "r1", [])
        assert resp["error"]["code"] == METHOD_NOT_FOUND
        assert "unknown method" in resp["error"]["message"]


class TestDispatchInternalError:
    @pytest.mark.asyncio
    async def test_internal_error(self, router: RequestRouter) -> None:
        """Handler exception should return internal error, not raise."""
        router._agent.respond = AsyncMock(side_effect=RuntimeError("boom"))
        resp = await router.dispatch("chat", {"text": "Hi"}, "r1", [])
        assert resp["error"]["code"] == INTERNAL_ERROR


class TestSessionLockBounds:
    def test_session_locks_bounded(self, router: RequestRouter) -> None:
        """Session lock cache should not grow beyond max_session_locks."""
        router._max_session_locks = 5
        for i in range(10):
            router._get_lock(f"session-{i}")
        assert len(router._session_locks) == 5
        # Most recent 5 should be present
        for i in range(5, 10):
            assert f"session-{i}" in router._session_locks

    def test_lru_reuse_preserves_lock(self, router: RequestRouter) -> None:
        """Reusing a session should move it to end (LRU)."""
        router._max_session_locks = 3
        lock_a = router._get_lock("a")
        router._get_lock("b")
        router._get_lock("c")
        # Reuse "a" — should move to end
        lock_a2 = router._get_lock("a")
        assert lock_a is lock_a2  # Same lock object
        # Now adding "d" should evict "b" (oldest), not "a"
        router._get_lock("d")
        assert "a" in router._session_locks
        assert "b" not in router._session_locks


class TestPerSessionLocking:
    @pytest.mark.asyncio
    async def test_same_session_serialized(self, router: RequestRouter) -> None:
        """Same session_id should serialize requests."""
        order: list[int] = []

        async def ordered_respond(*args: Any, **kwargs: Any) -> str:
            idx = len(order)
            order.append(idx)
            await asyncio.sleep(0.05)
            return f"resp-{idx}"

        router._agent.respond = ordered_respond

        # Same session — should serialize
        t1 = asyncio.create_task(
            router.dispatch("chat", {"text": "A"}, "r1", [], session_id="same")
        )
        await asyncio.sleep(0.01)  # Ensure t1 starts first
        t2 = asyncio.create_task(
            router.dispatch("chat", {"text": "B"}, "r2", [], session_id="same")
        )
        await asyncio.gather(t1, t2)
        assert order == [0, 1]  # Sequential
