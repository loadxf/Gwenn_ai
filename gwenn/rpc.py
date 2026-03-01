"""
Request Router — Transport-Independent JSON-RPC Dispatch.

Extracted from daemon.py to decouple request handling from the Unix socket
transport. Both the legacy Unix socket (daemon.py) and the new WebSocket
gateway (gateway.py) share this router.

Supports two lock patterns:
  - Global respond lock (legacy socket compat, heartbeat integration)
  - Per-session locks (concurrent sessions across channels/WebSocket)
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

import structlog

from gwenn.events import EventBus, UserActivityEvent
from gwenn.memory.session_store import SessionStore

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 helpers
# ---------------------------------------------------------------------------


def make_response(req_id: str | int | None, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(
    req_id: str | int | None,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def make_notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no id, no response expected)."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RequestRouter:
    """Transport-independent request router.

    Routes JSON-RPC-style messages to agent methods. Supports both the
    legacy NDJSON protocol (type/req_id) and JSON-RPC 2.0 (method/id).

    Lock strategy:
      - When session_id is provided: per-session lock (concurrent sessions)
      - When no session_id: global respond_lock (legacy socket compat)
    """

    def __init__(
        self,
        agent: SentientAgent,
        session_store: SessionStore,
        event_bus: EventBus,
        respond_lock: asyncio.Lock,
        shutdown_callback: Callable[[str], None] | None = None,
        active_connections_getter: Callable[[], int] | None = None,
        session_include_preview: bool = False,
    ) -> None:
        self._agent = agent
        self._session_store = session_store
        self._event_bus = event_bus
        self._respond_lock = respond_lock
        self._shutdown_callback = shutdown_callback
        self._get_active_connections = active_connections_getter or (lambda: 0)
        self._session_include_preview = session_include_preview
        # Per-session locks for concurrent session handling.
        # Bounded to prevent unbounded growth over long runtimes.
        self._session_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._max_session_locks = 512

    def _get_lock(self, session_id: str | None) -> asyncio.Lock:
        """Return per-session lock if session_id given, else global lock."""
        if session_id:
            if session_id in self._session_locks:
                self._session_locks.move_to_end(session_id)
                return self._session_locks[session_id]
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
            # Evict oldest unlocked entry if over capacity
            while len(self._session_locks) > self._max_session_locks:
                evicted = False
                for key in list(self._session_locks):
                    if not self._session_locks[key].locked():
                        del self._session_locks[key]
                        evicted = True
                        break
                if not evicted:
                    break  # All locks held; allow temporary overshoot
            return lock
        return self._respond_lock

    async def dispatch(
        self,
        method: str,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Route a request to the appropriate handler.

        Args:
            method: RPC method name (e.g., "chat", "status", "ping").
            params: Method parameters.
            req_id: Request ID for response correlation.
            history: Mutable conversation history for this connection.
            session_id: Optional session ID for per-session locking.

        Returns:
            JSON-RPC 2.0 response dict.
        """
        try:
            handler = self._handlers.get(method)
            if handler is None:
                return make_error(req_id, METHOD_NOT_FOUND, f"unknown method: {method}")
            return await handler(self, params, req_id, history, session_id)
        except Exception as e:
            logger.error("rpc.dispatch_error", method=method, error=str(e), exc_info=True)
            return make_error(req_id, INTERNAL_ERROR, "internal error")

    # ------------------------------------------------------------------
    # RPC method handlers
    # ------------------------------------------------------------------

    async def _handle_ping(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        return make_response(req_id, {"status": "pong"})

    async def _handle_chat(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        text = params.get("text", "")
        if not text:
            return make_error(req_id, INVALID_PARAMS, "empty text")

        # Notify heartbeat of user activity
        channel = params.get("channel", "cli")
        user_id = params.get("user_id", "")
        self._event_bus.emit(UserActivityEvent(channel=channel, user_id=user_id))

        # Notify heartbeat directly for timing-critical acceleration
        heartbeat = getattr(self._agent, "heartbeat", None)
        if heartbeat is not None:
            notify = getattr(heartbeat, "notify_user_activity", None)
            if callable(notify):
                notify()

        lock = self._get_lock(session_id)
        async with lock:
            response_text = await self._agent.respond(
                text,
                conversation_history=history,
            )
            emotion = "neutral"
            try:
                emotion = self._agent.affect_state.current_emotion.value
            except Exception:
                pass

        return make_response(req_id, {
            "text": str(response_text),
            "emotion": emotion,
        })

    async def _handle_status(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        status = self._agent.status if self._agent else {}
        return make_response(req_id, {
            "status": status,
            "active_connections": self._get_active_connections(),
        })

    async def _handle_heartbeat_status(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        hb: dict[str, Any] = {}
        if self._agent:
            heartbeat = getattr(self._agent, "heartbeat", None)
            if heartbeat is not None:
                hb = heartbeat.status
        return make_response(req_id, {"status": hb})

    async def _handle_list_sessions(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        limit = params.get("limit", 10)
        sessions = self._session_store.list_sessions(
            limit=limit,
            include_preview=self._session_include_preview,
        )
        return make_response(req_id, {"sessions": sessions})

    async def _handle_load_session(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        sid = params.get("session_id", "")
        if not SessionStore.is_valid_session_id(sid):
            return make_error(req_id, INVALID_PARAMS, "invalid session id")
        try:
            loaded = self._session_store.load_session(sid)
            history.clear()
            history.extend(loaded)
            return make_response(req_id, {"message_count": len(loaded)})
        except FileNotFoundError:
            return make_error(req_id, INVALID_PARAMS, "session not found")

    async def _handle_reset_session(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        cleared = len(history)
        history.clear()
        return make_response(req_id, {"cleared_messages": cleared})

    async def _handle_runtime_info(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        status = self._agent.status if self._agent else {}
        skills: list[dict[str, Any]] = []
        if self._agent is not None:
            skill_registry = getattr(self._agent, "skill_registry", None)
            all_skills = getattr(skill_registry, "all_skills", None)
            if callable(all_skills):
                try:
                    for skill in all_skills():
                        skills.append({
                            "name": str(getattr(skill, "name", "unknown")),
                            "category": str(getattr(skill, "category", "skill")),
                        })
                except Exception as e:
                    logger.debug("rpc.runtime_info_skill_list_failed", error=str(e))

        mcp_stats = getattr(getattr(self._agent, "_mcp_client", None), "stats", {})
        if not isinstance(mcp_stats, dict):
            mcp_stats = {}
        tool_registry = getattr(self._agent, "tool_registry", None)
        tools = {
            "registered": int(getattr(tool_registry, "count", 0)),
            "enabled": int(getattr(tool_registry, "enabled_count", 0)),
        }

        return make_response(req_id, {
            "status": status,
            "skills": skills,
            "mcp": mcp_stats,
            "tools": tools,
            "active_connections": self._get_active_connections(),
        })

    async def _handle_stop(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        logger.info("rpc.stop_requested")
        if self._shutdown_callback:
            self._shutdown_callback("rpc_stop_requested")
        return make_response(req_id, {"status": "stopping"})

    # ------------------------------------------------------------------
    # Checkpoint RPC handlers (Phase 7)
    # ------------------------------------------------------------------

    async def _handle_checkpoint_list(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        heartbeat = getattr(self._agent, "heartbeat", None)
        if heartbeat is None:
            return make_error(req_id, INVALID_PARAMS, "heartbeat not available")
        mgr = getattr(heartbeat, "_checkpoint_manager", None)
        if mgr is None:
            return make_response(req_id, {"checkpoints": [], "enabled": False})
        return make_response(req_id, {
            "checkpoints": mgr.list_checkpoints(),
            "enabled": True,
        })

    async def _handle_checkpoint_create(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        heartbeat = getattr(self._agent, "heartbeat", None)
        if heartbeat is None:
            return make_error(req_id, INVALID_PARAMS, "heartbeat not available")
        mgr = getattr(heartbeat, "_checkpoint_manager", None)
        if mgr is None:
            return make_error(req_id, INVALID_PARAMS, "checkpointing not enabled")
        try:
            checkpoint = await mgr.create_checkpoint(self._agent, heartbeat)
            path = await mgr.save_checkpoint(checkpoint)
            return make_response(req_id, {
                "checkpoint_id": checkpoint.checkpoint_id,
                "beat_count": checkpoint.beat_count,
                "size_bytes": path.stat().st_size,
            })
        except Exception as e:
            return make_error(req_id, INTERNAL_ERROR, f"checkpoint failed: {e}")

    async def _handle_checkpoint_restore(
        self,
        params: dict[str, Any],
        req_id: str | int | None,
        history: list[dict[str, Any]],
        session_id: str | None,
    ) -> dict[str, Any]:
        heartbeat = getattr(self._agent, "heartbeat", None)
        if heartbeat is None:
            return make_error(req_id, INVALID_PARAMS, "heartbeat not available")
        mgr = getattr(heartbeat, "_checkpoint_manager", None)
        if mgr is None:
            return make_error(req_id, INVALID_PARAMS, "checkpointing not enabled")

        checkpoint_id = params.get("checkpoint_id")
        if checkpoint_id is not None and (
            not isinstance(checkpoint_id, str) or len(checkpoint_id) > 64
        ):
            return make_error(req_id, INVALID_PARAMS, "invalid checkpoint_id")

        if checkpoint_id:
            checkpoint = await mgr.load_checkpoint(checkpoint_id)
        else:
            checkpoint = await mgr.load_latest_checkpoint()

        if checkpoint is None:
            return make_error(req_id, INVALID_PARAMS, "no checkpoint found")

        try:
            await mgr.restore_from_checkpoint(self._agent, checkpoint)
            # Sync heartbeat state with restored checkpoint.
            heartbeat._beat_count = checkpoint.beat_count
            mgr._last_checkpoint_beat = checkpoint.beat_count
            return make_response(req_id, {
                "checkpoint_id": checkpoint.checkpoint_id,
                "beat_count": checkpoint.beat_count,
                "restored": True,
            })
        except Exception as e:
            return make_error(req_id, INTERNAL_ERROR, f"restore failed: {e}")

    # ------------------------------------------------------------------
    # Handler registry
    # ------------------------------------------------------------------

    _handlers: dict[str, Any] = {
        "ping": _handle_ping,
        "chat": _handle_chat,
        "status": _handle_status,
        "heartbeat_status": _handle_heartbeat_status,
        "list_sessions": _handle_list_sessions,
        "load_session": _handle_load_session,
        "reset_session": _handle_reset_session,
        "runtime_info": _handle_runtime_info,
        "stop": _handle_stop,
        "checkpoint.list": _handle_checkpoint_list,
        "checkpoint.create": _handle_checkpoint_create,
        "checkpoint.restore": _handle_checkpoint_restore,
    }
