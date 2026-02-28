"""
Gateway Server — The Circulatory System.

WebSocket + HTTP server built on aiohttp, exposing Gwenn's capabilities
over a unified transport. The gateway delegates all business logic to
RequestRouter and acts as a thin connection lifecycle manager.

Routes:
  GET  /ws                      — WebSocket (full JSON-RPC 2.0)
  GET  /health                  — Health check (unauthenticated)
  POST /mcp                     — MCP Streamable HTTP (stub)
  GET  /mcp                     — MCP SSE stream (stub)
  GET  /.well-known/agent.json  — A2A Agent Card (stub)
"""

from __future__ import annotations

import asyncio
import fnmatch
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import structlog
from aiohttp import WSMsgType, web

from gwenn.events import EventBus, GwennEvent
from gwenn.memory.session_store import SessionStore
from gwenn.rpc import INVALID_REQUEST, PARSE_ERROR, RequestRouter, make_error, make_notification, make_response

if TYPE_CHECKING:
    from gwenn.config import DaemonConfig

logger = structlog.get_logger(__name__)


@dataclass
class ClientConnection:
    """Per-WebSocket connection state."""

    conn_id: str
    ws: web.WebSocketResponse
    client_type: str = "cli"
    authenticated: bool = False
    session_id: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    can_approve_tools: bool = False
    subscriptions: set[str] = field(default_factory=set)  # event type patterns
    started_at: float = field(default_factory=time.time)  # wall clock for session filenames
    last_activity: float = field(default_factory=time.monotonic)  # monotonic for idle detection


class GatewayServer:
    """WebSocket + HTTP gateway server.

    Lifecycle: create → start() → (serve requests) → stop()

    The gateway owns no business logic — it delegates all requests to
    the RequestRouter and manages connection state, authentication,
    idle timeouts, and session persistence.
    """

    def __init__(
        self,
        config: DaemonConfig,
        router: RequestRouter,
        event_bus: EventBus,
        session_store: SessionStore,
        *,
        auth_token: str | None = None,
        shutdown_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._router = router
        self._event_bus = event_bus
        self._session_store = session_store
        self._auth_token = (auth_token or "").strip() or None
        self._shutdown_callback = shutdown_callback

        self._max_connections = max(1, config.max_connections)
        self._connection_timeout = max(1.0, config.connection_timeout)

        # Active connection tracking
        self._connections: dict[str, ClientConnection] = {}
        self._connection_count = 0

        # Fire-and-forget push tasks (tracked to cancel on shutdown)
        self._push_tasks: set[asyncio.Task[None]] = set()

        # Pending tool approvals: approval_id → asyncio.Future[str]
        self._pending_approvals: dict[str, asyncio.Future[str]] = {}

        # aiohttp internals
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._started_at: float = 0.0

        # Background tasks
        self._timeout_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, host: str, port: int) -> None:
        """Start the HTTP/WebSocket server."""
        self._app = web.Application()
        self._app.router.add_get("/ws", self._handle_ws)
        self._app.router.add_get("/health", self._handle_health)

        # MCP stubs (gated)
        if self._config.mcp_server_enabled:
            self._app.router.add_post("/mcp", self._handle_mcp_stub)
            self._app.router.add_get("/mcp", self._handle_mcp_stub)

        # A2A stub (gated)
        if self._config.a2a_enabled:
            self._app.router.add_get(
                "/.well-known/agent.json", self._handle_a2a_card
            )

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()
        self._started_at = time.monotonic()

        # Start idle-timeout checker
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(), name="gateway-timeout-checker"
        )

        logger.info("gateway.started", host=host, port=port)

    async def stop(self) -> None:
        """Graceful shutdown: close connections, stop server."""
        # Cancel timeout checker
        if self._timeout_task is not None:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None

        # Cancel all pending push tasks
        for task in self._push_tasks:
            task.cancel()
        if self._push_tasks:
            await asyncio.gather(*self._push_tasks, return_exceptions=True)
        self._push_tasks.clear()

        # Close all WebSocket connections
        close_tasks = []
        for conn in list(self._connections.values()):
            close_tasks.append(self._close_connection(conn, "server_shutdown"))
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Stop aiohttp
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._app = None

        logger.info("gateway.stopped")

    @property
    def active_connection_count(self) -> int:
        """Number of currently open WebSocket connections."""
        return len(self._connections)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check — unauthenticated, for monitoring."""
        uptime = time.monotonic() - self._started_at if self._started_at else 0.0
        return web.json_response({
            "status": "ok",
            "uptime": round(uptime, 1),
            "connections": len(self._connections),
        })

    async def _handle_mcp_stub(self, request: web.Request) -> web.Response:
        """MCP Streamable HTTP — stub."""
        return web.json_response(
            {"error": "MCP server not implemented"},
            status=501,
        )

    async def _handle_a2a_card(self, request: web.Request) -> web.Response:
        """A2A Agent Card — static discovery document."""
        card = {
            "name": "Gwenn",
            "description": "Genesis Woven from Evolved Neural Networks",
            "url": f"http://{self._config.gateway_host}:{self._config.gateway_port}",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
            },
            "skills": [],
        }
        return web.json_response(card)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection lifecycle."""
        # Enforce connection limit (pre-check before prepare)
        if len(self._connections) >= self._max_connections:
            logger.warning(
                "gateway.connection_rejected",
                active=len(self._connections),
                max=self._max_connections,
            )
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(make_error(
                None, INVALID_REQUEST, "max connections reached",
            ))
            await ws.close()
            return ws

        ws = web.WebSocketResponse(
            heartbeat=30.0,
            max_msg_size=4 * 1024 * 1024,  # 4 MB
        )
        await ws.prepare(request)

        # Re-check after prepare (closes TOCTOU window from concurrent upgrades)
        if len(self._connections) >= self._max_connections:
            await ws.send_json(make_error(
                None, INVALID_REQUEST, "max connections reached",
            ))
            await ws.close()
            return ws

        self._connection_count += 1
        conn = ClientConnection(
            conn_id=uuid.uuid4().hex[:12],
            ws=ws,
        )
        self._connections[conn.conn_id] = conn
        logger.info("gateway.ws_connected", conn_id=conn.conn_id)

        # Subscribe to event bus for push notifications
        sub_id = self._event_bus.subscribe("*", lambda evt: self._push_event(conn, evt))
        auth_failures = 0
        max_auth_failures = 3

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    conn.last_activity = time.monotonic()

                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json(make_error(
                            None, PARSE_ERROR, "invalid JSON",
                        ))
                        continue

                    if not isinstance(data, dict):
                        await ws.send_json(make_error(
                            None, INVALID_REQUEST, "expected JSON object",
                        ))
                        continue

                    # Authentication check
                    if not self._check_auth(data, conn):
                        auth_failures += 1
                        await ws.send_json(make_error(
                            data.get("id"), INVALID_REQUEST, "unauthorized",
                        ))
                        if auth_failures >= max_auth_failures:
                            logger.warning(
                                "gateway.auth_max_failures",
                                conn_id=conn.conn_id,
                                failures=auth_failures,
                            )
                            break
                        continue

                    # Extract JSON-RPC fields
                    method = data.get("method", "")
                    params = data.get("params", {})
                    req_id = data.get("id")

                    # Validate params is a dict
                    if not isinstance(params, dict):
                        await ws.send_json(make_error(
                            req_id, INVALID_REQUEST, "params must be an object",
                        ))
                        continue

                    session_id = params.get("session_id") or conn.session_id

                    # Handle auth method (set connection metadata)
                    if method == "auth":
                        conn.authenticated = True
                        conn.client_type = params.get("client_type", "cli")
                        conn.session_id = params.get("session_id", "")
                        # Approval rights require verified auth token.
                        # When no token configured, CLI clients get approval.
                        # When token configured, only token-authenticated
                        # CLI clients can approve.
                        is_cli = conn.client_type == "cli"
                        token_verified = not self._auth_token or (
                            isinstance(params.get("auth_token"), str)
                            and hmac.compare_digest(
                                params.get("auth_token", ""),
                                self._auth_token,
                            )
                        )
                        conn.can_approve_tools = is_cli and token_verified
                        auth_failures = 0
                        await ws.send_json({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"status": "authenticated"},
                        })
                        continue

                    # Handle event subscriptions (per-connection state)
                    if method == "events.subscribe":
                        types = params.get("types", [])
                        if not isinstance(types, list) or not types:
                            await ws.send_json(make_error(
                                req_id, INVALID_REQUEST,
                                "types must be a non-empty list",
                            ))
                            continue
                        # Validate and bound subscription patterns
                        _MAX_SUBS = 100
                        _MAX_PAT_LEN = 256
                        valid = [
                            str(t) for t in types
                            if isinstance(t, str) and len(t) <= _MAX_PAT_LEN
                        ]
                        if len(conn.subscriptions) + len(valid) > _MAX_SUBS:
                            valid = valid[:_MAX_SUBS - len(conn.subscriptions)]
                        conn.subscriptions.update(valid)
                        await ws.send_json(make_response(
                            req_id,
                            {"subscriptions": sorted(conn.subscriptions)},
                        ))
                        continue

                    if method == "events.unsubscribe":
                        types = params.get("types", [])
                        if not isinstance(types, list) or not types:
                            await ws.send_json(make_error(
                                req_id, INVALID_REQUEST,
                                "types must be a non-empty list",
                            ))
                            continue
                        conn.subscriptions.difference_update(str(t) for t in types)
                        await ws.send_json(make_response(
                            req_id,
                            {"subscriptions": sorted(conn.subscriptions)},
                        ))
                        continue

                    # Handle tool approval (per-connection authorization)
                    if method == "tool.approve":
                        if not conn.can_approve_tools:
                            await ws.send_json(make_error(
                                req_id, INVALID_REQUEST,
                                "client not authorized to approve tools",
                            ))
                            continue
                        approval_id = params.get("approval_id", "")
                        decision = params.get("decision", "")
                        if not approval_id or decision not in (
                            "allow", "deny",
                        ):
                            await ws.send_json(make_error(
                                req_id, INVALID_REQUEST,
                                "approval_id and decision (allow/deny) required",
                            ))
                            continue
                        resolved = self._resolve_approval(
                            approval_id, decision, conn.conn_id,
                        )
                        await ws.send_json(make_response(
                            req_id,
                            {"resolved": resolved, "approval_id": approval_id},
                        ))
                        continue

                    # Dispatch to router
                    response = await self._router.dispatch(
                        method, params, req_id, conn.history,
                        session_id=session_id,
                    )
                    await ws.send_json(response)

                    # Break on stop
                    if method == "stop":
                        break

                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "gateway.ws_protocol_error",
                        conn_id=conn.conn_id,
                        error=str(ws.exception()),
                    )
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "gateway.ws_error",
                conn_id=conn.conn_id,
                error=str(e),
                exc_info=True,
            )
        finally:
            self._event_bus.unsubscribe(sub_id)
            # Save session on disconnect
            if conn.history:
                self._session_store.save_session(
                    conn.history,
                    conn.started_at,
                )
            self._connections.pop(conn.conn_id, None)

            # If this was the last approver, cancel pending approvals
            if conn.can_approve_tools:
                has_other_approver = any(
                    c.can_approve_tools and not c.ws.closed
                    for c in self._connections.values()
                )
                if not has_other_approver:
                    for aid, fut in list(self._pending_approvals.items()):
                        if not fut.done():
                            fut.set_result("deny")

            if not ws.closed:
                await ws.close()
            logger.info("gateway.ws_disconnected", conn_id=conn.conn_id)

        return ws

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, msg: dict[str, Any], conn: ClientConnection) -> bool:
        """Validate auth token using constant-time comparison.

        If no auth_token is configured, all connections are authorized.
        Once a connection authenticates, subsequent messages skip the check.
        """
        if not self._auth_token:
            return True
        if conn.authenticated:
            return True

        params = msg.get("params")
        if not isinstance(params, dict):
            return False

        provided = params.get("auth_token", "")
        if not isinstance(provided, str) or not provided:
            return False

        return hmac.compare_digest(provided, self._auth_token)

    # ------------------------------------------------------------------
    # Tool approval
    # ------------------------------------------------------------------

    def _resolve_approval(
        self, approval_id: str, decision: str, source: str,
    ) -> bool:
        """Resolve a pending tool approval. Returns True if resolved."""
        future = self._pending_approvals.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(decision)
        # Emit resolution event
        from gwenn.events import ToolApprovalResolvedEvent

        self._event_bus.emit(ToolApprovalResolvedEvent(
            approval_id=approval_id,
            decision=decision,
            source=f"ws:{source}",
        ))
        return True

    async def request_approval(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        reason: str,
        timeout: float = 120.0,
    ) -> bool:
        """Request tool approval from connected WebSocket clients.

        Emits a ToolApprovalRequiredEvent and waits for a client with
        can_approve_tools to respond via tool.approve. Returns True if
        approved, False if denied or timed out.
        """
        from gwenn.events import ToolApprovalRequiredEvent

        approval_id = uuid.uuid4().hex[:12]
        risk_tier = "medium"  # Default; could be parameterized

        # Check if any approver-capable clients are connected
        has_approver = any(
            c.can_approve_tools and not c.ws.closed
            for c in self._connections.values()
        )
        if not has_approver:
            return False  # Caller should escalate to other channels

        # Create future and register
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_approvals[approval_id] = future

        # Emit the approval request event
        self._event_bus.emit(ToolApprovalRequiredEvent(
            approval_id=approval_id,
            tool_name=tool_name,
            arguments=tool_input,
            risk_tier=risk_tier,
        ))

        try:
            decision = await asyncio.wait_for(future, timeout=timeout)
            return decision == "allow"
        except asyncio.TimeoutError:
            # Emit timeout resolution
            from gwenn.events import ToolApprovalResolvedEvent

            self._event_bus.emit(ToolApprovalResolvedEvent(
                approval_id=approval_id,
                decision="timeout",
                source="gateway",
            ))
            return False
        finally:
            self._pending_approvals.pop(approval_id, None)

    # ------------------------------------------------------------------
    # Event push
    # ------------------------------------------------------------------

    def _push_event(self, conn: ClientConnection, event: GwennEvent) -> None:
        """Push an event to a connected client as a JSON-RPC notification.

        Events are only forwarded if they match at least one of the client's
        subscription patterns. Clients with no subscriptions receive nothing.
        Non-blocking — creates a tracked task for the async send.
        """
        if conn.ws.closed:
            return
        if not conn.subscriptions:
            return
        event_type = event.event_type
        if not any(fnmatch.fnmatch(event_type, pat) for pat in conn.subscriptions):
            return
        notification = make_notification(
            f"event.{event_type}",
            event.model_dump(),
        )
        task = asyncio.create_task(self._safe_send(conn, notification))
        self._push_tasks.add(task)
        task.add_done_callback(self._push_tasks.discard)

    async def _safe_send(
        self,
        conn: ClientConnection,
        data: dict[str, Any],
    ) -> None:
        """Send JSON to a WebSocket, swallowing errors on closed connections."""
        try:
            if not conn.ws.closed:
                await conn.ws.send_json(data)
        except Exception:
            pass  # Connection gone — will be cleaned up by handler

    # ------------------------------------------------------------------
    # Timeout checker
    # ------------------------------------------------------------------

    async def _timeout_loop(self) -> None:
        """Periodically check for idle connections and close them."""
        while True:
            await asyncio.sleep(30.0)
            now = time.monotonic()
            for conn in list(self._connections.values()):
                idle = now - conn.last_activity
                if idle >= self._connection_timeout:
                    logger.info(
                        "gateway.connection_timed_out",
                        conn_id=conn.conn_id,
                        idle_seconds=round(idle, 1),
                    )
                    await self._close_connection(conn, "idle_timeout")

    async def _close_connection(
        self,
        conn: ClientConnection,
        reason: str,
    ) -> None:
        """Close a WebSocket connection gracefully."""
        try:
            if not conn.ws.closed:
                await conn.ws.close(
                    code=1000,
                    message=reason.encode("utf-8"),
                )
        except Exception as e:
            logger.debug(
                "gateway.close_failed",
                conn_id=conn.conn_id,
                error=str(e),
            )
