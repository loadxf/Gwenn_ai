"""
CLI Channel — Clients for connecting to the Gwenn daemon.

Provides two transport implementations with identical public APIs:

- ``CliChannel``: Unix domain socket (NDJSON) — the legacy transport.
- ``WebSocketCliChannel``: WebSocket via aiohttp — the new gateway transport.

If no daemon is reachable, both raise ``DaemonNotRunningError`` so the
caller can fall back to in-process mode.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Type alias for server-push event handlers.
ServerPushHandler = Callable[[dict[str, Any]], None]


class DaemonNotRunningError(Exception):
    """Raised when the daemon socket is missing or connection is refused."""


class CliChannel:
    """
    Async client for the Gwenn daemon Unix socket.

    Usage:
        channel = CliChannel()
        await channel.connect(socket_path)
        response = await channel.chat("hello")
        await channel.disconnect()
    """

    def __init__(
        self,
        auth_token: str | None = None,
        on_server_push: ServerPushHandler | None = None,
    ) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._socket_path: Path | None = None
        self._auth_token = auth_token
        self._on_server_push = on_server_push

    async def connect(self, socket_path: Path) -> None:
        """
        Open a connection to the daemon socket.

        Raises DaemonNotRunningError if socket is missing or refused.
        """
        if not socket_path.exists():
            raise DaemonNotRunningError(f"Daemon socket not found: {socket_path}")
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(str(socket_path))
            self._socket_path = socket_path
            logger.debug("cli_channel.connected", socket=str(socket_path))
        except (ConnectionRefusedError, OSError) as e:
            raise DaemonNotRunningError(f"Cannot connect to daemon at {socket_path}: {e}") from e

    async def disconnect(self) -> None:
        """Close the connection to the daemon."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.debug("cli_channel.disconnect_close_failed", error=str(e))
            self._writer = None
            self._reader = None

    # ------------------------------------------------------------------
    # Request methods
    # ------------------------------------------------------------------

    async def chat(self, text: str) -> dict[str, Any]:
        """Send a chat message and return the daemon response dict."""
        return await self._request({"type": "chat", "text": text})

    async def get_status(self) -> dict[str, Any]:
        """Request agent status from daemon."""
        return await self._request({"type": "status"})

    async def get_heartbeat_status(self) -> dict[str, Any]:
        """Request heartbeat status from daemon."""
        return await self._request({"type": "heartbeat_status"})

    async def ping(self) -> dict[str, Any]:
        """Ping the daemon."""
        return await self._request({"type": "ping"})

    async def list_sessions(self) -> list[dict]:
        """Request list of recent sessions."""
        resp = await self._request({"type": "list_sessions"})
        return resp.get("sessions", [])

    async def load_session(self, session_id: str) -> int:
        """
        Ask daemon to load a previous session into this connection's history.

        Returns the number of messages loaded.
        """
        resp = await self._request({"type": "load_session", "session_id": session_id})
        return resp.get("message_count", 0)

    async def reset_session(self) -> int:
        """Clear this connection's in-memory conversation history on the daemon."""
        resp = await self._request({"type": "reset_session"})
        return int(resp.get("cleared_messages", 0))

    async def get_runtime_info(self) -> dict[str, Any]:
        """Request runtime metadata used by advanced slash commands."""
        return await self._request({"type": "runtime_info"})

    async def stop_daemon(self) -> dict[str, Any]:
        """Send graceful stop request to daemon."""
        return await self._request({"type": "stop"})

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    # Default timeout for a single request/response round-trip (seconds).
    _REQUEST_TIMEOUT: float = 120.0

    async def _request(self, payload: dict) -> dict[str, Any]:
        """Send a JSON request and await the response with a timeout."""
        if not self._reader or not self._writer:
            raise RuntimeError("CliChannel is not connected")

        req_id = uuid.uuid4().hex[:8]
        payload["req_id"] = req_id
        if self._auth_token:
            payload["auth_token"] = self._auth_token

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

        try:
            return await asyncio.wait_for(
                self._read_response(req_id),
                timeout=self._REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Daemon did not respond within {self._REQUEST_TIMEOUT}s (req_id={req_id})"
            )

    async def _read_response(self, req_id: str) -> dict[str, Any]:
        """Read lines from the socket until a message matching req_id arrives.

        Non-matching messages (server-initiated events like proactive messages
        or heartbeat notifications) are dispatched to the ``on_server_push``
        callback if one was provided, otherwise logged and discarded.
        """
        while True:
            raw = await self._reader.readline()
            if not raw:
                raise ConnectionResetError("Daemon closed the connection")
            try:
                msg = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("cli_channel.bad_json", raw=raw[:200])
                continue
            if msg.get("req_id") == req_id:
                return msg
            # Server-push message — dispatch or log.
            if self._on_server_push is not None:
                try:
                    self._on_server_push(msg)
                except Exception:
                    logger.debug("cli_channel.server_push_handler_error", exc_info=True)
            else:
                logger.debug(
                    "cli_channel.server_push_ignored",
                    msg_type=msg.get("type"),
                )


class WebSocketCliChannel:
    """
    Async client for the Gwenn gateway WebSocket.

    Usage:
        channel = WebSocketCliChannel(auth_token="secret")
        await channel.connect("ws://127.0.0.1:18900/ws")
        response = await channel.chat("hello")
        await channel.disconnect()
    """

    def __init__(
        self,
        auth_token: str | None = None,
        on_server_push: ServerPushHandler | None = None,
    ) -> None:
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._auth_token = auth_token
        self._on_server_push = on_server_push
        self._url: str | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def connect(self, ws_url: str) -> None:
        """
        Open a WebSocket connection to the gateway.

        Raises DaemonNotRunningError if the gateway is unreachable.
        """
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(ws_url)
        except (aiohttp.ClientError, OSError) as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise DaemonNotRunningError(
                f"Cannot connect to gateway at {ws_url}: {e}"
            ) from e

        self._url = ws_url
        self._reader_task = asyncio.create_task(self._read_loop())

        if self._auth_token:
            try:
                await self._request(
                    "auth",
                    {"auth_token": self._auth_token, "client_type": "cli"},
                )
            except Exception:
                await self.disconnect()
                raise

        logger.debug("ws_cli_channel.connected", url=ws_url)

    async def disconnect(self) -> None:
        """Close the WebSocket connection and clean up resources."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._reader_task = None

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("ws_cli_channel.close_failed", error=str(e))
        self._ws = None

        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug("ws_cli_channel.session_close_failed", error=str(e))
            self._session = None

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionResetError("WebSocket disconnected"))
        self._pending.clear()

    # ------------------------------------------------------------------
    # Request methods
    # ------------------------------------------------------------------

    async def chat(self, text: str) -> dict[str, Any]:
        """Send a chat message and return the gateway response dict."""
        return await self._request("chat", {"text": text})

    async def get_status(self) -> dict[str, Any]:
        """Request agent status from gateway."""
        return await self._request("status", {})

    async def get_heartbeat_status(self) -> dict[str, Any]:
        """Request heartbeat status from gateway."""
        return await self._request("heartbeat_status", {})

    async def ping(self) -> dict[str, Any]:
        """Ping the gateway."""
        return await self._request("ping", {})

    async def list_sessions(self) -> list[dict]:
        """Request list of recent sessions."""
        resp = await self._request("list_sessions", {})
        return resp.get("sessions", [])

    async def load_session(self, session_id: str) -> int:
        """
        Ask gateway to load a previous session into this connection's history.

        Returns the number of messages loaded.
        """
        resp = await self._request("load_session", {"session_id": session_id})
        return resp.get("message_count", 0)

    async def reset_session(self) -> int:
        """Clear this connection's in-memory conversation history on the gateway."""
        resp = await self._request("reset_session", {})
        return int(resp.get("cleared_messages", 0))

    async def get_runtime_info(self) -> dict[str, Any]:
        """Request runtime metadata used by advanced slash commands."""
        return await self._request("runtime_info", {})

    async def stop_daemon(self) -> dict[str, Any]:
        """Send graceful stop request to gateway."""
        return await self._request("stop", {})

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    _REQUEST_TIMEOUT: float = 120.0

    async def _request(self, method: str, params: dict) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and return the result dict."""
        if (
            self._ws is None
            or self._ws.closed
            or self._reader_task is None
            or self._reader_task.done()
        ):
            raise RuntimeError("WebSocketCliChannel is not connected")

        req_id = uuid.uuid4().hex[:8]
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send_json(payload)
            return await asyncio.wait_for(fut, timeout=self._REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(
                f"Gateway did not respond within {self._REQUEST_TIMEOUT}s (req_id={req_id})"
            )
        except Exception:
            self._pending.pop(req_id, None)
            raise

    async def _read_loop(self) -> None:
        """Background reader that routes responses and dispatches push events."""
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "ws_cli_channel.bad_json", raw=msg.data[:200]
                        )
                        continue

                    # Response to a pending request (has matching id).
                    msg_id = data.get("id")
                    if msg_id is not None and msg_id in self._pending:
                        fut = self._pending.pop(msg_id)
                        if not fut.done():
                            if "error" in data:
                                err = data["error"]
                                if isinstance(err, dict):
                                    err_msg = err.get("message", "RPC error")
                                else:
                                    err_msg = str(err) if err else "RPC error"
                                fut.set_exception(RuntimeError(err_msg))
                            else:
                                fut.set_result(data.get("result", {}))
                        continue

                    # Server-push notification (no id, has method).
                    if self._on_server_push is not None:
                        try:
                            self._on_server_push(data)
                        except Exception:
                            logger.debug(
                                "ws_cli_channel.server_push_handler_error",
                                exc_info=True,
                            )
                    else:
                        logger.debug(
                            "ws_cli_channel.server_push_ignored",
                            method=data.get("method"),
                        )

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("ws_cli_channel.read_loop_error", error=str(e), exc_info=True)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(
                        ConnectionResetError("WebSocket connection lost")
                    )
            self._pending.clear()
