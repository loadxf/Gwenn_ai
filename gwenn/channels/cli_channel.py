"""
CLI Channel — Unix socket client for connecting to the Gwenn daemon.

The CLI becomes a thin interactive client when a daemon is running.
Communication is Newline-Delimited JSON (NDJSON) over a Unix domain socket.

If no daemon socket exists (or connection is refused), raises
DaemonNotRunningError so the caller can fall back to in-process mode.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
