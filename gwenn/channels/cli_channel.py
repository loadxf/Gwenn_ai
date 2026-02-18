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
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


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

    def __init__(self, auth_token: str | None = None) -> None:
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._socket_path: Optional[Path] = None
        self._auth_token = auth_token

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
            except Exception:
                pass
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

    async def stop_daemon(self) -> dict[str, Any]:
        """Send graceful stop request to daemon."""
        return await self._request({"type": "stop"})

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    async def _request(self, payload: dict) -> dict[str, Any]:
        """Send a JSON request and await the response."""
        if not self._reader or not self._writer:
            raise RuntimeError("CliChannel is not connected")

        req_id = uuid.uuid4().hex[:8]
        payload["id"] = req_id
        if self._auth_token:
            payload["auth_token"] = self._auth_token

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

        # Read until we get a response for our req_id
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
