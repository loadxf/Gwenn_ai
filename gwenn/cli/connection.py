"""Shared daemon connection helper for CLI commands.

Manages WebSocket connections to the Gwenn gateway, with authentication,
JSON-RPC calls, and event stream subscriptions.
"""

from __future__ import annotations

import asyncio
import functools
import json
import uuid
from typing import Any, AsyncIterator

import click
import structlog

logger = structlog.get_logger(__name__)


class DaemonNotRunning(click.ClickException):
    """Raised when the daemon is not reachable."""

    def __init__(self) -> None:
        super().__init__(
            "Gwenn daemon is not running. Start with: gwenn daemon"
        )


class DaemonConnection:
    """WebSocket connection to the daemon gateway.

    Handles connection, authentication, JSON-RPC calls, and event streaming.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 18900) -> None:
        self._host = host
        self._port = port
        self._ws: Any = None
        self._session: Any = None
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self, auth_token: str | None = None) -> None:
        """Open a WebSocket connection and authenticate."""
        import aiohttp

        url = f"ws://{self._host}:{self._port}/ws"
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(url)
        except (aiohttp.ClientError, OSError) as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise DaemonNotRunning() from e

        self._reader_task = asyncio.create_task(self._read_loop())

        if auth_token:
            await self.rpc("auth", {"auth_token": auth_token, "client_type": "cli"})

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        # Cancel any pending RPC calls
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    async def rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and return the result."""
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Not connected")

        req_id = uuid.uuid4().hex[:8]
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": req_id,
        }
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[req_id] = future

        await self._ws.send_json(payload)

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise click.ClickException("Daemon request timed out")

    async def subscribe(self, patterns: list[str]) -> None:
        """Subscribe to event patterns on the daemon."""
        await self.rpc("events.subscribe", {"types": patterns})

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield events from the subscription queue."""
        while True:
            event = await self._event_queue.get()
            yield event

    async def _read_loop(self) -> None:
        """Read messages from the WebSocket and dispatch."""
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    req_id = data.get("id")
                    if req_id is not None and req_id in self._pending:
                        # RPC response
                        future = self._pending.pop(req_id)
                        if not future.done():
                            if "error" in data:
                                future.set_exception(
                                    click.ClickException(
                                        data["error"].get("message", "RPC error")
                                    )
                                )
                            else:
                                future.set_result(data.get("result", {}))
                    elif data.get("method"):
                        # Event notification
                        try:
                            self._event_queue.put_nowait(data)
                        except asyncio.QueueFull:
                            pass  # Drop if consumer is slow
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


def require_daemon(func):
    """Decorator: connects to daemon, passes connection, handles errors."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from gwenn.config import DaemonConfig

        try:
            config = DaemonConfig()
        except Exception:
            config = None

        host = getattr(config, "gateway_host", "127.0.0.1") if config else "127.0.0.1"
        port = getattr(config, "gateway_port", 18900) if config else 18900
        auth_token = getattr(config, "auth_token", None) if config else None

        conn = DaemonConnection(host=str(host), port=int(port))
        try:
            await conn.connect(auth_token=(auth_token or "").strip() or None)
            return await func(*args, conn=conn, **kwargs)
        finally:
            await conn.disconnect()

    return wrapper
