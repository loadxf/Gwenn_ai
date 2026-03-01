"""
MCP Client — Gwenn's Connection to External Tool Servers.

This module implements a minimal but real MCP transport layer:
- stdio servers via JSON-RPC frames (Content-Length)
- streamable HTTP servers via JSON-RPC POST requests

Discovered tools are registered into ToolRegistry as callable proxy tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from gwenn import __version__
from gwenn.tools.registry import ToolDefinition, ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""

    name: str
    transport: str = "stdio"  # "stdio" or "streamable_http"
    command: Optional[str] = None  # For stdio: command to launch server
    args: list[str] = field(default_factory=list)
    url: Optional[str] = None  # For HTTP: server URL
    api_key: Optional[str] = None  # For HTTP: authentication
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 20.0


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


def _sanitize_mcp_name_part(s: str) -> str:
    """Replace characters invalid in Claude tool names with underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def _extract_jsonrpc_result(response: Any) -> Any:
    """Extract result payload from a JSON-RPC response."""

    if not isinstance(response, dict):
        raise RuntimeError(f"Invalid JSON-RPC response type: {type(response).__name__}")

    if "error" in response and response["error"] is not None:
        error_obj = response.get("error", {})
        if isinstance(error_obj, dict):
            code = error_obj.get("code", "unknown")
            message = error_obj.get("message", "Unknown MCP error")
            raise RuntimeError(f"MCP error {code}: {message}")
        raise RuntimeError(f"MCP error: {error_obj}")

    return response.get("result")


def _content_block_to_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return json.dumps(block, ensure_ascii=False)
    block_type = str(block.get("type", "")).lower()
    if block_type == "text":
        return str(block.get("text", ""))
    if "text" in block:
        return str(block.get("text", ""))
    return json.dumps(block, ensure_ascii=False)


def _render_mcp_call_result(result: Any) -> str:
    """Normalize MCP tools/call payloads to readable text for Claude tool_result."""

    if isinstance(result, dict):
        if result.get("isError") is True:
            err_content = result.get("content")
            if isinstance(err_content, list):
                err_text = "\n".join(_content_block_to_text(b) for b in err_content).strip()
                raise RuntimeError(err_text or "MCP tool returned an error result.")
            raise RuntimeError(str(err_content) or "MCP tool returned an error result.")

        content = result.get("content")
        if isinstance(content, list):
            text = "\n".join(_content_block_to_text(b) for b in content).strip()
            return text or json.dumps(result, ensure_ascii=False)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


class _BaseTransport:
    async def request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class _HTTPTransport(_BaseTransport):
    def __init__(self, config: MCPServerConfig):
        if not config.url:
            raise ValueError(f"MCP server '{config.name}' missing URL for streamable_http transport.")
        self._config = config
        self._request_id = 0
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    async def request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        self._request_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        response = await self._client.post(self._config.url, json=payload, headers=self._headers)
        response.raise_for_status()
        return _extract_jsonrpc_result(response.json())

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = await self._client.post(self._config.url, json=payload, headers=self._headers)
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()


class _StdioTransport(_BaseTransport):
    def __init__(self, config: MCPServerConfig, process: asyncio.subprocess.Process):
        self._config = config
        self._process = process
        self._request_id = 0
        self._io_lock = asyncio.Lock()
        self._broken = False
        self._stderr_task: Optional[asyncio.Task] = None
        if self._process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    @classmethod
    async def start(cls, config: MCPServerConfig) -> "_StdioTransport":
        if not config.command:
            raise ValueError(f"MCP server '{config.name}' missing 'command' for stdio transport.")

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (config.env or {}).items()})

        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        return cls(config=config, process=process)

    async def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug(
                        "mcp_client.server_stderr",
                        server=self._config.name,
                        message=text[:500],
                    )
        except Exception as exc:  # pragma: no cover - best-effort logging path
            logger.debug(
                "mcp_client.stderr_drain_failed",
                server=self._config.name,
                error=str(exc),
            )

    async def _send_message(self, payload: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + raw)
        await self._process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        assert self._process.stdout is not None

        headers: dict[str, str] = {}
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP stdio server '{self._config.name}' closed unexpectedly.")
            if line in (b"\n", b"\r\n"):
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = headers.get("content-length")
        if not content_length:
            raise RuntimeError(f"MCP stdio server '{self._config.name}' missing Content-Length header.")

        try:
            length = int(content_length)
        except ValueError as exc:
            raise RuntimeError(
                f"MCP stdio server '{self._config.name}' returned invalid Content-Length: {content_length}"
            ) from exc

        payload = await self._process.stdout.readexactly(length)
        decoded_payload = payload.decode("utf-8", errors="replace")
        parsed = json.loads(decoded_payload)
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"MCP stdio server '{self._config.name}' returned non-object JSON payload."
            )
        return parsed

    async def request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        if self._broken:
            raise RuntimeError(
                f"MCP stdio transport for '{self._config.name}' is broken "
                f"(previous timeout corrupted the stream). Restart the server."
            )
        async with self._io_lock:
            self._request_id += 1
            request_id = self._request_id
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params

            await self._send_message(payload)

            while True:
                try:
                    message = await asyncio.wait_for(
                        self._read_message(),
                        timeout=max(0.1, float(self._config.timeout_seconds)),
                    )
                except asyncio.TimeoutError as exc:
                    # Stream is now corrupt — mark transport as broken
                    self._broken = True
                    raise RuntimeError(
                        f"MCP stdio request '{method}' timed out after "
                        f"{self._config.timeout_seconds}s on server '{self._config.name}'."
                    ) from exc
                if message.get("id") != request_id:
                    logger.debug(
                        "mcp.stdio.unexpected_message",
                        expected_id=request_id,
                        received_id=message.get("id"),
                        method=message.get("method"),
                        server=self._config.name,
                    )
                    continue
                return _extract_jsonrpc_result(message)

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        async with self._io_lock:
            payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                payload["params"] = params
            await self._send_message(payload)

    async def close(self) -> None:
        process = self._process
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass


class MCPClient:
    """
    Client for connecting to MCP servers and discovering tools.

    Lifecycle:
    1. initialize() - Connect to all configured servers
    2. discover_tools() - Query each server for available tools
    3. register_tools() - Add discovered tools to the ToolRegistry
    4. execute_tool() - Proxy tool calls to the appropriate server
    5. shutdown() - Gracefully disconnect from all servers
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._servers: dict[str, MCPServerConfig] = {}
        self._discovered_tools: list[MCPTool] = []
        self._connected: set[str] = set()
        self._transports: dict[str, _BaseTransport] = {}

        logger.info("mcp_client.initialized")

    async def initialize(self, server_configs: list[dict[str, Any]]) -> None:
        """Connect to all configured MCP servers."""

        for config_dict in server_configs:
            try:
                config = MCPServerConfig(**config_dict)
                self._servers[config.name] = config

                transport_kind = config.transport.strip().lower()
                if transport_kind == "stdio":
                    transport = await _StdioTransport.start(config)
                elif transport_kind == "streamable_http":
                    transport = _HTTPTransport(config)
                else:
                    raise ValueError(
                        f"Unsupported MCP transport '{config.transport}' for server '{config.name}'."
                    )

                # Close any existing transport for this name (duplicate config)
                existing = self._transports.get(config.name)
                if existing is not None:
                    try:
                        await existing.close()
                    except Exception:
                        pass
                self._transports[config.name] = transport
                self._connected.add(config.name)

                await self._try_initialize_handshake(config.name)

                logger.info(
                    "mcp_client.server_connected",
                    name=config.name,
                    transport=config.transport,
                )
            except Exception as e:
                logger.error(
                    "mcp_client.connection_failed",
                    name=config_dict.get("name", "unknown"),
                    error=str(e),
                )

    async def _try_initialize_handshake(self, server_name: str) -> None:
        """Attempt MCP initialize handshake; continue if unsupported."""

        transport = self._transports.get(server_name)
        if transport is None:
            return

        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "gwenn", "version": __version__},
        }

        try:
            await transport.request("initialize", params)
            try:
                await transport.notify("notifications/initialized", {})
            except Exception:
                # Some servers don't require this notification.
                pass
        except Exception as exc:
            logger.debug(
                "mcp_client.initialize_handshake_skipped",
                server=server_name,
                error=str(exc),
            )

    async def discover_tools(self) -> list[MCPTool]:
        """Query all connected servers for available tools (tools/list)."""

        all_tools: list[MCPTool] = []

        for server_name in sorted(self._connected):
            transport = self._transports.get(server_name)
            if transport is None:
                continue
            try:
                logger.info("mcp_client.discovering_tools", server=server_name)
                result = await transport.request("tools/list", {})
                parsed_tools = self._parse_tools_list(server_name, result)
                all_tools.extend(parsed_tools)
                logger.info(
                    "mcp_client.server_tools_discovered",
                    server=server_name,
                    count=len(parsed_tools),
                )
            except Exception as e:
                logger.error(
                    "mcp_client.discovery_failed",
                    server=server_name,
                    error=str(e),
                )

        self._discovered_tools = list(all_tools)
        logger.info("mcp_client.discovery_complete", tool_count=len(all_tools))
        return all_tools

    def _parse_tools_list(self, server_name: str, result: Any) -> list[MCPTool]:
        tools_payload: list[Any]
        if isinstance(result, dict):
            raw_tools = result.get("tools", [])
            tools_payload = raw_tools if isinstance(raw_tools, list) else []
        elif isinstance(result, list):
            tools_payload = result
        else:
            tools_payload = []

        parsed: list[MCPTool] = []
        for item in tools_payload:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue

            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                description = f"MCP tool '{name}' from server '{server_name}'."

            schema = (
                item.get("inputSchema")
                or item.get("input_schema")
                or item.get("parameters")
                or {"type": "object", "properties": {}}
            )
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}

            parsed.append(
                MCPTool(
                    name=name.strip(),
                    description=description,
                    input_schema=schema,
                    server_name=server_name,
                )
            )
        return parsed

    async def register_tools(self) -> int:
        """Register all discovered MCP tools in the ToolRegistry."""

        count = 0
        registered_names: dict[str, str] = {}  # truncated_name → original tool key
        for tool in self._discovered_tools:
            server_name = tool.server_name
            tool_name = tool.name

            def handler_factory(sn: str, tn: str):
                async def handler(**kwargs):
                    return await self.execute_tool(sn, tn, kwargs)

                return handler

            handler = handler_factory(server_name, tool_name)
            # Sanitize both components so the combined name is a valid Claude tool
            # identifier (^[a-zA-Z0-9_-]{1,64}$).
            safe_server = _sanitize_mcp_name_part(server_name)
            safe_tool = _sanitize_mcp_name_part(tool_name)
            registered_name = f"mcp_{safe_server}_{safe_tool}"[:64]

            # Detect collisions caused by truncation and disambiguate
            tool_key = f"{server_name}/{tool_name}"
            existing_key = registered_names.get(registered_name)
            if existing_key is not None and existing_key != tool_key:
                logger.warning(
                    "mcp_client.name_collision",
                    truncated_name=registered_name,
                    existing_tool=existing_key,
                    new_tool=tool_key,
                )
                # Append a numeric suffix to disambiguate
                for suffix in range(2, 100):
                    candidate = f"{registered_name[:60]}_{suffix}"
                    if candidate not in registered_names:
                        registered_name = candidate
                        break

            registered_names[registered_name] = tool_key
            self._registry.register(
                ToolDefinition(
                    name=registered_name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    handler=handler,
                    risk_level="medium",
                    category=f"mcp:{server_name}",
                    is_builtin=False,
                    requires_approval=False,
                ),
                allow_override=True,
            )
            count += 1

        logger.info("mcp_client.tools_registered", count=count)
        return count

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool on an MCP server via tools/call."""

        if server_name not in self._connected:
            raise RuntimeError(f"MCP server '{server_name}' is not connected")

        transport = self._transports.get(server_name)
        if transport is None:
            raise RuntimeError(f"MCP server '{server_name}' has no active transport")

        logger.info("mcp_client.executing_tool", server=server_name, tool=tool_name)
        result = await transport.request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        return _render_mcp_call_result(result)

    async def shutdown(self) -> None:
        """Gracefully disconnect from all MCP servers."""

        for server_name in list(self._connected):
            transport = self._transports.get(server_name)
            try:
                if transport is not None:
                    await transport.close()
                logger.info("mcp_client.server_disconnected", name=server_name)
            except Exception as e:
                logger.error("mcp_client.shutdown_error", server=server_name, error=str(e))
            finally:
                self._connected.discard(server_name)
                self._transports.pop(server_name, None)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "configured_servers": len(self._servers),
            "connected_servers": len(self._connected),
            "discovered_tools": len(self._discovered_tools),
        }
