"""
Tool Proxy — Bridging Docker Subagents to Parent Tools.

Docker subagents are isolated processes with no direct access to the parent's
tool registry, MCP connections, or stateful builtins. This module provides the
bridge:

  ToolProxyServer (parent-side)
    Reads JSON-RPC ``tool/call`` requests from the subagent's stdout and
    dispatches them to the parent's real ToolExecutor/MCPClient. Returns
    results as JSON-RPC responses via the subagent's stdin.

  ToolProxyClient (subagent-side)
    Registered as the handler for each proxied tool inside the Docker
    container. When the subagent's AgenticLoop calls a tool, the client
    serializes it as a JSON-RPC request, writes it to stdout, and reads
    the response from stdin.

Security: The server validates every proxied call against the subagent's
allowed tool list before executing. Tools not in spec.tools are rejected.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Optional

import structlog

from gwenn.orchestration.models import JsonRpcRequest, JsonRpcResponse

logger = structlog.get_logger(__name__)


class ToolProxyServer:
    """Parent-side proxy: handles tool calls from Docker subagents.

    Reads JSON-RPC requests from a subprocess stdout stream and dispatches
    to the parent's tool handlers. Writes results back via stdin.
    """

    def __init__(
        self,
        tool_registry: Any,  # ToolRegistry
        tool_executor: Any,  # ToolExecutor
        allowed_tools: list[str],
    ):
        self._registry = tool_registry
        self._executor = tool_executor
        # Empty list means "use all medium-risk tools" (matches InProcessSubagentRunner)
        if allowed_tools:
            self._allowed_tools = set(allowed_tools)
        else:
            self._allowed_tools = {
                t.name for t in tool_registry.get_api_tools(max_risk="medium")
            }

    async def handle_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process a single JSON-RPC tool/call request and return the response."""
        req_id = request_data.get("id")
        method = request_data.get("method", "")
        params = request_data.get("params", {})

        if method != "tool/call":
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32601, "message": f"Unknown method: {method}"},
            ).model_dump()

        tool_name = params.get("name", "")
        tool_args = params.get("args", {})

        # Security: validate against allowed tools
        if tool_name not in self._allowed_tools:
            return JsonRpcResponse(
                id=req_id,
                error={
                    "code": -32600,
                    "message": f"Tool '{tool_name}' is not in the allowed list",
                },
            ).model_dump()

        # Look up handler
        handler = self._registry.get_handler(tool_name)
        if handler is None:
            return JsonRpcResponse(
                id=req_id,
                error={
                    "code": -32602,
                    "message": f"No handler found for tool '{tool_name}'",
                },
            ).model_dump()

        # Execute
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: handler(**tool_args))

            return JsonRpcResponse(
                id=req_id,
                result={"tool_result": result if isinstance(result, str) else json.dumps(result)},
            ).model_dump()

        except Exception as exc:
            logger.warning(
                "tool_proxy.execution_error",
                tool=tool_name,
                error=str(exc),
            )
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32000, "message": str(exc)},
            ).model_dump()

    async def serve_subprocess(
        self,
        proc_stdin: asyncio.StreamWriter,
        proc_stdout: asyncio.StreamReader,
    ) -> Optional[dict[str, Any]]:
        """Serve tool calls from a subprocess until it sends a final result.

        Reads line-delimited JSON-RPC from proc_stdout. Handles:
          - tool/call -> dispatches and returns result via proc_stdin
          - subagent/spawn -> returns the request for the Orchestrator to handle
          - Any response with no method -> treated as final result

        Returns the final result dict when the subagent completes.
        """
        while True:
            line = await proc_stdout.readline()
            if not line:
                return None  # EOF — subprocess ended

            try:
                data = json.loads(line.decode("utf-8").strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # Skip malformed lines

            method = data.get("method")

            if method == "tool/call":
                response = await self.handle_request(data)
                response_line = json.dumps(response) + "\n"
                proc_stdin.write(response_line.encode("utf-8"))
                await proc_stdin.drain()

            elif method == "subagent/spawn":
                # Return to orchestrator for handling
                return data

            elif "result" in data and method is None:
                # Final result from the subagent
                return data

            else:
                logger.debug("tool_proxy.unknown_message", data=data)


class ToolProxyClient:
    """Subagent-side proxy: wraps tool calls as JSON-RPC requests.

    Inside a Docker container, each proxied tool gets a ToolProxyClient
    instance as its handler. When the AgenticLoop calls the tool, this
    serializes it and sends it to the parent via stdout/stdin.
    """

    def __init__(self, tool_name: str):
        self._tool_name = tool_name

    async def __call__(self, **kwargs: Any) -> str:
        """Send a tool call to the parent and return the result."""
        request = JsonRpcRequest(
            method="tool/call",
            params={"name": self._tool_name, "args": kwargs},
        )

        # Write request to stdout (parent reads this)
        # Use run_in_executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        request_line = request.model_dump_json() + "\n"

        def _write() -> None:
            sys.stdout.write(request_line)
            sys.stdout.flush()

        await loop.run_in_executor(None, _write)

        # Read response from stdin (parent writes this)
        response_line = await loop.run_in_executor(None, sys.stdin.readline)
        if not response_line:
            return "[Error: parent process closed connection]"

        try:
            response = json.loads(response_line.strip())
        except json.JSONDecodeError:
            return "[Error: invalid response from parent]"

        if "error" in response and response["error"]:
            error = response["error"]
            return f"[Tool error: {error.get('message', 'unknown error')}]"

        result = response.get("result", {})
        return result.get("tool_result", str(result))
