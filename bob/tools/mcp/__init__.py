"""
MCP Client — Bob's Connection to External Tool Servers.

The Model Context Protocol (MCP) allows Bob to connect to external services
through a standardized protocol. Instead of hardcoding integrations for every
service, MCP servers expose tools/resources/prompts through a common interface.

This module implements the MCP client that:
1. Connects to configured MCP servers at startup
2. Discovers available tools via tools/list
3. Registers discovered tools in the ToolRegistry
4. Handles tool execution by proxying calls to the appropriate server

In production, this connects to servers like:
- File system servers (for persistent file access)
- Database servers (for querying data stores)
- API servers (for external service integration)
- Knowledge base servers (for RAG pipeline access)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from bob.tools.registry import ToolDefinition, ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""
    name: str
    transport: str = "stdio"           # "stdio" or "streamable_http"
    command: Optional[str] = None      # For stdio: command to launch server
    args: list[str] = field(default_factory=list)
    url: Optional[str] = None          # For HTTP: server URL
    api_key: Optional[str] = None      # For HTTP: authentication
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPClient:
    """
    Client for connecting to MCP servers and discovering tools.

    The client manages connections to multiple MCP servers, discovers their
    tools, and registers proxy handlers in the ToolRegistry so Claude can
    use MCP tools transparently alongside built-in tools.

    Lifecycle:
    1. initialize() - Connect to all configured servers
    2. discover_tools() - Query each server for available tools
    3. register_tools() - Add discovered tools to the ToolRegistry
    4. execute() - Proxy tool calls to the appropriate server
    5. shutdown() - Gracefully disconnect from all servers
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._servers: dict[str, MCPServerConfig] = {}
        self._discovered_tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._connected: set[str] = set()

        logger.info("mcp_client.initialized")

    async def initialize(self, server_configs: list[dict[str, Any]]) -> None:
        """
        Connect to all configured MCP servers.

        Each server configuration specifies the transport type and connection
        details. For stdio servers, we launch the server process. For HTTP
        servers, we establish the connection.
        """
        for config_dict in server_configs:
            try:
                config = MCPServerConfig(**config_dict)
                self._servers[config.name] = config

                # In production, this would actually launch/connect to the server.
                # For the foundational codebase, we track the configuration and
                # simulate the connection protocol.
                self._connected.add(config.name)

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

    async def discover_tools(self) -> list[MCPTool]:
        """
        Query all connected servers for available tools (tools/list).

        Returns all discovered tools. In production, this sends a JSON-RPC
        request to each server:
            {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        And receives the tool definitions in response.
        """
        all_tools = []

        for server_name in self._connected:
            try:
                # Production: send tools/list JSON-RPC request to server
                # For foundational codebase: log the discovery attempt
                logger.info("mcp_client.discovering_tools", server=server_name)

                # Tools would be returned from the server here
                # Each tool has: name, description, inputSchema
                # We'd parse them into MCPTool objects

            except Exception as e:
                logger.error(
                    "mcp_client.discovery_failed",
                    server=server_name,
                    error=str(e),
                )

        self._discovered_tools = {tool.name: tool for tool in all_tools}
        logger.info("mcp_client.discovery_complete", tool_count=len(all_tools))
        return all_tools

    async def register_tools(self) -> int:
        """
        Register all discovered MCP tools in the ToolRegistry.

        Each MCP tool gets a proxy handler that forwards execution to the
        appropriate MCP server. From Claude's perspective, MCP tools are
        indistinguishable from built-in tools.
        """
        count = 0
        for tool in self._discovered_tools.values():
            # Create a proxy handler that forwards to the MCP server
            server_name = tool.server_name

            async def make_handler(sn: str, tn: str):
                async def handler(**kwargs):
                    return await self.execute_tool(sn, tn, kwargs)
                return handler

            handler = await make_handler(server_name, tool.name)

            self._registry.register(ToolDefinition(
                name=f"mcp_{tool.server_name}_{tool.name}",
                description=tool.description,
                input_schema=tool.input_schema,
                handler=handler,
                risk_level="medium",  # MCP tools are medium-risk by default
                category=f"mcp:{tool.server_name}",
            ))
            count += 1

        logger.info("mcp_client.tools_registered", count=count)
        return count

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute a tool on an MCP server via tools/call.

        Production implementation sends:
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": unique_id
            }
        And returns the result from the server's response.
        """
        if server_name not in self._connected:
            raise RuntimeError(f"MCP server '{server_name}' is not connected")

        logger.info(
            "mcp_client.executing_tool",
            server=server_name,
            tool=tool_name,
        )

        # Production: send tools/call JSON-RPC to server and parse response
        # The response contains content blocks (text, images, etc.)
        # We'd extract and return the content

        return f"MCP tool {tool_name} executed on {server_name}"

    async def shutdown(self) -> None:
        """Gracefully disconnect from all MCP servers."""
        for server_name in list(self._connected):
            try:
                # Production: send graceful shutdown signal
                self._connected.discard(server_name)
                logger.info("mcp_client.server_disconnected", name=server_name)
            except Exception as e:
                logger.error("mcp_client.shutdown_error", server=server_name, error=str(e))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "configured_servers": len(self._servers),
            "connected_servers": len(self._connected),
            "discovered_tools": len(self._discovered_tools),
        }
