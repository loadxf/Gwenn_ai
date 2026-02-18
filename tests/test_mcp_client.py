from __future__ import annotations

import pytest

from gwenn.tools.mcp import MCPClient, MCPTool
from gwenn.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_register_tools_keeps_duplicate_tool_names_across_servers():
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._discovered_tools = [
        MCPTool(
            name="search",
            description="Search server A",
            input_schema={"type": "object", "properties": {}},
            server_name="server_a",
        ),
        MCPTool(
            name="search",
            description="Search server B",
            input_schema={"type": "object", "properties": {}},
            server_name="server_b",
        ),
    ]

    count = await client.register_tools()

    assert count == 2
    assert registry.get("mcp_server_a_search") is not None
    assert registry.get("mcp_server_b_search") is not None
