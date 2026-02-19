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


class _FakeTransport:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self.closed = False

    async def request(self, method: str, params: dict | None = None):
        self.calls.append((method, params))
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search docs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    }
                ]
            }
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "search ok"}]}
        raise AssertionError(f"Unexpected method: {method}")

    async def notify(self, method: str, params: dict | None = None):  # noqa: ARG002
        return None

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_discover_and_execute_tool_via_transport():
    registry = ToolRegistry()
    client = MCPClient(registry)
    fake = _FakeTransport()
    client._connected = {"demo"}
    client._transports = {"demo": fake}

    discovered = await client.discover_tools()
    assert len(discovered) == 1
    assert discovered[0].name == "search"
    assert discovered[0].server_name == "demo"

    count = await client.register_tools()
    assert count == 1
    assert registry.get("mcp_demo_search") is not None

    result = await client.execute_tool("demo", "search", {"q": "gwenn"})
    assert result == "search ok"
