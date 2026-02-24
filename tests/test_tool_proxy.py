"""Tests for gwenn.orchestration.tool_proxy — ToolProxyServer and ToolProxyClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gwenn.orchestration.tool_proxy import ToolProxyClient, ToolProxyServer
from gwenn.tools.registry import ToolDefinition, ToolRegistry


@pytest.fixture
def registry_with_tools():
    registry = ToolRegistry()

    async def handle_calculate(expression: str = "") -> str:
        return f"result: {eval(expression)}"

    def handle_greet(name: str = "world") -> str:
        return f"Hello, {name}!"

    registry.register(
        ToolDefinition(
            name="calculate",
            description="Calculate",
            input_schema={"type": "object", "properties": {}},
            handler=handle_calculate,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="greet",
            description="Greet",
            input_schema={"type": "object", "properties": {}},
            handler=handle_greet,
            risk_level="low",
        )
    )
    return registry


@pytest.fixture
def proxy_server(registry_with_tools):
    executor = MagicMock()
    return ToolProxyServer(
        tool_registry=registry_with_tools,
        tool_executor=executor,
        allowed_tools=["calculate", "greet"],
    )


class TestToolProxyServer:
    @pytest.mark.asyncio
    async def test_handle_allowed_tool_async(self, proxy_server):
        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "calculate", "args": {"expression": "2+2"}},
            "id": "1",
        }
        response = await proxy_server.handle_request(request)
        assert response["id"] == "1"
        assert response["result"]["tool_result"] == "result: 4"
        assert response.get("error") is None

    @pytest.mark.asyncio
    async def test_handle_allowed_tool_sync(self, proxy_server):
        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "greet", "args": {"name": "Gwenn"}},
            "id": "2",
        }
        response = await proxy_server.handle_request(request)
        assert response["result"]["tool_result"] == "Hello, Gwenn!"

    @pytest.mark.asyncio
    async def test_handle_disallowed_tool(self, registry_with_tools):
        executor = MagicMock()
        server = ToolProxyServer(
            tool_registry=registry_with_tools,
            tool_executor=executor,
            allowed_tools=["calculate"],  # greet NOT allowed
        )
        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "greet", "args": {}},
            "id": "3",
        }
        response = await server.handle_request(request)
        assert response["error"] is not None
        assert "not in the allowed list" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, proxy_server):
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": "4",
        }
        response = await proxy_server.handle_request(request)
        assert response["error"] is not None
        assert "Unknown method" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_missing_handler(self, proxy_server):
        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "nonexistent_tool", "args": {}},
            "id": "5",
        }
        response = await proxy_server.handle_request(request)
        assert response["error"] is not None

    @pytest.mark.asyncio
    async def test_handle_tool_execution_error(self):
        registry = ToolRegistry()

        async def handle_broken(**kwargs):
            raise ValueError("broken tool")

        registry.register(
            ToolDefinition(
                name="broken",
                description="broken",
                input_schema={"type": "object", "properties": {}},
                handler=handle_broken,
                risk_level="low",
            )
        )
        executor = MagicMock()
        server = ToolProxyServer(
            tool_registry=registry,
            tool_executor=executor,
            allowed_tools=["broken"],
        )
        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "broken", "args": {}},
            "id": "6",
        }
        response = await server.handle_request(request)
        assert response["error"] is not None
        assert "broken tool" in response["error"]["message"]


class TestToolProxyClient:
    def test_client_has_tool_name(self):
        client = ToolProxyClient("calculate")
        assert client._tool_name == "calculate"
