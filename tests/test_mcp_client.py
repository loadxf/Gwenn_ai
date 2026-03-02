from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.tools.mcp import (
    MCPClient,
    MCPServerConfig,
    MCPTool,
    _BaseTransport,
    _content_block_to_text,
    _extract_jsonrpc_result,
    _HTTPTransport,
    _render_mcp_call_result,
    _sanitize_mcp_name_part,
    _StdioTransport,
)
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


# ---------------------------------------------------------------------------
# MCPServerConfig defaults
# ---------------------------------------------------------------------------


def test_mcp_server_config_defaults():
    cfg = MCPServerConfig(name="test")
    assert cfg.transport == "stdio"
    assert cfg.command is None
    assert cfg.args == []
    assert cfg.url is None
    assert cfg.api_key is None
    assert cfg.env == {}
    assert cfg.timeout_seconds == 20.0


# ---------------------------------------------------------------------------
# _sanitize_mcp_name_part
# ---------------------------------------------------------------------------


def test_sanitize_mcp_name_part_basic():
    assert _sanitize_mcp_name_part("hello-world_123") == "hello-world_123"


def test_sanitize_mcp_name_part_special_chars():
    assert _sanitize_mcp_name_part("a.b/c@d") == "a_b_c_d"


# ---------------------------------------------------------------------------
# _extract_jsonrpc_result
# ---------------------------------------------------------------------------


def test_extract_jsonrpc_result_non_dict():
    with pytest.raises(RuntimeError, match="Invalid JSON-RPC response type"):
        _extract_jsonrpc_result("not a dict")


def test_extract_jsonrpc_result_non_dict_list():
    with pytest.raises(RuntimeError, match="Invalid JSON-RPC response type"):
        _extract_jsonrpc_result([1, 2, 3])


def test_extract_jsonrpc_result_error_dict():
    with pytest.raises(RuntimeError, match="MCP error 42: Something broke"):
        _extract_jsonrpc_result({"error": {"code": 42, "message": "Something broke"}})


def test_extract_jsonrpc_result_error_non_dict():
    with pytest.raises(RuntimeError, match="MCP error: some string"):
        _extract_jsonrpc_result({"error": "some string"})


def test_extract_jsonrpc_result_success():
    assert _extract_jsonrpc_result({"result": {"data": 1}}) == {"data": 1}


def test_extract_jsonrpc_result_none_result():
    assert _extract_jsonrpc_result({"result": None}) is None


def test_extract_jsonrpc_result_no_result_key():
    assert _extract_jsonrpc_result({}) is None


def test_extract_jsonrpc_result_error_none_is_not_error():
    # error key present but None — treated as no error
    assert _extract_jsonrpc_result({"error": None, "result": 42}) == 42


# ---------------------------------------------------------------------------
# _content_block_to_text
# ---------------------------------------------------------------------------


def test_content_block_to_text_string():
    assert _content_block_to_text("hello") == "hello"


def test_content_block_to_text_non_dict():
    assert _content_block_to_text(42) == "42"


def test_content_block_to_text_list():
    assert _content_block_to_text([1, 2]) == "[1, 2]"


def test_content_block_to_text_dict_type_text():
    assert _content_block_to_text({"type": "text", "text": "hello"}) == "hello"


def test_content_block_to_text_dict_type_text_empty():
    assert _content_block_to_text({"type": "text"}) == ""


def test_content_block_to_text_dict_text_but_not_type_text():
    # Has "text" key but type is "image"
    assert _content_block_to_text({"type": "image", "text": "caption"}) == "caption"


def test_content_block_to_text_dict_no_text():
    result = _content_block_to_text({"type": "image", "url": "http://x.png"})
    parsed = json.loads(result)
    assert parsed == {"type": "image", "url": "http://x.png"}


# ---------------------------------------------------------------------------
# _render_mcp_call_result
# ---------------------------------------------------------------------------


def test_render_mcp_call_result_is_error_with_list_content():
    with pytest.raises(RuntimeError, match="bad thing happened"):
        _render_mcp_call_result({
            "isError": True,
            "content": [{"type": "text", "text": "bad thing happened"}],
        })


def test_render_mcp_call_result_is_error_with_empty_list_content():
    with pytest.raises(RuntimeError, match="MCP tool returned an error result"):
        _render_mcp_call_result({"isError": True, "content": []})


def test_render_mcp_call_result_is_error_non_list_content():
    with pytest.raises(RuntimeError, match="something went wrong"):
        _render_mcp_call_result({"isError": True, "content": "something went wrong"})


def test_render_mcp_call_result_is_error_none_content():
    # str(None) == "None", which is truthy, so it raises RuntimeError("None")
    with pytest.raises(RuntimeError, match="None"):
        _render_mcp_call_result({"isError": True, "content": None})


def test_render_mcp_call_result_string():
    assert _render_mcp_call_result("just text") == "just text"


def test_render_mcp_call_result_non_dict_non_string():
    assert _render_mcp_call_result(123) == "123"


def test_render_mcp_call_result_none():
    assert _render_mcp_call_result(None) == "null"


def test_render_mcp_call_result_dict_with_content_list():
    result = _render_mcp_call_result({
        "content": [{"type": "text", "text": "hello"}],
    })
    assert result == "hello"


def test_render_mcp_call_result_dict_content_empty_text():
    # content list produces empty text => falls back to JSON
    result = _render_mcp_call_result({"content": []})
    assert result == '{"content": []}'


def test_render_mcp_call_result_dict_without_content():
    result = _render_mcp_call_result({"something": "else"})
    parsed = json.loads(result)
    assert parsed == {"something": "else"}


# ---------------------------------------------------------------------------
# _BaseTransport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_transport_not_implemented():
    t = _BaseTransport()
    with pytest.raises(NotImplementedError):
        await t.request("test")
    with pytest.raises(NotImplementedError):
        await t.notify("test")
    with pytest.raises(NotImplementedError):
        await t.close()


# ---------------------------------------------------------------------------
# _HTTPTransport
# ---------------------------------------------------------------------------


def test_http_transport_missing_url():
    cfg = MCPServerConfig(name="nourl", transport="streamable_http")
    with pytest.raises(ValueError, match="missing URL"):
        _HTTPTransport(cfg)


def test_http_transport_init_with_api_key():
    cfg = MCPServerConfig(
        name="test", transport="streamable_http", url="http://localhost:8080", api_key="secret"
    )
    transport = _HTTPTransport(cfg)
    assert transport._headers["Authorization"] == "Bearer secret"
    assert transport._request_id == 0


def test_http_transport_init_no_api_key():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)
    assert "Authorization" not in transport._headers


@pytest.mark.asyncio
async def test_http_transport_request():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)

    mock_response = MagicMock()
    mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    mock_response.raise_for_status = MagicMock()

    transport._client = AsyncMock()
    transport._client.post = AsyncMock(return_value=mock_response)

    result = await transport.request("test/method", {"key": "value"})
    assert result == {"ok": True}
    assert transport._request_id == 1

    call_args = transport._client.post.call_args
    assert call_args[0][0] == "http://localhost:8080"
    payload = call_args[1]["json"]
    assert payload["method"] == "test/method"
    assert payload["params"] == {"key": "value"}
    assert payload["id"] == 1


@pytest.mark.asyncio
async def test_http_transport_request_no_params():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)

    mock_response = MagicMock()
    mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": None}
    mock_response.raise_for_status = MagicMock()

    transport._client = AsyncMock()
    transport._client.post = AsyncMock(return_value=mock_response)

    result = await transport.request("test/method")
    payload = transport._client.post.call_args[1]["json"]
    assert "params" not in payload


@pytest.mark.asyncio
async def test_http_transport_notify():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    transport._client = AsyncMock()
    transport._client.post = AsyncMock(return_value=mock_response)

    await transport.notify("some/notification", {"data": 1})

    call_args = transport._client.post.call_args
    payload = call_args[1]["json"]
    assert payload["method"] == "some/notification"
    assert payload["params"] == {"data": 1}
    assert "id" not in payload


@pytest.mark.asyncio
async def test_http_transport_notify_no_params():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    transport._client = AsyncMock()
    transport._client.post = AsyncMock(return_value=mock_response)

    await transport.notify("some/notification")

    payload = transport._client.post.call_args[1]["json"]
    assert "params" not in payload


@pytest.mark.asyncio
async def test_http_transport_close():
    cfg = MCPServerConfig(name="test", transport="streamable_http", url="http://localhost:8080")
    transport = _HTTPTransport(cfg)
    transport._client = AsyncMock()
    transport._client.aclose = AsyncMock()

    await transport.close()
    transport._client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# _StdioTransport
# ---------------------------------------------------------------------------


def _make_mock_process(stdout_data=b"", stderr_data=b"", returncode=None):
    """Create a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()

    # stdout as a StreamReader-like mock
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(return_value=stdout_data)
    proc.stdout.readexactly = AsyncMock(return_value=stdout_data)

    # stderr as a StreamReader-like mock
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=stderr_data)

    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    return proc


def test_stdio_transport_init_with_stderr():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process(stderr_data=b"")
    # stderr task is created
    with patch("asyncio.create_task") as mock_create_task:
        mock_create_task.return_value = MagicMock()
        transport = _StdioTransport(cfg, proc)
        assert transport._stderr_task is not None
        # Close the unawaited coroutine passed to the mocked create_task.
        coro = mock_create_task.call_args[0][0]
        coro.close()


def test_stdio_transport_init_without_stderr():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)
    assert transport._stderr_task is None


@pytest.mark.asyncio
async def test_stdio_transport_start():
    cfg = MCPServerConfig(name="test", command="echo", args=["hello"], env={"FOO": "BAR"})
    mock_proc = _make_mock_process()

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = MagicMock()
            transport = await _StdioTransport.start(cfg)
            assert transport._process is mock_proc
            # Close the unawaited _drain_stderr coroutine passed to mocked create_task.
            if mock_create_task.call_args:
                coro = mock_create_task.call_args[0][0]
                coro.close()


@pytest.mark.asyncio
async def test_stdio_transport_start_missing_command():
    cfg = MCPServerConfig(name="test")  # no command
    with pytest.raises(ValueError, match="missing 'command'"):
        await _StdioTransport.start(cfg)


@pytest.mark.asyncio
async def test_stdio_transport_drain_stderr():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    # stderr returns lines then empty (EOF)
    proc.stderr.readline = AsyncMock(side_effect=[b"some error\n", b"", b""])

    with patch("asyncio.create_task", side_effect=lambda coro: asyncio.ensure_future(coro)):
        transport = _StdioTransport(cfg, proc)
        # Let the drain task run
        await asyncio.sleep(0.05)
        if transport._stderr_task:
            await transport._stderr_task


@pytest.mark.asyncio
async def test_stdio_transport_send_message():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    payload = {"jsonrpc": "2.0", "id": 1, "method": "test"}
    await transport._send_message(payload)

    proc.stdin.write.assert_called_once()
    written = proc.stdin.write.call_args[0][0]
    assert b"Content-Length:" in written
    proc.stdin.drain.assert_awaited_once()


@pytest.mark.asyncio
async def test_stdio_transport_read_message_success():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"}).encode("utf-8")
    header_line = f"Content-Length: {len(body)}\r\n".encode("utf-8")

    proc.stdout.readline = AsyncMock(side_effect=[header_line, b"\r\n"])
    proc.stdout.readexactly = AsyncMock(return_value=body)

    transport = _StdioTransport(cfg, proc)
    msg = await transport._read_message()
    assert msg == {"jsonrpc": "2.0", "id": 1, "result": "ok"}


@pytest.mark.asyncio
async def test_stdio_transport_read_message_eof():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    proc.stdout.readline = AsyncMock(return_value=b"")

    transport = _StdioTransport(cfg, proc)
    with pytest.raises(RuntimeError, match="closed unexpectedly"):
        await transport._read_message()


@pytest.mark.asyncio
async def test_stdio_transport_read_message_no_content_length():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    # Header line without Content-Length, then blank line
    proc.stdout.readline = AsyncMock(side_effect=[b"X-Custom: foo\r\n", b"\r\n"])

    transport = _StdioTransport(cfg, proc)
    with pytest.raises(RuntimeError, match="missing Content-Length"):
        await transport._read_message()


@pytest.mark.asyncio
async def test_stdio_transport_read_message_invalid_content_length():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    proc.stdout.readline = AsyncMock(side_effect=[b"Content-Length: abc\r\n", b"\r\n"])

    transport = _StdioTransport(cfg, proc)
    with pytest.raises(RuntimeError, match="invalid Content-Length"):
        await transport._read_message()


@pytest.mark.asyncio
async def test_stdio_transport_read_message_non_object_json():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None

    body = json.dumps([1, 2, 3]).encode("utf-8")
    proc.stdout.readline = AsyncMock(
        side_effect=[f"Content-Length: {len(body)}\r\n".encode(), b"\r\n"]
    )
    proc.stdout.readexactly = AsyncMock(return_value=body)

    transport = _StdioTransport(cfg, proc)
    with pytest.raises(RuntimeError, match="non-object JSON"):
        await transport._read_message()


@pytest.mark.asyncio
async def test_stdio_transport_read_message_skips_headerless_line():
    """Lines without ':' in the header section are skipped."""
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"}).encode("utf-8")
    proc.stdout.readline = AsyncMock(side_effect=[
        b"GARBAGE_NO_COLON\r\n",
        f"Content-Length: {len(body)}\r\n".encode(),
        b"\r\n",
    ])
    proc.stdout.readexactly = AsyncMock(return_value=body)

    transport = _StdioTransport(cfg, proc)
    msg = await transport._read_message()
    assert msg["result"] == "ok"


@pytest.mark.asyncio
async def test_stdio_transport_request():
    cfg = MCPServerConfig(name="test", command="echo", timeout_seconds=5.0)
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"data": 42}}).encode("utf-8")

    proc.stdout.readline = AsyncMock(side_effect=[
        f"Content-Length: {len(body)}\r\n".encode(),
        b"\r\n",
    ])
    proc.stdout.readexactly = AsyncMock(return_value=body)

    result = await transport.request("test/method", {"key": "value"})
    assert result == {"data": 42}
    assert transport._request_id == 1

    # Check that payload was written
    written = proc.stdin.write.call_args[0][0]
    assert b'"method": "test/method"' in written
    assert b'"params"' in written


@pytest.mark.asyncio
async def test_stdio_transport_request_no_params():
    cfg = MCPServerConfig(name="test", command="echo", timeout_seconds=5.0)
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": None}).encode("utf-8")

    proc.stdout.readline = AsyncMock(side_effect=[
        f"Content-Length: {len(body)}\r\n".encode(),
        b"\r\n",
    ])
    proc.stdout.readexactly = AsyncMock(return_value=body)

    result = await transport.request("test/method")
    written = proc.stdin.write.call_args[0][0]
    assert b'"params"' not in written


@pytest.mark.asyncio
async def test_stdio_transport_request_timeout():
    cfg = MCPServerConfig(name="test", command="echo", timeout_seconds=0.01)
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    # _read_message hangs forever
    async def hang_forever():
        await asyncio.sleep(999)
        return {}

    with patch.object(transport, "_read_message", side_effect=hang_forever):
        with pytest.raises(RuntimeError, match="timed out"):
            await transport.request("test/method")


@pytest.mark.asyncio
async def test_stdio_transport_request_skips_mismatched_ids():
    cfg = MCPServerConfig(name="test", command="echo", timeout_seconds=5.0)
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    # First response has wrong id, second has correct id
    wrong_msg = {"jsonrpc": "2.0", "id": 999, "method": "notification"}
    correct_msg = {"jsonrpc": "2.0", "id": 1, "result": "correct"}

    with patch.object(
        transport,
        "_read_message",
        AsyncMock(side_effect=[wrong_msg, correct_msg]),
    ):
        result = await transport.request("test/method")
        assert result == "correct"


@pytest.mark.asyncio
async def test_stdio_transport_notify():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    await transport.notify("some/notification", {"data": 1})

    written = proc.stdin.write.call_args[0][0]
    assert b'"method": "some/notification"' in written
    assert b'"params"' in written
    assert b'"id"' not in written


@pytest.mark.asyncio
async def test_stdio_transport_notify_no_params():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process()
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    await transport.notify("some/notification")

    written = proc.stdin.write.call_args[0][0]
    assert b'"params"' not in written


@pytest.mark.asyncio
async def test_stdio_transport_close_terminate_succeeds():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process(returncode=None)
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    await transport.close()

    proc.terminate.assert_called_once()
    proc.wait.assert_awaited()
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_stdio_transport_close_terminate_timeout_then_kill():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process(returncode=None)
    proc.stderr = None
    # wait() times out on first call, succeeds on second (after kill)
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError, None])
    transport = _StdioTransport(cfg, proc)

    await transport.close()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_stdio_transport_close_already_exited():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process(returncode=0)  # already exited
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    await transport.close()

    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_stdio_transport_close_cancels_stderr_task():
    cfg = MCPServerConfig(name="test", command="echo")
    proc = _make_mock_process(returncode=0)
    proc.stderr = None
    transport = _StdioTransport(cfg, proc)

    # Create a real task that sleeps forever so we can cancel it
    async def never_finish():
        await asyncio.sleep(9999)

    task = asyncio.create_task(never_finish())
    transport._stderr_task = task

    await transport.close()
    assert task.cancelled()


# ---------------------------------------------------------------------------
# MCPClient.initialize - stdio transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_stdio():
    registry = ToolRegistry()
    client = MCPClient(registry)

    mock_transport = MagicMock(spec=_StdioTransport)
    mock_transport.request = AsyncMock(return_value={"result": {}})
    mock_transport.notify = AsyncMock()

    with patch(
        "gwenn.tools.mcp._StdioTransport.start",
        new_callable=AsyncMock,
        return_value=mock_transport,
    ):
        await client.initialize([{"name": "test_server", "transport": "stdio", "command": "echo"}])

    assert "test_server" in client._connected
    assert "test_server" in client._transports


@pytest.mark.asyncio
async def test_initialize_http():
    registry = ToolRegistry()
    client = MCPClient(registry)

    fake_transport = AsyncMock(spec=_HTTPTransport)
    fake_transport.request = AsyncMock(return_value={"capabilities": {}})
    fake_transport.notify = AsyncMock()

    with patch("gwenn.tools.mcp._HTTPTransport", return_value=fake_transport):
        await client.initialize([
            {
                "name": "http_server",
                "transport": "streamable_http",
                "url": "http://localhost:8080",
            }
        ])

    assert "http_server" in client._connected


@pytest.mark.asyncio
async def test_initialize_unsupported_transport():
    registry = ToolRegistry()
    client = MCPClient(registry)

    # Should not raise, but log error and not connect
    await client.initialize([{"name": "bad", "transport": "grpc"}])
    assert "bad" not in client._connected


@pytest.mark.asyncio
async def test_initialize_connection_failure():
    registry = ToolRegistry()
    client = MCPClient(registry)

    with patch(
        "gwenn.tools.mcp._StdioTransport.start",
        new_callable=AsyncMock,
        side_effect=RuntimeError("process failed"),
    ):
        await client.initialize([{"name": "broken", "transport": "stdio", "command": "bad_cmd"}])

    assert "broken" not in client._connected


# ---------------------------------------------------------------------------
# MCPClient._try_initialize_handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_initialize_handshake_no_transport():
    registry = ToolRegistry()
    client = MCPClient(registry)

    # Should just return without error
    await client._try_initialize_handshake("nonexistent")


@pytest.mark.asyncio
async def test_try_initialize_handshake_request_fails():
    registry = ToolRegistry()
    client = MCPClient(registry)
    mock_transport = AsyncMock()
    mock_transport.request = AsyncMock(side_effect=RuntimeError("handshake failed"))
    client._transports["test"] = mock_transport

    # Should not raise
    await client._try_initialize_handshake("test")


@pytest.mark.asyncio
async def test_try_initialize_handshake_notify_fails():
    registry = ToolRegistry()
    client = MCPClient(registry)
    mock_transport = AsyncMock()
    mock_transport.request = AsyncMock(return_value={"result": {}})
    mock_transport.notify = AsyncMock(side_effect=RuntimeError("notify failed"))
    client._transports["test"] = mock_transport

    # Should not raise — notify failure is swallowed
    await client._try_initialize_handshake("test")


# ---------------------------------------------------------------------------
# MCPClient.discover_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_tools_no_transport():
    """Connected server with missing transport is skipped."""
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._connected = {"ghost"}
    # No transport registered for "ghost"

    tools = await client.discover_tools()
    assert tools == []


@pytest.mark.asyncio
async def test_discover_tools_request_error():
    """Error during tools/list is caught gracefully."""
    registry = ToolRegistry()
    client = MCPClient(registry)
    mock_transport = AsyncMock()
    mock_transport.request = AsyncMock(side_effect=RuntimeError("connection reset"))
    client._connected = {"flaky"}
    client._transports = {"flaky": mock_transport}

    tools = await client.discover_tools()
    assert tools == []


# ---------------------------------------------------------------------------
# MCPClient._parse_tools_list edge cases
# ---------------------------------------------------------------------------


def test_parse_tools_list_result_is_list():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", [
        {"name": "tool1", "description": "desc", "inputSchema": {"type": "object", "properties": {}}},
    ])
    assert len(tools) == 1
    assert tools[0].name == "tool1"


def test_parse_tools_list_result_not_dict_or_list():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", "bad result")
    assert tools == []


def test_parse_tools_list_tools_key_not_list():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": "not a list"})
    assert tools == []


def test_parse_tools_list_item_not_dict():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": ["not_a_dict"]})
    assert tools == []


def test_parse_tools_list_missing_name():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"description": "no name"}]})
    assert tools == []


def test_parse_tools_list_empty_name():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"name": "  ", "description": "blank"}]})
    assert tools == []


def test_parse_tools_list_name_not_string():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"name": 123}]})
    assert tools == []


def test_parse_tools_list_missing_description():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"name": "tool1"}]})
    assert len(tools) == 1
    assert "MCP tool 'tool1'" in tools[0].description


def test_parse_tools_list_empty_description():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"name": "tool1", "description": "  "}]})
    assert "MCP tool 'tool1'" in tools[0].description


def test_parse_tools_list_description_not_string():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [{"name": "tool1", "description": 42}]})
    assert "MCP tool 'tool1'" in tools[0].description


def test_parse_tools_list_schema_fallbacks():
    """Test input_schema, parameters, and default fallback."""
    registry = ToolRegistry()
    client = MCPClient(registry)

    # uses input_schema
    tools = client._parse_tools_list("srv", {"tools": [
        {"name": "t1", "description": "d", "input_schema": {"type": "object", "properties": {"a": {}}}},
    ]})
    assert "a" in tools[0].input_schema.get("properties", {})

    # uses parameters
    tools = client._parse_tools_list("srv", {"tools": [
        {"name": "t2", "description": "d", "parameters": {"type": "object", "properties": {"b": {}}}},
    ]})
    assert "b" in tools[0].input_schema.get("properties", {})

    # default fallback when no schema provided
    tools = client._parse_tools_list("srv", {"tools": [{"name": "t3", "description": "d"}]})
    assert tools[0].input_schema == {"type": "object", "properties": {}}


def test_parse_tools_list_schema_not_dict():
    registry = ToolRegistry()
    client = MCPClient(registry)
    tools = client._parse_tools_list("srv", {"tools": [
        {"name": "t1", "description": "d", "inputSchema": "not a dict"},
    ]})
    assert tools[0].input_schema == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# MCPClient.register_tools - name collision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_tools_name_collision():
    """When truncated names collide, numeric suffix is appended."""
    registry = ToolRegistry()
    client = MCPClient(registry)

    # Both tools must produce the SAME truncated registered name but have different
    # server_name/tool_name keys.  The registered name is `mcp_{server}_{tool}`[:64].
    # Use long enough names that both truncate to the same 64-char prefix.
    # "mcp_" is 4 chars, "srv_" is 4 chars, leaving 56 chars for tool name.
    # If two tools share the same server but differ after char 56, they collide.
    base_tool_name = "t" * 56  # exactly fills the 64-char limit
    tool_name_a = base_tool_name + "_alpha"  # truncated to same 64 chars
    tool_name_b = base_tool_name + "_beta"   # truncated to same 64 chars

    client._discovered_tools = [
        MCPTool(
            name=tool_name_a,
            description="First",
            input_schema={"type": "object", "properties": {}},
            server_name="srv",
        ),
        MCPTool(
            name=tool_name_b,
            description="Second",
            input_schema={"type": "object", "properties": {}},
            server_name="srv",
        ),
    ]

    count = await client.register_tools()
    assert count == 2

    # The second tool should have gotten a disambiguation suffix
    # Find all tools with "mcp_srv_t" prefix
    all_names = [name for name in registry._tools if name.startswith("mcp_srv_t")]
    assert len(all_names) == 2
    # One should end with "_2"
    assert any(n.endswith("_2") for n in all_names)


# ---------------------------------------------------------------------------
# MCPClient.execute_tool errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_not_connected():
    registry = ToolRegistry()
    client = MCPClient(registry)

    with pytest.raises(RuntimeError, match="not connected"):
        await client.execute_tool("absent", "tool", {})


@pytest.mark.asyncio
async def test_execute_tool_no_transport():
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._connected.add("ghost")
    # No transport set for "ghost"

    with pytest.raises(RuntimeError, match="no active transport"):
        await client.execute_tool("ghost", "tool", {})


# ---------------------------------------------------------------------------
# MCPClient.shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_success():
    registry = ToolRegistry()
    client = MCPClient(registry)
    mock_transport = AsyncMock()
    mock_transport.close = AsyncMock()
    client._connected = {"srv"}
    client._transports = {"srv": mock_transport}

    await client.shutdown()

    mock_transport.close.assert_awaited_once()
    assert "srv" not in client._connected
    assert "srv" not in client._transports


@pytest.mark.asyncio
async def test_shutdown_error():
    registry = ToolRegistry()
    client = MCPClient(registry)
    mock_transport = AsyncMock()
    mock_transport.close = AsyncMock(side_effect=RuntimeError("close failed"))
    client._connected = {"srv"}
    client._transports = {"srv": mock_transport}

    # Should not raise
    await client.shutdown()

    assert "srv" not in client._connected
    assert "srv" not in client._transports


@pytest.mark.asyncio
async def test_shutdown_no_transport():
    """Server connected but transport already removed."""
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._connected = {"orphan"}
    # No transport for "orphan"

    await client.shutdown()
    assert "orphan" not in client._connected


# ---------------------------------------------------------------------------
# MCPClient.stats
# ---------------------------------------------------------------------------


def test_stats_property():
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._servers = {"a": MCPServerConfig(name="a"), "b": MCPServerConfig(name="b")}
    client._connected = {"a"}
    client._discovered_tools = [
        MCPTool(name="t1", description="d", input_schema={}, server_name="a"),
    ]

    stats = client.stats
    assert stats["configured_servers"] == 2
    assert stats["connected_servers"] == 1
    assert stats["discovered_tools"] == 1


# ---------------------------------------------------------------------------
# MCPClient.register_tools - handler execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registered_handler_calls_execute_tool():
    """The handler created during register_tools correctly proxies to execute_tool."""
    registry = ToolRegistry()
    client = MCPClient(registry)
    client._discovered_tools = [
        MCPTool(
            name="greet",
            description="Say hello",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            server_name="srv",
        ),
    ]
    client._connected = {"srv"}

    mock_transport = AsyncMock()
    mock_transport.request = AsyncMock(
        return_value={"content": [{"type": "text", "text": "Hello World"}]}
    )
    client._transports = {"srv": mock_transport}

    await client.register_tools()

    tool_def = registry.get("mcp_srv_greet")
    assert tool_def is not None
    result = await tool_def.handler(name="World")
    assert result == "Hello World"
