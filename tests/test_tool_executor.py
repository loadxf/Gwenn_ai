from __future__ import annotations

import asyncio
import time

import pytest

from gwenn.tools.executor import ToolExecutor
from gwenn.tools.registry import ToolDefinition, ToolRegistry


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="external_echo",
            description="external test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text="": f"echo:{text}",
            risk_level="low",
            category="test",
            is_builtin=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="builtin_echo",
            description="builtin test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text="": f"builtin:{text}",
            risk_level="low",
            category="test",
            is_builtin=True,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_sandbox_blocks_non_builtin_tools_by_default() -> None:
    registry = _make_registry()
    executor = ToolExecutor(registry=registry, sandbox_enabled=True)

    result = await executor.execute("call_1", "external_echo", {"text": "x"})

    assert result.success is False
    assert "blocked by sandbox policy" in (result.error or "")


@pytest.mark.asyncio
async def test_sandbox_allows_builtin_tools() -> None:
    registry = _make_registry()
    executor = ToolExecutor(registry=registry, sandbox_enabled=True)

    result = await executor.execute("call_2", "builtin_echo", {"text": "ok"})

    assert result.success is True
    assert result.result == "builtin:ok"


@pytest.mark.asyncio
async def test_sync_timeout_does_not_allow_unbounded_thread_growth() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow",
            description="slow sync tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: (time.sleep(0.2), "slow")[1],
            risk_level="low",
            category="test",
            is_builtin=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="fast",
            description="fast sync tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "fast",
            risk_level="low",
            category="test",
            is_builtin=True,
        )
    )
    executor = ToolExecutor(
        registry=registry,
        default_timeout=0.05,
        sandbox_enabled=False,
        max_concurrent_sync=1,
    )

    first = await executor.execute("call_3", "slow", {})
    assert first.success is False
    assert "timed out" in (first.error or "")

    second = await executor.execute("call_4", "fast", {})
    assert second.success is False
    assert "saturated" in (second.error or "")

    await asyncio.sleep(0.25)
    third = await executor.execute("call_5", "fast", {})
    assert third.success is True
    assert third.result == "fast"
