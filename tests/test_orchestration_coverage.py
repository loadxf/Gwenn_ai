"""Comprehensive coverage tests for the orchestration package.

Covers all missing lines across:
  - gwenn/orchestration/runners.py
  - gwenn/orchestration/subagent_entry.py
  - gwenn/orchestration/docker_manager.py
  - gwenn/orchestration/orchestrator.py
  - gwenn/orchestration/tool_proxy.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from gwenn.orchestration.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    SubagentProgress,
    SubagentResult,
    SubagentSpec,
    SwarmResult,
    SwarmSpec,
)
from gwenn.orchestration.runners import (
    SUBAGENT_SYSTEM_PROMPT,
    DockerSubagentRunner,
    InProcessSubagentRunner,
    SubagentRunnerBase,
)
from gwenn.orchestration.orchestrator import Orchestrator
from gwenn.orchestration.tool_proxy import ToolProxyClient, ToolProxyServer
from gwenn.orchestration.docker_manager import (
    DENIED_MOUNT_PATTERNS,
    DockerManager,
)


# =====================================================================
# Shared fixtures / helpers
# =====================================================================


class MockConfig:
    enabled = True
    max_concurrent_subagents = 3
    default_timeout = 30.0
    default_max_iterations = 5
    max_nesting_depth = 3
    subagent_model = ""
    max_total_api_calls = 50
    max_active_swarms = 2
    max_concurrent_api_calls = 2
    default_tools: list[str] = []
    default_runtime = "in_process"
    autonomous_spawn_enabled = True
    autonomous_spawn_cooldown = 0.0
    autonomous_spawn_max_per_hour = 100


class StubLoopResult:
    def __init__(self, text="done", tool_calls=None, iterations=1):
        self.text = text
        self.tool_calls = tool_calls or []
        self.iterations = iterations


class QuickRunner(SubagentRunnerBase):
    def __init__(self, result_text="done", delay=0.0, status="completed", error=None):
        self._result_text = result_text
        self._delay = delay
        self._status = status
        self._error = error

    async def run(self, spec: SubagentSpec) -> SubagentResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        return SubagentResult(
            task_id=spec.task_id,
            status=self._status,
            result_text=self._result_text,
            iterations=1,
            elapsed_seconds=self._delay,
            error=self._error,
        )


def _patch_inprocess_deps():
    """Return a context manager that patches the local imports inside
    InProcessSubagentRunner.run()."""
    return _MultiPatch(
        "gwenn.config.SafetyConfig",
        "gwenn.config.ContextConfig",
        "gwenn.harness.safety.SafetyGuard",
        "gwenn.harness.context.ContextManager",
        "gwenn.harness.loop.AgenticLoop",
    )


class _MultiPatch:
    """Convenience: patches multiple targets, storing mocks by short name."""

    def __init__(self, *targets: str):
        self._targets = targets
        self._patchers = []
        self.mocks: dict[str, MagicMock] = {}

    def __enter__(self):
        for t in self._targets:
            p = patch(t)
            m = p.start()
            self._patchers.append(p)
            short = t.rsplit(".", 1)[-1]
            self.mocks[short] = m
        return self

    def __exit__(self, *args):
        for p in self._patchers:
            p.stop()


# =====================================================================
# runners.py  InProcessSubagentRunner
# =====================================================================


class TestInProcessSubagentRunnerInit:
    def test_init_stores_attributes(self):
        engine = MagicMock()
        registry = MagicMock()
        executor = MagicMock()
        runner = InProcessSubagentRunner(
            engine=engine, tool_registry=registry,
            tool_executor=executor, parent_model="claude-test",
        )
        assert runner._engine is engine
        assert runner._tool_registry is registry
        assert runner._tool_executor is executor
        assert runner._parent_model == "claude-test"


class TestInProcessSubagentRunnerRun:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_api_tools.return_value = []
        executor = MagicMock()
        runner = InProcessSubagentRunner(
            engine=engine, tool_registry=registry,
            tool_executor=executor,
        )

        stub_result = StubLoopResult(text="finished", tool_calls=[1, 2], iterations=3)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(return_value=stub_result)
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            spec = SubagentSpec(task_description="test task", max_iterations=5, timeout_seconds=10.0)
            result = await runner.run(spec)

        assert result.status == "completed"
        assert result.result_text == "finished"
        assert result.tool_calls_made == 2
        assert result.iterations == 3
        assert result.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_with_custom_system_prompt(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_api_tools.return_value = []
        executor = MagicMock()
        runner = InProcessSubagentRunner(engine=engine, tool_registry=registry, tool_executor=executor)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(return_value=StubLoopResult())
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            spec = SubagentSpec(task_description="test", system_prompt="Custom prompt")
            result = await runner.run(spec)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_with_specified_tools(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_tools_by_name.return_value = [{"name": "tool_a"}]
        executor = MagicMock()
        runner = InProcessSubagentRunner(engine=engine, tool_registry=registry, tool_executor=executor)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(return_value=StubLoopResult())
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            spec = SubagentSpec(task_description="test", tools=["tool_a"])
            result = await runner.run(spec)

        assert result.status == "completed"
        registry.get_tools_by_name.assert_called_once_with(["tool_a"])

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_api_tools.return_value = []
        executor = MagicMock()
        runner = InProcessSubagentRunner(engine=engine, tool_registry=registry, tool_executor=executor)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(side_effect=asyncio.TimeoutError())
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                spec = SubagentSpec(task_description="slow", timeout_seconds=0.01)
                result = await runner.run(spec)

        assert result.status == "timeout"
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_run_cancelled(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_api_tools.return_value = []
        executor = MagicMock()
        runner = InProcessSubagentRunner(engine=engine, tool_registry=registry, tool_executor=executor)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(side_effect=asyncio.CancelledError())
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            with patch("asyncio.wait_for", side_effect=asyncio.CancelledError()):
                spec = SubagentSpec(task_description="cancel me")
                result = await runner.run(spec)

        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_run_generic_exception(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.get_api_tools.return_value = []
        executor = MagicMock()
        runner = InProcessSubagentRunner(engine=engine, tool_registry=registry, tool_executor=executor)

        with _patch_inprocess_deps() as mp:
            mock_loop_inst = AsyncMock()
            mock_loop_inst.run = AsyncMock(side_effect=RuntimeError("boom"))
            mp.mocks["AgenticLoop"].return_value = mock_loop_inst

            with patch("asyncio.wait_for", side_effect=RuntimeError("boom")):
                spec = SubagentSpec(task_description="error")
                result = await runner.run(spec)

        assert result.status == "failed"
        assert "boom" in result.error


# =====================================================================
# runners.py  DockerSubagentRunner
# =====================================================================


class TestDockerSubagentRunnerInit:
    def test_init_stores_attributes(self):
        dm = MagicMock()
        reg = MagicMock()
        exe = MagicMock()
        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=reg, tool_executor=exe, api_key="sk-test")
        assert runner._docker is dm
        assert runner._tool_registry is reg
        assert runner._tool_executor is exe
        assert runner._api_key == "sk-test"


class TestDockerSubagentRunnerRun:
    @pytest.mark.asyncio
    async def test_docker_unavailable(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=False)
        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())
        spec = SubagentSpec(task_description="test")
        result = await runner.run(spec)
        assert result.status == "failed"
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_successful_docker_run(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(
            docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock(), api_key="sk-test",
        )

        final_data = {
            "result": {
                "status": "completed", "result_text": "docker result",
                "tool_calls_made": 3, "iterations": 2, "tokens_used": 100,
            }
        }

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(return_value=final_data)
            MockProxy.return_value = proxy_inst

            spec = SubagentSpec(task_description="docker task", timeout_seconds=30)
            result = await runner.run(spec)

        assert result.status == "completed"
        assert result.result_text == "docker result"
        assert result.tool_calls_made == 3

    @pytest.mark.asyncio
    async def test_docker_run_no_result_key(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(return_value={"error": "bad"})
            MockProxy.return_value = proxy_inst
            spec = SubagentSpec(task_description="test")
            result = await runner.run(spec)

        assert result.status == "failed"
        assert "No result received" in result.error

    @pytest.mark.asyncio
    async def test_docker_run_none_result(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(return_value=None)
            MockProxy.return_value = proxy_inst
            spec = SubagentSpec(task_description="test")
            result = await runner.run(spec)

        assert result.status == "failed"
        assert "No result received" in result.error

    @pytest.mark.asyncio
    async def test_docker_run_timeout_with_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(side_effect=asyncio.TimeoutError())
            MockProxy.return_value = proxy_inst

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                spec = SubagentSpec(task_description="test", timeout_seconds=0.01)
                result = await runner.run(spec)

        assert result.status == "timeout"
        dm.kill_container.assert_called_once_with("gwenn-sub-test")

    @pytest.mark.asyncio
    async def test_docker_run_timeout_no_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        dm.run_container = AsyncMock(side_effect=asyncio.TimeoutError())

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())
        spec = SubagentSpec(task_description="test", timeout_seconds=0.01)
        result = await runner.run(spec)

        assert result.status == "timeout"
        dm.kill_container.assert_not_called()

    @pytest.mark.asyncio
    async def test_docker_run_cancelled_with_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(side_effect=asyncio.CancelledError())
            MockProxy.return_value = proxy_inst

            with patch("asyncio.wait_for", side_effect=asyncio.CancelledError()):
                spec = SubagentSpec(task_description="test")
                result = await runner.run(spec)

        assert result.status == "cancelled"
        dm.kill_container.assert_called_once_with("gwenn-sub-test")

    @pytest.mark.asyncio
    async def test_docker_run_cancelled_no_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        dm.run_container = AsyncMock(side_effect=asyncio.CancelledError())

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())
        spec = SubagentSpec(task_description="test")
        result = await runner.run(spec)

        assert result.status == "cancelled"
        dm.kill_container.assert_not_called()

    @pytest.mark.asyncio
    async def test_docker_run_generic_exception_with_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        dm.run_container = AsyncMock(return_value=("gwenn-sub-test", mock_proc))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())

        with patch("gwenn.orchestration.tool_proxy.ToolProxyServer") as MockProxy:
            proxy_inst = AsyncMock()
            proxy_inst.serve_subprocess = AsyncMock(side_effect=RuntimeError("container crash"))
            MockProxy.return_value = proxy_inst
            with patch("asyncio.wait_for", side_effect=RuntimeError("container crash")):
                spec = SubagentSpec(task_description="test")
                result = await runner.run(spec)

        assert result.status == "failed"
        assert "container crash" in result.error
        dm.kill_container.assert_called_once_with("gwenn-sub-test")

    @pytest.mark.asyncio
    async def test_docker_run_generic_exception_no_container(self):
        dm = AsyncMock()
        dm.check_available = AsyncMock(return_value=True)
        dm.kill_container = AsyncMock()
        dm.run_container = AsyncMock(side_effect=ValueError("setup failed"))

        runner = DockerSubagentRunner(docker_manager=dm, tool_registry=MagicMock(), tool_executor=MagicMock())
        spec = SubagentSpec(task_description="test")
        result = await runner.run(spec)

        assert result.status == "failed"
        assert "setup failed" in result.error
        dm.kill_container.assert_not_called()


# =====================================================================
# subagent_entry.py
# =====================================================================


def _subagent_entry_patches():
    """Return a dict of context managers for _run_subagent local imports."""
    return {
        "ClaudeConfig": patch("gwenn.config.ClaudeConfig"),
        "CognitiveEngine": patch("gwenn.api.claude.CognitiveEngine"),
        "ToolRegistry": patch("gwenn.tools.registry.ToolRegistry"),
        "ToolDefinition": patch("gwenn.tools.registry.ToolDefinition"),
        "SafetyConfig": patch("gwenn.config.SafetyConfig"),
        "SafetyGuard": patch("gwenn.harness.safety.SafetyGuard"),
        "ContextConfig": patch("gwenn.config.ContextConfig"),
        "ContextManager": patch("gwenn.harness.context.ContextManager"),
        "ToolExecutor": patch("gwenn.tools.executor.ToolExecutor"),
        "AgenticLoop": patch("gwenn.harness.loop.AgenticLoop"),
        "ToolProxyClient": patch("gwenn.orchestration.tool_proxy.ToolProxyClient"),
    }


class _SubagentPatchContext:
    """Context manager that starts/stops all subagent_entry patches."""

    def __init__(self):
        self._raw = _subagent_entry_patches()
        self.mocks: dict[str, MagicMock] = {}
        self._active = []

    def __enter__(self):
        for name, p in self._raw.items():
            m = p.start()
            self._active.append(p)
            self.mocks[name] = m
        return self

    def __exit__(self, *args):
        for p in self._active:
            p.stop()


class TestRunSubagent:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(task_description="test", tools=[]).model_dump()

        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.exists", return_value=False):
            result = await _run_subagent(spec_data)

        assert result.status == "failed"
        assert "No API key" in result.error

    @pytest.mark.asyncio
    async def test_api_key_from_secret_file(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(
            task_description="test", tools=["tool_a"],
            tool_schemas={"tool_a": {"description": "A tool", "input_schema": {"type": "object", "properties": {}}}},
        ).model_dump()

        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.exists", return_value=True), \
             patch("builtins.open", MagicMock(return_value=io.StringIO("sk-secret"))):

            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(return_value=StubLoopResult(text="ok", tool_calls=[1], iterations=2))
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                result = await _run_subagent(spec_data)

        assert result.status == "completed"
        assert result.result_text == "ok"

    @pytest.mark.asyncio
    async def test_successful_run_with_env_key(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(
            task_description="research", tools=["search"],
            tool_schemas={"search": {"description": "Search", "input_schema": {"type": "object", "properties": {}}}},
            model="claude-test",
        ).model_dump()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test123"}):
            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(return_value=StubLoopResult(text="research result", tool_calls=[1], iterations=5))
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                result = await _run_subagent(spec_data)

        assert result.status == "completed"
        assert result.result_text == "research result"

    @pytest.mark.asyncio
    async def test_run_with_default_model(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(task_description="test", model="", tools=[]).model_dump()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "GWENN_MODEL": "claude-haiku"}):
            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(return_value=StubLoopResult())
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                result = await _run_subagent(spec_data)

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_with_custom_system_prompt(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(
            task_description="test", system_prompt="Custom agent prompt", tools=[],
        ).model_dump()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(return_value=StubLoopResult())
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                result = await _run_subagent(spec_data)

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(task_description="slow", timeout_seconds=0.01, tools=[]).model_dump()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(side_effect=asyncio.TimeoutError())
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    result = await _run_subagent(spec_data)

        assert result.status == "timeout"
        assert "Timed out" in result.error

    @pytest.mark.asyncio
    async def test_run_exception(self):
        from gwenn.orchestration.subagent_entry import _run_subagent

        spec_data = SubagentSpec(task_description="fail", tools=[]).model_dump()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with _SubagentPatchContext() as ctx:
                mock_reg = MagicMock()
                mock_reg.get_api_tools.return_value = []
                ctx.mocks["ToolRegistry"].return_value = mock_reg

                mock_loop = AsyncMock()
                mock_loop.run = AsyncMock(side_effect=RuntimeError("agent crash"))
                ctx.mocks["AgenticLoop"].return_value = mock_loop

                with patch("asyncio.wait_for", side_effect=RuntimeError("agent crash")):
                    result = await _run_subagent(spec_data)

        assert result.status == "failed"
        assert "agent crash" in result.error


class TestSubagentEntryMain:
    @pytest.mark.asyncio
    async def test_main_empty_stdin(self):
        from gwenn.orchestration.subagent_entry import main
        with patch("sys.stdin", io.StringIO("")), pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_invalid_json(self):
        from gwenn.orchestration.subagent_entry import main
        stdout_buf = io.StringIO()
        with patch("sys.stdin", io.StringIO("not json\n")), \
             patch("sys.stdout", stdout_buf), \
             pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1
        parsed = json.loads(stdout_buf.getvalue().strip())
        assert parsed["error"]["code"] == -32700

    @pytest.mark.asyncio
    async def test_main_wrong_method(self):
        from gwenn.orchestration.subagent_entry import main
        request = {"jsonrpc": "2.0", "method": "wrong/method", "id": "1"}
        stdout_buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(request) + "\n")), \
             patch("sys.stdout", stdout_buf), \
             pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1
        parsed = json.loads(stdout_buf.getvalue().strip())
        assert parsed["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_main_successful_run(self):
        from gwenn.orchestration.subagent_entry import main
        spec = SubagentSpec(task_description="test", tools=[])
        request = {"jsonrpc": "2.0", "method": "subagent/run", "params": spec.model_dump(), "id": "req-1"}
        mock_result = SubagentResult(task_id=spec.task_id, status="completed", result_text="success")

        stdout_buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(request) + "\n")), \
             patch("sys.stdout", stdout_buf), \
             patch("gwenn.orchestration.subagent_entry._run_subagent", new_callable=AsyncMock, return_value=mock_result):
            await main()

        parsed = json.loads(stdout_buf.getvalue().strip())
        assert parsed["id"] == "req-1"
        assert parsed["result"]["status"] == "completed"


# =====================================================================
# docker_manager.py
# =====================================================================


class TestDockerManagerCheckAvailableNonZero:
    @pytest.mark.asyncio
    async def test_check_available_nonzero_returncode(self):
        manager = DockerManager()
        manager._available = None
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.check_available()
        assert result is False


class TestDockerManagerEnsureImage:
    @pytest.mark.asyncio
    async def test_ensure_image_no_dockerfile(self, tmp_path):
        manager = DockerManager(project_root=tmp_path)
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await manager.ensure_image()
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_image_build_failure(self, tmp_path):
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile.subagent").write_text("FROM python:3.11")
        manager = DockerManager(project_root=tmp_path)
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.wait = AsyncMock(return_value=None)
                proc.returncode = 1
            else:
                proc.communicate = AsyncMock(return_value=(b"", b"build error"))
                proc.returncode = 1
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess):
            result = await manager.ensure_image()
        assert result is False


class TestDockerManagerRunContainer:
    @pytest.mark.asyncio
    async def test_run_container_success(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        spec = SubagentSpec(task_id="test-123", task_description="test", filesystem_access=["/tmp/safe_dir"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("os.open", return_value=3), \
             patch("os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), \
             patch("asyncio.ensure_future"):
            container_name, proc = await manager.run_container(spec, "sk-test")

        assert container_name == "gwenn-sub-test-123"
        assert proc is mock_proc

    @pytest.mark.asyncio
    async def test_run_container_denied_mount(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        spec = SubagentSpec(
            task_id="test-456", task_description="test",
            filesystem_access=["/home/user/gwenn_data/db.sqlite", "/tmp/safe"],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("os.open", return_value=3), \
             patch("os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), \
             patch("asyncio.ensure_future") as mock_ensure:
            container_name, proc = await manager.run_container(spec, "sk-test")

        # Close unawaited coroutine passed to the mocked ensure_future.
        if mock_ensure.call_args:
            coro = mock_ensure.call_args[0][0]
            if hasattr(coro, "close"):
                coro.close()

        assert container_name == "gwenn-sub-test-456"

    @pytest.mark.asyncio
    async def test_run_container_launch_failure(self):
        manager = DockerManager()
        spec = SubagentSpec(task_id="test-789", task_description="test")

        with patch("os.open", return_value=3), \
             patch("os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), \
             patch("asyncio.create_subprocess_exec", side_effect=OSError("docker failed")), \
             patch.object(DockerManager, "_cleanup_secret") as mock_cleanup:
            with pytest.raises(OSError, match="docker failed"):
                await manager.run_container(spec, "sk-test")
            mock_cleanup.assert_called_once()


class TestDockerManagerKillContainer:
    @pytest.mark.asyncio
    async def test_kill_container_timeout(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await manager.kill_container("test-container")

    @pytest.mark.asyncio
    async def test_kill_container_exception(self):
        manager = DockerManager()
        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("fail")):
            await manager.kill_container("test-container")


class TestDockerManagerCleanupOrphans:
    @pytest.mark.asyncio
    async def test_cleanup_orphans_no_containers(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            killed = await manager.cleanup_orphans()
        assert killed == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans_with_dead_parent(self):
        manager = DockerManager()
        ps_output = b"abc123\tgwenn-subagent=true,gwenn-pid=999999"
        inspect_output = b"999999"
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(ps_output, b""))
                proc.returncode = 0
            elif call_count == 2:
                proc.communicate = AsyncMock(return_value=(inspect_output, b""))
                proc.returncode = 0
            else:
                proc.wait = AsyncMock(return_value=None)
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch("os.getpid", return_value=12345), \
             patch("os.kill", side_effect=OSError("No such process")), \
             patch("asyncio.wait_for", return_value=None):
            killed = await manager.cleanup_orphans()
        assert killed == 1

    @pytest.mark.asyncio
    async def test_cleanup_orphans_parent_alive(self):
        manager = DockerManager()
        ps_output = b"abc123\tgwenn-subagent=true,gwenn-pid=999999"
        inspect_output = b"999999"
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(ps_output, b""))
                proc.returncode = 0
            elif call_count == 2:
                proc.communicate = AsyncMock(return_value=(inspect_output, b""))
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch("os.getpid", return_value=12345), \
             patch("os.kill", return_value=None):
            killed = await manager.cleanup_orphans()
        assert killed == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans_same_pid(self):
        manager = DockerManager()
        ps_output = b"abc123\tgwenn-subagent=true"
        inspect_output = b"12345"
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(ps_output, b""))
            elif call_count == 2:
                proc.communicate = AsyncMock(return_value=(inspect_output, b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch("os.getpid", return_value=12345):
            killed = await manager.cleanup_orphans()
        assert killed == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans_empty_lines(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"\n\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            killed = await manager.cleanup_orphans()
        assert killed == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans_inspect_exception(self):
        manager = DockerManager()
        ps_output = b"abc123\tgwenn-subagent=true"
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(ps_output, b""))
                proc.returncode = 0
            elif call_count == 2:
                proc.communicate = AsyncMock(side_effect=RuntimeError("inspect fail"))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess):
            killed = await manager.cleanup_orphans()
        assert killed == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans_exception(self):
        manager = DockerManager()
        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("fail")):
            killed = await manager.cleanup_orphans()
        assert killed == 0


class TestDockerManagerCleanupAll:
    @pytest.mark.asyncio
    async def test_cleanup_all_with_containers(self):
        manager = DockerManager()
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(b"container1\ncontainer2\n", b""))
            else:
                proc.wait = AsyncMock(return_value=None)
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch("asyncio.wait_for", return_value=None):
            await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_cleanup_all_no_containers(self):
        manager = DockerManager()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_cleanup_all_exception(self):
        manager = DockerManager()
        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("fail")):
            await manager.cleanup_all()


class TestDockerManagerCleanupSecret:
    def test_cleanup_secret_success(self, tmp_path):
        secret = tmp_path / "secret.key"
        secret.write_text("test")
        DockerManager._cleanup_secret(str(secret))
        assert not secret.exists()

    def test_cleanup_secret_missing(self):
        DockerManager._cleanup_secret("/nonexistent/path/secret.key")


class TestDockerManagerMountAllowed:
    def test_git_denied(self):
        manager = DockerManager()
        assert not manager._is_mount_allowed(Path("/repo/.git/config"))

    def test_github_allowed(self):
        manager = DockerManager()
        assert manager._is_mount_allowed(Path("/repo/.github/workflows/ci.yml"))


# =====================================================================
# orchestrator.py
# =====================================================================


class TestOrchestratorSpawnDefaults:
    @pytest.mark.asyncio
    async def test_spawn_applies_config_defaults(self):
        config = MockConfig()
        config.subagent_model = "claude-haiku"
        config.default_tools = ["tool_a", "tool_b"]
        orch = Orchestrator(config=config, runner=QuickRunner())

        spec = SubagentSpec(task_description="test", model="", timeout_seconds=0, max_iterations=0, tools=[])
        task_id = await orch.spawn(spec)
        assert spec.model == "claude-haiku"
        assert spec.timeout_seconds == config.default_timeout
        assert spec.max_iterations == config.default_max_iterations
        assert spec.tools == ["tool_a", "tool_b"]
        await orch.collect_result(task_id)

    @pytest.mark.asyncio
    async def test_spawn_preserves_existing_values(self):
        config = MockConfig()
        config.subagent_model = "claude-haiku"
        config.default_tools = ["default_tool"]
        orch = Orchestrator(config=config, runner=QuickRunner())

        spec = SubagentSpec(
            task_description="test", model="claude-opus",
            timeout_seconds=60.0, max_iterations=20, tools=["custom_tool"],
        )
        task_id = await orch.spawn(spec)
        assert spec.model == "claude-opus"
        assert spec.timeout_seconds == 60.0
        assert spec.max_iterations == 20
        assert spec.tools == ["custom_tool"]
        await orch.collect_result(task_id)


class TestOrchestratorSpawnSwarm:
    @pytest.mark.asyncio
    async def test_spawn_swarm_disabled(self):
        config = MockConfig()
        config.enabled = False
        orch = Orchestrator(config=config, runner=QuickRunner())
        with pytest.raises(RuntimeError, match="disabled"):
            await orch.spawn_swarm(SwarmSpec(description="test"))

    @pytest.mark.asyncio
    async def test_spawn_swarm_max_active_reached(self):
        config = MockConfig()
        config.max_active_swarms = 0
        orch = Orchestrator(config=config, runner=QuickRunner())
        with pytest.raises(RuntimeError, match="Maximum active swarms"):
            await orch.spawn_swarm(SwarmSpec(description="test"))

    @pytest.mark.asyncio
    async def test_spawn_swarm_partial_failure_rollback(self):
        config = MockConfig()
        config.max_total_api_calls = 1
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._total_api_calls = 1  # pre-exhaust budget

        agents = [SubagentSpec(task_description=f"task {i}") for i in range(3)]
        swarm = SwarmSpec(description="test", agents=agents)

        with pytest.raises(RuntimeError, match="budget exhausted"):
            await orch.spawn_swarm(swarm)

    @pytest.mark.asyncio
    async def test_spawn_swarm_timeout_inherited(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())

        agent = SubagentSpec(task_description="test", timeout_seconds=0)
        swarm = SwarmSpec(description="test", agents=[agent], timeout_seconds=99.0)
        swarm_id = await orch.spawn_swarm(swarm)
        assert agent.timeout_seconds == 99.0
        await orch.collect_swarm(swarm_id)


class TestOrchestratorCheckStatusSwarm:
    @pytest.mark.asyncio
    async def test_check_status_swarm(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())

        agents = [SubagentSpec(task_description="task A")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)
        await orch.collect_swarm(swarm_id)

        # Re-add the swarm entry so check_status hits the swarm branch
        orch._active_swarms[swarm_id] = swarm
        orch._swarm_tasks[swarm_id] = [agents[0].task_id]

        status = await orch.check_status(swarm_id)
        assert status.get("type") == "swarm"


class TestOrchestratorCollectResult:
    @pytest.mark.asyncio
    async def test_collect_result_while_running(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(delay=0.1))
        spec = SubagentSpec(task_description="test")
        await orch.spawn(spec)
        result = await orch.collect_result(spec.task_id)
        assert result is not None
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_collect_result_not_found(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        result = await orch.collect_result("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_result_task_exception(self):
        config = MockConfig()

        class ExplodingRunner(SubagentRunnerBase):
            async def run(self, spec: SubagentSpec) -> SubagentResult:
                raise RuntimeError("kaboom")

        orch = Orchestrator(config=config, runner=ExplodingRunner())
        spec = SubagentSpec(task_description="test")
        await orch.spawn(spec)
        result = await orch.collect_result(spec.task_id)
        assert result is not None
        assert result.status == "failed"


class TestOrchestratorCollectSwarm:
    @pytest.mark.asyncio
    async def test_collect_swarm_not_found(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        result = await orch.collect_swarm("nonexistent-swarm")
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_swarm_waits_for_active(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(delay=0.1))
        agents = [SubagentSpec(task_description="A"), SubagentSpec(task_description="B")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)
        result = await orch.collect_swarm(swarm_id)
        assert result is not None
        assert result.status == "completed"
        assert len(result.individual_results) == 2

    @pytest.mark.asyncio
    async def test_collect_swarm_all_failed(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(status="failed", error="boom"))
        agents = [SubagentSpec(task_description="A"), SubagentSpec(task_description="B")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)
        result = await orch.collect_swarm(swarm_id)
        assert result is not None
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_collect_swarm_with_cancelled(self):
        config = MockConfig()
        call_count = 0

        class MixedRunner(SubagentRunnerBase):
            async def run(self, spec: SubagentSpec) -> SubagentResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return SubagentResult(task_id=spec.task_id, status="completed", result_text="ok")
                return SubagentResult(task_id=spec.task_id, status="cancelled")

        orch = Orchestrator(config=config, runner=MixedRunner())
        agents = [SubagentSpec(task_description="A"), SubagentSpec(task_description="B")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)
        result = await orch.collect_swarm(swarm_id)
        assert result is not None
        assert result.status == "cancelled"


class TestOrchestratorCancel:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        result = await orch.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_swarm(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(delay=10.0))
        agents = [SubagentSpec(task_description="A"), SubagentSpec(task_description="B")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)
        cancelled = await orch.cancel(swarm_id)
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_with_children(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(delay=10.0))
        parent_spec = SubagentSpec(task_description="parent")
        parent_id = await orch.spawn(parent_spec)
        child_spec = SubagentSpec(task_description="child", parent_task_id=parent_id)
        await orch.spawn(child_spec)
        cancelled = await orch.cancel(parent_id)
        assert cancelled is True
        await asyncio.sleep(0.1)


class TestOrchestratorHandleNestedSpawn:
    @pytest.mark.asyncio
    async def test_handle_nested_spawn(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        spec = SubagentSpec(task_description="nested task", depth=1)
        task_id = await orch.handle_nested_spawn("parent-1", spec)
        assert spec.parent_task_id == "parent-1"
        result = await orch.collect_result(task_id)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_handle_nested_spawn_depth_exceeded(self):
        config = MockConfig()
        config.max_nesting_depth = 2
        orch = Orchestrator(config=config, runner=QuickRunner())
        spec = SubagentSpec(task_description="too deep", depth=2)
        with pytest.raises(ValueError, match="nesting depth"):
            await orch.handle_nested_spawn("parent-1", spec)


class TestOrchestratorAutonomousSpawn:
    def test_autonomous_spawn_disabled(self):
        config = MockConfig()
        config.autonomous_spawn_enabled = False
        orch = Orchestrator(config=config, runner=QuickRunner())
        assert orch.can_autonomous_spawn() is False


class TestOrchestratorShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_timeout(self):
        config = MockConfig()

        class NeverEndingRunner(SubagentRunnerBase):
            async def run(self, spec: SubagentSpec) -> SubagentResult:
                await asyncio.sleep(1000)
                return SubagentResult(task_id=spec.task_id)

        orch = Orchestrator(config=config, runner=NeverEndingRunner())
        spec = SubagentSpec(task_description="forever")
        await orch.spawn(spec)

        with patch("gwenn.orchestration.orchestrator.asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await orch.shutdown()
        assert orch.active_count == 0


class TestOrchestratorOnTaskDone:
    def test_on_task_done_cancelled(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        mock_task = MagicMock()
        mock_task.result.side_effect = asyncio.CancelledError()
        orch._active_tasks["test-1"] = mock_task
        orch._on_task_done("test-1", mock_task)
        result = orch._completed_results.get("test-1")
        assert result is not None
        assert result.status == "cancelled"

    def test_on_task_done_exception(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        mock_task = MagicMock()
        mock_task.result.side_effect = RuntimeError("task failed")
        orch._active_tasks["test-2"] = mock_task
        orch._on_task_done("test-2", mock_task)
        result = orch._completed_results.get("test-2")
        assert result is not None
        assert result.status == "failed"
        assert "task failed" in result.error

    def test_on_task_done_with_pii_redaction(self):
        config = MockConfig()
        redactor = MagicMock()
        redactor.enabled = True
        redactor.redact.return_value = "[REDACTED]"
        orch = Orchestrator(config=config, runner=QuickRunner(), redactor=redactor)
        mock_task = MagicMock()
        mock_task.result.return_value = SubagentResult(task_id="t3", status="completed", result_text="sensitive data")
        orch._active_tasks["t3"] = mock_task
        orch._on_task_done("t3", mock_task)
        assert orch._completed_results["t3"].result_text == "[REDACTED]"

    def test_on_task_done_redaction_disabled(self):
        config = MockConfig()
        redactor = MagicMock()
        redactor.enabled = False
        orch = Orchestrator(config=config, runner=QuickRunner(), redactor=redactor)
        mock_task = MagicMock()
        mock_task.result.return_value = SubagentResult(task_id="t4", status="completed", result_text="private")
        orch._active_tasks["t4"] = mock_task
        orch._on_task_done("t4", mock_task)
        assert orch._completed_results["t4"].result_text == "private"

    def test_on_task_done_redaction_exception(self):
        config = MockConfig()
        redactor = MagicMock()
        redactor.enabled = True
        redactor.redact.side_effect = RuntimeError("redact fail")
        orch = Orchestrator(config=config, runner=QuickRunner(), redactor=redactor)
        mock_task = MagicMock()
        mock_task.result.return_value = SubagentResult(task_id="t5", status="completed", result_text="data")
        orch._active_tasks["t5"] = mock_task
        orch._on_task_done("t5", mock_task)
        assert orch._completed_results["t5"].result_text == "data"

    def test_on_task_done_evicts_oldest(self):
        config = MockConfig()
        config.max_total_api_calls = 1  # max_completed = max(50, 1) = 50
        orch = Orchestrator(config=config, runner=QuickRunner())
        for i in range(50):
            orch._completed_results[f"old-{i}"] = SubagentResult(task_id=f"old-{i}", status="completed")
        mock_task = MagicMock()
        mock_task.result.return_value = SubagentResult(task_id="new-one", status="completed")
        orch._active_tasks["new-one"] = mock_task
        orch._on_task_done("new-one", mock_task)
        assert "new-one" in orch._completed_results
        assert "old-0" not in orch._completed_results
        assert len(orch._completed_results) == 50

    def test_on_task_done_updates_progress(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._progress["t6"] = SubagentProgress(task_id="t6", status="running")
        mock_task = MagicMock()
        mock_task.result.return_value = SubagentResult(task_id="t6", status="completed", elapsed_seconds=1.5, iterations=3)
        orch._active_tasks["t6"] = mock_task
        orch._on_task_done("t6", mock_task)
        progress = orch._progress["t6"]
        assert progress.status == "completed"
        assert progress.elapsed_seconds == 1.5
        assert progress.iterations == 3


class TestOrchestratorGetSwarmStatus:
    def test_swarm_status_running(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["s1"] = ["a", "b"]
        orch._active_tasks["a"] = MagicMock()
        orch._completed_results["b"] = SubagentResult(task_id="b", status="completed")
        status = orch._get_swarm_status("s1")
        assert status["status"] == "running"

    def test_swarm_status_completed(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["s2"] = ["a", "b"]
        orch._completed_results["a"] = SubagentResult(task_id="a", status="completed")
        orch._completed_results["b"] = SubagentResult(task_id="b", status="completed")
        status = orch._get_swarm_status("s2")
        assert status["status"] == "completed"

    def test_swarm_status_all_failed(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["s3"] = ["a", "b"]
        orch._completed_results["a"] = SubagentResult(task_id="a", status="failed")
        orch._completed_results["b"] = SubagentResult(task_id="b", status="timeout")
        status = orch._get_swarm_status("s3")
        assert status["status"] == "failed"

    def test_swarm_status_partial(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["s4"] = ["a", "b"]
        orch._completed_results["a"] = SubagentResult(task_id="a", status="completed")
        orch._completed_results["b"] = SubagentResult(task_id="b", status="failed")
        status = orch._get_swarm_status("s4")
        assert status["status"] == "partial"

    def test_swarm_status_unknown(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["s5"] = ["x"]
        status = orch._get_swarm_status("s5")
        assert status["status"] == "unknown"

    def test_swarm_status_empty(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        orch._swarm_tasks["se"] = []
        status = orch._get_swarm_status("se")
        assert status["total_agents"] == 0


class TestOrchestratorAggregateResults:
    @pytest.mark.asyncio
    async def test_aggregate_no_successful(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        results = [SubagentResult(task_id="t1", status="failed", error="fail")]
        text = await orch._aggregate_results(results, "concatenate")
        assert "No successful results" in text

    @pytest.mark.asyncio
    async def test_aggregate_concatenate(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        results = [
            SubagentResult(task_id="t1", status="completed", result_text="result A"),
            SubagentResult(task_id="t2", status="completed", result_text="result B"),
        ]
        text = await orch._aggregate_results(results, "concatenate")
        assert "Result 1" in text and "Result 2" in text

    @pytest.mark.asyncio
    async def test_aggregate_synthesize_with_engine(self):
        config = MockConfig()
        mock_engine = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Synthesized result"
        mock_engine.think = AsyncMock(return_value=mock_response)
        orch = Orchestrator(config=config, runner=QuickRunner(), engine=mock_engine)
        results = [
            SubagentResult(task_id="t1", status="completed", result_text="A"),
            SubagentResult(task_id="t2", status="completed", result_text="B"),
        ]
        text = await orch._aggregate_results(results, "synthesize")
        assert text == "Synthesized result"

    @pytest.mark.asyncio
    async def test_aggregate_synthesize_no_engine(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner(), engine=None)
        results = [SubagentResult(task_id="t1", status="completed", result_text="A")]
        text = await orch._aggregate_results(results, "synthesize")
        assert "Result 1" in text

    @pytest.mark.asyncio
    async def test_aggregate_synthesize_engine_failure(self):
        config = MockConfig()
        mock_engine = AsyncMock()
        mock_engine.think = AsyncMock(side_effect=RuntimeError("API error"))
        orch = Orchestrator(config=config, runner=QuickRunner(), engine=mock_engine)
        results = [SubagentResult(task_id="t1", status="completed", result_text="A")]
        text = await orch._aggregate_results(results, "synthesize")
        assert "Result 1" in text

    @pytest.mark.asyncio
    async def test_aggregate_synthesize_response_no_text(self):
        config = MockConfig()
        mock_engine = AsyncMock()
        mock_engine.think = AsyncMock(return_value="plain string response")
        orch = Orchestrator(config=config, runner=QuickRunner(), engine=mock_engine)
        results = [SubagentResult(task_id="t1", status="completed", result_text="A")]
        text = await orch._aggregate_results(results, "synthesize")
        assert text == "plain string response"

    @pytest.mark.asyncio
    async def test_aggregate_vote(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        results = [
            SubagentResult(task_id="t1", status="completed", result_text="yes"),
            SubagentResult(task_id="t2", status="completed", result_text="yes"),
            SubagentResult(task_id="t3", status="completed", result_text="no"),
        ]
        text = await orch._aggregate_results(results, "vote")
        assert "Majority" in text and "2/3" in text and "yes" in text

    @pytest.mark.asyncio
    async def test_aggregate_unknown_strategy(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        results = [SubagentResult(task_id="t1", status="completed", result_text="A")]
        text = await orch._aggregate_results(results, "unknown_strategy")
        assert "Result 1" in text


# =====================================================================
# tool_proxy.py  ToolProxyServer.serve_subprocess
# =====================================================================


class TestToolProxyServerServeSubprocess:
    @pytest.mark.asyncio
    async def test_serve_eof(self):
        from gwenn.tools.registry import ToolRegistry
        server = ToolProxyServer(tool_registry=ToolRegistry(), tool_executor=MagicMock(), allowed_tools=["x"])
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")
        result = await server.serve_subprocess(AsyncMock(), mock_stdout)
        assert result is None

    @pytest.mark.asyncio
    async def test_serve_tool_call(self):
        from gwenn.tools.registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()

        async def handle_calc(expression: str = "") -> str:
            return "42"

        registry.register(ToolDefinition(
            name="calc", description="calc",
            input_schema={"type": "object", "properties": {}},
            handler=handle_calc, risk_level="low",
        ))

        server = ToolProxyServer(
            tool_registry=registry, tool_executor=MagicMock(), allowed_tools=["calc"],
        )

        call_count = 0

        async def mock_readline():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                req = {"jsonrpc": "2.0", "method": "tool/call", "params": {"name": "calc", "args": {"expression": "1+1"}}, "id": "1"}
                return json.dumps(req).encode() + b"\n"
            elif call_count == 2:
                return json.dumps({"result": {"status": "completed"}}).encode() + b"\n"
            return b""

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.readline = mock_readline

        result = await server.serve_subprocess(mock_stdin, mock_stdout)
        assert result is not None and "result" in result
        mock_stdin.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_serve_subagent_spawn(self):
        from gwenn.tools.registry import ToolRegistry
        server = ToolProxyServer(tool_registry=ToolRegistry(), tool_executor=MagicMock(), allowed_tools=["x"])
        spawn_req = {"jsonrpc": "2.0", "method": "subagent/spawn", "params": {"task_description": "child"}, "id": "2"}
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=json.dumps(spawn_req).encode() + b"\n")
        result = await server.serve_subprocess(AsyncMock(), mock_stdout)
        assert result["method"] == "subagent/spawn"

    @pytest.mark.asyncio
    async def test_serve_final_result(self):
        from gwenn.tools.registry import ToolRegistry
        server = ToolProxyServer(tool_registry=ToolRegistry(), tool_executor=MagicMock(), allowed_tools=["x"])
        final = {"result": {"task_id": "t1", "status": "completed"}}
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=json.dumps(final).encode() + b"\n")
        result = await server.serve_subprocess(AsyncMock(), mock_stdout)
        assert result == final

    @pytest.mark.asyncio
    async def test_serve_malformed_json(self):
        from gwenn.tools.registry import ToolRegistry
        server = ToolProxyServer(tool_registry=ToolRegistry(), tool_executor=MagicMock(), allowed_tools=["x"])
        call_count = 0

        async def mock_readline():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"not json at all\n"
            elif call_count == 2:
                return json.dumps({"result": {"status": "done"}}).encode() + b"\n"
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.readline = mock_readline
        result = await server.serve_subprocess(AsyncMock(), mock_stdout)
        assert result is not None

    @pytest.mark.asyncio
    async def test_serve_unknown_message(self):
        from gwenn.tools.registry import ToolRegistry
        server = ToolProxyServer(tool_registry=ToolRegistry(), tool_executor=MagicMock(), allowed_tools=["x"])
        call_count = 0

        async def mock_readline():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"method": "unknown/thing", "id": "x"}).encode() + b"\n"
            elif call_count == 2:
                return json.dumps({"result": {"status": "done"}}).encode() + b"\n"
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.readline = mock_readline
        result = await server.serve_subprocess(AsyncMock(), mock_stdout)
        assert result is not None


class TestToolProxyServerEmptyAllowedTools:
    def test_empty_allowed_tools(self):
        """When allowed_tools is empty, use medium-risk tools from registry."""
        registry = MagicMock()
        tool_obj = MagicMock()
        tool_obj.name = "medium_tool"
        registry.get_api_tools.return_value = [tool_obj]

        server = ToolProxyServer(
            tool_registry=registry, tool_executor=MagicMock(), allowed_tools=[],
        )
        assert "medium_tool" in server._allowed_tools


class TestToolProxyServerNonStringResult:
    @pytest.mark.asyncio
    async def test_handle_dict_result(self):
        from gwenn.tools.registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()

        async def handle_data() -> dict:
            return {"key": "value", "count": 42}

        registry.register(ToolDefinition(
            name="data_tool", description="Returns dict",
            input_schema={"type": "object", "properties": {}},
            handler=handle_data, risk_level="low",
        ))

        server = ToolProxyServer(
            tool_registry=registry, tool_executor=MagicMock(), allowed_tools=["data_tool"],
        )

        request = {"jsonrpc": "2.0", "method": "tool/call", "params": {"name": "data_tool", "args": {}}, "id": "1"}
        response = await server.handle_request(request)
        assert response.get("error") is None
        tool_result = response["result"]["tool_result"]
        parsed = json.loads(tool_result)
        assert parsed["key"] == "value"


# =====================================================================
# tool_proxy.py  ToolProxyClient
# =====================================================================


class TestToolProxyClientCall:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        client = ToolProxyClient("test_tool")
        response = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"tool_result": "hello world"}}) + "\n"
        mock_stdout = io.StringIO()
        mock_stdin = io.StringIO(response)

        with patch("sys.stdout", mock_stdout), patch("sys.stdin", mock_stdin):
            result = await client(name="test")

        assert result == "hello world"
        written = mock_stdout.getvalue()
        parsed = json.loads(written.strip())
        assert parsed["method"] == "tool/call"
        assert parsed["params"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_call_empty_response(self):
        client = ToolProxyClient("test_tool")
        mock_stdout = io.StringIO()
        mock_stdin = io.StringIO("")
        with patch("sys.stdout", mock_stdout), patch("sys.stdin", mock_stdin):
            result = await client()
        assert "closed connection" in result

    @pytest.mark.asyncio
    async def test_call_invalid_json_response(self):
        client = ToolProxyClient("test_tool")
        mock_stdout = io.StringIO()
        mock_stdin = io.StringIO("not json\n")
        with patch("sys.stdout", mock_stdout), patch("sys.stdin", mock_stdin):
            result = await client()
        assert "invalid response" in result

    @pytest.mark.asyncio
    async def test_call_error_response(self):
        client = ToolProxyClient("test_tool")
        response = json.dumps({"jsonrpc": "2.0", "id": "1", "error": {"code": -32000, "message": "tool broken"}}) + "\n"
        mock_stdout = io.StringIO()
        mock_stdin = io.StringIO(response)
        with patch("sys.stdout", mock_stdout), patch("sys.stdin", mock_stdin):
            result = await client()
        assert "Tool error" in result and "tool broken" in result

    @pytest.mark.asyncio
    async def test_call_result_without_tool_result(self):
        client = ToolProxyClient("test_tool")
        response = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"some_other_key": "value"}}) + "\n"
        mock_stdout = io.StringIO()
        mock_stdin = io.StringIO(response)
        with patch("sys.stdout", mock_stdout), patch("sys.stdin", mock_stdin):
            result = await client()
        assert "some_other_key" in result


# =====================================================================
# Miscellaneous edge cases
# =====================================================================


class TestSubagentSystemPromptConstant:
    def test_system_prompt_format(self):
        formatted = SUBAGENT_SYSTEM_PROMPT.format(task_description="test task", max_iterations=10)
        assert "test task" in formatted and "10" in formatted


class TestSubagentRunnerBaseAbstract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            SubagentRunnerBase()


class TestDockerManagerInit:
    def test_custom_init(self, tmp_path):
        dm = DockerManager(image="custom:v1", network="bridge", memory_limit="512m", cpu_limit=1.0, project_root=tmp_path)
        assert dm._image == "custom:v1"
        assert dm._network == "bridge"
        assert dm._memory_limit == "512m"
        assert dm._cpu_limit == 1.0
        assert dm._project_root == tmp_path
        assert dm._available is None


class TestOrchestratorProperties:
    @pytest.mark.asyncio
    async def test_properties(self):
        config = MockConfig()
        orch = Orchestrator(config=config, runner=QuickRunner())
        assert orch.active_count == 0
        assert orch.completed_count == 0
        spec = SubagentSpec(task_description="test")
        await orch.spawn(spec)
        await orch.collect_result(spec.task_id)
        assert orch.completed_count == 1


# =====================================================================
# Final gap coverage
# =====================================================================


class TestToolProxyServerHandlerNone:
    """Cover tool_proxy.py line 88: tool in allowed list but no handler found."""

    @pytest.mark.asyncio
    async def test_tool_allowed_but_no_handler(self):
        """Tool name is in allowed_tools but registry has no handler for it."""
        from gwenn.tools.registry import ToolRegistry

        registry = ToolRegistry()
        # Create server with 'phantom' in allowed_tools, but phantom is not registered
        server = ToolProxyServer(
            tool_registry=registry,
            tool_executor=MagicMock(),
            allowed_tools=["phantom"],
        )

        request = {
            "jsonrpc": "2.0",
            "method": "tool/call",
            "params": {"name": "phantom", "args": {}},
            "id": "99",
        }
        response = await server.handle_request(request)
        assert response["error"] is not None
        assert "No handler found" in response["error"]["message"]


class TestOrchestratorSwarmPartialRollback:
    """Cover orchestrator.py lines 167-168: cancel already-spawned on partial failure."""

    @pytest.mark.asyncio
    async def test_swarm_spawn_partial_rollback(self):
        config = MockConfig()
        config.max_nesting_depth = 3

        call_count = 0

        class CountingRunner(SubagentRunnerBase):
            async def run(self, spec: SubagentSpec) -> SubagentResult:
                await asyncio.sleep(10.0)  # never finishes (will be cancelled)
                return SubagentResult(task_id=spec.task_id, status="completed")

        orch = Orchestrator(config=config, runner=CountingRunner())

        # First agent is fine, second exceeds nesting depth
        good_agent = SubagentSpec(task_description="ok", depth=0)
        bad_agent = SubagentSpec(task_description="bad", depth=3)  # at depth limit

        swarm = SwarmSpec(description="test", agents=[good_agent, bad_agent])
        with pytest.raises(ValueError, match="nesting depth"):
            await orch.spawn_swarm(swarm)

        # Give cancellation time to propagate
        await asyncio.sleep(0.1)

        # The first spawned task should have been cancelled
        assert orch.active_count == 0 or good_agent.task_id not in orch._active_tasks


class TestOrchestratorCollectSwarmExceptionWaiting:
    """Cover orchestrator.py lines 253-254: exception while waiting for active swarm task."""

    @pytest.mark.asyncio
    async def test_collect_swarm_with_exception_in_task(self):
        config = MockConfig()

        class ExplodingRunner(SubagentRunnerBase):
            async def run(self, spec: SubagentSpec) -> SubagentResult:
                raise RuntimeError("explosion")

        orch = Orchestrator(config=config, runner=ExplodingRunner())

        agents = [SubagentSpec(task_description="A")]
        swarm = SwarmSpec(description="test", agents=agents)
        swarm_id = await orch.spawn_swarm(swarm)

        # collect_swarm needs to handle the exception in the awaited task
        result = await orch.collect_swarm(swarm_id)
        assert result is not None
        # The task that threw should have been caught by _on_task_done
        assert len(result.individual_results) == 1
        assert result.individual_results[0].status == "failed"


class TestDockerManagerCleanupAfterExit:
    """Cover docker_manager.py lines 201-204: _cleanup_after_exit inner function."""

    @pytest.mark.asyncio
    async def test_cleanup_after_exit_runs(self):
        manager = DockerManager()

        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        # Make proc.wait resolve immediately
        mock_proc.wait = AsyncMock(return_value=0)

        spec = SubagentSpec(task_id="cleanup-test", task_description="test")

        # We need to actually let ensure_future schedule the cleanup coroutine
        # and then run it
        captured_coro = None
        original_ensure_future = asyncio.ensure_future

        def capture_ensure_future(coro):
            nonlocal captured_coro
            captured_coro = coro
            return original_ensure_future(coro)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("os.open", return_value=3), \
             patch("os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), \
             patch("asyncio.ensure_future", side_effect=capture_ensure_future), \
             patch.object(DockerManager, "_cleanup_secret") as mock_cleanup:

            container_name, proc = await manager.run_container(spec, "sk-test")

            # The ensure_future should have scheduled _cleanup_after_exit
            # Let it run
            if captured_coro:
                # Give the task a chance to run
                await asyncio.sleep(0.05)

        # _cleanup_secret should have been called from the _cleanup_after_exit function
        mock_cleanup.assert_called_once()
