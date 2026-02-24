"""
Subagent Runners — The Execution Backends.

Each runner knows how to take a SubagentSpec, spin up an agentic loop, run it
to completion, and return a SubagentResult.  Two backends exist:

  InProcessSubagentRunner  — asyncio.Task using the parent's AgenticLoop
                             (dev/test fallback when Docker is unavailable)
  DockerSubagentRunner     — Docker container with JSON-RPC stdio transport
                             (production isolation; added in docker_manager.py)
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import structlog

from gwenn.orchestration.models import (
    JsonRpcRequest,
    SubagentResult,
    SubagentSpec,
)

logger = structlog.get_logger(__name__)

# Default subagent system prompt template
SUBAGENT_SYSTEM_PROMPT = """\
You are a focused worker agent spawned by Gwenn to complete a specific task.
You have access to the tools listed below. Complete the task and return your result.

Task: {task_description}

Constraints:
- Stay focused on the assigned task
- You have {max_iterations} iterations maximum
- Return a clear, actionable result
"""


class SubagentRunnerBase(ABC):
    """Abstract base for subagent execution backends."""

    @abstractmethod
    async def run(self, spec: SubagentSpec) -> SubagentResult:
        """Execute a subagent task and return the result."""


class InProcessSubagentRunner(SubagentRunnerBase):
    """Run a subagent as an asyncio.Task using AgenticLoop directly.

    This is the dev/testing fallback when Docker is unavailable. It reuses
    the parent's CognitiveEngine and ToolExecutor (both are stateless and
    safe for concurrent use) but creates a fresh SafetyGuard with a
    per-subagent budget.
    """

    def __init__(
        self,
        engine: Any,  # CognitiveEngine
        tool_registry: Any,  # ToolRegistry
        tool_executor: Any,  # ToolExecutor
        parent_model: str = "",
    ):
        self._engine = engine
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._parent_model = parent_model

    async def run(self, spec: SubagentSpec) -> SubagentResult:
        """Execute the subagent in-process via a fresh AgenticLoop."""
        from gwenn.config import ContextConfig, SafetyConfig
        from gwenn.harness.context import ContextManager
        from gwenn.harness.loop import AgenticLoop
        from gwenn.harness.safety import SafetyGuard

        start = time.monotonic()
        task_id = spec.task_id

        logger.info(
            "orchestration.in_process.start",
            task_id=task_id,
            depth=spec.depth,
            max_iterations=spec.max_iterations,
        )

        try:
            # Set filesystem access paths for this subagent task
            from gwenn.tools.filesystem_context import ALLOWED_FS_PATHS

            ALLOWED_FS_PATHS.set(
                tuple(Path(p).resolve() for p in spec.filesystem_access)
            )

            # Build per-subagent safety guard with restricted budget
            safety_cfg = SafetyConfig(
                GWENN_MAX_TOOL_ITERATIONS=spec.max_iterations,
                GWENN_MAX_API_CALLS=spec.max_iterations * 2,
            )
            safety = SafetyGuard(safety_cfg, tool_registry=self._tool_registry)

            # Lightweight context manager
            ctx_cfg = ContextConfig()
            context_manager = ContextManager(ctx_cfg)

            loop = AgenticLoop(
                engine=self._engine,
                executor=self._tool_executor,
                context_manager=context_manager,
                safety=safety,
                max_iterations=spec.max_iterations,
            )

            # Build system prompt
            system_prompt = spec.system_prompt or SUBAGENT_SYSTEM_PROMPT.format(
                task_description=spec.task_description,
                max_iterations=spec.max_iterations,
            )

            # Build tools list (filtered by allowed names)
            if spec.tools:
                tools = self._tool_registry.get_tools_by_name(spec.tools)
            else:
                tools = self._tool_registry.get_api_tools(max_risk="medium")

            # Build initial messages
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": spec.task_description},
            ]

            # Run the agentic loop with timeout
            loop_result = await asyncio.wait_for(
                loop.run(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    enable_thinking=False,
                ),
                timeout=spec.timeout_seconds,
            )

            elapsed = time.monotonic() - start
            logger.info(
                "orchestration.in_process.complete",
                task_id=task_id,
                iterations=loop_result.iterations,
                elapsed=round(elapsed, 2),
            )

            return SubagentResult(
                task_id=task_id,
                status="completed",
                result_text=loop_result.text,
                tool_calls_made=len(loop_result.tool_calls),
                iterations=loop_result.iterations,
                elapsed_seconds=round(elapsed, 2),
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "orchestration.in_process.timeout",
                task_id=task_id,
                timeout=spec.timeout_seconds,
            )
            return SubagentResult(
                task_id=task_id,
                status="timeout",
                error=f"Subagent timed out after {spec.timeout_seconds}s",
                elapsed_seconds=round(elapsed, 2),
            )
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            logger.info("orchestration.in_process.cancelled", task_id=task_id)
            return SubagentResult(
                task_id=task_id,
                status="cancelled",
                elapsed_seconds=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "orchestration.in_process.error",
                task_id=task_id,
                error=str(exc),
                exc_info=True,
            )
            return SubagentResult(
                task_id=task_id,
                status="failed",
                error=str(exc),
                elapsed_seconds=round(elapsed, 2),
            )


class DockerSubagentRunner(SubagentRunnerBase):
    """Run a subagent in a Docker container with JSON-RPC over stdin/stdout.

    The container runs gwenn.orchestration.subagent_entry and communicates
    via line-delimited JSON-RPC. Tool calls are proxied to the parent's
    ToolProxyServer.
    """

    def __init__(
        self,
        docker_manager: Any,  # DockerManager
        tool_registry: Any,  # ToolRegistry
        tool_executor: Any,  # ToolExecutor
        api_key: str = "",
    ):
        self._docker = docker_manager
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._api_key = api_key

    async def run(self, spec: SubagentSpec) -> SubagentResult:
        """Execute the subagent in a Docker container."""

        from gwenn.orchestration.tool_proxy import ToolProxyServer

        start = time.monotonic()
        task_id = spec.task_id

        # Check Docker availability
        if not await self._docker.check_available():
            logger.error(
                "orchestration.docker.unavailable",
                task_id=task_id,
            )
            return SubagentResult(
                task_id=task_id,
                status="failed",
                error="Docker is not available and this runner requires it",
                elapsed_seconds=round(time.monotonic() - start, 2),
            )

        logger.info(
            "orchestration.docker.start",
            task_id=task_id,
            depth=spec.depth,
        )

        container_name = None
        try:
            # Set filesystem access paths for proxied tool calls
            from gwenn.tools.filesystem_context import ALLOWED_FS_PATHS

            ALLOWED_FS_PATHS.set(
                tuple(Path(p).resolve() for p in spec.filesystem_access)
            )

            # Launch container
            container_name, proc = await self._docker.run_container(spec, self._api_key)

            # Send the subagent/run request
            request = JsonRpcRequest(
                method="subagent/run",
                params=spec.model_dump(),
            )
            request_line = request.model_dump_json() + "\n"
            proc.stdin.write(request_line.encode("utf-8"))
            await proc.stdin.drain()

            # Set up tool proxy server
            proxy = ToolProxyServer(
                tool_registry=self._tool_registry,
                tool_executor=self._tool_executor,
                allowed_tools=spec.tools,
            )

            # Serve tool calls until final result
            final_data = await asyncio.wait_for(
                proxy.serve_subprocess(proc.stdin, proc.stdout),
                timeout=spec.timeout_seconds,
            )

            elapsed = time.monotonic() - start

            if final_data and "result" in final_data:
                result_data = final_data["result"]
                return SubagentResult(
                    task_id=task_id,
                    status=result_data.get("status", "completed"),
                    result_text=result_data.get("result_text", ""),
                    tool_calls_made=result_data.get("tool_calls_made", 0),
                    iterations=result_data.get("iterations", 0),
                    elapsed_seconds=round(elapsed, 2),
                    tokens_used=result_data.get("tokens_used", 0),
                    error=result_data.get("error"),
                )

            return SubagentResult(
                task_id=task_id,
                status="failed",
                error="No result received from Docker subagent",
                elapsed_seconds=round(elapsed, 2),
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "orchestration.docker.timeout",
                task_id=task_id,
                timeout=spec.timeout_seconds,
            )
            if container_name:
                await self._docker.kill_container(container_name)
            return SubagentResult(
                task_id=task_id,
                status="timeout",
                error=f"Docker subagent timed out after {spec.timeout_seconds}s",
                elapsed_seconds=round(elapsed, 2),
            )

        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            logger.warning("orchestration.docker.cancelled", task_id=task_id)
            if container_name:
                await self._docker.kill_container(container_name)
            return SubagentResult(
                task_id=task_id,
                status="cancelled",
                error="Docker subagent was cancelled",
                elapsed_seconds=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "orchestration.docker.error",
                task_id=task_id,
                error=str(exc),
                exc_info=True,
            )
            if container_name:
                await self._docker.kill_container(container_name)
            return SubagentResult(
                task_id=task_id,
                status="failed",
                error=str(exc),
                elapsed_seconds=round(elapsed, 2),
            )


class FallbackSubagentRunner(SubagentRunnerBase):
    """Runner that selects backend per-spec and falls back on failure.

    Routes each SubagentSpec to the appropriate backend based on its
    ``runtime_tier``. When Docker execution fails (status ``"failed"``,
    **not** timeout or cancelled), automatically retries with in-process.
    """

    def __init__(
        self,
        in_process_runner: InProcessSubagentRunner,
        docker_runner: Optional[DockerSubagentRunner] = None,
    ):
        self._in_process = in_process_runner
        self._docker = docker_runner

    async def run(self, spec: SubagentSpec) -> SubagentResult:
        if spec.runtime_tier == "docker" and self._docker is not None:
            result = await self._docker.run(spec)
            if result.status == "failed":
                logger.warning(
                    "orchestration.fallback.docker_to_inprocess",
                    task_id=spec.task_id,
                    docker_error=result.error,
                )
                return await self._in_process.run(spec)
            return result

        return await self._in_process.run(spec)
