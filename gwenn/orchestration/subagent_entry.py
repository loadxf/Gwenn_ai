"""
Subagent Entry Point — The Docker Container's Main Loop.

    python -m gwenn.orchestration.subagent_entry

Reads a JSON-RPC ``subagent/run`` request from stdin containing a SubagentSpec,
executes the task using a stripped-down AgenticLoop, and writes the result
back to stdout as a JSON-RPC response.

Tool calls are proxied to the parent via JSON-RPC over stdin/stdout:
  - Subagent writes ``tool/call`` request to stdout
  - Parent executes the tool and writes the result to subagent's stdin
  - Subagent reads the result and continues

Nested spawn requests work similarly: the subagent writes ``subagent/spawn``
to stdout and reads the completed SubagentResult from stdin.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from gwenn.orchestration.models import (
    JsonRpcResponse,
    SubagentResult,
    SubagentSpec,
)
from gwenn.orchestration.runners import SUBAGENT_SYSTEM_PROMPT


async def _run_subagent(spec_data: dict[str, Any]) -> SubagentResult:
    """Execute a subagent task with tool proxying back to the parent."""
    from gwenn.config import ClaudeConfig, ContextConfig, SafetyConfig
    from gwenn.api.claude import CognitiveEngine
    from gwenn.harness.context import ContextManager
    from gwenn.harness.loop import AgenticLoop
    from gwenn.harness.safety import SafetyGuard
    from gwenn.tools.registry import ToolDefinition, ToolRegistry
    from gwenn.tools.executor import ToolExecutor
    from gwenn.orchestration.tool_proxy import ToolProxyClient

    import time

    spec = SubagentSpec(**spec_data)
    start = time.monotonic()

    # Read API key from Docker secret mount or environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        secret_path = "/run/secrets/api_key"
        if os.path.exists(secret_path):
            with open(secret_path) as f:
                api_key = f.read().strip()

    if not api_key:
        return SubagentResult(
            task_id=spec.task_id,
            status="failed",
            error="No API key available",
        )

    # Build engine
    model = spec.model or os.environ.get("GWENN_MODEL", "claude-sonnet-4-5-20250929")
    claude_cfg = ClaudeConfig(ANTHROPIC_API_KEY=api_key, GWENN_MODEL=model)
    engine = CognitiveEngine(claude_cfg)

    # Build tool registry with proxy clients (using schemas from parent)
    registry = ToolRegistry()
    for tool_name in spec.tools:
        proxy = ToolProxyClient(tool_name)
        schema_info = spec.tool_schemas.get(tool_name, {})
        registry.register(
            ToolDefinition(
                name=tool_name,
                description=schema_info.get("description", f"Proxied tool: {tool_name}"),
                input_schema=schema_info.get("input_schema", {"type": "object", "properties": {}}),
                handler=proxy,
                risk_level="low",
                is_builtin=True,
            )
        )

    # Build safety + context + executor
    safety_cfg = SafetyConfig(
        GWENN_MAX_TOOL_ITERATIONS=spec.max_iterations,
        GWENN_MAX_API_CALLS=spec.max_iterations * 2,
    )
    safety = SafetyGuard(safety_cfg, tool_registry=registry)
    ctx_cfg = ContextConfig()
    context = ContextManager(ctx_cfg)
    executor = ToolExecutor(registry=registry, sandbox_enabled=False)

    # Build loop
    loop = AgenticLoop(
        engine=engine,
        executor=executor,
        context_manager=context,
        safety=safety,
        max_iterations=spec.max_iterations,
    )

    # System prompt
    system_prompt = spec.system_prompt or SUBAGENT_SYSTEM_PROMPT.format(
        task_description=spec.task_description,
        max_iterations=spec.max_iterations,
    )

    # Build tools for API
    tools = registry.get_api_tools()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": spec.task_description},
    ]

    try:
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
        return SubagentResult(
            task_id=spec.task_id,
            status="completed",
            result_text=loop_result.text,
            tool_calls_made=len(loop_result.tool_calls),
            iterations=loop_result.iterations,
            elapsed_seconds=round(elapsed, 2),
        )

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return SubagentResult(
            task_id=spec.task_id,
            status="timeout",
            error=f"Timed out after {spec.timeout_seconds}s",
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return SubagentResult(
            task_id=spec.task_id,
            status="failed",
            error=str(exc),
            elapsed_seconds=round(elapsed, 2),
        )


async def main() -> None:
    """Entry point: read spec from stdin, run, write result to stdout."""
    # Redirect all logging to stderr to keep stdout clean for JSON-RPC
    import logging
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    logging.root.handlers = [handler]

    # Also redirect structlog output to stderr so that any module using
    # structlog.get_logger() does not pollute the JSON-RPC stdout channel.
    # Without this, structlog's default PrintLogger writes to stdout, bypassing
    # the stdlib handler above and corrupting the JSON-RPC protocol.
    import structlog as _structlog
    _structlog.configure(
        wrapper_class=_structlog.stdlib.BoundLogger,
        logger_factory=_structlog.stdlib.LoggerFactory(),
    )


    # Read the initial subagent/run request
    line = sys.stdin.readline()
    if not line:
        sys.exit(1)

    try:
        request = json.loads(line.strip())
    except json.JSONDecodeError as exc:
        error_response = JsonRpcResponse(
            id=None,
            error={"code": -32700, "message": f"Parse error: {exc}"},
        )
        sys.stdout.write(error_response.model_dump_json() + "\n")
        sys.stdout.flush()
        sys.exit(1)

    method = request.get("method", "")
    if method != "subagent/run":
        error_response = JsonRpcResponse(
            id=request.get("id"),
            error={"code": -32601, "message": f"Expected subagent/run, got {method}"},
        )
        sys.stdout.write(error_response.model_dump_json() + "\n")
        sys.stdout.flush()
        sys.exit(1)

    spec_data = request.get("params", {})
    result = await _run_subagent(spec_data)

    # Write final result
    response = JsonRpcResponse(
        id=request.get("id"),
        result=result.model_dump(),
    )
    sys.stdout.write(response.model_dump_json() + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
