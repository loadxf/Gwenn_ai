"""
Tool Executor — Gwenn's Sandboxed Action Engine.

When Claude decides to use a tool, this module handles the actual execution.
It's the boundary between "deciding to do something" and "doing it."

The executor enforces:
1. SANDBOXING: Tools run in restricted contexts where possible
2. TIMEOUT PROTECTION: No tool can run forever
3. ERROR HANDLING: Failures are captured and fed back to Claude as tool results
4. OBSERVABILITY: Every execution is logged for audit and debugging

The critical design philosophy: let Claude reason freely about what to do,
but let the executor enforce what's ALLOWED. Claude can ask to delete a file;
the executor decides whether that's permitted.

Note: Approval gating is handled by the safety guard (harness/safety.py) in the
agentic loop BEFORE the executor runs. The executor focuses on sandboxing,
timeouts, and error handling.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from typing import Any, Callable, Optional

import structlog

from gwenn.tools.registry import ToolDefinition, ToolRegistry

logger = structlog.get_logger(__name__)


def _safe_release(sem: asyncio.BoundedSemaphore) -> None:
    """Release a BoundedSemaphore, ignoring ValueError from double-release."""
    try:
        sem.release()
    except ValueError:
        pass  # Already released by timeout handler


class ToolExecutionResult:
    """
    The result of executing a tool — success or failure.

    This gets converted into the tool_result message that's sent back
    to Claude in the conversation, allowing it to see what happened
    and decide what to do next.
    """
    def __init__(
        self,
        tool_use_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
    ):
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time


# JSON Schema type → Python types (for lightweight validation)
_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _validate_tool_input(
    schema: dict[str, Any],
    tool_input: dict[str, Any],
) -> Optional[str]:
    """
    Lightweight JSON Schema validation for tool inputs.

    Checks required fields and basic type constraints.  Returns an error
    message string on failure, or None if the input is valid.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # Check required fields
    missing = [name for name in required if name not in tool_input]
    if missing:
        return f"Missing required parameter(s): {', '.join(missing)}"

    # Check types for provided fields
    for name, value in tool_input.items():
        prop_schema = properties.get(name)
        if not prop_schema or not isinstance(prop_schema, dict):
            continue
        expected_type = prop_schema.get("type")
        if not expected_type:
            continue
        py_types = _JSON_TYPE_MAP.get(expected_type)
        if py_types is None:
            continue
        # In Python bool is a subclass of int, but JSON booleans are distinct
        if isinstance(value, bool) and expected_type in ("integer", "number"):
            return f"Parameter '{name}' expected {expected_type}, got boolean"
        if not isinstance(value, py_types):
            return f"Parameter '{name}' expected {expected_type}, got {type(value).__name__}"

    return None


class ToolExecutor:
    """
    Executes tools with safety checks, sandboxing, and observability.

    The executor sits between Claude's tool_use decisions and the actual
    tool handlers. It intercepts each call to:
    1. Check if the tool exists and is enabled
    2. Apply sandbox policy
    3. Apply timeout limits
    4. Execute the handler (sync or async)
    5. Capture results or errors
    6. Log everything for observability
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_output_length: int = 25000,  # Match Claude Code's limit
        sandbox_enabled: bool = False,
        sandbox_allowed_tools: Optional[list[str]] = None,
        max_concurrent_sync: int = 8,
    ):
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_output_length = max_output_length
        self._sandbox_enabled = sandbox_enabled
        self._sandbox_allowed_tools = set(sandbox_allowed_tools or [])
        self._sync_slot = asyncio.BoundedSemaphore(max(1, int(max_concurrent_sync)))

        # Execution statistics
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0

        logger.info(
            "tool_executor.initialized",
            timeout=default_timeout,
            max_output=max_output_length,
            sandbox_enabled=sandbox_enabled,
            sandbox_allowlist_count=len(self._sandbox_allowed_tools),
            max_concurrent_sync=max(1, int(max_concurrent_sync)),
        )

    async def execute(
        self,
        tool_use_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        Execute a tool call from Claude.

        This is the main entry point. It receives the tool_use block from
        Claude's response and returns a ToolExecutionResult that will be
        sent back as a tool_result message.

        Args:
            tool_use_id: The ID from Claude's tool_use block (for correlation)
            tool_name: Which tool to execute
            tool_input: The parameters Claude provided

        Returns:
            ToolExecutionResult with success/failure and output
        """
        start_time = time.monotonic()
        self._total_executions += 1

        logger.info(
            "tool_executor.executing",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input_keys=list(tool_input.keys()),
        )

        # Step 1: Look up the tool
        tool_def = self._registry.get(tool_name)
        if not tool_def:
            self._total_failures += 1
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        if not tool_def.enabled:
            self._total_failures += 1
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' is currently disabled.",
            )

        if self._sandbox_enabled and not self._is_sandbox_allowed(tool_def):
            self._total_failures += 1
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=(
                    f"Tool '{tool_name}' is blocked by sandbox policy. "
                    "Only built-in tools and explicitly allowed tools are permitted."
                ),
            )

        # Step 2: Get the handler
        handler = tool_def.handler
        if not handler:
            self._total_failures += 1
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=f"No handler registered for tool: {tool_name}",
            )

        # Step 3: Validate input against the tool's JSON Schema
        validation_error = _validate_tool_input(tool_def.input_schema, tool_input)
        if validation_error:
            self._total_failures += 1
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=validation_error,
            )

        # Step 4: Execute with timeout
        timeout = tool_def.timeout if tool_def.timeout is not None else self._default_timeout
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**tool_input),
                    timeout=timeout,
                )
            else:
                result = await self._execute_sync_handler(handler, tool_input, timeout)

            # Truncate overly long results
            result_str = str(result)
            if len(result_str) > self._max_output_length:
                result_str = (
                    result_str[: self._max_output_length - 100]
                    + f"\n\n[Output truncated — {len(result_str)} chars total, "
                    f"showing first {self._max_output_length - 100}]"
                )
                result = result_str

            elapsed = time.monotonic() - start_time
            self._total_successes += 1

            logger.info(
                "tool_executor.success",
                tool_name=tool_name,
                elapsed=round(elapsed, 2),
                result_length=len(str(result)),
            )

            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=elapsed,
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            self._total_failures += 1
            logger.warning(
                "tool_executor.timeout",
                tool_name=tool_name,
                timeout=timeout,
            )
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=f"Tool execution timed out after {timeout}s",
                execution_time=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            self._total_failures += 1
            error_detail = f"{type(e).__name__}: {str(e)}"
            logger.error(
                "tool_executor.error",
                tool_name=tool_name,
                error=error_detail,
                traceback=traceback.format_exc(),
            )
            return ToolExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=error_detail,
                execution_time=elapsed,
            )

    async def _execute_sync_handler(
        self,
        handler: Callable[..., Any],
        tool_input: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute a synchronous handler in a dedicated daemon thread.

        Using a per-call thread avoids deadlocks we have seen with repeated
        run_in_executor/to_thread usage in this runtime.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        try:
            await asyncio.wait_for(
                self._sync_slot.acquire(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Sync tool executor is saturated with long-running tasks."
            ) from exc

        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        result_box: dict[str, Any] = {}
        released = threading.Event()

        def _invoke() -> None:
            try:
                result_box["result"] = handler(**tool_input)
            except Exception as exc:  # pragma: no cover - surfaced to caller
                result_box["error"] = exc
            finally:
                if not released.is_set():
                    released.set()
                    try:
                        loop.call_soon_threadsafe(_safe_release, self._sync_slot)
                    except RuntimeError:  # pragma: no cover – shutdown race
                        # Loop may already be closed if shutdown races tool completion.
                        try:
                            self._sync_slot.release()
                        except ValueError:
                            pass
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:  # pragma: no cover – shutdown race
                    pass

        try:
            thread = threading.Thread(target=_invoke, daemon=True)
            thread.start()
        except Exception:
            self._sync_slot.release()
            raise

        try:
            await asyncio.wait_for(done.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            # Release the semaphore since the stuck thread may never do so.
            # Use the released flag to prevent double-release if the thread
            # eventually completes.
            if not released.is_set():
                released.set()
                try:
                    self._sync_slot.release()
                except ValueError:  # pragma: no cover
                    pass
            raise RuntimeError(
                f"Synchronous tool handler timed out after {effective_timeout}s"
            ) from None

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")

    def _is_sandbox_allowed(self, tool_def: ToolDefinition) -> bool:
        """Enforce the sandbox boundary for externally sourced tools."""
        if tool_def.is_builtin:
            return True
        return tool_def.name in self._sandbox_allowed_tools

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "successes": self._total_successes,
            "failures": self._total_failures,
            "success_rate": (
                self._total_successes / max(1, self._total_executions)
            ),
        }
