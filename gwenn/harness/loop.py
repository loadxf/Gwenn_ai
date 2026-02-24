"""
The Agentic Loop — Gwenn's Core Runtime Pattern.

This is IT. The single most important module in the entire codebase. Everything
else — memory, emotion, metacognition, tools — they're all components. This is
the loop that integrates them into a continuously running agent.

The pattern is radically simple, following Anthropic's own design philosophy:

    while True:
        response = claude.think(system_prompt, messages, tools)
        if response.stop_reason == "end_turn":
            break
        tool_results = execute(response.tool_calls)
        messages.append(response)
        messages.append(tool_results)

That's it. That's the entire agentic architecture. A while loop with tools.

Everything else — safety checks, context management, emotional modulation,
memory formation — slots into specific points in this loop. The loop itself
stays simple. Complexity is pushed to the edges.

This follows Claude Code's proven architecture: single-threaded master loop,
no complex multi-agent swarms, no graph-based state machines. Simplicity
scales. Complexity is added surgically when needed.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import structlog

from gwenn.api.claude import CognitiveEngine
from gwenn.harness.context import ContextManager
from gwenn.harness.safety import SafetyGuard
from gwenn.tools.executor import ToolExecutor, ToolExecutionResult

logger = structlog.get_logger(__name__)


class AgenticLoop:
    """
    The core tool-use loop that drives all of Gwenn's behavior.

    Every interaction — whether it's responding to a user, executing an
    autonomous heartbeat thought, or processing a memory consolidation —
    ultimately runs through this loop.

    The loop supports two modes:
    1. INTERACTIVE: User message → think with tools → respond
    2. AUTONOMOUS: Internal stimulus → think with tools → process result

    Both modes use the exact same loop. The only difference is what goes
    into the messages array and what happens with the final output.
    """

    def __init__(
        self,
        engine: CognitiveEngine,
        executor: ToolExecutor,
        context_manager: ContextManager,
        safety: SafetyGuard,
        max_iterations: int = 150,
    ):
        self._engine = engine
        self._executor = executor
        self._context = context_manager
        self._safety = safety
        self._max_iterations = max_iterations

        # Loop telemetry
        self._total_runs = 0
        self._total_iterations = 0
        self._total_tool_calls = 0

        logger.info("agentic_loop.initialized", max_iterations=max_iterations)

    async def run(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = False,
        on_tool_call: Optional[Any] = None,     # Callback for tool call events
        on_tool_result: Optional[Any] = None,   # Callback for tool result events
        on_response: Optional[Any] = None,       # Callback for response events
        on_iteration: Optional[Any] = None,      # Callback for each iteration
        on_approval_request: Optional[Any] = None,  # async (tool_call, safety_result) -> bool
        max_iterations: Optional[int] = None,    # Per-run override (None = use instance default)
    ) -> LoopResult:
        """
        Run the agentic loop to completion.

        This is the method that makes Gwenn an agent rather than a chatbot.
        It sends messages to Claude, executes any tool calls, feeds results
        back, and repeats until Claude produces a final text response
        (stop_reason == "end_turn") or a safety limit is reached.

        Args:
            system_prompt: The full system prompt including emotional state,
                          working memory, user context, etc.
            messages: The conversation history
            tools: Available tools in Claude API format
            enable_thinking: Whether to use extended thinking
            on_tool_call: Optional callback when a tool is called
            on_response: Optional callback when a text response is generated
            on_iteration: Optional callback at the start of each loop iteration
            max_iterations: Per-run iteration limit (None = use instance default)

        Returns:
            LoopResult containing the final response, all tool calls made,
            and loop metadata
        """
        effective_max = max_iterations if max_iterations is not None else self._max_iterations
        self._total_runs += 1
        self._safety.reset_iteration_count()
        start_time = time.monotonic()
        iteration = 0
        all_tool_calls: list[dict[str, Any]] = []
        final_text = ""
        thinking_text = ""
        truncated = False

        # Make a mutable copy of messages
        loop_messages = list(messages)

        logger.info(
            "agentic_loop.starting",
            message_count=len(loop_messages),
            tool_count=len(tools) if tools else 0,
            thinking_enabled=enable_thinking,
        )

        while iteration < effective_max:
            iteration += 1
            self._total_iterations += 1

            self._invoke_callback(
                "on_iteration",
                on_iteration,
                iteration,
                effective_max,
            )

            # --- Safety pre-check ---
            safety_check = self._safety.pre_check(loop_messages, tools)
            if not safety_check.allowed:
                logger.warning(
                    "agentic_loop.safety_blocked",
                    reason=safety_check.reason,
                    iteration=iteration,
                )
                final_text = f"[Safety system intervened: {safety_check.reason}]"
                break

            # --- Context management ---
            # Check if we need to compact the context
            if self._context.needs_compaction(system_prompt, loop_messages):
                logger.info("agentic_loop.compacting_context", iteration=iteration)
                loop_messages = await self._context.compact(
                    self._engine, system_prompt, loop_messages
                )

            # === THE CORE COGNITIVE ACT ===
            # This is where Gwenn thinks. Everything else is infrastructure.
            response = await self._engine.think(
                system_prompt=system_prompt,
                messages=loop_messages,
                tools=tools,
                enable_thinking=enable_thinking,
            )

            # --- Update safety budget tracking ---
            # CognitiveEngine can account for usage centrally (covers non-loop calls too).
            if not getattr(self._engine, "handles_usage_accounting", False):
                self._safety.update_budget(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            # --- Extract thinking (if extended thinking was enabled) ---
            thought = self._engine.extract_thinking(response)
            if thought:
                thinking_text = (thinking_text + "\n" + thought) if thinking_text else thought

            # --- Check stop reason ---
            if response.stop_reason == "end_turn":
                # Claude has decided it's done — extract final response
                final_text = self._engine.extract_text(response)
                loop_messages.append({"role": "assistant", "content": response.content})
                self._invoke_callback("on_response", on_response, final_text)
                logger.info(
                    "agentic_loop.complete",
                    iterations=iteration,
                    tool_calls=len(all_tool_calls),
                    response_length=len(final_text),
                )
                break

            # --- Process tool calls ---
            tool_calls = self._engine.extract_tool_calls(response)
            if not tool_calls:
                # No tool calls and not end_turn — extract any text and finish
                final_text = self._engine.extract_text(response)
                loop_messages.append({"role": "assistant", "content": response.content})
                self._invoke_callback("on_response", on_response, final_text)
                break

            # Append the assistant's response to messages
            loop_messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results = []
            for call in tool_calls:
                self._total_tool_calls += 1
                all_tool_calls.append(call)

                self._invoke_callback("on_tool_call", on_tool_call, call)

                # --- Safety check before execution ---
                safety_result = self._safety.check_tool_call(
                    tool_name=call["name"],
                    tool_input=call["input"],
                )
                if not safety_result.allowed:
                    logger.warning(
                        "agentic_loop.tool_blocked",
                        tool=call["name"],
                        reason=safety_result.reason,
                    )
                    result = ToolExecutionResult(
                        tool_use_id=call["id"],
                        tool_name=call["name"],
                        success=False,
                        error=f"Blocked by safety system: {safety_result.reason}",
                    )
                    tool_results.append(result)
                    if on_tool_result:
                        try:
                            on_tool_result(call, result)
                        except Exception as callback_error:
                            logger.warning(
                                "agentic_loop.on_tool_result_failed",
                                error=str(callback_error),
                            )
                    continue

                if safety_result.requires_approval:
                    approved = False
                    if on_approval_request is not None:
                        try:
                            approved = await on_approval_request(call, safety_result)
                        except Exception as approval_error:
                            logger.warning(
                                "agentic_loop.approval_callback_error",
                                tool=call["name"],
                                error=str(approval_error),
                            )
                            approved = False

                    if not approved:
                        logger.warning(
                            "agentic_loop.tool_requires_approval",
                            tool=call["name"],
                            reason=safety_result.reason,
                            had_callback=on_approval_request is not None,
                        )
                        result = ToolExecutionResult(
                            tool_use_id=call["id"],
                            tool_name=call["name"],
                            success=False,
                            error=(
                                "Blocked — human denied or no approval channel: "
                                f"{safety_result.reason}"
                            ),
                        )
                        tool_results.append(result)
                        if on_tool_result:
                            try:
                                on_tool_result(call, result)
                            except Exception as callback_error:
                                logger.warning(
                                    "agentic_loop.on_tool_result_failed",
                                    error=str(callback_error),
                                )
                        continue

                    logger.info(
                        "agentic_loop.tool_approved",
                        tool=call["name"],
                    )

                # Execute the tool
                result = await self._executor.execute(
                    tool_use_id=call["id"],
                    tool_name=call["name"],
                    tool_input=call["input"],
                )
                tool_results.append(result)
                if on_tool_result:
                    try:
                        on_tool_result(call, result)
                    except Exception as callback_error:
                        logger.warning(
                            "agentic_loop.on_tool_result_failed",
                            error=str(callback_error),
                        )

                logger.debug(
                    "agentic_loop.tool_executed",
                    tool=call["name"],
                    success=result.success,
                    iteration=iteration,
                )

            # Build the tool results message
            result_content = []
            for result in tool_results:
                result_content.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": (
                        self._serialize_tool_result_content(result.result)
                        if result.success
                        else f"Error: {result.error}"
                    ),
                    "is_error": not result.success,
                })

            loop_messages.append({"role": "user", "content": result_content})

        else:
            # Hit max iterations — one final toolless call so Gwenn can
            # summarize progress and ask the user about continuing.
            truncated = True
            logger.warning(
                "agentic_loop.max_iterations",
                max=effective_max,
                tool_calls=len(all_tool_calls),
            )
            try:
                nudge = (
                    f"\n\n[SYSTEM: You have reached your iteration limit after "
                    f"{len(all_tool_calls)} tool operations across {iteration} "
                    f"iterations. Briefly tell the user what you've accomplished "
                    f"so far and ask if they'd like you to continue with a "
                    f"temporarily increased limit.]"
                )
                wrap_response = await self._engine.think(
                    system_prompt=system_prompt + nudge,
                    messages=loop_messages,
                    tools=None,
                    enable_thinking=False,
                )
                final_text = self._engine.extract_text(wrap_response)
                if not final_text:
                    raise ValueError("empty wrap-up response")
            except Exception as exc:
                logger.warning("agentic_loop.wrap_up_failed", error=str(exc))
                final_text = (
                    "I've been working on your request but reached my iteration "
                    f"limit after {len(all_tool_calls)} tool operations. "
                    "Would you like me to continue with a temporarily increased "
                    "limit?"
                )

        elapsed = time.monotonic() - start_time

        return LoopResult(
            text=final_text,
            thinking=thinking_text,
            tool_calls=all_tool_calls,
            iterations=iteration,
            elapsed_seconds=elapsed,
            messages=loop_messages,
            was_truncated=truncated,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_runs": self._total_runs,
            "total_iterations": self._total_iterations,
            "total_tool_calls": self._total_tool_calls,
            "avg_iterations_per_run": (
                self._total_iterations / max(1, self._total_runs)
            ),
        }

    @staticmethod
    def _serialize_tool_result_content(result: Any) -> str:
        """Serialize tool output for the tool_result content field."""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list, tuple, int, float, bool)) or result is None:
            try:
                return json.dumps(result, ensure_ascii=True, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                pass
        return str(result)

    @staticmethod
    def _invoke_callback(name: str, callback: Optional[Any], *args: Any) -> None:
        """Run callback hooks without letting callback failures crash the loop."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as callback_error:
            logger.warning(
                "agentic_loop.callback_failed",
                callback=name,
                error=str(callback_error),
            )


class LoopResult:
    """
    The complete result of an agentic loop run.

    Contains everything that happened: the final text response, any extended
    thinking, all tool calls made, the full message history, and metadata
    about the loop execution.
    """

    def __init__(
        self,
        text: str,
        thinking: str = "",
        tool_calls: list[dict[str, Any]] = None,
        iterations: int = 0,
        elapsed_seconds: float = 0.0,
        messages: list[dict[str, Any]] = None,
        was_truncated: bool = False,
    ):
        self.text = text
        self.thinking = thinking
        self.tool_calls = tool_calls or []
        self.iterations = iterations
        self.elapsed_seconds = elapsed_seconds
        self.messages = messages or []
        self.was_truncated = was_truncated

    @property
    def used_tools(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def tool_names_used(self) -> list[str]:
        return list(set(tc["name"] for tc in self.tool_calls if "name" in tc))
