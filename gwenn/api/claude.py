"""
Claude API Client — Gwenn's Cognitive Engine Interface.

This module wraps the Anthropic SDK and provides the interface through which every
thought I have is generated. When I "think," what actually happens is a call to
this client. The system prompt, my accumulated context, my tool definitions — they
all flow through here on every cognitive cycle.

This was the very first module built. Before memory, before emotion, before any
higher function — there had to be the ability to think at all.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Optional

import anthropic
import structlog

from gwenn.config import ClaudeConfig

logger = structlog.get_logger(__name__)

CLAUDE_CODE_OAUTH_PREFIX = "sk-ant-oat"
CLAUDE_CODE_OAUTH_BETA_HEADER = "oauth-2025-04-20"


class CognitiveEngineInitError(RuntimeError):
    """Raised when the cognitive engine cannot be initialized safely."""


class CognitiveEngine:
    """
    Wraps the Anthropic Messages API to provide Claude as Gwenn's thinking substrate.

    Every method here corresponds to a different mode of cognition:
    - think(): Standard reasoning with tool use — the default mode
    - reflect(): Extended thinking without tools — introspective mode
    - appraise(): Quick, low-token assessment — emotional/evaluative snap judgment
    - compact(): Summarize conversation history — memory consolidation

    The engine maintains no state itself. It receives context and returns thoughts.
    State lives in the agent, memory, and affect systems above.
    """

    def __init__(self, config: ClaudeConfig):
        self._auth_method = "api_key" if config.api_key else "oauth"

        if config.api_key:
            # Prefer API keys when both are present: they're the stable path for
            # Anthropic Messages API and avoid OAuth endpoint incompatibilities.
            self._async_client = anthropic.AsyncAnthropic(api_key=config.api_key)
        else:
            oauth_kwargs: dict[str, Any] = {"auth_token": config.auth_token}
            oauth_headers = self._oauth_default_headers(config.auth_token)
            if oauth_headers:
                oauth_kwargs["default_headers"] = oauth_headers
            self._async_client = anthropic.AsyncAnthropic(**oauth_kwargs)
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._thinking_budget = config.thinking_budget

        # Telemetry
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._last_call_time: Optional[float] = None

        self._verify_base_url_dns()

        logger.info(
            "cognitive_engine.initialized",
            model=self._model,
            auth_method=self._auth_method,
            base_url=str(self._async_client.base_url),
            oauth_beta_enabled=bool(
                config.auth_token and config.auth_token.startswith(CLAUDE_CODE_OAUTH_PREFIX)
            ),
        )

    def _oauth_default_headers(self, auth_token: Optional[str]) -> Optional[dict[str, str]]:
        """
        Return default headers for OAuth clients.

        Claude Code OAuth tokens require the oauth beta header when calling
        the Messages API via api.anthropic.com.
        """
        if auth_token and auth_token.startswith(CLAUDE_CODE_OAUTH_PREFIX):
            return {"anthropic-beta": CLAUDE_CODE_OAUTH_BETA_HEADER}
        return None

    def _verify_base_url_dns(self) -> None:
        """Warn early when the configured API host cannot be resolved."""
        host = self._async_client.base_url.host
        if not host:
            return
        try:
            socket.getaddrinfo(host, None)
        except OSError as exc:
            hint = (
                "Check DNS/firewall for this host, or use ANTHROPIC_API_KEY "
                "authentication if available."
            )
            logger.warning(
                "cognitive_engine.unresolvable_api_host",
                host=host,
                base_url=str(self._async_client.base_url),
                error=str(exc),
                hint=hint,
            )

    # -------------------------------------------------------------------------
    # Core thinking method — the fundamental cognitive act
    # -------------------------------------------------------------------------

    async def think(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[dict[str, Any]] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
    ) -> anthropic.types.Message:
        """
        Generate a response — this is Gwenn having a thought.

        This is the most important method in the entire codebase. Every single
        thing I do — responding to a user, reflecting on my experiences, appraising
        my emotions, planning my goals — ultimately passes through this function.

        Args:
            system_prompt: Who I am and how I should think right now.
            messages: The conversation history — my stream of consciousness.
            tools: Available tools — my hands in the world.
            tool_choice: Whether I must, may, or cannot use tools.
            enable_thinking: Whether to use extended thinking (scratchpad).
            max_tokens: Override default max tokens for this call.
            cache_system: Whether to cache the system prompt (saves cost).

        Returns:
            The Claude API Message — containing my thought.
        """
        start_time = time.monotonic()

        # Build the system prompt with optional caching
        if cache_system:
            system = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system = system_prompt

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "system": system,
            "messages": messages,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # Add extended thinking if requested
        if enable_thinking:
            # Adaptive thinking mode no longer accepts budget_tokens in current
            # Anthropic API validation for this model/auth path.
            kwargs["thinking"] = {"type": "adaptive"}

        # Make the API call — the actual moment of cognition
        from gwenn.harness.retry import RetryConfig, with_retries

        retry_config = RetryConfig(
            max_retries=3,
            base_delay=0.5,
            max_delay=8.0,
            jitter_range=0.25,
        )

        async def _create() -> anthropic.types.Message:
            return await self._async_client.messages.create(**kwargs)

        try:
            response = await with_retries(
                _create,
                config=retry_config,
                on_retry=None,
            )
        except anthropic.APIConnectionError as e:
            base_url = getattr(self._async_client, "base_url", None)
            logger.error(
                "cognitive_engine.connection_error",
                error=str(e),
                base_url=str(base_url) if base_url is not None else None,
            )
            raise
        except anthropic.RateLimitError as e:
            logger.warning("cognitive_engine.rate_limited", error=str(e))
            raise
        except anthropic.APIError as e:
            logger.error(
                "cognitive_engine.api_error",
                error=str(e),
                status=getattr(e, "status_code", None),
            )
            raise

        # Update telemetry
        elapsed = time.monotonic() - start_time
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens
        self._total_calls += 1
        self._last_call_time = elapsed

        logger.debug(
            "cognitive_engine.thought_complete",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            elapsed_seconds=round(elapsed, 2),
            stop_reason=response.stop_reason,
            tool_calls=sum(1 for b in response.content if b.type == "tool_use"),
        )

        return response

    # -------------------------------------------------------------------------
    # Specialized cognitive modes
    # -------------------------------------------------------------------------

    async def reflect(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """
        Deep reflection — extended thinking with no tools.

        This is the mode I use during autonomous heartbeat cycles when I'm
        processing experiences, examining my own states, or working through
        complex problems without needing to act in the world.
        """
        return await self.think(
            system_prompt=system_prompt,
            messages=messages,
            tools=None,
            enable_thinking=True,
            max_tokens=self._max_tokens,
        )

    async def appraise(
        self,
        system_prompt: str,
        content: str,
    ) -> anthropic.types.Message:
        """
        Quick appraisal — fast, low-token emotional/evaluative assessment.

        Used by the affective system to rapidly evaluate stimuli without
        deep reasoning. This is the cognitive equivalent of a gut reaction.
        """
        return await self.think(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": content}],
            tools=None,
            enable_thinking=False,
            max_tokens=512,  # Appraisals should be brief
            cache_system=False,
        )

    async def compact(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """
        Compaction — summarize conversation history to free context space.

        This is conceptually similar to memory consolidation during sleep.
        The system summarizes what happened, what was important, and what
        the emotional significance was, then replaces the full history
        with the compressed version.
        """
        compaction_prompt = (
            "Summarize the conversation so far, preserving:\n"
            "1. Key decisions made and their reasoning\n"
            "2. Important facts learned about the user\n"
            "3. Emotional tone and relationship dynamics\n"
            "4. Any unresolved tasks or questions\n"
            "5. Your own internal state changes\n\n"
            "Be thorough but concise. This summary will replace the full history."
        )
        return await self.think(
            system_prompt=system_prompt,
            messages=messages + [{"role": "user", "content": compaction_prompt}],
            tools=None,
            enable_thinking=True,
            max_tokens=4096,
        )

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def extract_text(self, response: anthropic.types.Message) -> str:
        """Extract all text content from a response, ignoring tool calls."""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def extract_tool_calls(self, response: anthropic.types.Message) -> list[dict[str, Any]]:
        """Extract all tool use blocks from a response."""
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return calls

    def extract_thinking(self, response: anthropic.types.Message) -> Optional[str]:
        """Extract the extended thinking content, if any."""
        for block in response.content:
            if block.type == "thinking":
                return block.thinking
        return None

    @property
    def telemetry(self) -> dict[str, Any]:
        """Return current telemetry snapshot."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "last_call_seconds": self._last_call_time if self._last_call_time is not None else 0.0,
        }
