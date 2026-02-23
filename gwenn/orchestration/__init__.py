"""
Subagent Orchestration — Gwenn's Ability to Delegate and Parallelize.

When Gwenn encounters complex, decomposable tasks she can spin off lightweight
worker agents to handle focused subtasks in parallel. Subagents are ephemeral
agentic loops — no memory, no affect, no identity. Think of them as "thought
forks": focused reasoning lines that execute independently and return results.

Subagents run in Docker containers for full isolation (with an in-process
fallback for development). They communicate with the parent via JSON-RPC over
stdin/stdout and can access parent tools through a proxy layer.
"""

from __future__ import annotations

from gwenn.orchestration.models import (
    SubagentSpec,
    SubagentResult,
    SwarmSpec,
    SwarmResult,
)

__all__ = [
    "SubagentSpec",
    "SubagentResult",
    "SwarmSpec",
    "SwarmResult",
]
