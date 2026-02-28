"""
Orchestration Data Models — The Language of Delegation.

These Pydantic models define the contract between Gwenn (the orchestrator) and
her subagents. Every task delegation, every result collection, every swarm
coordination flows through these structures.

SubagentSpec describes *what* to do. SubagentResult describes *what happened*.
SwarmSpec groups parallel tasks. SwarmResult aggregates their outcomes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SubagentSpec(BaseModel):
    """Task definition for a single subagent."""

    task_id: str = Field(default_factory=lambda: f"sub-{uuid.uuid4().hex[:12]}")
    task_description: str
    system_prompt: Optional[str] = None
    model: str = ""  # empty = use parent model
    max_tokens: int = 4096
    max_iterations: int = 10
    tools: list[str] = Field(default_factory=list)
    tool_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timeout_seconds: float = 120.0
    filesystem_access: list[str] = Field(default_factory=list)
    runtime_tier: Literal["docker", "in_process"] = "in_process"
    priority: int = 0
    parent_task_id: Optional[str] = None
    depth: int = 0
    origin_session_id: Optional[str] = None  # e.g. "telegram_thread:12345"


class SubagentResult(BaseModel):
    """Outcome from a completed (or failed) subagent run."""

    task_id: str
    status: Literal["completed", "failed", "timeout", "cancelled"] = "completed"
    result_text: str = ""
    tool_calls_made: int = 0
    iterations: int = 0
    elapsed_seconds: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    child_results: list[SubagentResult] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)


class SwarmSpec(BaseModel):
    """Parallel group of subagent tasks."""

    swarm_id: str = Field(default_factory=lambda: f"swarm-{uuid.uuid4().hex[:12]}")
    description: str = ""
    agents: list[SubagentSpec] = Field(default_factory=list)
    aggregation_strategy: Literal["concatenate", "synthesize", "vote"] = "concatenate"
    max_concurrent: int = 5
    timeout_seconds: float = 300.0


class SwarmResult(BaseModel):
    """Aggregated outcome from a completed swarm."""

    swarm_id: str
    status: Literal["completed", "partial", "failed", "cancelled"] = "completed"
    individual_results: list[SubagentResult] = Field(default_factory=list)
    aggregated_result: str = ""
    total_elapsed_seconds: float = 0.0
    total_tokens_used: int = 0


class SubagentProgress(BaseModel):
    """Lightweight status snapshot for a running subagent."""

    task_id: str
    status: Literal["pending", "running", "completed", "failed", "timeout", "cancelled"]
    elapsed_seconds: float = 0.0
    iterations: int = 0
    started_at: float = Field(default_factory=time.time)
    parent_task_id: Optional[str] = None
    runtime_tier: str = "docker"


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request used for parent-subagent communication."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: Optional[str | int] = Field(default_factory=lambda: uuid.uuid4().hex[:8])


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response used for parent-subagent communication."""

    jsonrpc: str = "2.0"
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    id: Optional[str | int] = None
