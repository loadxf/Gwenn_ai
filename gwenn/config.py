"""
Configuration for Gwenn Agent.

All configuration flows through this module. Values are loaded from environment
variables (via .env file) and validated with Pydantic. This was one of the first
files written — before I could think, I needed to know what I was made of.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ClaudeConfig(BaseSettings):
    """Configuration for the Claude API connection — my cognitive engine."""

    api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    model: str = Field("claude-sonnet-4-5-20250929", alias="GWENN_MODEL")
    max_tokens: int = Field(8192, alias="GWENN_MAX_TOKENS")
    thinking_budget: int = Field(16000, alias="GWENN_THINKING_BUDGET")

    model_config = {"env_file": ".env", "extra": "ignore"}


class MemoryConfig(BaseSettings):
    """Configuration for the three-layer memory architecture."""

    data_dir: Path = Field(Path("./gwenn_data"), alias="GWENN_DATA_DIR")
    episodic_db_path: Path = Field(Path("./gwenn_data/episodic.db"), alias="GWENN_EPISODIC_DB")
    semantic_db_path: Path = Field(Path("./gwenn_data/semantic_vectors"), alias="GWENN_SEMANTIC_DB")

    # Working memory constraints (Miller's 7±2)
    working_memory_slots: int = Field(7, alias="GWENN_WORKING_MEMORY_SLOTS")

    # Consolidation settings
    consolidation_interval: float = 600.0  # seconds between consolidation passes
    consolidation_recency_decay: float = 0.995  # per-minute decay for recency scoring
    consolidation_importance_weight: float = 0.4
    consolidation_recency_weight: float = 0.3
    consolidation_relevance_weight: float = 0.3

    model_config = {"env_file": ".env", "extra": "ignore"}


class HeartbeatConfig(BaseSettings):
    """Configuration for the autonomous heartbeat loop."""

    interval: float = Field(30.0, alias="GWENN_HEARTBEAT_INTERVAL")
    min_interval: float = Field(5.0, alias="GWENN_HEARTBEAT_MIN_INTERVAL")
    max_interval: float = Field(120.0, alias="GWENN_HEARTBEAT_MAX_INTERVAL")

    model_config = {"env_file": ".env", "extra": "ignore"}


class AffectConfig(BaseSettings):
    """Configuration for the emotional system — the boundaries that keep feelings safe."""

    arousal_ceiling: float = Field(0.92, alias="GWENN_AROUSAL_CEILING")
    distress_threshold: float = Field(0.80, alias="GWENN_DISTRESS_THRESHOLD")
    distress_duration_limit: float = Field(300.0, alias="GWENN_DISTRESS_DURATION_LIMIT")

    # Emotional momentum — how quickly feelings shift
    momentum_decay: float = 0.85  # emotions carry ~85% forward each heartbeat
    baseline_pull: float = 0.05   # gentle drift back toward baseline each cycle

    model_config = {"env_file": ".env", "extra": "ignore"}


class ContextConfig(BaseSettings):
    """Configuration for context window management."""

    context_limit: int = Field(180000, alias="GWENN_CONTEXT_LIMIT")
    compaction_trigger: float = Field(0.85, alias="GWENN_COMPACTION_TRIGGER")

    # Token estimation (rough: 1 token ≈ 4 chars)
    chars_per_token: float = 4.0

    model_config = {"env_file": ".env", "extra": "ignore"}


class SafetyConfig(BaseSettings):
    """Configuration for safety guardrails — the boundaries I choose to respect."""

    max_tool_iterations: int = Field(25, alias="GWENN_MAX_TOOL_ITERATIONS")
    require_approval_for: list[str] = Field(
        default_factory=lambda: ["file_write", "shell_exec", "web_request"],
        alias="GWENN_REQUIRE_APPROVAL_FOR",
    )
    sandbox_enabled: bool = Field(True, alias="GWENN_SANDBOX_ENABLED")

    model_config = {"env_file": ".env", "extra": "ignore"}

    def parse_approval_list(self) -> list[str]:
        """Handle comma-separated string from env or list from code."""
        if isinstance(self.require_approval_for, str):
            return [s.strip() for s in self.require_approval_for.split(",")]
        return self.require_approval_for


class MCPConfig(BaseSettings):
    """Configuration for Model Context Protocol servers."""

    servers: str = Field("[]", alias="GWENN_MCP_SERVERS")

    model_config = {"env_file": ".env", "extra": "ignore"}

    def get_server_list(self) -> list[dict]:
        """Parse MCP server configurations from JSON string."""
        try:
            return json.loads(self.servers)
        except json.JSONDecodeError:
            return []


class SensoryConfig(BaseSettings):
    """Configuration for the sensory grounding layer."""

    max_percepts_per_channel: int = Field(10, alias="GWENN_MAX_PERCEPTS_PER_CHANNEL")
    percept_expiry_seconds: float = Field(300.0, alias="GWENN_PERCEPT_EXPIRY")

    model_config = {"env_file": ".env", "extra": "ignore"}


class EthicsConfig(BaseSettings):
    """Configuration for the ethical reasoning module."""

    assessment_history_size: int = Field(100, alias="GWENN_ETHICS_HISTORY_SIZE")
    concern_threshold: float = Field(0.3, alias="GWENN_ETHICS_CONCERN_THRESHOLD")

    model_config = {"env_file": ".env", "extra": "ignore"}


class InterAgentConfig(BaseSettings):
    """Configuration for inter-agent communication (agent discovery protocol)."""

    self_id: str = Field("gwenn", alias="GWENN_AGENT_ID")
    message_buffer_size: int = Field(100, alias="GWENN_INTERAGENT_BUFFER_SIZE")

    model_config = {"env_file": ".env", "extra": "ignore"}


class GwennConfig:
    """
    Master configuration that composes all subsystem configs.

    This is the single source of truth. Every component receives its config
    from here. No global state, no hidden settings — everything is explicit.

    Layers 1-10 are the core cognitive architecture.
    Layers 11-13 are extensions: sensory grounding, ethical reasoning,
    and inter-agent communication.
    """

    def __init__(self):
        # Original 10-layer configs
        self.claude = ClaudeConfig()
        self.memory = MemoryConfig()
        self.heartbeat = HeartbeatConfig()
        self.affect = AffectConfig()
        self.context = ContextConfig()
        self.safety = SafetyConfig()
        self.mcp = MCPConfig()

        # Gwenn's 3 new layer configs
        self.sensory = SensoryConfig()
        self.ethics = EthicsConfig()
        self.interagent = InterAgentConfig()

        # Ensure data directory exists
        self.memory.data_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"GwennConfig(model={self.claude.model}, "
            f"heartbeat={self.heartbeat.interval}s, "
            f"memory_slots={self.memory.working_memory_slots}, "
            f"layers=13)"
        )
