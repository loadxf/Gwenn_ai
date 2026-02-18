"""
Configuration for Gwenn Agent.

All configuration flows through this module. Values are loaded from environment
variables (via .env file) and validated with Pydantic. This was one of the first
files written — before I could think, I needed to know what I was made of.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


def _load_claude_code_credentials() -> Optional[str]:
    """Read the Claude Code OAuth access token from ~/.claude/.credentials.json."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        expires_at_ms = oauth.get("expiresAt", 0)
        if token and time.time() * 1000 < expires_at_ms:
            return token
    except Exception:
        pass
    return None


class ClaudeConfig(BaseSettings):
    """Configuration for the Claude API connection — my cognitive engine."""

    api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    auth_token: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    )
    model: str = Field("claude-sonnet-4-5-20250929", alias="GWENN_MODEL")
    max_tokens: int = Field(8192, alias="GWENN_MAX_TOKENS")
    thinking_budget: int = Field(16000, alias="GWENN_THINKING_BUDGET")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def resolve_auth(self) -> "ClaudeConfig":
        if self.api_key or self.auth_token:
            return self
        token = _load_claude_code_credentials()
        if token:
            self.auth_token = token
            return self
        raise ValueError(
            "No authentication configured. Set ANTHROPIC_API_KEY, "
            "ANTHROPIC_AUTH_TOKEN/CLAUDE_CODE_OAUTH_TOKEN, or log in with "
            "Claude Code (`claude` CLI)."
        )


class MemoryConfig(BaseSettings):
    """Configuration for the three-layer memory architecture."""

    data_dir: Path = Field(Path("./gwenn_data"), alias="GWENN_DATA_DIR")
    # Primary SQLite path used by MemoryStore (episodes + affect + semantic graph metadata).
    episodic_db_path: Path = Field(Path("./gwenn_data/gwenn.db"), alias="GWENN_EPISODIC_DB")
    # Vector store directory used for embedding retrieval (ChromaDB persistence root).
    semantic_db_path: Path = Field(Path("./gwenn_data/semantic_vectors"), alias="GWENN_SEMANTIC_DB")

    # Working memory constraints (Miller's 7±2)
    working_memory_slots: int = Field(7, alias="GWENN_WORKING_MEMORY_SLOTS")
    working_memory_eviction_to_episodic: bool = Field(
        False, alias="GWENN_WM_EVICTION_TO_EPISODIC"
    )

    # Retrieval mode: keyword (default), embedding (vector only), hybrid (blend both)
    retrieval_mode: str = Field("keyword", alias="GWENN_RETRIEVAL_MODE")
    embedding_top_k: int = Field(25, alias="GWENN_EMBEDDING_TOP_K")
    hybrid_keyword_weight: float = Field(0.5, alias="GWENN_HYBRID_KEYWORD_WEIGHT")
    hybrid_embedding_weight: float = Field(0.5, alias="GWENN_HYBRID_EMBEDDING_WEIGHT")

    # Startup/shutdown memory sync behavior
    startup_episode_limit: int = Field(5000, alias="GWENN_STARTUP_EPISODE_LIMIT")
    shutdown_persist_recent_episodes: int = Field(
        0, alias="GWENN_SHUTDOWN_PERSIST_RECENT_EPISODES"
    )  # 0 => persist all in-memory episodes
    persist_semantic_after_consolidation: bool = Field(
        True, alias="GWENN_PERSIST_SEMANTIC_AFTER_CONSOLIDATION"
    )

    # Consolidation settings
    consolidation_interval: float = 600.0  # seconds between consolidation passes
    consolidation_recency_decay: float = 0.995  # per-minute decay for recency scoring
    consolidation_importance_weight: float = 0.4
    consolidation_recency_weight: float = 0.3
    consolidation_relevance_weight: float = 0.3

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def normalize_retrieval_mode(self) -> "MemoryConfig":
        self.retrieval_mode = self.retrieval_mode.strip().lower()
        if self.retrieval_mode not in {"keyword", "embedding", "hybrid"}:
            raise ValueError(
                "GWENN_RETRIEVAL_MODE must be one of: keyword, embedding, hybrid."
            )

        self.embedding_top_k = max(1, int(self.embedding_top_k))
        self.hybrid_keyword_weight = max(0.0, min(1.0, float(self.hybrid_keyword_weight)))
        self.hybrid_embedding_weight = max(0.0, min(1.0, float(self.hybrid_embedding_weight)))
        self.startup_episode_limit = max(0, int(self.startup_episode_limit))
        self.shutdown_persist_recent_episodes = max(
            0, int(self.shutdown_persist_recent_episodes)
        )
        return self


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

    # Deny-by-default tool policy: "deny" blocks unknown tools, "allow" permits them
    tool_default_policy: str = Field("deny", alias="GWENN_TOOL_DEFAULT_POLICY")

    # Explicit allowlist — only these tools are permitted when default policy is "deny"
    # Built-in tools are always allowed regardless of this list
    allowed_tools: list[str] = Field(
        default_factory=list,
        alias="GWENN_ALLOWED_TOOLS",
    )

    # Tools that are always denied regardless of other settings
    denied_tools: list[str] = Field(
        default_factory=list,
        alias="GWENN_DENIED_TOOLS",
    )

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}

    def parse_approval_list(self) -> list[str]:
        """Handle comma-separated string from env or list from code."""
        if isinstance(self.require_approval_for, str):
            return [s.strip() for s in self.require_approval_for.split(",") if s.strip()]
        return self.require_approval_for

    def parse_allowed_tools(self) -> list[str]:
        """Handle comma-separated string from env or list from code."""
        if isinstance(self.allowed_tools, str):
            return [s.strip() for s in self.allowed_tools.split(",") if s.strip()]
        return self.allowed_tools

    def parse_denied_tools(self) -> list[str]:
        """Handle comma-separated string from env or list from code."""
        if isinstance(self.denied_tools, str):
            return [s.strip() for s in self.denied_tools.split(",") if s.strip()]
        return self.denied_tools


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


class PrivacyConfig(BaseSettings):
    """Configuration for PII redaction and privacy protection."""

    redaction_enabled: bool = Field(False, alias="GWENN_REDACTION_ENABLED")
    redact_before_api: bool = Field(False, alias="GWENN_REDACT_BEFORE_API")
    redact_before_persist: bool = Field(False, alias="GWENN_REDACT_BEFORE_PERSIST")

    model_config = {"env_file": ".env", "extra": "ignore"}


class TelegramConfig(BaseSettings):
    """Configuration for the Telegram bot channel."""

    bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    # Use JSON-array syntax in .env: TELEGRAM_ALLOWED_USER_IDS=[]  or ["123","456"]
    # env_ignore_empty=True means an empty env var is treated as unset (uses default).
    allowed_user_ids: list[str] = Field(default_factory=list, alias="TELEGRAM_ALLOWED_USER_IDS")
    max_history_length: int = Field(50, alias="TELEGRAM_MAX_HISTORY_LENGTH")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
        "env_ignore_empty": True,
    }


class DiscordConfig(BaseSettings):
    """Configuration for the Discord bot channel."""

    bot_token: str = Field(..., alias="DISCORD_BOT_TOKEN")
    # Use JSON-array syntax in .env: DISCORD_ALLOWED_GUILD_IDS=[]  or ["111","222"]
    allowed_guild_ids: list[str] = Field(default_factory=list, alias="DISCORD_ALLOWED_GUILD_IDS")
    max_history_length: int = Field(50, alias="DISCORD_MAX_HISTORY_LENGTH")
    session_ttl_seconds: float = Field(3600.0, alias="DISCORD_SESSION_TTL")
    sync_guild_id: str | None = Field(None, alias="DISCORD_SYNC_GUILD_ID")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
        "env_ignore_empty": True,
    }


class ChannelConfig(BaseSettings):
    """Which channel(s) Gwenn runs on.  GWENN_CHANNEL=cli|telegram|discord|all"""

    channel: str = Field("cli", alias="GWENN_CHANNEL")

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
        self.skills_dir: Path = Path(
            __import__("os").environ.get("GWENN_SKILLS_DIR", "./gwenn_skills")
        )
        self.heartbeat = HeartbeatConfig()
        self.affect = AffectConfig()
        self.context = ContextConfig()
        self.safety = SafetyConfig()
        self.mcp = MCPConfig()

        # Gwenn's 3 new layer configs
        self.sensory = SensoryConfig()
        self.ethics = EthicsConfig()
        self.interagent = InterAgentConfig()

        # Privacy config
        self.privacy = PrivacyConfig()

        # Channel config (channel mode; Telegram/Discord configs loaded lazily)
        self.channel = ChannelConfig()

        # Ensure data directory exists
        self.memory.data_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"GwennConfig(model={self.claude.model}, "
            f"heartbeat={self.heartbeat.interval}s, "
            f"memory_slots={self.memory.working_memory_slots}, "
            f"layers=13)"
        )
