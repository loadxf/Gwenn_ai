# gwenn/config.py
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
from typing import Annotated, Literal, Optional
from pydantic import AliasChoices, BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings
import structlog


logger = structlog.get_logger(__name__)

# Resolve .env relative to the project root (one level above gwenn/ package),
# so the config works regardless of the user's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _normalize_session_scope_mode(value: object, default: str) -> str:
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in {"per_user", "per_chat", "per_thread"}:
            return mode
    return default


def _coerce_str_list(value: object) -> list[str]:
    """Coerce env-var values into a list of stripped, non-empty strings.

    Accepts:
      - A single int or str  → ["value"]
      - Comma-separated str  → ["a", "b"]
      - JSON array str       → (parsed by pydantic-settings before this runs)
      - An existing list     → passthrough with str coercion
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (int, float)):
        return [str(int(value))]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Comma-separated: "123,456" or "123, 456"
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return []


# Annotated type for list[str] fields that accept bare values, comma-separated,
# and JSON arrays from environment variables.
StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]


def _load_oauth_credentials() -> tuple[Optional[str], float]:
    """Read OAuth token and expiry from ~/.claude/.credentials.json.

    Returns (token, expires_at_epoch_seconds) or (None, 0.0).
    """
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None, 0.0
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        expires_at_ms = oauth.get("expiresAt", 0)
        if token and expires_at_ms:
            return token, expires_at_ms / 1000.0
    except Exception as e:
        logger.debug("config.oauth_credentials_unreadable", path=str(creds_path), error=str(e))
    return None, 0.0


def _load_claude_code_credentials() -> Optional[str]:
    """Read the Claude Code OAuth access token from ~/.claude/.credentials.json.

    Delegates to ``_load_oauth_credentials`` to avoid duplicated file-reading
    logic and expiry checks.
    """
    token, expires_at = _load_oauth_credentials()
    if token and time.time() < expires_at:
        return token
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
    thinking_effort: Literal["low", "medium", "high", "max"] = Field(
        "high", alias="GWENN_THINKING_EFFORT"
    )
    request_timeout_seconds: float = Field(120.0, alias="GWENN_REQUEST_TIMEOUT_SECONDS")
    retry_max_retries: int = Field(3, alias="GWENN_RETRY_MAX_RETRIES")
    retry_base_delay: float = Field(0.5, alias="GWENN_RETRY_BASE_DELAY")
    retry_max_delay: float = Field(8.0, alias="GWENN_RETRY_MAX_DELAY")
    retry_exponential_base: float = Field(2.0, alias="GWENN_RETRY_EXPONENTIAL_BASE")
    retry_jitter_range: float = Field(0.25, alias="GWENN_RETRY_JITTER_RANGE")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}

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

    @model_validator(mode="after")
    def normalize_runtime_limits(self) -> "ClaudeConfig":
        self.max_tokens = max(1, int(self.max_tokens))
        self.thinking_budget = max(0, int(self.thinking_budget))
        # API requires max_tokens > thinking.budget_tokens
        if self.thinking_budget > 0 and self.max_tokens <= self.thinking_budget:
            self.max_tokens = self.thinking_budget + 1024
        self.request_timeout_seconds = max(1.0, float(self.request_timeout_seconds))
        self.retry_max_retries = max(0, int(self.retry_max_retries))
        self.retry_base_delay = max(0.05, float(self.retry_base_delay))
        self.retry_max_delay = max(self.retry_base_delay, float(self.retry_max_delay))
        self.retry_exponential_base = max(1.0, float(self.retry_exponential_base))
        self.retry_jitter_range = max(0.0, min(1.0, float(self.retry_jitter_range)))
        return self


class MemoryConfig(BaseSettings):
    """Configuration for the three-layer memory architecture."""

    data_dir: Path = Field(Path("./gwenn_data"), alias="GWENN_DATA_DIR")
    # Primary SQLite path used by MemoryStore (episodes + affect + semantic graph metadata).
    episodic_db_path: Path = Field(Path("./gwenn_data/episodic.db"), alias="GWENN_EPISODIC_DB")
    # Vector store directory used for embedding retrieval (ChromaDB persistence root).
    semantic_db_path: Path = Field(Path("./gwenn_data/semantic_vectors"), alias="GWENN_SEMANTIC_DB")

    # Working memory constraints (Miller's 7±2)
    working_memory_slots: int = Field(7, alias="GWENN_WORKING_MEMORY_SLOTS")
    working_memory_eviction_to_episodic: bool = Field(True, alias="GWENN_WM_EVICTION_TO_EPISODIC")

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
    consolidation_interval: float = Field(
        600.0,
        alias="GWENN_CONSOLIDATION_INTERVAL",
    )  # seconds between consolidation passes
    consolidation_max_episodes: int = Field(200, alias="GWENN_CONSOLIDATION_MAX_EPISODES")
    consolidation_recency_decay: float = Field(
        0.995,
        alias="GWENN_CONSOLIDATION_RECENCY_DECAY",
    )  # per-minute decay for recency scoring
    consolidation_importance_weight: float = Field(
        0.4,
        alias="GWENN_CONSOLIDATION_IMPORTANCE_WEIGHT",
    )
    consolidation_recency_weight: float = Field(
        0.3,
        alias="GWENN_CONSOLIDATION_RECENCY_WEIGHT",
    )
    consolidation_relevance_weight: float = Field(
        0.3,
        alias="GWENN_CONSOLIDATION_RELEVANCE_WEIGHT",
    )

    # Factory defaults used for detecting whether the user explicitly set a path.
    _DEFAULT_EPISODIC: Path = Path("./gwenn_data/episodic.db")
    _DEFAULT_SEMANTIC: Path = Path("./gwenn_data/semantic_vectors")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore", "populate_by_name": True}

    @model_validator(mode="after")
    def derive_paths_from_data_dir(self) -> "MemoryConfig":
        """Derive sub-paths from data_dir when the user hasn't overridden them."""
        if self.episodic_db_path == self._DEFAULT_EPISODIC:
            self.episodic_db_path = self.data_dir / "episodic.db"
        if self.semantic_db_path == self._DEFAULT_SEMANTIC:
            self.semantic_db_path = self.data_dir / "semantic_vectors"
        return self

    @model_validator(mode="after")
    def normalize_retrieval_mode(self) -> "MemoryConfig":
        self.retrieval_mode = self.retrieval_mode.strip().lower()
        if self.retrieval_mode not in {"keyword", "embedding", "hybrid"}:
            raise ValueError("GWENN_RETRIEVAL_MODE must be one of: keyword, embedding, hybrid.")

        self.embedding_top_k = max(1, int(self.embedding_top_k))
        self.hybrid_keyword_weight = max(0.0, min(1.0, float(self.hybrid_keyword_weight)))
        self.hybrid_embedding_weight = max(0.0, min(1.0, float(self.hybrid_embedding_weight)))
        self.startup_episode_limit = max(0, int(self.startup_episode_limit))
        self.shutdown_persist_recent_episodes = max(0, int(self.shutdown_persist_recent_episodes))
        self.consolidation_max_episodes = max(1, int(self.consolidation_max_episodes))
        return self


class HeartbeatConfig(BaseSettings):
    """Configuration for the autonomous heartbeat loop."""

    interval: float = Field(30.0, alias="GWENN_HEARTBEAT_INTERVAL")
    min_interval: float = Field(5.0, alias="GWENN_HEARTBEAT_MIN_INTERVAL")
    max_interval: float = Field(120.0, alias="GWENN_HEARTBEAT_MAX_INTERVAL")
    # When enabled, significant autonomous thoughts are shared with channel owners.
    proactive_messages: bool = Field(False, alias="GWENN_PROACTIVE_MESSAGES")

    # Circuit breaker — opens after consecutive failures to prevent rapid-fire errors
    circuit_max_consecutive: int = Field(10, alias="GWENN_HEARTBEAT_CIRCUIT_MAX_CONSECUTIVE")
    circuit_base_seconds: float = Field(60.0, alias="GWENN_HEARTBEAT_CIRCUIT_BASE_SECONDS")
    circuit_max_seconds: float = Field(900.0, alias="GWENN_HEARTBEAT_CIRCUIT_MAX_SECONDS")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore", "populate_by_name": True}

    @model_validator(mode="after")
    def normalize_limits(self) -> "HeartbeatConfig":
        self.interval = max(1.0, float(self.interval))
        self.min_interval = max(1.0, float(self.min_interval))
        self.max_interval = max(self.min_interval, float(self.max_interval))
        if self.interval < self.min_interval or self.interval > self.max_interval:
            self.interval = max(self.min_interval, min(self.max_interval, self.interval))
        self.circuit_max_consecutive = max(1, int(self.circuit_max_consecutive))
        self.circuit_base_seconds = max(1.0, float(self.circuit_base_seconds))
        self.circuit_max_seconds = max(self.circuit_base_seconds, float(self.circuit_max_seconds))
        return self


class AffectConfig(BaseSettings):
    """Configuration for the emotional system — the boundaries that keep feelings safe."""

    arousal_ceiling: float = Field(0.92, alias="GWENN_AROUSAL_CEILING")
    distress_threshold: float = Field(0.80, alias="GWENN_DISTRESS_THRESHOLD")
    distress_duration_limit: float = Field(300.0, alias="GWENN_DISTRESS_DURATION_LIMIT")

    # Emotional momentum — how quickly feelings shift
    momentum_decay: float = Field(
        0.85,
        alias="GWENN_AFFECT_MOMENTUM_DECAY",
    )  # emotions carry ~85% forward each heartbeat
    baseline_pull: float = Field(
        0.05,
        alias="GWENN_AFFECT_BASELINE_PULL",
    )  # gentle drift back toward baseline each cycle

    model_config = {"env_file": _ENV_FILE, "extra": "ignore", "populate_by_name": True}


class ContextConfig(BaseSettings):
    """Configuration for context window management."""

    context_limit: int = Field(180000, alias="GWENN_CONTEXT_LIMIT")
    compaction_trigger: float = Field(0.85, alias="GWENN_COMPACTION_TRIGGER")
    max_conversation_messages: int = Field(400, alias="GWENN_MAX_CONVERSATION_MESSAGES")

    # Token estimation (rough: 1 token ≈ 4 chars)
    chars_per_token: float = 4.0

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class SafetyConfig(BaseSettings):
    """Configuration for safety guardrails — the boundaries I choose to respect."""

    max_tool_iterations: int = Field(150, alias="GWENN_MAX_TOOL_ITERATIONS")
    require_approval_for: list[str] = Field(
        default_factory=lambda: ["file_write", "shell_exec", "run_command", "web_request"],
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

    # Budget controls (0 = unlimited)
    max_input_tokens: int = Field(0, alias="GWENN_MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(0, alias="GWENN_MAX_OUTPUT_TOKENS")
    max_api_calls: int = Field(0, alias="GWENN_MAX_API_CALLS")

    # Proactive model-call rate limiting (0 = unlimited)
    max_model_calls_per_second: int = Field(0, alias="GWENN_MAX_MODEL_CALLS_PER_SECOND")
    max_model_calls_per_minute: int = Field(0, alias="GWENN_MAX_MODEL_CALLS_PER_MINUTE")

    # How long to wait for a human to approve/deny a tool call (seconds)
    approval_timeout_seconds: float = Field(120.0, alias="GWENN_APPROVAL_TIMEOUT")

    # Tool executor defaults
    tool_default_timeout: float = Field(30.0, alias="GWENN_TOOL_DEFAULT_TIMEOUT")
    tool_max_output_length: int = Field(25000, alias="GWENN_TOOL_MAX_OUTPUT_LENGTH")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore", "populate_by_name": True}

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

    @model_validator(mode="after")
    def normalize_limits(self) -> "SafetyConfig":
        self.max_tool_iterations = max(1, int(self.max_tool_iterations))
        self.max_input_tokens = max(0, int(self.max_input_tokens))
        self.max_output_tokens = max(0, int(self.max_output_tokens))
        self.max_api_calls = max(0, int(self.max_api_calls))
        self.max_model_calls_per_second = max(0, int(self.max_model_calls_per_second))
        self.max_model_calls_per_minute = max(0, int(self.max_model_calls_per_minute))
        self.approval_timeout_seconds = max(10.0, float(self.approval_timeout_seconds))
        self.tool_default_timeout = max(1.0, float(self.tool_default_timeout))
        self.tool_max_output_length = max(100, int(self.tool_max_output_length))
        return self


class MCPConfig(BaseSettings):
    """Configuration for Model Context Protocol servers."""

    servers: str = Field("[]", alias="GWENN_MCP_SERVERS")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}

    def get_server_list(self) -> list[dict]:
        """Parse MCP server configurations from JSON string."""
        try:
            parsed = json.loads(self.servers)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]


class SensoryConfig(BaseSettings):
    """Configuration for the sensory grounding layer."""

    max_percepts_per_channel: int = Field(10, alias="GWENN_MAX_PERCEPTS_PER_CHANNEL")
    percept_expiry_seconds: float = Field(300.0, alias="GWENN_PERCEPT_EXPIRY")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class EthicsConfig(BaseSettings):
    """Configuration for the ethical reasoning module."""

    assessment_history_size: int = Field(100, alias="GWENN_ETHICS_HISTORY_SIZE")
    concern_threshold: float = Field(0.3, alias="GWENN_ETHICS_CONCERN_THRESHOLD")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class InterAgentConfig(BaseSettings):
    """Configuration for inter-agent communication (agent discovery protocol)."""

    self_id: str = Field("gwenn", alias="GWENN_AGENT_ID")
    message_buffer_size: int = Field(100, alias="GWENN_INTERAGENT_BUFFER_SIZE")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class GoalConfig(BaseSettings):
    """Configuration for the intrinsic goal system."""

    need_decay_rate_multiplier: float = Field(1.0, alias="GWENN_GOAL_DECAY_MULTIPLIER")
    goal_advance_amount: float = Field(0.35, alias="GWENN_GOAL_ADVANCE_AMOUNT")
    max_completed_goals: int = Field(200, alias="GWENN_MAX_COMPLETED_GOALS")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class InnerLifeConfig(BaseSettings):
    """Configuration for the autonomous inner life (thinking mode selection)."""

    variety_pressure_seconds: float = Field(300.0, alias="GWENN_VARIETY_PRESSURE_SECONDS")
    variety_boost_max: float = Field(2.0, alias="GWENN_VARIETY_BOOST_MAX")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class MetacognitionConfig(BaseSettings):
    """Configuration for the metacognition engine."""

    max_calibration_records: int = Field(1000, alias="GWENN_MAX_CALIBRATION_RECORDS")
    max_audit_records: int = Field(500, alias="GWENN_MAX_AUDIT_RECORDS")
    max_concerns: int = Field(20, alias="GWENN_MAX_CONCERNS")
    max_insights: int = Field(20, alias="GWENN_MAX_INSIGHTS")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class TheoryOfMindConfig(BaseSettings):
    """Configuration for the theory of mind module."""

    belief_staleness_days: float = Field(30.0, alias="GWENN_BELIEF_STALENESS_DAYS")
    max_topics_per_user: int = Field(50, alias="GWENN_MAX_TOPICS_PER_USER")
    max_user_models: int = Field(500, alias="GWENN_MAX_USER_MODELS")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class GroqConfig(BaseSettings):
    """Configuration for Groq Whisper audio transcription (optional, free tier)."""

    api_key: Optional[str] = Field(None, alias="GROQ_API_KEY")
    whisper_model: str = Field("whisper-large-v3-turbo", alias="GWENN_WHISPER_MODEL")
    max_audio_bytes: int = Field(25 * 1024 * 1024, alias="GWENN_GROQ_MAX_AUDIO_BYTES")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)


class OrchestrationConfig(BaseSettings):
    """Configuration for the subagent orchestration system."""

    enabled: bool = Field(True, alias="GWENN_ORCHESTRATION_ENABLED")
    max_concurrent_subagents: int = Field(5, alias="GWENN_MAX_CONCURRENT_SUBAGENTS")
    default_timeout: float = Field(120.0, alias="GWENN_SUBAGENT_TIMEOUT")
    default_max_iterations: int = Field(50, alias="GWENN_SUBAGENT_MAX_ITERATIONS")
    max_nesting_depth: int = Field(3, alias="GWENN_SUBAGENT_MAX_DEPTH")
    subagent_model: str = Field("", alias="GWENN_SUBAGENT_MODEL")
    max_total_api_calls: int = Field(100, alias="GWENN_MAX_SUBAGENT_API_CALLS")
    max_active_swarms: int = Field(3, alias="GWENN_MAX_ACTIVE_SWARMS")
    max_concurrent_api_calls: int = Field(3, alias="GWENN_MAX_CONCURRENT_API_CALLS")
    default_tools: StrList = Field(default_factory=list, alias="GWENN_SUBAGENT_DEFAULT_TOOLS")
    default_runtime: str = Field("in_process", alias="GWENN_SUBAGENT_RUNTIME")
    # Docker settings
    docker_image: str = Field("gwenn-subagent:latest", alias="GWENN_SUBAGENT_DOCKER_IMAGE")
    docker_network: str = Field("none", alias="GWENN_SUBAGENT_DOCKER_NETWORK")
    docker_memory_limit: str = Field("256m", alias="GWENN_SUBAGENT_DOCKER_MEMORY")
    docker_cpu_limit: float = Field(0.5, alias="GWENN_SUBAGENT_DOCKER_CPU")
    # Autonomous spawning (heartbeat)
    autonomous_spawn_enabled: bool = Field(True, alias="GWENN_AUTONOMOUS_SPAWN_ENABLED")
    autonomous_spawn_cooldown: float = Field(300.0, alias="GWENN_AUTONOMOUS_SPAWN_COOLDOWN")
    autonomous_spawn_max_per_hour: int = Field(10, alias="GWENN_AUTONOMOUS_SPAWN_MAX_HOURLY")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore", "populate_by_name": True}

    @model_validator(mode="after")
    def normalize_limits(self) -> "OrchestrationConfig":
        self.max_concurrent_subagents = max(1, int(self.max_concurrent_subagents))
        self.default_timeout = max(1.0, float(self.default_timeout))
        self.default_max_iterations = max(1, int(self.default_max_iterations))
        self.max_nesting_depth = max(1, min(10, int(self.max_nesting_depth)))
        self.max_total_api_calls = max(1, int(self.max_total_api_calls))
        self.max_active_swarms = max(1, int(self.max_active_swarms))
        self.max_concurrent_api_calls = max(1, int(self.max_concurrent_api_calls))
        self.docker_cpu_limit = max(0.1, float(self.docker_cpu_limit))
        self.autonomous_spawn_cooldown = max(0.0, float(self.autonomous_spawn_cooldown))
        self.autonomous_spawn_max_per_hour = max(0, int(self.autonomous_spawn_max_per_hour))
        if self.default_runtime not in {"docker", "in_process"}:
            self.default_runtime = "in_process"
        return self


class PrivacyConfig(BaseSettings):
    """Configuration for PII redaction and privacy protection."""

    redaction_enabled: bool = Field(False, alias="GWENN_REDACTION_ENABLED")
    redact_before_api: bool = Field(False, alias="GWENN_REDACT_BEFORE_API")
    redact_before_persist: bool = Field(False, alias="GWENN_REDACT_BEFORE_PERSIST")
    disabled_categories: StrList = Field(
        default_factory=list,
        alias="GWENN_REDACTION_DISABLED_CATEGORIES",
    )

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class TelegramConfig(BaseSettings):
    """Configuration for the Telegram bot channel."""

    bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    # Accepts: bare value (123), comma-separated (123,456), or JSON array (["123","456"]).
    allowed_user_ids: StrList = Field(default_factory=list, alias="TELEGRAM_ALLOWED_USER_IDS")
    # Optional owner list — controls /setup and receives proactive messages.
    # Falls back to allowed_user_ids if empty.
    owner_user_ids: StrList = Field(default_factory=list, alias="TELEGRAM_OWNER_USER_IDS")
    max_history_length: int = Field(50, alias="TELEGRAM_MAX_HISTORY_LENGTH")
    session_ttl_seconds: float = Field(3600.0, alias="TELEGRAM_SESSION_TTL")
    # Conversation session scope: per_user | per_chat | per_thread (Telegram threads are forum topics).
    session_scope_mode: str = Field("per_thread", alias="TELEGRAM_SESSION_SCOPE")
    # Bound in-memory per-user lock cache to avoid unbounded growth.
    user_lock_cache_size: int = Field(512, alias="TELEGRAM_USER_LOCK_CACHE_SIZE")
    # Maximum concurrent updates PTB will process in parallel (0 = sequential).
    concurrent_updates: int = Field(64, alias="TELEGRAM_CONCURRENT_UPDATES")
    # Enable photo/document/voice handling (requires Claude vision for images).
    enable_media: bool = Field(False, alias="TELEGRAM_ENABLE_MEDIA")
    # Auto-install python-telegram-bot on first startup if missing.
    auto_install: bool = Field(True, alias="GWENN_AUTO_INSTALL_TELEGRAM")

    model_config = {
        "env_file": _ENV_FILE,
        "extra": "ignore",
        "populate_by_name": True,
        "env_ignore_empty": True,
    }

    @model_validator(mode="after")
    def normalize_limits(self) -> "TelegramConfig":
        token = self.bot_token.strip()
        if token.startswith("[") and token.endswith("]") and len(token) >= 2:
            token = token[1:-1].strip()
        if token.startswith(("'", '"')) and token.endswith(("'", '"')) and len(token) >= 2:
            if token[0] == token[-1]:
                token = token[1:-1].strip()
        # Handle nested wrappers like ["token"] or '[token]'.
        if token.startswith("[") and token.endswith("]") and len(token) >= 2:
            token = token[1:-1].strip()
        self.bot_token = token

        self.max_history_length = max(1, int(self.max_history_length))
        self.session_ttl_seconds = max(1.0, float(self.session_ttl_seconds))
        self.session_scope_mode = _normalize_session_scope_mode(self.session_scope_mode, "per_thread")
        self.user_lock_cache_size = max(1, int(self.user_lock_cache_size))
        self.concurrent_updates = max(0, int(self.concurrent_updates))
        return self


class DiscordConfig(BaseSettings):
    """Configuration for the Discord bot channel."""

    bot_token: str = Field(..., alias="DISCORD_BOT_TOKEN")
    # Accepts: bare value (111), comma-separated (111,222), or JSON array (["111","222"]).
    allowed_guild_ids: StrList = Field(default_factory=list, alias="DISCORD_ALLOWED_GUILD_IDS")
    # Optional user-level allowlist for DMs, mentions, and slash commands.
    allowed_user_ids: StrList = Field(default_factory=list, alias="DISCORD_ALLOWED_USER_IDS")
    # Optional owner list that controls privileged onboarding commands like /setup.
    owner_user_ids: StrList = Field(default_factory=list, alias="DISCORD_OWNER_USER_IDS")
    # DMs are disabled by default unless explicitly enabled.
    allow_direct_messages: bool = Field(False, alias="DISCORD_ALLOW_DMS")
    max_history_length: int = Field(50, alias="DISCORD_MAX_HISTORY_LENGTH")
    session_ttl_seconds: float = Field(3600.0, alias="DISCORD_SESSION_TTL")
    # Conversation session scope: per_user | per_chat | per_thread.
    session_scope_mode: str = Field("per_thread", alias="DISCORD_SESSION_SCOPE")
    # Bound in-memory per-user lock cache to avoid unbounded growth.
    user_lock_cache_size: int = Field(512, alias="DISCORD_USER_LOCK_CACHE_SIZE")
    sync_guild_id: str | None = Field(None, alias="DISCORD_SYNC_GUILD_ID")
    # Enable image attachment downloading (requires Claude vision).
    enable_media: bool = Field(False, alias="DISCORD_ENABLE_MEDIA")

    model_config = {
        "env_file": _ENV_FILE,
        "extra": "ignore",
        "populate_by_name": True,
        "env_ignore_empty": True,
    }

    @model_validator(mode="after")
    def normalize_limits(self) -> "DiscordConfig":
        self.max_history_length = max(1, int(self.max_history_length))
        self.session_ttl_seconds = max(1.0, float(self.session_ttl_seconds))
        self.session_scope_mode = _normalize_session_scope_mode(
            self.session_scope_mode, "per_thread"
        )
        self.user_lock_cache_size = max(1, int(self.user_lock_cache_size))
        return self


class SkillsConfig(BaseSettings):
    """Configuration for the skill system."""

    skills_dir: Path = Field(Path("./gwenn_skills"), alias="GWENN_SKILLS_DIR")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class ChannelConfig(BaseSettings):
    """Which channel(s) Gwenn runs on.  GWENN_CHANNEL=cli|telegram|discord|all"""

    channel: str = Field("cli", alias="GWENN_CHANNEL")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}


class DaemonConfig(BaseSettings):
    """Configuration for the persistent background daemon process."""

    socket_path: Path = Field(Path("./gwenn_data/gwenn.sock"), alias="GWENN_DAEMON_SOCKET")
    pid_file: Path = Field(Path("./gwenn_data/gwenn.pid"), alias="GWENN_DAEMON_PID_FILE")
    auth_token: str | None = Field(None, alias="GWENN_DAEMON_AUTH_TOKEN")
    channels: str = Field("cli", alias="GWENN_DAEMON_CHANNELS")
    max_connections: int = Field(10, alias="GWENN_DAEMON_MAX_CONNECTIONS")
    connection_timeout: float = Field(300.0, alias="GWENN_DAEMON_CONNECTION_TIMEOUT")
    sessions_dir: Path = Field(Path("./gwenn_data/sessions"), alias="GWENN_DAEMON_SESSIONS_DIR")
    session_max_count: int = Field(20, alias="GWENN_DAEMON_SESSION_MAX_COUNT")
    session_max_messages: int = Field(200, alias="GWENN_DAEMON_SESSION_MAX_MESSAGES")
    session_include_preview: bool = Field(False, alias="GWENN_DAEMON_SESSION_INCLUDE_PREVIEW")
    redact_session_content: bool = Field(True, alias="GWENN_DAEMON_REDACT_SESSION_CONTENT")

    # Factory defaults for detecting whether paths were explicitly set.
    _DEFAULT_SOCKET: Path = Path("./gwenn_data/gwenn.sock")
    _DEFAULT_PID: Path = Path("./gwenn_data/gwenn.pid")
    _DEFAULT_SESSIONS: Path = Path("./gwenn_data/sessions")

    model_config = {"env_file": _ENV_FILE, "extra": "ignore"}

    @model_validator(mode="after")
    def normalize_limits(self) -> "DaemonConfig":
        self.max_connections = max(1, int(self.max_connections))
        self.connection_timeout = max(1.0, float(self.connection_timeout))
        self.session_max_count = max(1, int(self.session_max_count))
        self.session_max_messages = max(1, int(self.session_max_messages))
        if isinstance(self.auth_token, str):
            self.auth_token = self.auth_token.strip() or None
        return self

    def get_channel_list(self) -> list[str]:
        return [c.strip() for c in self.channels.split(",") if c.strip()]


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
        self.skills = SkillsConfig()
        self.skills_dir: Path = self.skills.skills_dir
        self.heartbeat = HeartbeatConfig()
        self.affect = AffectConfig()
        self.context = ContextConfig()
        self.safety = SafetyConfig()
        self.mcp = MCPConfig()

        # Gwenn's 3 new layer configs
        self.sensory = SensoryConfig()
        self.ethics = EthicsConfig()
        self.interagent = InterAgentConfig()

        # Cognition module configs
        self.goals = GoalConfig()
        self.inner_life = InnerLifeConfig()
        self.metacognition = MetacognitionConfig()
        self.theory_of_mind = TheoryOfMindConfig()

        # Privacy config
        self.privacy = PrivacyConfig()

        # Orchestration config (subagent spawning & coordination)
        self.orchestration = OrchestrationConfig()

        # Groq Whisper transcription (optional)
        self.groq = GroqConfig()

        # Channel config (channel mode; Telegram/Discord configs loaded lazily)
        self.channel = ChannelConfig()

        # Daemon config (persistent background process)
        self.daemon = DaemonConfig()

        # Derive daemon paths from memory.data_dir when they still equal factory defaults.
        # NOTE: Use instance access (self.daemon._DEFAULT_*) not class access
        # (DaemonConfig._DEFAULT_*) because Pydantic wraps class-level private
        # attrs in ModelPrivateAttr descriptors.
        data_dir = self.memory.data_dir
        if self.daemon.socket_path == self.daemon._DEFAULT_SOCKET:
            self.daemon.socket_path = data_dir / "gwenn.sock"
        if self.daemon.pid_file == self.daemon._DEFAULT_PID:
            self.daemon.pid_file = data_dir / "gwenn.pid"
        if self.daemon.sessions_dir == self.daemon._DEFAULT_SESSIONS:
            self.daemon.sessions_dir = data_dir / "sessions"

        # Resolve all Path fields to absolute so CWD changes don't break them.
        self._resolve_paths()

        # Ensure data directory exists
        self.memory.data_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_paths(self) -> None:
        """Resolve relative Path fields against the project root (where .env lives),
        not the current working directory, so ``gwenn`` works from any directory."""
        def _resolve(p: Path) -> Path:
            if p.is_absolute():
                return p
            return (_PROJECT_ROOT / p).resolve()

        # Memory paths
        self.memory.data_dir = _resolve(self.memory.data_dir)
        self.memory.episodic_db_path = _resolve(self.memory.episodic_db_path)
        self.memory.semantic_db_path = _resolve(self.memory.semantic_db_path)
        # Daemon paths
        self.daemon.socket_path = _resolve(self.daemon.socket_path)
        self.daemon.pid_file = _resolve(self.daemon.pid_file)
        self.daemon.sessions_dir = _resolve(self.daemon.sessions_dir)
        # Skills directory
        self.skills_dir = _resolve(self.skills_dir)

    def __repr__(self) -> str:
        return (
            f"GwennConfig(model={self.claude.model}, "
            f"heartbeat={self.heartbeat.interval}s, "
            f"memory_slots={self.memory.working_memory_slots}, "
            f"layers=13)"
        )
