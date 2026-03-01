# Gwenn Configuration Reference

All configuration is via environment variables, loaded from `.env` using
pydantic-settings. Copy `.env.example` to `.env` and adjust values as needed.

---

## Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(unset)* | Anthropic API key. Takes priority over OAuth if both set. |
| `ANTHROPIC_AUTH_TOKEN` | *(unset)* | Claude Code OAuth token (explicit). |
| `CLAUDE_CODE_OAUTH_TOKEN` | *(unset)* | Alias for `ANTHROPIC_AUTH_TOKEN`. |
| `ANTHROPIC_BASE_URL` | *(unset)* | Custom API base URL (proxy). Must forward `anthropic-beta` header. |

If no key or token is set, Gwenn auto-detects from `~/.claude/.credentials.json`.

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_LOG_LEVEL` | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

## Claude API

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_MODEL` | `claude-sonnet-4-5-20250929` | Claude model to use. Recommended: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| `GWENN_MAX_TOKENS` | `8192` | Max output tokens per response |
| `GWENN_THINKING_BUDGET` | `16000` | Token budget for extended thinking (used as fallback when adaptive is unavailable) |
| `GWENN_THINKING_EFFORT` | `high` | Thinking effort level: `low`, `medium`, `high`, or `max`. Controls depth of reasoning in adaptive mode. |
| `GWENN_REQUEST_TIMEOUT_SECONDS` | `120.0` | API request timeout |
| `GWENN_RETRY_MAX_RETRIES` | `3` | Max retry attempts on transient failure |
| `GWENN_RETRY_BASE_DELAY` | `0.5` | Base delay between retries (seconds) |
| `GWENN_RETRY_MAX_DELAY` | `8.0` | Maximum retry delay (seconds) |
| `GWENN_RETRY_EXPONENTIAL_BASE` | `2.0` | Exponential backoff multiplier |
| `GWENN_RETRY_JITTER_RANGE` | `0.25` | Random jitter range for retry timing |

## Channel Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `CLI_ENABLED` | `true` | Enable the CLI terminal channel |
| `TELEGRAM_ENABLED` | `false` | Enable the Telegram bot channel |
| `DISCORD_ENABLED` | `false` | Enable the Discord bot channel |
| `SLACK_ENABLED` | `false` | Enable the Slack bot channel |

## Memory & Persistence

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_DATA_DIR` | `./gwenn_data` | Root data directory for all persistence |
| `GWENN_EPISODIC_DB` | `./gwenn_data/episodic.db` | SQLite database for episodic memory |
| `GWENN_SEMANTIC_DB` | `./gwenn_data/semantic_vectors` | ChromaDB directory (only used in embedding/hybrid mode) |
| `GWENN_RETRIEVAL_MODE` | `keyword` | Memory retrieval: `keyword`, `embedding`, or `hybrid` |
| `GWENN_EMBEDDING_TOP_K` | `25` | Top-K results for embedding retrieval |
| `GWENN_HYBRID_KEYWORD_WEIGHT` | `0.5` | Keyword component weight in hybrid mode |
| `GWENN_HYBRID_EMBEDDING_WEIGHT` | `0.5` | Embedding component weight in hybrid mode |
| `GWENN_STARTUP_EPISODE_LIMIT` | `5000` | Max recent episodes to preload (0 = disable) |
| `GWENN_SHUTDOWN_PERSIST_RECENT_EPISODES` | `0` | Episodes to persist on shutdown |
| `GWENN_PERSIST_SEMANTIC_AFTER_CONSOLIDATION` | `True` | Flush semantic graph after each consolidation |
| `GWENN_WM_EVICTION_TO_EPISODIC` | `True` | Promote evicted working memory items to episodic |

## Consolidation

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_CONSOLIDATION_INTERVAL` | `600.0` | Seconds between consolidation eligibility checks |
| `GWENN_CONSOLIDATION_MAX_EPISODES` | `200` | Max episodes per consolidation batch |
| `GWENN_CONSOLIDATION_RECENCY_DECAY` | `0.995` | Recency decay factor for scoring |
| `GWENN_CONSOLIDATION_IMPORTANCE_WEIGHT` | `0.4` | Importance weight in scoring formula |
| `GWENN_CONSOLIDATION_RECENCY_WEIGHT` | `0.3` | Recency weight in scoring formula |
| `GWENN_CONSOLIDATION_RELEVANCE_WEIGHT` | `0.3` | Relevance weight in scoring formula |

## Heartbeat

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_HEARTBEAT_INTERVAL` | `30.0` | Starting heartbeat interval (seconds) |
| `GWENN_HEARTBEAT_MIN_INTERVAL` | `5.0` | Fastest interval during active conversation |
| `GWENN_HEARTBEAT_MAX_INTERVAL` | `120.0` | Slowest interval when idle |
| `GWENN_PROACTIVE_MESSAGES` | `false` | Share significant autonomous thoughts with channel owners |
| `GWENN_HEARTBEAT_CIRCUIT_MAX_CONSECUTIVE` | `10` | Consecutive failures before circuit breaker opens |
| `GWENN_HEARTBEAT_CIRCUIT_BASE_SECONDS` | `60.0` | Base backoff delay when circuit breaker opens (seconds) |
| `GWENN_HEARTBEAT_CIRCUIT_MAX_SECONDS` | `900.0` | Maximum backoff delay (15 min cap) |

## Affect (Emotional System)

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_AROUSAL_CEILING` | `0.92` | Maximum arousal level (prevents overexcitation) |
| `GWENN_DISTRESS_THRESHOLD` | `0.80` | Arousal level triggering distress circuit breaker |
| `GWENN_DISTRESS_DURATION_LIMIT` | `300` | Max seconds of sustained distress before intervention |
| `GWENN_AFFECT_MOMENTUM_DECAY` | `0.85` | How quickly emotions carry forward (0-1) |
| `GWENN_AFFECT_BASELINE_PULL` | `0.05` | Pull strength toward emotional baseline (0-1) |

## Context Window

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_CONTEXT_LIMIT` | `180000` | Token budget for context window |
| `GWENN_COMPACTION_TRIGGER` | `0.85` | Context usage ratio triggering compaction (0-1) |
| `GWENN_WORKING_MEMORY_SLOTS` | `7` | Working memory capacity |
| `GWENN_MAX_CONVERSATION_MESSAGES` | `400` | Max messages in conversation history before trimming |

## Safety

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_MAX_TOOL_ITERATIONS` | `150` | Max iterations in the agentic tool-use loop |
| `GWENN_REQUIRE_APPROVAL_FOR` | `["file_write","shell_exec","run_command","web_request"]` | Tools requiring human approval |
| `GWENN_SANDBOX_ENABLED` | `True` | Enable deny-by-default for non-builtin tools |
| `GWENN_ALLOWED_TOOLS` | `[]` | Allowlisted tool names (under deny-by-default) |
| `GWENN_DENIED_TOOLS` | `[]` | Blocklisted tool names (overrides allowlist) |
| `GWENN_MAX_INPUT_TOKENS` | `0` | Input token budget (0 = unlimited) |
| `GWENN_MAX_OUTPUT_TOKENS` | `0` | Output token budget (0 = unlimited) |
| `GWENN_MAX_API_CALLS` | `0` | API call budget (0 = unlimited) |
| `GWENN_MAX_MODEL_CALLS_PER_SECOND` | `0` | Rate limit per second (0 = unlimited) |
| `GWENN_MAX_MODEL_CALLS_PER_MINUTE` | `0` | Rate limit per minute (0 = unlimited) |
| `GWENN_TOOL_DEFAULT_POLICY` | `deny` | Default policy for non-builtin tools: `deny` or `allow` |
| `GWENN_APPROVAL_TIMEOUT` | `120.0` | Seconds to wait for human approval of a tool call |
| `GWENN_TOOL_DEFAULT_TIMEOUT` | `30.0` | Default timeout for tool execution (seconds) |
| `GWENN_TOOL_MAX_OUTPUT_LENGTH` | `25000` | Max characters in tool output before truncation |

## Privacy

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_REDACTION_ENABLED` | `False` | Master switch for PII redaction |
| `GWENN_REDACT_BEFORE_API` | `False` | Redact PII before sending to Claude |
| `GWENN_REDACT_BEFORE_PERSIST` | `False` | Redact PII before writing to disk |
| `GWENN_REDACTION_DISABLED_CATEGORIES` | *(empty)* | Comma-separated categories to skip |

## Sensory Grounding

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_MAX_PERCEPTS_PER_CHANNEL` | `10` | Max sensory percepts per channel |
| `GWENN_PERCEPT_EXPIRY` | `300.0` | Percept expiry in seconds |

## Ethics

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_ETHICS_HISTORY_SIZE` | `100` | Max ethical assessment records |
| `GWENN_ETHICS_CONCERN_THRESHOLD` | `0.3` | Minimum score to flag an ethical concern |

## Inter-Agent Communication

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_AGENT_ID` | `gwenn` | This agent's identifier for inter-agent messaging |
| `GWENN_INTERAGENT_BUFFER_SIZE` | `100` | Max messages in inbox/outbox buffers |

## Goal System

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_GOAL_DECAY_MULTIPLIER` | `1.0` | Multiplier for intrinsic need decay rate |
| `GWENN_GOAL_ADVANCE_AMOUNT` | `0.35` | Default progress per goal advance call |
| `GWENN_MAX_COMPLETED_GOALS` | `200` | Max completed goals in history |

## Inner Life

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_VARIETY_PRESSURE_SECONDS` | `300.0` | Seconds before an unused thinking mode gets a variety boost |
| `GWENN_VARIETY_BOOST_MAX` | `2.0` | Maximum variety boost multiplier |

## Metacognition

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_MAX_CALIBRATION_RECORDS` | `1000` | Max confidence calibration records |
| `GWENN_MAX_AUDIT_RECORDS` | `500` | Max honesty audit records |
| `GWENN_MAX_CONCERNS` | `20` | Max active metacognitive concerns |
| `GWENN_MAX_INSIGHTS` | `20` | Max metacognitive insights carried forward |

## Theory of Mind

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_BELIEF_STALENESS_DAYS` | `30.0` | Days before unconfirmed beliefs decay |
| `GWENN_MAX_TOPICS_PER_USER` | `50` | Max topics tracked per user model |
| `GWENN_MAX_USER_MODELS` | `500` | Max user models before LRU eviction |

## Orchestration (Subagents)

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_ORCHESTRATION_ENABLED` | `True` | Enable subagent system |
| `GWENN_MAX_CONCURRENT_SUBAGENTS` | `5` | Max parallel subagents |
| `GWENN_SUBAGENT_TIMEOUT` | `120.0` | Per-subagent timeout (seconds) |
| `GWENN_SUBAGENT_MAX_ITERATIONS` | `50` | Max agentic loop iterations per subagent |
| `GWENN_SUBAGENT_MAX_DEPTH` | `3` | Max nesting depth |
| `GWENN_SUBAGENT_MODEL` | *(empty)* | Model for subagents (empty = use parent) |
| `GWENN_MAX_SUBAGENT_API_CALLS` | `100` | Session-wide API call cap for subagents |
| `GWENN_MAX_ACTIVE_SWARMS` | `3` | Max concurrent swarms |
| `GWENN_MAX_CONCURRENT_API_CALLS` | `3` | Max simultaneous API calls |
| `GWENN_SUBAGENT_DEFAULT_TOOLS` | *(empty)* | Comma-separated default tools for subagents |
| `GWENN_SUBAGENT_RUNTIME` | `in_process` | Runtime: `in_process` or `docker` |
| `GWENN_SUBAGENT_DOCKER_IMAGE` | `gwenn-subagent:latest` | Docker image for subagent containers |
| `GWENN_SUBAGENT_DOCKER_NETWORK` | `none` | Docker network mode |
| `GWENN_SUBAGENT_DOCKER_MEMORY` | `256m` | Docker memory limit |
| `GWENN_SUBAGENT_DOCKER_CPU` | `0.5` | Docker CPU limit |
| `GWENN_AUTONOMOUS_SPAWN_ENABLED` | `True` | Allow heartbeat-driven subagent spawning |
| `GWENN_AUTONOMOUS_SPAWN_COOLDOWN` | `300.0` | Cooldown between autonomous spawns (seconds) |
| `GWENN_AUTONOMOUS_SPAWN_MAX_HOURLY` | `10` | Max autonomous spawns per hour |

## Skills

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_SKILLS_DIR` | `./gwenn_skills` | Directory containing skill `.md` files |

## MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_MCP_SERVERS` | `[]` | JSON array of MCP server configurations |

Each server object supports:
- `name` (string) -- server identifier
- `transport` (string) -- `stdio` or `streamable_http`
- `command` (string, stdio only) -- executable to run
- `args` (array, stdio only) -- command arguments
- `env` (object, stdio only) -- environment variables
- `url` (string, HTTP only) -- server URL
- `api_key` (string, HTTP only) -- optional Bearer token

## Voice Transcription (Groq Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(unset)* | Groq API key (free tier at https://console.groq.com) |
| `GWENN_WHISPER_MODEL` | `whisper-large-v3-turbo` | Whisper model for voice transcription |
| `GWENN_GROQ_MAX_AUDIO_BYTES` | `26214400` | Max audio file size (25 MB) |

## Telegram Channel

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | *(required)* | Bot token from @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | `[]` | User ID whitelist (empty = public) |
| `TELEGRAM_OWNER_USER_IDS` | `[]` | Owner IDs for `/setup` and proactive messages |
| `TELEGRAM_MAX_HISTORY_LENGTH` | `50` | Max messages in conversation history |
| `TELEGRAM_SESSION_TTL` | `3600.0` | Session time-to-live (seconds) |
| `TELEGRAM_SESSION_SCOPE` | `per_thread` | Session scope: `per_user`, `per_chat`, `per_thread` |
| `TELEGRAM_CONCURRENT_UPDATES` | `64` | Max concurrent update processing (0 = sequential) |
| `TELEGRAM_USER_LOCK_CACHE_SIZE` | `512` | Per-user lock cache size (bounds memory) |
| `TELEGRAM_ENABLE_MEDIA` | `false` | Enable photo/document/voice handling |
| `GWENN_AUTO_INSTALL_TELEGRAM` | `true` | Auto-install `python-telegram-bot` if missing |

## Discord Channel

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | *(required)* | Bot token from Discord Developer Portal |
| `DISCORD_ALLOWED_GUILD_IDS` | `[]` | Guild whitelist (empty = all guilds) |
| `DISCORD_ALLOWED_USER_IDS` | `[]` | User whitelist (empty = all users) |
| `DISCORD_OWNER_USER_IDS` | `[]` | Owner IDs for `/setup` and proactive messages |
| `DISCORD_ALLOW_DMS` | `false` | Allow direct messages |
| `DISCORD_MAX_HISTORY_LENGTH` | `50` | Max messages in conversation history |
| `DISCORD_SESSION_TTL` | `3600.0` | Session TTL (seconds) |
| `DISCORD_SESSION_SCOPE` | `per_thread` | Session scope: `per_user`, `per_chat`, `per_thread` |
| `DISCORD_USER_LOCK_CACHE_SIZE` | `512` | Per-user lock cache size (bounds memory) |
| `DISCORD_SYNC_GUILD_ID` | *(empty)* | Guild ID for instant slash-command sync in dev |
| `DISCORD_ENABLE_MEDIA` | `false` | Enable image attachment downloading |

## Daemon

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_DAEMON_SOCKET` | `./gwenn_data/gwenn.sock` | Unix socket path |
| `GWENN_DAEMON_PID_FILE` | `./gwenn_data/gwenn.pid` | PID file path |
| `GWENN_DAEMON_AUTH_TOKEN` | *(empty)* | Shared auth token (recommended) |
| *(uses CLI/TELEGRAM/DISCORD_ENABLED)* | | Channels managed by daemon |
| `GWENN_DAEMON_MAX_CONNECTIONS` | `10` | Max concurrent connections |
| `GWENN_DAEMON_CONNECTION_TIMEOUT` | `300.0` | Connection timeout (seconds) |
| `GWENN_DAEMON_SESSIONS_DIR` | `./gwenn_data/sessions` | Session storage directory |
| `GWENN_DAEMON_SESSION_MAX_COUNT` | `20` | Max stored sessions |
| `GWENN_DAEMON_SESSION_MAX_MESSAGES` | `200` | Max messages per session |
| `GWENN_DAEMON_SESSION_INCLUDE_PREVIEW` | `False` | Show message previews in `/resume` |
| `GWENN_DAEMON_REDACT_SESSION_CONTENT` | `True` | PII-redact stored session content |

## Checkpoint/Restore

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_CHECKPOINT_ENABLED` | `true` | Enable periodic state checkpoints |
| `GWENN_CHECKPOINT_INTERVAL_BEATS` | `50` | Heartbeats between checkpoints |
| `GWENN_CHECKPOINT_MAX_COUNT` | `10` | Max stored checkpoints |

## Self-Healing

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_SELF_HEALING_ENABLED` | `true` | Enable autonomous recovery |
| `GWENN_SELF_HEALING_COOLDOWN` | `300` | Cooldown between healing actions (seconds) |
| `GWENN_SELF_HEALING_MAX_ACTIONS_HOUR` | `20` | Max healing actions per hour |
| `GWENN_SELF_HEALING_CHANNEL_RESTART` | `true` | Allow channel restart as healing action |
| `GWENN_SELF_HEALING_STUCK_MULTIPLIER` | `2.0` | Multiplier for stuck-heartbeat detection |
| `GWENN_SELF_HEALING_MEMORY_THRESHOLD` | `85.0` | Memory usage % triggering healing |
| `GWENN_SELF_HEALING_ERROR_RATE_THRESHOLD` | `5.0` | Error rate triggering healing |

## Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `GWENN_GATEWAY_ENABLED` | `true` | Enable WebSocket/HTTP gateway |
| `GWENN_GATEWAY_HOST` | `127.0.0.1` | Gateway bind address |
| `GWENN_GATEWAY_PORT` | `18900` | Gateway port |
| `GWENN_LEGACY_SOCKET_ENABLED` | `true` | Keep Unix socket alongside gateway |
| `GWENN_MCP_SERVER_ENABLED` | `false` | Expose MCP server endpoint |
| `GWENN_A2A_ENABLED` | `false` | Enable agent-to-agent protocol |
| `GWENN_HEARTBEAT_CORE` | `true` | Run heartbeat in gateway process |

## Slack Channel

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_ENABLED` | `false` | Enable Slack bot channel |
| `GWENN_SLACK_BOT_TOKEN` | *(required)* | Slack bot token (`xoxb-...`) |
| `GWENN_SLACK_APP_TOKEN` | *(required)* | Slack app-level token (`xapp-...`) |
| `GWENN_SLACK_ALLOWED_USER_IDS` | `[]` | User ID whitelist |
| `GWENN_SLACK_OWNER_USER_IDS` | `[]` | Owner IDs |
| `GWENN_SLACK_MAX_HISTORY_LENGTH` | `50` | Max messages per session |
| `GWENN_SLACK_SESSION_TTL` | `3600.0` | Session TTL (seconds) |
| `GWENN_SLACK_SESSION_SCOPE` | `per_thread` | Session scope: `per_user`, `per_chat`, `per_thread` |
| `GWENN_SLACK_USER_LOCK_CACHE_SIZE` | `512` | Lock cache size |
