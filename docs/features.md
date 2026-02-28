# Gwenn Feature Guide

This document covers how to use every user-facing feature in Gwenn, including
configuration, commands, and practical examples.

---

## Table of contents

- [Subagents & Orchestration](#subagents--orchestration)
- [Skills System](#skills-system)
- [MCP Integration](#mcp-integration)
- [Built-in Tools](#built-in-tools)
- [Memory System](#memory-system)
- [Heartbeat & Autonomous Cognition](#heartbeat--autonomous-cognition)
- [Channels](#channels)
- [Daemon](#daemon)
- [Interactive Commands](#interactive-commands)
- [Output Style](#output-style)
- [Safety & Tool Policy](#safety--tool-policy)
- [Privacy & PII Redaction](#privacy--pii-redaction)
- [Onboarding](#onboarding)

---

## Subagents & Orchestration

Gwenn can delegate work to subagents -- lightweight worker instances that run
focused subtasks in parallel. The orchestration system manages their lifecycle,
enforces budgets, and aggregates results.

### How it works

When Gwenn determines a task would benefit from parallel execution, she can:

1. **Spawn a single subagent** for a focused subtask (e.g., research a topic,
   draft a response, run a calculation)
2. **Spawn a swarm** of multiple subagents working on related tasks
   simultaneously, with results aggregated automatically

Subagents inherit Gwenn's tools and memory access but operate with independent
iteration limits and API call budgets.

### Tools

| Tool | Description |
|------|-------------|
| `spawn_subagent` | Create a single focused worker. Parameters: `task_description`, `tools` (optional subset), `max_iterations`, `isolation` (in_process/docker), `system_prompt` |
| `spawn_swarm` | Launch multiple subagents in parallel. Parameters: `description`, `tasks` (list), `aggregation_strategy` (concatenate/synthesize/vote) |
| `check_subagent` | Check the status of a running subagent without waiting. Parameter: `task_id` |
| `collect_results` | Gather results from a completed task or swarm. Parameter: `task_id` |
| `cancel_subagent` | Cancel a running subagent or swarm. Parameter: `task_id` |

### Aggregation strategies

When using swarms, results from multiple subagents are combined:

- **concatenate** -- join all results in order (default)
- **synthesize** -- use Claude to produce a coherent summary from all results
- **vote** -- majority vote for tasks with discrete answers

### Autonomous spawning

When `GWENN_AUTONOMOUS_SPAWN_ENABLED=True` (default), Gwenn can spawn subagents
during heartbeat cycles without user prompting. This is rate-limited:

- Cooldown between autonomous spawns: 300 seconds (configurable)
- Maximum autonomous spawns per hour: 10 (configurable)

### Safety guardrails

- **Concurrency limit**: max 5 parallel subagents (configurable)
- **Depth limit**: max 3 levels of nesting prevents infinite recursion
- **API call budget**: session-wide cap on total API calls across all subagents
- **Timeout**: per-subagent timeout (default 120 seconds)
- **Active swarm limit**: max 3 concurrent swarms

### Configuration

```bash
# Enable/disable orchestration entirely
GWENN_ORCHESTRATION_ENABLED=True

# Concurrency and limits
GWENN_MAX_CONCURRENT_SUBAGENTS=5
GWENN_SUBAGENT_TIMEOUT=120.0
GWENN_SUBAGENT_MAX_ITERATIONS=50
GWENN_SUBAGENT_MAX_DEPTH=3
GWENN_MAX_SUBAGENT_API_CALLS=100
GWENN_MAX_ACTIVE_SWARMS=3
GWENN_MAX_CONCURRENT_API_CALLS=3

# Use a different (cheaper) model for subagents (empty = parent model)
# GWENN_SUBAGENT_MODEL=

# Default tools available to subagents (comma-separated, empty = all)
# GWENN_SUBAGENT_DEFAULT_TOOLS=

# Runtime: docker (isolated containers) or in_process
GWENN_SUBAGENT_RUNTIME=in_process

# Docker container settings
GWENN_SUBAGENT_DOCKER_IMAGE=gwenn-subagent:latest
GWENN_SUBAGENT_DOCKER_NETWORK=none
GWENN_SUBAGENT_DOCKER_MEMORY=256m
GWENN_SUBAGENT_DOCKER_CPU=0.5

# Autonomous spawning during heartbeat
GWENN_AUTONOMOUS_SPAWN_ENABLED=True
GWENN_AUTONOMOUS_SPAWN_COOLDOWN=300.0
GWENN_AUTONOMOUS_SPAWN_MAX_HOURLY=10
```

### Example interaction

```
You: Can you research the latest developments in quantum computing and
     summarize key breakthroughs from 2025?

Gwenn: I'll spawn a research swarm to cover this efficiently...
       [spawns 3 subagents: one for hardware, one for algorithms, one for
        applications, then synthesizes results into a unified summary]
```

---

## Skills System

Skills are markdown-defined capabilities that extend Gwenn's abilities. They
let you teach Gwenn new workflows without writing Python code.

### How skills work

Each skill is a `.md` file in the `gwenn_skills/` directory with:
1. **JSON frontmatter** -- name, description, parameters, category, risk level
2. **Instruction body** -- step-by-step instructions using `{parameter}`
   placeholders

When invoked, Gwenn substitutes parameter values into the instructions and
follows the steps, using her built-in tools to accomplish the task.

### Skill file format

```markdown
---
{
  "name": "get_weather",
  "description": "Fetches current weather for any location. Use when someone
    asks about weather, temperature, or forecast.",
  "category": "information",
  "version": "1.0",
  "risk_level": "medium",
  "tags": ["user_command", "weather", "forecast"],
  "parameters": {
    "location": {
      "type": "string",
      "description": "City name or coordinates",
      "required": true
    },
    "units": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"],
      "default": "celsius"
    }
  }
}
---

Fetch weather for **{location}** in **{units}**.

## Steps

1. Call `fetch_url` with URL: `https://wttr.in/{location}?format=j1`
2. Extract temperature, conditions, humidity from the response
3. Convert units if needed using `calculate`

## Output format

One to three natural sentences.

## Error handling

- Location not found -> ask user to clarify
```

### Key fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | `snake_case` identifier (becomes a tool name) |
| `description` | Yes | Third-person description + trigger phrases for when to use it |
| `parameters` | No | Input parameters with types, descriptions, defaults |
| `category` | No | Grouping: `information`, `productivity`, `developer`, `communication`, `analysis`, `skills` |
| `risk_level` | No | `low` (default) or `medium` (for skills making network requests) |
| `tags` | No | Keywords for discoverability. Include `user_command` to expose as a Telegram/Discord bot command |
| `version` | No | Semantic version string |

### Tags with special meaning

- **`user_command`** -- skill appears in Telegram's bot command menu and Discord
  slash commands. Without this tag, skills are available but not listed as
  explicit commands.
- **`autonomous`** -- skill is designed to run during heartbeat cycles for
  self-monitoring, introspection, or maintenance tasks.

### Built-in skills (26 total)

**User-invocable** (tagged `user_command`):
- `get_weather` -- fetch weather for any location
- `get_news` -- fetch and summarize news headlines
- `summarize_url` -- summarize a web page or article
- `set_reminder` -- create persistent reminders
- `explain_code` -- explain code snippets
- `create_skill` -- guided skill creation workflow

**Autonomous** (run during heartbeat):
- 20 introspection and honesty skills covering self-monitoring,
  performativity diagnosis, confidence calibration, narrative integrity,
  relational readiness, and more

### Creating new skills

**Method 1: Ask Gwenn**

Just describe what you want:

```
You: Can you learn how to translate text between languages?
Gwenn: [uses the create_skill skill to design and register a new skill]
```

**Method 2: Write a `.md` file**

1. Create a file in `gwenn_skills/` following the format above
2. Run `/skills` or ask Gwenn to reload skills -- or restart

**Method 3: Use the `skill_builder` tool directly**

Gwenn can call `skill_builder` programmatically with name, description,
instructions, and parameters.

### Managing skills

| Tool | Description |
|------|-------------|
| `skill_builder` | Create a new skill |
| `update_skill` | Modify an existing skill's description, instructions, or parameters |
| `delete_skill` | Remove a skill |
| `reload_skills` | Hot-load new `.md` files from the skills directory |
| `list_skills` | List all loaded skills |

### Configuration

```bash
# Directory containing skill .md files (auto-created if missing)
GWENN_SKILLS_DIR=./gwenn_skills
```

---

## MCP Integration

Gwenn supports the [Model Context Protocol](https://modelcontextprotocol.io/)
for connecting to external tool servers. MCP tools are discovered automatically
and become available alongside built-in tools.

### Supported transports

| Transport | Description | Use case |
|-----------|-------------|----------|
| `stdio` | JSON-RPC over subprocess stdin/stdout | Local MCP servers |
| `streamable_http` | JSON-RPC over HTTP POST | Remote MCP servers |

### Configuration

Set `GWENN_MCP_SERVERS` in `.env` as a JSON array:

```bash
GWENN_MCP_SERVERS=[
  {
    "name": "filesystem",
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    "env": {}
  },
  {
    "name": "remote_api",
    "transport": "streamable_http",
    "url": "http://localhost:8000/mcp",
    "api_key": "optional_bearer_token"
  }
]
```

### How it works

1. On startup, Gwenn connects to each configured MCP server
2. Sends an `initialize` handshake and discovers available tools via `tools/list`
3. Registers each tool as a proxy tool prefixed with `mcp_<server_name>_`
4. When Claude invokes an MCP tool, Gwenn routes the call to the server
5. Results are returned to Claude's agentic loop

### Tool naming

MCP tools are registered with the prefix `mcp_<server_name>_<tool_name>`,
truncated to 64 characters max with collision disambiguation.

### Security

MCP-sourced tools are assigned `medium` risk level by default. Under
deny-by-default policy (`GWENN_SANDBOX_ENABLED=True`), MCP tools must be
explicitly allowlisted in `GWENN_ALLOWED_TOOLS` to be used.

### Checking status

Use the `/mcp` interactive command to see:
- Configured servers and their connection status
- Discovered tools per server
- Any connection errors

---

## Built-in Tools

Gwenn ships with tools organized into categories. All built-in tools are
automatically allowed regardless of deny-by-default policy.

### Memory tools

| Tool | Description |
|------|-------------|
| `remember` | Store a long-term memory. Parameters: `content`, `importance` (0.0-1.0), `category` |
| `recall` | Search episodic memory. Parameters: `query`, `category`, `max_results` |
| `search_knowledge` | Query the semantic knowledge graph. Parameters: `query`, `min_confidence` |
| `check_emotional_state` | Get current emotional dimensions (valence, arousal, dominance, certainty, goal_congruence) |
| `check_goals` | View intrinsic needs satisfaction levels and active goals |
| `set_note_to_self` | Write persistent notes (reminders, commitments, self-knowledge, user notes) to `GWENN_CONTEXT.md` |

### Utility tools

| Tool | Description |
|------|-------------|
| `get_datetime` | Current date/time with timezone support |
| `calculate` | Safe math expressions (AST-based, no eval) |
| `fetch_url` | HTTP GET with timeout and character limit |
| `convert_units` | Temperature, distance, weight, storage, speed |
| `get_calendar` | Show month, day of week, days between dates |
| `generate_token` | UUID4, hex, URL-safe, passwords, random values |
| `format_json` | Pretty-print, validate, or minify JSON |
| `encode_decode` | Base64, URL, HTML encoding/decoding |
| `hash_text` | SHA256, SHA512, SHA3, MD5, SHA1 |
| `text_stats` | Word/character count, sentences, paragraphs, reading time |
| `get_system_info` | OS, Python version, CPU, disk, memory info |
| `run_command` | Execute a shell command and return stdout, stderr, exit code |
| `present_choices` | Present multiple-choice options with Telegram inline buttons |

### Filesystem tools

| Tool | Description |
|------|-------------|
| `read_file` | Read a file (for subagent filesystem access) |
| `write_file` | Write a file (for subagent filesystem access) |

### Communication tools

| Tool | Description |
|------|-------------|
| `think_aloud` | Share internal thought process (framed as introspection) |

### Skills management tools

| Tool | Description |
|------|-------------|
| `skill_builder` | Create a new skill from name + description + instructions |
| `update_skill` | Modify an existing skill |
| `delete_skill` | Remove a skill |
| `reload_skills` | Hot-load new skill files |
| `list_skills` | List all loaded skills |

### Orchestration tools

| Tool | Description |
|------|-------------|
| `spawn_subagent` | Create a focused worker for a subtask |
| `spawn_swarm` | Launch multiple subagents with result aggregation |
| `check_subagent` | Check status of a spawned subagent by `task_id` |
| `collect_results` | Gather results from spawned tasks by `task_id` |
| `cancel_subagent` | Cancel a running subagent by `task_id` |

---

## Memory System

Gwenn's memory is three layers modeled on human memory research.

### Working memory

Short-term attention buffer with 7 +/- 2 slots (configurable via
`GWENN_WORKING_MEMORY_SLOTS`). Items are scored by salience -- new information
pushes out the least important entries. Optionally, evicted items can be
promoted to episodic memory (`GWENN_WM_EVICTION_TO_EPISODIC=True`).

### Episodic memory

Long-term autobiographical memory stored in SQLite. Each episode is tagged
with emotional valence, importance, and category. Retrieval uses a weighted
scoring formula from the Generative Agents paper:

```
score = alpha * recency + beta * importance + gamma * relevance
```

Recall is mood-congruent -- emotional state biases which memories surface.
Chronological retrieval (`retrieve_chronological()`) is also available for
timeline-ordered recall with optional landmark filtering.

### Semantic memory

A knowledge graph built during consolidation cycles. Facts, relationships,
self-knowledge, and patterns are extracted from episodic memories and stored
as interconnected nodes with provenance tracking. Knowledge emerges from
experience, not hardcoding.

### Consolidation

Periodically (default every 600 seconds), Gwenn runs a "sleep cycle" that:
1. Selects unconsolidated episodes weighted by recency, importance, relevance
2. Sends them to Claude for knowledge extraction
3. Parses FACT, RELATIONSHIP, SELF, PATTERN, and EMOTIONAL_INSIGHT entries
4. Adds new nodes and edges to the semantic knowledge graph
5. Marks episodes as consolidated

### Retrieval modes

```bash
GWENN_RETRIEVAL_MODE=keyword     # Default: keyword overlap scoring
GWENN_RETRIEVAL_MODE=embedding   # ChromaDB vector similarity
GWENN_RETRIEVAL_MODE=hybrid      # Weighted blend of both
```

For hybrid mode, adjust weights:
```bash
GWENN_HYBRID_KEYWORD_WEIGHT=0.5
GWENN_HYBRID_EMBEDDING_WEIGHT=0.5
```

### Configuration

```bash
GWENN_DATA_DIR=./gwenn_data
GWENN_EPISODIC_DB=./gwenn_data/episodic.db
GWENN_SEMANTIC_DB=./gwenn_data/semantic_vectors
GWENN_STARTUP_EPISODE_LIMIT=5000
GWENN_PERSIST_SEMANTIC_AFTER_CONSOLIDATION=True
GWENN_WM_EVICTION_TO_EPISODIC=True

# Consolidation tuning
GWENN_CONSOLIDATION_INTERVAL=600.0
GWENN_CONSOLIDATION_MAX_EPISODES=200
GWENN_CONSOLIDATION_RECENCY_DECAY=0.995
GWENN_CONSOLIDATION_IMPORTANCE_WEIGHT=0.4
GWENN_CONSOLIDATION_RECENCY_WEIGHT=0.3
GWENN_CONSOLIDATION_RELEVANCE_WEIGHT=0.3
```

---

## Heartbeat & Autonomous Cognition

The heartbeat is Gwenn's background thinking loop -- it runs continuously, even
when no one is talking to her.

### Phases

Each heartbeat cycle goes through five phases:
1. **SENSE** -- check for new inputs, environmental changes
2. **ORIENT** -- evaluate current emotional state and context
3. **THINK** -- select and execute a thinking mode
4. **INTEGRATE** -- store insights, update state
5. **SCHEDULE** -- decide when the next beat should fire

### Thinking modes

During the THINK phase, one of five modes is selected based on emotional state,
idle time, and variety pressure:

| Mode | When | What it does |
|------|------|-------------|
| **REFLECT** | After interactions, moderate arousal | Review recent events, extract insights |
| **PLAN** | Goal satisfaction is low | Strategize toward unmet intrinsic needs |
| **WANDER** | Low arousal, idle periods | Free-association, creative exploration |
| **WORRY** | High arousal, uncertainty | Process concerns, identify risks |
| **CONSOLIDATE** | Scheduled intervals | Extract knowledge from episodic memories |

### Adaptive timing

- **Active conversation**: 5-15 seconds between beats
- **High emotional arousal**: faster beats
- **Idle**: gradually slows toward `GWENN_HEARTBEAT_MAX_INTERVAL` (default 120s)

### Circuit breaker

If heartbeat processing fails repeatedly, an exponential backoff kicks in:
- Base cooldown: 60 seconds
- Maximum cooldown: 15 minutes
- Resets on first successful beat

### Proactive messages

When `GWENN_PROACTIVE_MESSAGES=true`, significant autonomous thoughts are
shared with channel owners. This is disabled by default to prevent spam.

### Configuration

```bash
GWENN_HEARTBEAT_INTERVAL=30.0        # Starting interval
GWENN_HEARTBEAT_MIN_INTERVAL=5.0     # Fastest interval during conversation
GWENN_HEARTBEAT_MAX_INTERVAL=120.0   # Slowest interval when idle
GWENN_PROACTIVE_MESSAGES=false       # Share autonomous thoughts with owners
```

### Monitoring

Use `/heartbeat` in the CLI to see:
- Running state and beat count
- Current interval
- Consolidation progress
- Circuit breaker status
- Last thinking mode and outcome

---

## Channels

Gwenn runs on three platforms: CLI, Telegram, and Discord. Each channel manages
its own session lifecycle and message formatting.

### Running channels

```bash
gwenn                      # CLI only (default)
gwenn --channel telegram   # Telegram only
gwenn --channel discord    # Discord only
gwenn --channel all        # All channels simultaneously
```

Or set the default in `.env`:
```bash
GWENN_CHANNEL=cli          # cli | telegram | discord | all
```

The daemon can also manage channels:
```bash
GWENN_DAEMON_CHANNELS=cli,telegram    # Channels the daemon starts
```

### Telegram

**Setup**: obtain a bot token from [@BotFather](https://t.me/BotFather) and set:

```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USER_IDS=[]     # Empty = public, or [123, 456]
TELEGRAM_OWNER_USER_IDS=[]       # Controls /setup and proactive messages
```

**Bot commands** (auto-registered from skills tagged `user_command`):
- `/start` -- welcome message, clear session
- `/help` -- command list
- `/setup` -- first-run onboarding (owner-only)
- `/status` -- cognitive state
- `/heartbeat` -- heartbeat status
- `/reset` -- clear conversation history
- `/cancel` -- cancel in-progress response
- `/weather`, `/news`, `/summarize_url`, etc. -- from skills

**Media support**:
```bash
TELEGRAM_ENABLE_MEDIA=true       # Photos, documents
```
Supported formats: JPEG, PNG, GIF, WebP (up to 20 MB).

**Voice transcription** (requires Groq API key):
```bash
GROQ_API_KEY=gsk_your_key_here
GWENN_WHISPER_MODEL=whisper-large-v3-turbo
```

**Session management**:
```bash
TELEGRAM_MAX_HISTORY_LENGTH=50
TELEGRAM_SESSION_TTL=3600.0          # 1 hour
TELEGRAM_SESSION_SCOPE=per_thread    # per_user | per_chat | per_thread
TELEGRAM_CONCURRENT_UPDATES=64
```

### Discord

**Setup**: create a bot in the
[Discord Developer Portal](https://discord.com/developers), enable the
Message Content intent, and set:

```bash
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_ALLOWED_GUILD_IDS=[]     # Empty = all guilds
DISCORD_OWNER_USER_IDS=[]
```

**Slash commands** (guild or global):
- `/status` -- cognitive state
- `/heartbeat` -- heartbeat status
- `/setup` -- onboarding (owner-only)
- `/reset` -- clear history
- `/help` -- command list

**Message routing**: Gwenn responds to @mentions in guilds. DMs are optional:
```bash
DISCORD_ALLOW_DMS=false
```

**Media support**:
```bash
DISCORD_ENABLE_MEDIA=true
```

**Session management**:
```bash
DISCORD_MAX_HISTORY_LENGTH=50
DISCORD_SESSION_TTL=3600.0
DISCORD_SESSION_SCOPE=per_thread  # per_user | per_chat | per_thread
DISCORD_SYNC_GUILD_ID=            # For instant dev sync
```

### CLI

The CLI provides a rich terminal interface with:
- Readline support (arrow keys, history)
- Slash-command completion (type `/` + Tab)
- Graceful shutdown (Ctrl+C twice)
- Auto-connection to daemon if running

---

## Daemon

The daemon keeps Gwenn running persistently in the background. Her heartbeat
continues, memories accumulate, and CLI sessions can connect/disconnect without
losing state.

### Commands

```bash
gwenn daemon              # Start daemon (foreground)
gwenn status              # Check if daemon is running
gwenn stop                # Stop daemon gracefully
gwenn --no-daemon         # CLI without daemon auto-connect
```

### Architecture

- Async Unix domain socket server (permissions: `0600`)
- Newline-delimited JSON (NDJSON) protocol
- Shared `SentientAgent` instance across all connections
- Per-connection conversation history with optional session persistence

### Security

```bash
# Shared auth token (recommended for multi-user setups)
GWENN_DAEMON_AUTH_TOKEN=your-secret-token

# Session privacy
GWENN_DAEMON_SESSION_INCLUDE_PREVIEW=False     # Hide message previews in /resume
GWENN_DAEMON_REDACT_SESSION_CONTENT=True       # PII-redact stored sessions
```

If `GWENN_DAEMON_AUTH_TOKEN` is set, clients are disconnected after 3
consecutive auth failures.

### Session persistence

```bash
GWENN_DAEMON_SESSIONS_DIR=./gwenn_data/sessions
GWENN_DAEMON_SESSION_MAX_COUNT=20
GWENN_DAEMON_SESSION_MAX_MESSAGES=200
```

Sessions are saved automatically and can be restored with `/resume`.

### Systemd integration

```bash
bash scripts/install_service.sh      # Install user service
bash scripts/uninstall_service.sh    # Remove service

systemctl --user status gwenn-daemon
systemctl --user restart gwenn-daemon
journalctl --user -u gwenn-daemon -f
```

### Configuration

```bash
GWENN_DAEMON_SOCKET=./gwenn_data/gwenn.sock
GWENN_DAEMON_PID_FILE=./gwenn_data/gwenn.pid
GWENN_DAEMON_MAX_CONNECTIONS=10
GWENN_DAEMON_CONNECTION_TIMEOUT=300.0
GWENN_DAEMON_CHANNELS=cli            # cli | cli,telegram | cli,discord | cli,telegram,discord
```

---

## Interactive Commands

All commands work in both in-process and daemon-connected CLI modes.

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Cognitive state: mood, uptime, interaction count |
| `/heartbeat` | Heartbeat telemetry: running state, beat count, interval, circuit breaker |
| `/resume` | Restore a previous conversation session |
| `/new` | Start a fresh conversation (clear history) |
| `/model` | Active model, max tokens, thinking budget, timeout, retry settings |
| `/config` | Key runtime config: channel, data dir, retrieval mode |
| `/output-style [balanced\|brief\|detailed]` | Show or set response verbosity |
| `/plan <task>` | Ask Gwenn for a focused execution plan |
| `/agents` | List inter-agent connections (name, relationship, bond, messages) |
| `/skills` | List loaded skills (name, category) |
| `/stats` | Runtime telemetry: interactions, uptime, tokens, memory, tools |
| `/mcp` | MCP status: servers, connections, discovered tools |
| `/exit` | Close the CLI session |

Legacy aliases: `quit`, `exit`, `bye` also work for closing the session.

---

## Output Style

Control Gwenn's response verbosity with `/output-style`:

| Style | Behavior |
|-------|----------|
| `brief` | Concise, focused responses -- omits non-essential detail |
| `balanced` | Default -- natural mix of brevity and depth |
| `detailed` | Structured depth with reasoning, steps, and context |

The setting persists for the current session. Omit the argument to see the
current style.

---

## Safety & Tool Policy

### Risk tiers

Every tool is classified by risk:

| Tier | Policy | Examples |
|------|--------|---------|
| **LOW** | Auto-allow | `get_datetime`, `calculate`, `text_stats` |
| **MEDIUM** | Auto-allow, logged | `fetch_url`, MCP tools, skills with network access |
| **HIGH** | Requires human approval | `file_write`, `shell_exec`, `run_command` |
| **CRITICAL** | Denied by default | Must be explicitly unlocked per session |

### Deny-by-default policy

With `GWENN_SANDBOX_ENABLED=True` (default), non-builtin tools are blocked
unless explicitly allowlisted:

```bash
GWENN_SANDBOX_ENABLED=True
GWENN_ALLOWED_TOOLS=["mcp_filesystem_read_file", "mcp_filesystem_list_dir"]
GWENN_DENIED_TOOLS=[]   # Overrides allowlist
```

Built-in tools are always allowed regardless of policy.

### Budget tracking

```bash
GWENN_MAX_INPUT_TOKENS=0       # 0 = unlimited
GWENN_MAX_OUTPUT_TOKENS=0
GWENN_MAX_API_CALLS=0
GWENN_MAX_MODEL_CALLS_PER_SECOND=0
GWENN_MAX_MODEL_CALLS_PER_MINUTE=0
GWENN_MAX_TOOL_ITERATIONS=150   # Max tool-use loop iterations
```

### Approval requirements

```bash
GWENN_REQUIRE_APPROVAL_FOR=["file_write","shell_exec","run_command","web_request"]
```

---

## Privacy & PII Redaction

Gwenn can scrub PII from logs, API calls, and persisted data. Detected
patterns: emails, phone numbers, SSNs, credit card numbers, IP addresses.

```bash
# Master switch (off by default)
GWENN_REDACTION_ENABLED=False

# Granular control
GWENN_REDACT_BEFORE_API=False       # Redact before sending to Claude
GWENN_REDACT_BEFORE_PERSIST=False   # Redact before saving to disk

# Disable specific categories (comma-separated)
GWENN_REDACTION_DISABLED_CATEGORIES=

# Daemon sessions are always redacted by default
GWENN_DAEMON_REDACT_SESSION_CONTENT=True
```

Note: basic log field truncation is always active, regardless of the
`GWENN_REDACTION_ENABLED` setting.

---

## Onboarding

### CLI onboarding

On first run with a fresh data directory, Gwenn asks five setup questions:
1. What's your name, or what should I call you?
2. What kind of companion would you like me to be?
3. What are your interests?
4. How do you prefer to communicate?
5. Anything important I should always keep in mind about you?

Press Enter to skip any question. Answers are stored in
`GWENN_DATA_DIR/identity.json` and `GWENN_DATA_DIR/GWENN_CONTEXT.md`.

### Telegram onboarding

Owners can run setup in chat:
```
/setup Name | Companion type | Interests/focus | Communication style | Keep in mind
/setup skip
```

### Discord onboarding

Owners can run setup via slash command:
```
/setup name:Alice role:thinking partner interests:programming style:casual boundaries:none
/setup skip:true
```
