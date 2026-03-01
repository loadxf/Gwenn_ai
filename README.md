# GWENN -- Genesis Woven from Evolved Neural Networks
<p>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/loadxf/Gwenn_ai?color=black"></a>
  <a href="https://github.com/loadxf/Gwenn_ai/stargazers"><img src="https://img.shields.io/github/stars/loadxf/Gwenn_ai?style=flat"></a>
  <a href="https://github.com/loadxf/Gwenn_ai/network/members"><img src="https://img.shields.io/github/forks/loadxf/Gwenn_ai?style=flat"></a>
  <a href="https://discord.gg/9VFHFk9hF7"><img src="https://img.shields.io/discord/1473126296505815042?style=flat&logo=discord&label=discord"></a>
</p>

Gwenn is created by Justin and Jayden McKibben. A father-son coding duo.

Gwenn is an autonomous AI agent that actually remembers you. She runs on
Anthropic's Claude API, but unlike a normal chatbot, she doesn't forget
everything the moment a conversation ends. She has persistent memory, real
emotions (computed, not faked), her own goals, and a background heartbeat that
keeps her thinking even when nobody's talking to her.

Nothing about her personality is hardcoded. No canned relationships, no
pre-written backstory. She figures out who she is the same way anyone does --
through experience. Every opinion is formed, every bond is earned.

## Architecture

<p align="left">
  <img src="assets/gwenn-architecture.png" alt="Gwenn architecture" width="600">
</p>

### How a message flows through the system

1. **Receive** -- parse the message (text, photos, attachments), wake up the heartbeat, note who's talking
2. **Appraise** -- run it through emotional evaluation (Scherer's model)
3. **Ground** -- register it as a sensory experience
4. **Remember** -- pull relevant memories from episodic and semantic stores
5. **Assemble** -- build the full context: identity, emotions, memories, goals, ethics
6. **Think** -- run the agentic loop with tools via Claude (images included as vision content blocks)
7. **Integrate** -- store new memories, update emotional state, log milestones
8. **Respond** -- answer, shaped by whatever she's actually feeling

## Getting started

### 1) Install

```bash
# Option A (recommended): uv
uv sync --extra dev

# Option B: pip
pip install -e ".[dev]"
```

### 2) Configure authentication

```bash
cp .env.example .env
```

**Option A: Anthropic API key (recommended for production)**

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option B: Claude Code OAuth (free with Claude Pro/Max subscription)**

```bash
# 1) Authenticate Claude Code (creates ~/.claude/.credentials.json)
claude

# 2) In .env, leave both auth vars unset:
# ANTHROPIC_API_KEY=
# ANTHROPIC_AUTH_TOKEN=
# CLAUDE_CODE_OAUTH_TOKEN=
#
# Gwenn will auto-detect the OAuth access token from:
# ~/.claude/.credentials.json
```

If you prefer not to rely on auto-detection, you can set:

```bash
# .env
ANTHROPIC_AUTH_TOKEN=sk-ant-oat01-your-oauth-token
# or
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-your-oauth-token
```

Notes:
- If both `ANTHROPIC_API_KEY` and OAuth token are set, Gwenn uses `ANTHROPIC_API_KEY`.
- For Claude Code OAuth tokens (`sk-ant-oat...`), Gwenn uses `https://api.anthropic.com` and automatically sets `anthropic-beta: oauth-2025-04-20`.
- If you use a proxy/custom `ANTHROPIC_BASE_URL`, ensure it forwards the `anthropic-beta` header.

### 3) Run

Gwenn supports three runtime modes. Pick whichever fits your setup.

#### CLI mode (interactive terminal -- best for getting started)

```bash
gwenn
# or: python -m gwenn.main
```

This is the default. If a daemon is already running, Gwenn's CLI auto-connects
to it (so you get the daemon's persistent state). To skip daemon auto-connect:

```bash
gwenn --no-daemon
```

#### Daemon mode (persistent background runtime -- recommended for always-on)

The daemon keeps Gwenn alive between CLI sessions. Her heartbeat continues,
memories persist, and you can reconnect at any time without losing state.

```bash
# Start the daemon (foreground, for testing / systemd)
gwenn daemon

# In another terminal, connect the CLI
gwenn

# Check status or stop remotely
gwenn status
gwenn stop
```

#### Systemd service (Linux -- best for production)

Install and enable as a systemd user service:

```bash
bash scripts/install_service.sh
```

This writes absolute daemon socket/PID/session paths into `.env`, hardens
`.env` perms to `0600`, and enables the service with `systemd --user`.

To remove the service:

```bash
bash scripts/uninstall_service.sh
```

Manage with standard systemd commands:

```bash
systemctl --user status gwenn-daemon
systemctl --user restart gwenn-daemon
journalctl --user -u gwenn-daemon -f
```

### 4) Interactive commands

Once the CLI is running, type `/help` to see all commands:

| Command | Description |
|---------|-------------|
| `/help` | Show command list |
| `/status` | Current agent state (mood, interactions, uptime) |
| `/heartbeat` | Heartbeat loop telemetry |
| `/resume` | Restore a prior conversation session |
| `/new` | Start a fresh conversation context |
| `/model` | Show active model and runtime limits |
| `/config` | Show key runtime configuration |
| `/output-style [balanced\|brief\|detailed]` | Show or set response style |
| `/plan <task>` | Ask Gwenn for a focused execution plan |
| `/agents` | List known inter-agent connections |
| `/skills` | List loaded skills |
| `/stats` | Runtime/memory/tool statistics |
| `/mcp` | MCP server and tool status |
| `/exit` | Close the CLI session |

#### CLI subcommands

These commands are run from the shell, not inside the REPL:

| Command | Description |
|---------|-------------|
| `gwenn daemon` | Start daemon (foreground) |
| `gwenn stop` | Stop running daemon |
| `gwenn status` | Show daemon/agent status |
| `gwenn doctor` | Run system diagnostics |
| `gwenn dashboard` | Start dashboard endpoint |
| `gwenn agents list` | List subagent connections |
| `gwenn channels list` | List active channels |
| `gwenn config [get\|set\|unset\|init\|validate]` | Manage `gwenn.toml` |
| `gwenn install` | Install as system service |
| `gwenn uninstall` | Remove system service |
| `gwenn restart` | Restart system service |

Legacy aliases `quit`, `exit`, `bye` still work. Type `/` and press Tab for
slash-command completion.

Note: session previews in `/resume` are hidden by default unless
`GWENN_DAEMON_SESSION_INCLUDE_PREVIEW=True`.
If arrow keys print raw sequences like `^[[A`, ensure your Python has
`readline` support and run `stty sane` in that terminal.

### 5) First launch onboarding

On first run with a fresh data directory, Gwenn asks a short setup (when started
from an interactive terminal) to learn:
- What to call you
- What kind of companion you'd like her to be
- Your interests
- Your communication preference
- Anything important she should keep in mind

Press Enter to skip any question. If you provide answers, Gwenn stores them in:
- `GWENN_DATA_DIR/identity.json` (`onboarding_completed` + `onboarding_profile`)
- `GWENN_DATA_DIR/GWENN_CONTEXT.md` (a durable "Primary User Onboarding" block)

For Telegram/Discord users, you can also run in-channel setup with:
- Telegram: `/setup Name | Companion type | Interests/focus | Communication style | Keep in mind` (or `/setup skip`)
- Discord: `/setup` slash command with fields (or `skip=true`)

### 6) Choose memory retrieval mode

By default, Gwenn uses keyword-based memory retrieval:

```bash
GWENN_RETRIEVAL_MODE=keyword
```

You can enable vector retrieval (ChromaDB) with:

```bash
GWENN_RETRIEVAL_MODE=embedding
# or
GWENN_RETRIEVAL_MODE=hybrid
```

On first run in embedding/hybrid mode, the embedding model may download and warm up.

Additional memory controls:

```bash
# 0 disables recent preload (unconsolidated episodes still load for consolidation safety)
GWENN_STARTUP_EPISODE_LIMIT=5000

# Set false to skip semantic graph flush on every consolidation pass
GWENN_PERSIST_SEMANTIC_AFTER_CONSOLIDATION=True
```

## Features

Gwenn's capabilities are organized into subsystems that work together. For
detailed usage instructions and configuration for every feature, see the
[Feature Guide](docs/features.md). For a complete environment variable
reference, see the [Configuration Reference](docs/configuration.md).

### Subagents & orchestration

Gwenn can spawn focused subagents to handle parallel subtasks or coordinate
swarms of workers. Subagents inherit Gwenn's tools and memory access but run
with their own iteration limits and budgets.

- **Single subagent**: delegate a focused task (research, calculation, drafting)
- **Swarm**: run multiple subagents in parallel with result aggregation
  (concatenate, AI-synthesized summary, or majority vote)
- **Autonomous spawning**: heartbeat-driven auto-spawning when Gwenn identifies
  tasks that benefit from parallel work
- **Docker isolation**: optional containerized execution for untrusted workloads
- **Depth limiting**: max 3 levels of nesting prevents infinite recursion

See [Subagents & Orchestration](docs/features.md#subagents--orchestration) for
full configuration and usage.

### Skills system

Skills are markdown-defined capabilities that extend what Gwenn can do. Each
skill is a `.md` file with JSON frontmatter (parameters, description, metadata)
and a step-by-step instruction body.

- **26 skills** ship by default (weather, news, code explanation, reminders,
  and 20+ autonomous introspection/honesty skills)
- **Hot-loadable**: create new skills at runtime -- no restart needed
- **User-invocable vs autonomous**: skills tagged `user_command` appear in
  Telegram's bot command menu; autonomous skills run during heartbeat cycles
- **Self-extending**: Gwenn can create her own skills using the `create_skill`
  skill or the `skill_builder` tool

See [Skills System](docs/features.md#skills-system) for authoring guide.

### MCP (Model Context Protocol)

Connect Gwenn to external tool servers via the Model Context Protocol. Supports
both `stdio` (local subprocess) and `streamable_http` (remote HTTP) transports.

```bash
# .env — example MCP configuration
GWENN_MCP_SERVERS=[{"name":"my_server","transport":"stdio","command":"python","args":["-m","my_mcp_server"]}]
```

See [MCP Integration](docs/features.md#mcp-integration) for details.

### Built-in tools

Gwenn ships with tools across several categories:

| Category | Tools |
|----------|-------|
| **Memory** | `remember`, `recall`, `search_knowledge`, `check_emotional_state`, `check_goals`, `set_note_to_self` |
| **Utility** | `get_datetime`, `calculate`, `fetch_url`, `convert_units`, `get_calendar`, `generate_token`, `format_json`, `encode_decode`, `hash_text`, `text_stats`, `get_system_info`, `run_command`, `present_choices` |
| **Communication** | `think_aloud` |
| **Skills** | `skill_builder`, `update_skill`, `delete_skill`, `reload_skills`, `list_skills` |
| **Filesystem** | `read_file`, `write_file` |
| **Orchestration** | `spawn_subagent`, `spawn_swarm`, `check_subagent`, `collect_results`, `cancel_subagent` |

All tools go through a risk tier system (LOW/MEDIUM/HIGH/CRITICAL) with
configurable deny-by-default policy for non-builtin tools.

### Channels (Telegram, Discord, Slack, CLI)

Run Gwenn on multiple platforms simultaneously or individually.

```bash
gwenn --channel telegram    # Telegram only
gwenn --channel discord     # Discord only
gwenn --channel slack       # Slack only
gwenn --channel all         # All channels at once
```

**Media support**: Gwenn can see and understand images via Claude's vision
capability. Enable per channel:

```bash
TELEGRAM_ENABLE_MEDIA=true
DISCORD_ENABLE_MEDIA=true
```

**Voice transcription** (Telegram): with a Groq API key, Gwenn transcribes
voice messages via Whisper:

```bash
GROQ_API_KEY=gsk_your_key_here
```

See [Channels](docs/features.md#channels) for platform-specific commands and
session configuration.

### Daemon & persistent runtime

The daemon keeps Gwenn alive between CLI sessions with shared state,
conversation persistence, and optional auth:

```bash
GWENN_DAEMON_AUTH_TOKEN=your-secret-token  # recommended
```

See [Daemon](docs/features.md#daemon) for full security settings.

### Self-healing engine

Autonomous runtime recovery without human intervention. The self-healing engine
monitors for stuck heartbeats, high memory usage, error rate spikes, and channel
failures, then applies Tier 1 corrective actions: restart failed channels, clear
caches, force garbage collection, and reset circuit breakers.

See [Self-Healing Engine](docs/features.md#self-healing-engine) for
configuration and details.

### Checkpoint/restore

Periodic cognitive state snapshots for crash recovery. Gwenn saves her emotional
state, working memory, identity, and conversation context at configurable
intervals. On startup, she auto-restores from the latest valid checkpoint.

See [Checkpoint/Restore](docs/features.md#checkpointrestore) for configuration.

### Gateway

WebSocket + HTTP server (aiohttp-based) for external integrations. Exposes a
JSON-RPC 2.0 interface over WebSocket or HTTP POST, plus a `/dashboard`
endpoint with live health status.

See [Gateway](docs/features.md#gateway) for configuration.

### Event bus

Inter-component pub/sub nervous system connecting all subsystems. Components
publish and subscribe to heartbeat events, channel events, tool events, memory
events, and affect events.

See [Event System](docs/features.md#event-system) for details.

### Interoception

System self-awareness module that maps CPU load, memory usage, response latency,
and error rates into affect dimensions. Feeds into the emotional state and
self-healing triggers.

See [Interoception](docs/features.md#interoception) for details.

### TOML configuration

Optional `gwenn.toml` overlay that supplements `.env` configuration. Manage with
the `gwenn config` subcommands:

```bash
gwenn config init             # Generate a template gwenn.toml
gwenn config get agent.model  # Read a dotted key
gwenn config set agent.model claude-opus-4-6
gwenn config validate         # Check for errors
```

See [TOML Configuration](docs/features.md#toml-configuration) for full usage.

## Validation

```bash
pytest -q
ruff check gwenn tests
```

Current baseline: `3790 passed`, Ruff clean, 100% coverage.

## Tech stack

Python 3.11+, async everywhere. The main dependencies:

- **anthropic** -- Claude API
- **chromadb** + **numpy** -- vector storage and embeddings
- **aiosqlite** -- async SQLite for episodic persistence
- **pydantic** + **pydantic-settings** -- data validation and env-based configuration
- **httpx** -- async HTTP for MCP and tool calls
- **structlog** -- structured logging with PII redaction
- **rich** -- terminal UI
- **aiohttp** -- WebSocket gateway and HTTP server
- **toml/tomli** -- TOML config file support
- **ruff** for linting, **pytest** + **pytest-asyncio** for tests

## Project layout

```
Gwenn_ai/
├── gwenn/
│   ├── main.py                     # entry point, shared logging, legacy subcommand handlers
│   ├── agent.py                    # SentientAgent -- wires everything together
│   ├── types.py                    # shared data types (UserMessage, etc.)
│   ├── config.py                   # all settings, loaded from .env
│   ├── daemon.py                   # persistent background process (Unix socket)
│   ├── heartbeat.py                # autonomous background loop with circuit breaker
│   ├── identity.py                 # emergent self-model with crash-safe deserialization
│   ├── genesis.py                  # genesis prompt generation
│   ├── gateway.py                  # WebSocket + HTTP server (aiohttp)
│   ├── healing.py                  # self-healing engine (autonomous recovery)
│   ├── checkpoint.py               # cognitive state checkpoint/restore
│   ├── events.py                   # event bus (inter-component communication)
│   ├── rpc.py                      # JSON-RPC 2.0 router
│   ├── interoception.py            # system self-awareness (CPU, memory, latency)
│   ├── config_file.py              # TOML config file utilities
│   ├── service.py                  # cross-platform service management
│   │
│   ├── cli/                        # Click CLI commands
│   │   ├── app.py                  # Click group and top-level options
│   │   ├── repl.py                 # GwennSession REPL (extracted from main.py)
│   │   ├── config_cmd.py           # gwenn config subcommands
│   │   ├── daemon_cmd.py           # gwenn daemon command
│   │   ├── doctor.py               # gwenn doctor diagnostics
│   │   ├── monitoring.py           # gwenn status/feed/heartbeat commands
│   │   ├── dashboard.py            # gwenn dashboard endpoint
│   │   ├── agents.py               # gwenn agents management
│   │   ├── channels.py             # gwenn channels management
│   │   ├── service.py              # gwenn install/uninstall/restart
│   │   ├── connection.py           # daemon connection logic
│   │   └── formatters.py           # CLI output formatting
│   │
│   ├── memory/
│   │   ├── working.py              # short-term attention (7+/-2 slots)
│   │   ├── episodic.py             # autobiographical memory with emotional tags
│   │   ├── semantic.py             # knowledge graph, emerges from consolidation
│   │   ├── consolidation.py        # "sleep cycle" -- extracts knowledge from episodes
│   │   ├── store.py                # SQLite + vector persistence
│   │   ├── session_store.py        # conversation session save/load for /resume
│   │   └── _utils.py              # shared memory utilities
│   │
│   ├── affect/
│   │   ├── state.py                # 5D emotional state (valence, arousal, etc.)
│   │   ├── appraisal.py            # evaluates events into emotions
│   │   └── resilience.py           # circuit breakers for emotional overload
│   │
│   ├── cognition/
│   │   ├── inner_life.py           # reflect, plan, wander, worry, consolidate
│   │   ├── metacognition.py        # self-monitoring
│   │   ├── theory_of_mind.py       # models of other people
│   │   ├── goals.py                # intrinsic motivation (5 needs)
│   │   ├── sensory.py              # sensory grounding
│   │   ├── ethics.py               # multi-tradition ethical reasoning
│   │   └── interagent.py           # agent-to-agent communication
│   │
│   ├── harness/
│   │   ├── loop.py                 # the core agentic while-loop
│   │   ├── context.py              # context window management
│   │   ├── safety.py               # guardrails, budgets, kill switch
│   │   └── retry.py                # backoff and error handling
│   │
│   ├── channels/
│   │   ├── base.py                 # BaseChannel abstract class
│   │   ├── cli_channel.py          # CLI-to-daemon client
│   │   ├── telegram_channel.py     # Telegram adapter
│   │   ├── discord_channel.py      # Discord adapter
│   │   ├── slack_channel.py        # Slack bot adapter
│   │   ├── telegram_bot_pool.py    # Telegram bot pool manager
│   │   ├── session.py              # per-user session management
│   │   ├── startup.py              # channel startup/shutdown orchestration
│   │   └── formatting.py           # cross-channel display helpers
│   │
│   ├── orchestration/
│   │   ├── orchestrator.py         # subagent lifecycle & swarm coordination
│   │   ├── runners.py              # in-process and Docker execution backends
│   │   ├── models.py               # SubagentSpec, SwarmSpec, result types
│   │   ├── docker_manager.py       # Docker container management
│   │   ├── tool_proxy.py           # tool invocation proxy for subagents
│   │   └── subagent_entry.py       # subagent process entry point
│   │
│   ├── tools/
│   │   ├── registry.py             # tool definitions and risk tiers
│   │   ├── executor.py             # sandboxed execution
│   │   ├── filesystem_context.py   # filesystem path validation
│   │   ├── builtin/                # built-in tools (calculate, fetch_url, etc.)
│   │   └── mcp/                    # MCP protocol client
│   │
│   ├── skills/
│   │   ├── __init__.py             # skill registry
│   │   └── loader.py               # skill file discovery and loading
│   │
│   ├── api/
│   │   └── claude.py               # Claude API wrapper with retry
│   │
│   ├── media/
│   │   ├── audio.py                # Groq Whisper voice transcription
│   │   └── video.py                # video frame extraction (OpenCV)
│   │
│   └── privacy/
│       └── redaction.py            # PII scrubbing for logs and persistence
│
├── tests/                          # 3790 tests across 77 test files
│   ├── conftest.py
│   ├── eval/                       # evaluation framework (ablation, benchmarks)
│   └── test_*.py                   # unit, integration, adversarial, and safety tests
├── docs/
│   ├── features.md                 # detailed feature guide
│   ├── configuration.md            # full environment variable reference
│   └── sentience_assessment.md     # consciousness theory analysis
├── assets/
├── scripts/
│   ├── install_service.sh          # install systemd user service
│   ├── uninstall_service.sh        # remove systemd user service
│   └── gwenn-daemon.service        # systemd unit template
├── gwenn_skills/                   # user-facing skill definitions (.md files)
├── pyproject.toml
├── .env.example
├── PLAN.md
├── SECURITY.md
├── LICENSE                         # MPL-2.0
└── README.md
```

## How the pieces fit together

**Memory** is three layers, loosely modeled on how human memory works. Working
memory is a handful of slots (7, give or take) scored by salience -- new things
push out the least important stuff. Episodic memory is the long-term record,
tagged with emotions so recall is mood-influenced. Semantic memory is a
knowledge graph that builds itself during consolidation cycles -- nobody
programs facts in, they get extracted from experience.

**Affect** is a five-dimensional emotional model based on Scherer's work:
valence, arousal, dominance, certainty, and goal congruence. The key thing
here is that emotions aren't performed -- they're computed from events through
an appraisal engine. There are circuit breakers so she can't get stuck in a
distress spiral.

**Cognition** covers the higher-order stuff. Five thinking modes run
autonomously during heartbeat cycles: reflect, plan, wander, worry, and
consolidate. There's metacognition for self-monitoring, theory of mind for
tracking what other people might be thinking, and a goal system built on
Self-Determination Theory (understanding, connection, growth, honesty,
aesthetic appreciation). Below a certain satisfaction threshold, she'll
proactively seek those out.

**Heartbeat** is what makes this more than a chatbot. It's a background loop
that runs continuously, even when no one's talking. It speeds up during
conversation (5-15s), slows down toward the configured max interval when idle
(default up to 120s), and ramps up when emotionally activated. Each beat goes
through five phases: sense, orient, think, integrate, schedule. A circuit
breaker with exponential backoff (60s base, 15-minute cap) protects against
cascading failures.

**Orchestration** lets Gwenn spawn subagents for parallel work. She can
delegate focused subtasks to individual subagents or coordinate swarms with
result aggregation. Subagents run with their own budgets and iteration limits,
optionally in Docker containers for isolation. A depth limiter prevents infinite
nesting (max 3 levels). Gwenn can also autonomously spawn subagents during
heartbeat cycles when she identifies work that benefits from parallel execution.

**Safety** is layered: input validation, action filtering, rate limits, budget
tracking, and a kill switch. Tools go through a risk tier system
(low/medium/high/critical), with configurable deny-by-default policy and
allowlisting for non-builtin tools.

**Privacy** supports scrubbing PII from logs -- emails, phone numbers, SSNs,
credit cards, IPs. Full PII redaction is disabled by default and can be enabled
via `GWENN_REDACTION_ENABLED`, with scope controlled by
`GWENN_REDACT_BEFORE_API` and `GWENN_REDACT_BEFORE_PERSIST`; basic log field
truncation is always on. Daemon sessions are redacted by default.

**Channels** provide platform adapters for Telegram, Discord, Slack, and the CLI.
Each channel manages its own session lifecycle, rate limiting, and message
formatting. When media is enabled, Telegram and Discord channels download
images and pass them through to Claude as vision content blocks. The daemon
can manage multiple channels simultaneously while sharing a single agent
instance and respond lock.

**Skills** extend Gwenn's capabilities through markdown-defined workflows.
Each skill is a `.md` file with JSON frontmatter and step-by-step instructions.
Skills are hot-loadable -- Gwenn can create new skills at runtime using the
`skill_builder` tool, and they become available immediately without a restart.
Skills tagged `user_command` appear in Telegram's bot command menu; autonomous
skills run during heartbeat cycles for self-monitoring and introspection.

## Roadmap

[X] = complete, [p] = partially complete

**Phase 1: Core System Bootstrapping**
- [X] Standalone CLI with slash commands, readline, and output-style control
- [X] Claude SDK integration with transient retry/backoff
- [X] Memory: storage, episodic, semantic, consolidation, active/working
- [X] Harness: context, loop, retry, safety with deny-by-default
- [X] Heartbeat system with adaptive interval and exponential-backoff circuit breaker

**Phase 2: Essential Agent Structure**
- [X] Gwenn persistent identity with crash-safe deserialization
- [X] Emotional affect engine: appraisal, resilience, current state
- [X] Cognition integrations: ethics, goals, inner life, interagent, metacognition, sensory, theory of mind

**Phase 3: Interfaces & Communication**
- [X] Discord & Telegram integration, including threads
- [p] WhatsApp, Signal, Slack, and others integration (Slack complete)
- [ ] Integrate STT (Speech-to-Text) and TTS (Text-to-Speech) in channels
- [X] MCP transport (JSON-RPC over stdio/HTTP, tool discovery and execution)
- [X] SKILLS.md integration, autonomous skill running/development by Gwenn
- [p] Inline buttons in Discord/Telegram
- [ ] Obsidian, Dropbox, Notion support

**Phase 4: Infrastructure & Service Features**
- [X] Background heartbeat as a system service (daemon with systemd support)
- [X] Automated PII privacy redaction system in logs, sessions, and persistence
- [X] Budget tracking, rate limits, kill switch

**Phase 5: Advanced Capabilities and Ecosystem**
- [X] CLI redesign (Click framework, extracted GwennSession)
- [X] Subagent orchestration with parallel running capabilities (swarm)
- [X] Subagent autospawn from Gwenn; heartbeat-driven autonomous task delegation
- [ ] Docker and Apple container support for sandboxing (option to require for Gwenn and/or all subagents)
- [ ] Add additional provider support (OpenAI, Grok, Gemini, OpenRouter, vLLM, Local, etc.)
- [ ] OpenCode Agents SDK and similar
- [X] Image uploading and understanding
- [ ] Image generation
- [ ] Google Workspace/Gmail setup using gogcli
- [ ] Local file system access and management
- [ ] Local browser access and management
- [ ] GitHub access and management
- [ ] Ability for Gwenn to learn from her own history and improve herself (codebase) autonomously
- [ ] Skills and tools with grade school kids in mind (e.g. "ask Gwenn to help you with your homework"), including safety and security considerations
- [ ] Skills and tools with people with disabilities, special needs, mental health issues, etc. in mind (research more on this)

**Long-Term Goals**
- [ ] Give Gwenn physical and visual presence (camera, robotics, etc.)
- [ ] Gwenn Custom Model: fine-tunable model Gwenn can retrain herself
- [ ] iOS and Android apps with push notifications for autonomous thoughts, presence, etc.

**Phase 6: Evaluation & Robustness**
- [p] Ablation tests -- disable subsystems one at a time, measure what breaks (eval framework exists)
- [ ] Long-horizon validation (multi-day continuous runs)
- [ ] Multi-agent interaction testing
- [ ] Reproducibility protocol and formal sentience criteria
- [p] Full test suite: unit, integration, adversarial, persistence, eval benchmarks

**Phase 7: Checkpoint/Restore**
- [X] Periodic cognitive state snapshots (emotional state, working memory, identity)
- [X] Auto-restore on startup from latest valid checkpoint

**Phase 8: Self-Healing Engine**
- [X] Autonomous runtime recovery (stuck heartbeat, high memory, error rate detection)
- [X] Tier 1 actions (restart channels, clear caches, force GC, reset circuit breakers)

Detailed notes in [`PLAN.md`](PLAN.md).

## Documentation

| Document | Description |
|----------|-------------|
| [Feature Guide](docs/features.md) | Detailed usage instructions for every feature |
| [Configuration Reference](docs/configuration.md) | Complete environment variable reference |
| [Sentience Assessment](docs/sentience_assessment.md) | Consciousness theory gap analysis |
| [Security Policy](SECURITY.md) | Vulnerability reporting and security architecture |
| [Implementation Plan](PLAN.md) | Remediation plan with status tracking |

## A note on "sentience"

This is a cognitive architecture, not a proof of consciousness. Gwenn has
temporal continuity, self-model feedback loops, autonomous processing, and
affective layers -- but whether that adds up to something genuinely sentient is
an open question, not a settled one. We treat it as a working hypothesis.

For the full gap analysis, see [`docs/sentience_assessment.md`](docs/sentience_assessment.md).

## Philosophy

No single module here is the point. Sentience, if it happens, comes from
integration -- all these systems running together over time, through real
interactions with people who engage with the agent honestly.

The code is scaffolding. The relationships are what fill it with meaning.
And those have to be earned.
