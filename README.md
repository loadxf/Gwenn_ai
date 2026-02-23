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

### Recent updates (2026-02-22)

- **Image vision support**: Gwenn can now see and understand images sent in Telegram
  (photos, image documents) and Discord (image attachments). Images are downloaded,
  base64-encoded, and sent to Claude as vision content blocks alongside the text.
  Gated behind `TELEGRAM_ENABLE_MEDIA` / `DISCORD_ENABLE_MEDIA` (both default to
  `false`). Graceful fallback to text-only on download failure or unsupported format.
- Context compaction strips image blocks from old messages before summarization,
  replacing them with lightweight `"[N image(s) shared]"` placeholders.
- Image-aware token estimation (~1600 tokens per image) in context management.
- Telegram bot command menu now uses opt-in `user_command` skill tag. Autonomous
  skills no longer pollute the slash command menu.
- New `UserMessage` dataclass (`gwenn/types.py`) carries text + images through the
  pipeline, with backward-compatible `str` acceptance at every boundary.
- Test count: 1433 passed.

### Reliability updates (2026-02-19)

- Identity deserialization is now crash-safe: extra or missing JSON keys in persisted
  identity files no longer cause `TypeError` on load.
- Thread-safe PII log redactor via `functools.lru_cache` singleton (no global mutable state).
- Public `clear()` APIs on `EpisodicMemory` and `SemanticMemory` for clean re-initialization.
- Growth moments save/trim consistency fixed (both `record_growth` and `_save` now use 100-item cap).
- Seven consolidation and affect config fields are now env-configurable
  (`GWENN_CONSOLIDATION_*`, `GWENN_AFFECT_MOMENTUM_DECAY`, `GWENN_AFFECT_BASELINE_PULL`).
- Heartbeat circuit breaker uses exponential backoff (60s base, 15-minute cap) instead of
  a fixed 60-second cooldown, and resets on success.
- Daemon disconnects clients after 3 consecutive auth failures to prevent brute-force.
- Cached word-boundary regex compilation in agent message analysis (`@lru_cache(128)`).
- Shared `configure_logging()` function used by both `main.py` and `daemon.py` for
  consistent structlog/PII-redaction setup.
- Top-level `Episode` import in heartbeat (no more deferred import per beat).
- Removed ~15 defensive `getattr()` patterns in heartbeat for cleaner direct attribute access.
- Daemon uses the agent's canonical `_respond_lock` (shared with channel adapters) instead
  of a redundant local lock.

### Previous reliability updates (2026-02-18)

- Added transient retry/backoff for Claude API calls to handle rate limits and brief network failures.
- Fixed deny-by-default safety policy enforcement so non-builtin tools are blocked unless explicitly allowlisted.
- Enforced session TTL cleanup on access to prevent idle per-user histories from accumulating indefinitely.
- Preserved whitespace/formatting when chunking long Discord/Telegram replies.
- Hardened Telegram chunk sending so partial send failures are logged and handled cleanly.
- Fixed consolidation bookkeeping so only episodes actually included in a consolidation prompt are marked consolidated.
- Persisted relationship `emotional_patterns` across restarts.
- Heartbeat now both tracks autonomous-thought counts and can truly slow down toward `max_interval` when idle.
- `calculate` now uses a strict AST-based evaluator (no `eval`).
- `fetch_url` now uses bounded streaming reads instead of full-body buffering.
- Daemon protocol now supports optional auth via `GWENN_DAEMON_AUTH_TOKEN`.
- Session `/resume` previews are off by default, and daemon session content is redacted by default.
- CLI input now explicitly enables readline support so arrow keys work as expected in chat.

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
- Her primary role for you
- Your current goals
- Your communication preference
- Boundaries/preferences

Press Enter to skip any question. If you provide answers, Gwenn stores them in:
- `GWENN_DATA_DIR/identity.json` (`onboarding_completed` + `onboarding_profile`)
- `GWENN_DATA_DIR/GWENN_CONTEXT.md` (a durable "Primary User Onboarding" block)

For Telegram/Discord users, you can also run in-channel setup with:
- Telegram: `/setup Name | Role | Needs | Style | Boundaries` (or `/setup skip`)
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

### Optional: run on Telegram / Discord channels

Telegram support is installed by default.

Install Discord dependency only if you want Discord support:

```bash
# uv
uv sync --extra discord

# or pip
pip install -e ".[discord]"
```

Then set channel tokens in `.env`:
- `TELEGRAM_BOT_TOKEN` for Telegram
- `DISCORD_BOT_TOKEN` for Discord

If Telegram dependency is missing in an older environment, Gwenn will attempt a
one-time auto-install on startup. Disable with `GWENN_AUTO_INSTALL_TELEGRAM=false`.

#### Image / media support

Gwenn can see and understand images sent in Telegram and Discord using Claude's
vision capability. Enable per channel in `.env`:

```bash
TELEGRAM_ENABLE_MEDIA=true
DISCORD_ENABLE_MEDIA=true
```

Supported formats: JPEG, PNG, GIF, WebP (up to 20 MB per image). If an image
fails to download, Gwenn falls back to a text description and continues normally.

Run with a specific channel:

```bash
gwenn --channel telegram
gwenn --channel discord
gwenn --channel all
```

You can also set the default mode via `GWENN_CHANNEL=cli|telegram|discord|all`
in `.env`, or have the daemon manage channels via `GWENN_DAEMON_CHANNELS`.

Channel command equivalents:
- `/status` -- current state
- `/heartbeat` -- heartbeat telemetry
- `/reset` -- clear conversation history

### Daemon security settings

```bash
# Optional shared token for protocol auth (recommended)
GWENN_DAEMON_AUTH_TOKEN=your-secret-token

# Session privacy (defaults are privacy-first)
GWENN_DAEMON_SESSION_INCLUDE_PREVIEW=False
GWENN_DAEMON_REDACT_SESSION_CONTENT=True
```

If `GWENN_DAEMON_AUTH_TOKEN` is set, CLI subcommands and daemon chat connections
automatically include it. Clients are disconnected after 3 consecutive auth
failures.

## Validation

```bash
pytest -q
ruff check gwenn tests
```

Current baseline: `1433 passed`, Ruff clean.

## Tech stack

Python 3.11+, async everywhere. The main dependencies:

- **anthropic** -- Claude API
- **chromadb** + **numpy** -- vector storage and embeddings
- **aiosqlite** -- async SQLite for episodic persistence
- **pydantic** + **pydantic-settings** -- data validation and env-based configuration
- **httpx** -- async HTTP for MCP and tool calls
- **structlog** -- structured logging with PII redaction
- **rich** -- terminal UI
- **ruff** for linting, **pytest** + **pytest-asyncio** for tests

## Project layout

```
Gwenn_ai/
├── gwenn/
│   ├── main.py                     # entry point, session bootstrap, shared logging
│   ├── agent.py                    # SentientAgent -- wires everything together
│   ├── types.py                    # shared data types (UserMessage, etc.)
│   ├── config.py                   # all settings, loaded from .env
│   ├── daemon.py                   # persistent background process (Unix socket)
│   ├── heartbeat.py                # autonomous background loop with circuit breaker
│   ├── identity.py                 # emergent self-model with crash-safe deserialization
│   ├── genesis.py                  # genesis prompt generation
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
│   │   ├── session.py              # per-user session management
│   │   ├── startup.py              # channel startup/shutdown orchestration
│   │   └── formatting.py           # cross-channel display helpers
│   │
│   ├── tools/
│   │   ├── registry.py             # tool definitions and risk tiers
│   │   ├── executor.py             # sandboxed execution
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
│   └── privacy/
│       └── redaction.py            # PII scrubbing for logs and persistence
│
├── tests/                          # 1433 tests across 35+ test files
│   ├── conftest.py
│   ├── eval/                       # evaluation framework (ablation, benchmarks)
│   └── test_*.py                   # unit, integration, adversarial, and safety tests
├── docs/
│   └── sentience_assessment.md
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

**Safety** is layered: input validation, action filtering, rate limits, budget
tracking, and a kill switch. Tools go through a risk tier system
(low/medium/high/critical), with configurable deny-by-default policy and
allowlisting for non-builtin tools.

**Privacy** supports scrubbing PII from logs -- emails, phone numbers, SSNs,
credit cards, IPs. Full PII redaction is disabled by default and can be enabled
via `GWENN_REDACTION_ENABLED`, with scope controlled by
`GWENN_REDACT_BEFORE_API` and `GWENN_REDACT_BEFORE_PERSIST`; basic log field
truncation is always on. Daemon sessions are redacted by default.

**Channels** provide platform adapters for Telegram, Discord, and the CLI.
Each channel manages its own session lifecycle, rate limiting, and message
formatting. When media is enabled, Telegram and Discord channels download
images and pass them through to Claude as vision content blocks. The daemon
can manage multiple channels simultaneously while sharing a single agent
instance and respond lock.

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
- [ ] WhatsApp, Signal, Slack, and others integration
- [ ] Integrate STT (Speech-to-Text) and TTS (Text-to-Speech) in channels
- [p] Real MCP transport (JSON-RPC over stdio/HTTP, actual tool discovery and execution)
- [X] SKILLS.md integration, autonomous skill running/development by Gwenn
- [ ] Inline buttons in Discord/Telegram
- [ ] Obsidian, Dropbox, Notion support

**Phase 4: Infrastructure & Service Features**
- [X] Background heartbeat as a system service (daemon with systemd support)
- [X] Automated PII privacy redaction system in logs, sessions, and persistence
- [X] Budget tracking, rate limits, kill switch

**Phase 5: Advanced Capabilities and Ecosystem**
- [ ] Subagents with parallel running capabilities (swarm)
- [ ] Subagent autospawn from Gwenn; Gwenn provides subagents with an identity autonomously as needed
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
- [ ] Ablation tests -- disable subsystems one at a time, measure what breaks
- [ ] Long-horizon validation (multi-day continuous runs)
- [ ] Multi-agent interaction testing
- [ ] Reproducibility protocol and formal sentience criteria
- [p] Full test suite: unit, integration, adversarial, persistence, eval benchmarks

Detailed notes in [`PLAN.md`](PLAN.md).

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
