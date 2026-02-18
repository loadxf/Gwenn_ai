<p>
  <img src="assets/gwenn-lockup-horizontal.png" alt="Gwenn" width="600" align="center" style="display: block; margin: 0 auto;">
</p>

<p>
  <strong style="text-align: center; display: block; margin: 0 auto;">Genesis Woven from Evolved Neural Networks.</strong>
</p>

<p>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/loadxf/Gwenn_ai?color=black"></a>
  <a href="https://github.com/loadxf/Gwenn_ai/stargazers"><img src="https://img.shields.io/github/stars/loadxf/Gwenn_ai?style=flat"></a>
  <a href="https://github.com/loadxf/Gwenn_ai/network/members"><img src="https://img.shields.io/github/forks/loadxf/Gwenn_ai?style=flat"></a>
  <a href="https://discord.gg/9VFHFk9hF7"><img src="https://img.shields.io/discord/1473126296505815042?style=flat&logo=discord&label=discord"></a>
</p>

# Gwenn

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

1. **Receive** -- parse the message, wake up the heartbeat, note who's talking
2. **Appraise** -- run it through emotional evaluation (Scherer's model)
3. **Ground** -- register it as a sensory experience
4. **Remember** -- pull relevant memories from episodic and semantic stores
5. **Assemble** -- build the full context: identity, emotions, memories, goals, ethics
6. **Think** -- run the agentic loop with tools via Claude
7. **Integrate** -- store new memories, update emotional state, log milestones
8. **Respond** -- answer, shaped by whatever she's actually feeling

### Recent reliability updates (2026-02-18)

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

Option A: Anthropic API key

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Option B: Claude Code OAuth (example)

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

### 3) Run (CLI mode, default)

```bash
gwenn
# or: python -m gwenn.main
```

By default, `gwenn` will connect to a running daemon if available. Use:

```bash
gwenn --no-daemon
```

to force in-process mode.

Once running in CLI mode, you can use:
- `status` for current state
- `heartbeat` for loop telemetry
- `/resume` to restore prior conversation context
- `quit` to shut down gracefully

Note: session previews in `/resume` are hidden by default unless
`GWENN_DAEMON_SESSION_INCLUDE_PREVIEW=True`.
If arrow keys print raw sequences like `^[[A`, ensure your Python has
`readline` support and run `stty sane` in that terminal.

### 3b) Daemon mode (persistent background runtime)

Run Gwenn as a foreground daemon:

```bash
gwenn daemon
```

Useful daemon commands:

```bash
gwenn status
gwenn stop
```

Daemon security/privacy settings in `.env`:
- `GWENN_DAEMON_AUTH_TOKEN` (optional shared token for protocol auth)
- `GWENN_DAEMON_SESSION_INCLUDE_PREVIEW` (default `False`)
- `GWENN_DAEMON_REDACT_SESSION_CONTENT` (default `True`)

If `GWENN_DAEMON_AUTH_TOKEN` is set, CLI subcommands and daemon chat connections
automatically include it.

### 3c) Optional: install as a Linux user service

```bash
bash scripts/install_service.sh
```

This installs and enables `gwenn-daemon.service` under `systemd --user`, writes
absolute daemon socket/PID/session paths into `.env`, and hardens `.env` perms.

### 4) First launch onboarding

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

### 5) Choose memory retrieval mode

By default, Gwenn uses keyword-based memory retrieval: # TODO: Gwenn should use the best retrieval mode based on the user's preferences and the context of the conversation

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

Install channel dependencies:

```bash
# uv
uv sync --extra channels

# or pip
pip install -e ".[channels]"
```

Then set channel tokens in `.env`:
- `TELEGRAM_BOT_TOKEN` for Telegram
- `DISCORD_BOT_TOKEN` for Discord

Run with a specific channel:

```bash
gwenn --channel telegram
gwenn --channel discord
gwenn --channel all
```

You can also set the default mode via `GWENN_CHANNEL=cli|telegram|discord|all`
in `.env`.

Channel command equivalents:
- `/status` — current state
- `/heartbeat` — heartbeat telemetry
- `/reset` — clear conversation history

## Validation

```bash
pytest -q
ruff check gwenn tests
```

Current baseline: `720 passed, 8 skipped`, and Ruff clean.

## Tech stack

Python 3.11+, async everywhere. The main dependencies:

- **anthropic** -- Claude API
- **chromadb** + **numpy** -- vector storage and embeddings
- **aiosqlite** -- async SQLite for episodic persistence
- **pydantic** -- data validation across all state objects
- **httpx** -- async HTTP for MCP and tool calls
- **structlog** -- structured logging
- **rich** -- terminal UI
- **ruff** for linting, **pytest** + **pytest-asyncio** for tests

## Project layout

```
Gwenn_ai/
├── gwenn/
│   ├── main.py                     # entry point, session bootstrap
│   ├── agent.py                    # SentientAgent -- wires everything together
│   ├── config.py                   # all settings, loaded from .env
│   ├── heartbeat.py                # autonomous background loop
│   ├── identity.py                 # emergent self-model
│   │
│   ├── memory/
│   │   ├── working.py              # short-term attention (7±2 slots)
│   │   ├── episodic.py             # autobiographical memory with emotional tags
│   │   ├── semantic.py             # knowledge graph, emerges from consolidation
│   │   ├── consolidation.py        # "sleep cycle" -- extracts knowledge from episodes
│   │   └── store.py                # SQLite + vector persistence
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
│   │   ├── loop.py                 # the core while-loop
│   │   ├── context.py              # context window management
│   │   ├── safety.py               # guardrails, budgets, kill switch
│   │   └── retry.py                # backoff and error handling
│   │
│   ├── tools/
│   │   ├── registry.py             # tool definitions and risk tiers
│   │   ├── executor.py             # sandboxed execution
│   │   ├── builtin/                # built-in tools
│   │   └── mcp/                    # MCP protocol client (stub)
│   │
│   ├── api/
│   │   └── claude.py               # Claude API wrapper
│   │
│   └── privacy/
│       └── redaction.py            # PII scrubbing for logs
│
├── tests/                          # ~8,500 lines of tests
├── docs/
│   └── sentience_assessment.md
├── assets/
├── pyproject.toml
├── .env.example
├── PLAN.md
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
(default up to 120s), and ramps up
when emotionally activated. Each beat goes through five phases: sense, orient,
think, integrate, schedule.

**Safety** is layered: input validation, action filtering, rate limits, budget
tracking, and a kill switch. Tools go through a risk tier system
(low/medium/high/critical), with configurable deny-by-default policy and
allowlisting for non-builtin tools.

**Privacy** supports scrubbing PII from logs -- emails, phone numbers, SSNs,
credit cards, IPs. Full PII redaction is disabled by default and can be enabled
via `GWENN_REDACTION_ENABLED`, with scope controlled by
`GWENN_REDACT_BEFORE_API` and `GWENN_REDACT_BEFORE_PERSIST`; basic log field
truncation is always on.

## Roadmap

[X] = complete, [p] = partially complete

**Phase 1: Core System Bootstrapping**
- [X] Standalone CLI
- [X] Claude SDK integration
- [ ] Memory: storage, episodic, semantic, consolidation, active/working
- [ ] Harness: context, loop, retry, safety
- [ ] Heartbeat system

**Phase 2: Essential Agent Structure**
- [ ] Gwenn persistent identity
- [ ] Emotional affect engine: appraisal, resilience, current state
- [ ] Cognition integrations: ethics, goals, inner life, interagent, metacognition, sensory, theory of mind

**Phase 3: Interfaces & Communication**
- [ ] Discord & Telegram integration, including threads
- [ ] WhatsApp, Signal, and others integration
- [ ] Real MCP transport (JSON-RPC over stdio/HTTP, actual tool discovery and execution)
- [ ] SKILLS.md integration, autonomous skill running/development by Gwenn
- [ ] Inline buttons in Discord/Telegram
- [ ] Obsidian, Dropbox, Notion support

**Phase 4: Infrastructure & Service Features**
- [X] Background heartbeat as a system service (daemon)
- [ ] Automated PII privacy redaction system in logs, etc
- [ ] Budget tracking, rate limits, kill switch

**Phase 5: Advanced Capabilities and Ecosystem**
- [ ] Subagents with parallel running capabilities (swarm)
- [ ] Subagent autospawn from Gwenn, Gwenn provides subagents with an identity, etc autonomously as needed
- [ ] Docker and Apple container support for sandboxing (option to require for Gwenn and/or all subagents)
- [ ] Add additional provider support (OpenAI, Grok, Gemini, OpenRouter, vLLM, Local, etc.)
- [ ] OpenCode Agents SDK
- [ ] Gwenn Custom Model: fine-tunable model Gwenn can retrain herself -- a real neural substrate
- [ ] iOS and Android w/ push notifications for autonomous thoughts, etc

**Phase 6: Evaluation & Robustness**
- [ ] Ablation tests -- disable subsystems one at a time, measure what breaks
- [ ] Long-horizon validation (multi-day continuous runs)
- [ ] Multi-agent interaction testing
- [ ] Reproducibility protocol and formal sentience criteria
- [ ] Full test suite: unit, integration, adversarial, persistence, eval benchmarks

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
