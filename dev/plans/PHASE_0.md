# Gwenn Infrastructure Upgrade: Master Guide

---

## Implementation Instructions

These instructions govern how ALL phase files (PHASE_1 through PHASE_6) are implemented. Treat this section as the system prompt for the entire upgrade. **Read this section in full before beginning any work.**

### Autonomous Execution

Work **autonomously and iteratively** until all phases are complete (Phase 1 → 2 → 3 → 4 → 5 → 6). Do not stop between phases to ask for permission to continue. After completing each phase and confirming all tests pass, immediately proceed to the next phase. Only pause to ask the user when:
- A genuine ambiguity in the plan cannot be resolved by reading the code
- A technical decision has multiple valid approaches with meaningfully different tradeoffs
- An external blocker is encountered (missing API keys, broken dependencies, etc.)

### Think Step by Step

For every sub-step (e.g., 1a, 1b, ...) within a phase:

1. **Read first.** Read all files that will be modified or that the new code depends on. Understand existing patterns, naming conventions, and import structure before writing a single line.
2. **Plan the edit.** Before writing code, state exactly what will change and why. Identify the minimal set of lines that need to change.
3. **Ask: is this the best way?** Before implementing, explicitly consider: Is there a simpler approach? Does the codebase already have a pattern for this? Am I over-engineering? Am I introducing unnecessary coupling? Would a different API surface be cleaner? Document the reasoning.
4. **Be surgical.** Make the minimum change necessary. Do not refactor surrounding code, add comments to untouched lines, rename variables you didn't introduce, or "clean up" anything outside the scope of the sub-step. Every changed line must be justified by the sub-step's goal.
5. **Write tests.** Every sub-step must include tests for the code it introduces or modifies. Tests come before or alongside the implementation, not after.
6. **Verify.** Run the full test suite after each sub-step. Do not proceed to the next sub-step until all tests pass.

### Adversarial Review Protocol

For **every sub-step**, before finalizing the implementation:

1. **Spawn an adversarial review agent.** This agent's job is to critique the implementation. It should:
   - Read the code that was just written or modified
   - Check for: correctness, edge cases, race conditions, security issues, unnecessary complexity, violations of existing patterns, missing error handling, missing tests, performance concerns
   - Propose specific objections or improvements
   - Challenge assumptions: "Why did you use X instead of Y?", "What happens when Z fails?", "This introduces coupling between A and B — is that necessary?"

2. **Resolve disagreements.** If the adversarial agent raises valid concerns:
   - Address each concern explicitly (fix the code, add a test, or explain why the concern doesn't apply)
   - Do not dismiss concerns without reasoning
   - The goal is consensus on the best path forward, not speed

3. **Only continue once resolved.** Do not move to the next sub-step until the adversarial review is satisfied or concerns are explicitly addressed with reasoning.

### Test Requirements

- **100% test coverage** for all new code. Every new file, every new function, every new class method, every branch must have a corresponding test.
- **Use `pytest-cov`** to measure coverage. The coverage report must show 100% for all new files before moving to the next sub-step.
- **Test types required:**
  - Unit tests for each new module (isolated, fast, no I/O)
  - Integration tests for cross-module interactions (e.g., heartbeat → gateway → agent lifecycle)
  - Edge case tests for error handling, timeouts, boundary conditions
  - Regression tests: the full existing test suite (~3,082 tests) must pass after each sub-step
- **Mock external dependencies.** Never call real APIs (Anthropic, Telegram, Slack, Discord) in tests. Mock the CognitiveEngine, bot APIs, and network I/O.
- **Async tests** use `@pytest.mark.asyncio` — consistent with the existing test suite.
- **Test file naming**: `tests/test_<module>.py` matching the source file.

### Code Quality Standards

- Follow existing codebase patterns exactly: structlog for logging, pydantic for models, type hints everywhere, docstrings on public APIs
- No new dependencies beyond those specified in the phase files
- No TODO/FIXME/HACK comments — resolve issues before committing
- No dead code, no commented-out code, no placeholder stubs that do nothing
- Security: validate all external input, no command injection, no path traversal, no leaked secrets in logs
- Every commit must leave the test suite green. Never commit broken code.

### Commit Discipline

- One commit per sub-step (e.g., `1a`, `1b`, ...)
- Commit message format: `Phase X.Y: <concise description of what changed>`
- Each commit is atomic — it can be reverted independently without breaking other sub-steps
- Run `pytest` before every commit. If tests fail, fix before committing.

### Progress Tracking

- Mark each sub-step as complete only after: code written, tests written, tests passing (including existing suite), adversarial review passed
- If a sub-step takes more than expected, note the deviation and adjust the approach rather than rushing

---

## How to Use This Plan

This document is the **master reference** for the Gwenn Infrastructure Upgrade. The full plan is broken into numbered phase files:

| File | Phase | Description | Risk |
|------|-------|-------------|------|
| `PHASE_0.md` | — | This file. Architecture, guide, cross-cutting concerns | — |
| `PHASE_1.md` | Heartbeat-as-Core | Invert ownership: heartbeat owns agent, gateway, channels | **High** |
| `PHASE_2.md` | Service Management | Python-native systemd/launchd install/uninstall | Low |
| `PHASE_3.md` | TOML Configuration | `gwenn.toml` via pydantic-settings built-in TOML source | Low |
| `PHASE_4.md` | Event Enrichment | Client subscriptions, tool approval over WebSocket, moral events | Medium |
| `PHASE_5.md` | CLI Redesign | click-based CLI with dashboard, monitoring, fleet management | Medium |
| `PHASE_6.md` | Telegram Swarm Viz | Visible subagent bots + typed inter-agent messaging | Medium |
| `PHASE_7.md` | Checkpoint/Restore | Periodic cognitive snapshots for crash recovery | Medium |
| `PHASE_8.md` | Self-Healing Tier 1 | Autonomous runtime recovery (channels, subagents, memory) | Low-Medium |
| `FUTURE_VISION.md` | — | Research-backed roadmap for post-infrastructure enhancements | — |

**Implementation order is strict**: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each phase builds on the previous.

**Within each phase**, follow the sub-steps (e.g., 1a, 1b, ...) in order. Each sub-step should be a single commit with tests passing.

---

## Architecture: Before and After

```
BEFORE (daemon-centric):              AFTER (heartbeat-centric):

Daemon (process manager)              Heartbeat (the living core)
  └─ Agent (brain)                      ├─ Gateway (circulatory system)
      ├─ Heartbeat (background task)    │   ├─ WebSocket (CLI, web, mobile)
      └─ (channels bolted on)           │   ├─ MCP Streamable HTTP (stub)
  └─ Socket Server (bolted on)          │   ├─ A2A Agent Card (stub)
  └─ Channels (bolted on)               │   └─ Health endpoint
                                        ├─ Event Bus (nervous system)
                                        │   └─ Each beat pumps events
                                        ├─ Channels (sensory organs)
                                        │   ├─ Telegram
                                        │   ├─ Discord
                                        │   └─ Slack
                                        ├─ Agent (brain — cognitive subsystems)
                                        │   ├─ Memory, Affect, Cognition
                                        │   ├─ Identity, Goals, Ethics
                                        │   └─ Theory of Mind, Metacognition
                                        └─ Safety (immune system)
```

### Two-Layer Timing (like the human body)

The heartbeat runs on 5-120 second cycles. Users expect sub-second responses. Resolution:
- **Heartbeat (slow, 5-120s):** Autonomous cognition, consolidation, proactive thinking, event pulse
- **Neural response (immediate):** Messages handled instantly by the gateway → agent.respond(). The heartbeat is notified (accelerates) but doesn't gate responses.

This is how humans work: the heart beats at ~60bpm, neurons fire at ~1000Hz. Different timescales, same organism.

### Breaking the Agent <-> Heartbeat Circular Dependency

**Current state** (verified against code):
- Agent creates heartbeat in `initialize()` at line 618: `self.heartbeat = Heartbeat(self._config.heartbeat, self)`
- Agent calls `heartbeat.start()` in `start()` (line 731) and `heartbeat.stop()` in `shutdown()` (line 754)
- Heartbeat holds `self._agent` reference and calls back into 20+ agent methods/attributes (see Phase 1 for full list)
- Heartbeat accesses private APIs: `_respond_lock`, `_persist_episode()`, `theory_of_mind._user_models`

**New dependency direction:**
1. **Heartbeat → Agent (downward, direct):** Heartbeat holds a reference to agent after `_wake_up()`. Direct method calls for cognitive operations — this is clean downward access.
2. **Agent → Heartbeat (upward, via event bus):** Agent never holds a heartbeat reference. When signaling "user activity detected, accelerate", it emits `UserActivityEvent` AND the heartbeat provides a direct `notify_user_activity()` method called by the `RequestRouter` (timing-critical path — event bus has no latency guarantee).
3. **Subsystems → World (lateral, via event bus):** Affect, memory, ethics emit events. The heartbeat, gateway, and other subsystems subscribe as needed.

---

## Verified Current State Reference

These facts were verified against the actual codebase (not assumptions):

| File | Lines | Key Facts |
|------|-------|-----------|
| `gwenn/heartbeat.py` | 775 | `Heartbeat.__init__(config: HeartbeatConfig, agent: SentientAgent)`. 5-phase cycle: sense→orient→think→integrate→schedule. Uses `_respond_lock` at 3 points in `_integrate()`. |
| `gwenn/daemon.py` | 581 | `GwennDaemon.__init__(config: GwennConfig)`. 10 RPC methods in `_dispatch()`. Unix socket via `asyncio.start_unix_server`. Channel startup at lines 137-144 with `done_callback` crash monitoring. |
| `gwenn/agent.py` | 4581 | `SentientAgent.__init__(config: GwennConfig)`. 15 named subsystem layers. `_respond_lock` defined at line 288 but never acquired inside agent.py — external callers are responsible. Per-session locks in `_session_respond_locks`. |
| `gwenn/config.py` | 771 | **22 config classes** (21 BaseSettings + 1 plain GwennConfig master). GwennConfig is NOT a BaseSettings subclass. |
| `gwenn/main.py` | 1935 | argparse with 3 subcommands (daemon/stop/status). `GwennSession` class is ~1600 lines for the interactive REPL. |
| `gwenn/channels/base.py` | ~350 | `BaseChannel` ABC with `start()`, `stop()`, `send_message()`, `channel_name`. Has `handle_message()` bridge and per-session lock helper. |
| `gwenn/channels/cli_channel.py` | ~190 | **NOT a BaseChannel subclass.** Standalone Unix socket client. NDJSON protocol. |
| `gwenn/channels/telegram_channel.py` | 1460 | Extends BaseChannel. Forum topic support. Tool approval via inline keyboards. `_thread_to_chat` mapping. |
| `gwenn/channels/startup.py` | ~150 | `build_channels()` and `run_channels_until_shutdown()` — the daemon delegates channel lifecycle here. |
| `gwenn/channels/session.py` | ~180 | `SessionManager` for per-channel conversation histories. Separate from `SessionStore` in memory/. |
| `gwenn/orchestration/models.py` | ~100 | `SubagentSpec`, `SubagentResult`, `SwarmSpec`, `SwarmResult`, `SubagentProgress`, `JsonRpcRequest/Response`. |
| `gwenn/orchestration/orchestrator.py` | ~530 | `Orchestrator` class — spawn, swarm, cancel, collect, aggregate. |
| `gwenn/affect/state.py` | ~250 | `AffectiveState`, `EmotionalDimensions` (5D), `EmotionLabel` (14 emotions). |
| `gwenn/affect/appraisal.py` | ~280 | `AppraisalEngine` — Scherer's Sequential Check Theory. 13 stimulus types. |
| `pyproject.toml` | 63 | `pydantic-settings>=2.1.0`, `httpx>=0.27.0`, `rich>=13.7.0`. No aiohttp/click/slack-bolt. |

**Critical: No event bus exists.** All inter-component communication is direct method calls, shared references, and asyncio locks.

---

## Cross-Phase File Map

### Modified Files

| File | Phases | What Changes |
|------|--------|-------------|
| `gwenn/heartbeat.py` | **1** | **Major refactor** — becomes the core. Owns gateway, agent, channels, event bus. New `run()` lifecycle. |
| `gwenn/daemon.py` | 1 | **Drastically simplified** — PID file + `heartbeat.run()`. From 581 lines to ~80 (with feature flag). |
| `gwenn/agent.py` | 1 | Remove heartbeat ownership. Accept event_bus. Expose private APIs. |
| `gwenn/channels/cli_channel.py` | 1 | Add `WebSocketCliChannel`. |
| `gwenn/config.py` | 1, 3, 6, 7, 8 | Gateway + Slack config (1); `GwennSettingsBase` for TOML (3); swarm bot pool (6); checkpoint config (7); self-healing config (8). |
| `gwenn/main.py` | 1, 2, 3, **5** | WebSocket connection (1); install/uninstall (2); config subcommand (3); gutted → delegates to `gwenn/cli/` (5). |
| `gwenn/affect/appraisal.py` | 4 | `MoralConcernEvent` emission. Optional interoceptive signal integration. |
| `gwenn/affect/state.py` | 4 | `EmotionChangedEvent` emission. |
| `gwenn/gateway.py` | 1, 4, 5, 7, 8 | WebSocket + HTTP server (1); event subscriptions + tool approval (4); `/dashboard` endpoint (5); checkpoint RPC (7); health status (8). |
| `gwenn/channels/telegram_channel.py` | 6 | Integrate bot pool for swarm visualization. |
| `gwenn/orchestration/models.py` | 6 | Add `SubagentPersona`, extend `SubagentSpec` with persona. |
| `gwenn/orchestration/orchestrator.py` | 6 | Wire bot pool + typed messages into swarm lifecycle + turn-taking. |
| `gwenn/events.py` | 1, 4, 6, 7, 8 | Event bus + base events (1); enrichment + interoceptive events (4); swarm viz + agent message types (6); checkpoint events (7); self-heal events (8). |
| `pyproject.toml` | 1, 2, 3, 5 | Add aiohttp, slack-bolt (1); package-data for templates (2); bump pydantic-settings, add tomli-w (3); add click (5). |
| `.env.example` | 1, 3, 7, 8 | Gateway, Slack, feature flag fields (1); TOML reference (3); checkpoint fields (7); self-healing fields (8). |

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| `gwenn/rpc.py` | 1 | `RequestRouter` + JSON-RPC 2.0 helpers |
| `gwenn/gateway.py` | 1 | WebSocket + HTTP + MCP/A2A stubs |
| `gwenn/events.py` | 1 | Event bus + Pydantic event definitions |
| `gwenn/channels/slack_channel.py` | 1 | Slack channel via slack-bolt async Socket Mode |
| `gwenn/service.py` | 2 | `ServiceManager` ABC + systemd/launchd implementations |
| `gwenn/templates/gwenn-daemon.service` | 2 | systemd template (moved from `scripts/`) |
| `gwenn/templates/com.gwenn.daemon.plist` | 2 | launchd template |
| `gwenn/config_file.py` | 3 | TOML find/load/write utilities |
| `gwenn/templates/gwenn.toml.example` | 3 | Annotated config template |
| `gwenn/interoception.py` | 4 | `InteroceptiveState` + `InteroceptiveMonitor` |
| `gwenn/cli/__init__.py` | 5 | CLI package init |
| `gwenn/cli/app.py` | 5 | Main CLI group + global flags (click) |
| `gwenn/cli/repl.py` | 5 | `GwennSession` extracted from main.py |
| `gwenn/cli/monitoring.py` | 5 | status, feed, logs, costs commands |
| `gwenn/cli/dashboard.py` | 5 | Live TUI dashboard (rich) |
| `gwenn/cli/agents.py` | 5 | agents + swarm management commands |
| `gwenn/cli/channels.py` | 5 | channels list/restart/test commands |
| `gwenn/cli/service.py` | 5 | install/uninstall/restart (delegates to `gwenn.service`) |
| `gwenn/cli/config_cmd.py` | 5 | config subcommands (delegates to `gwenn.config_file`) |
| `gwenn/cli/doctor.py` | 5 | Health check system |
| `gwenn/cli/formatters.py` | 5 | Color helpers, state indicators, tables |
| `gwenn/cli/connection.py` | 5 | Shared `DaemonConnection` WebSocket client helper |
| `gwenn/channels/telegram_bot_pool.py` | 6 | Bot pool manager for swarm visualization |
| `gwenn/checkpoint.py` | 7 | `CognitiveCheckpoint` model + `CheckpointManager` |
| `gwenn/healing.py` | 8 | `SelfHealingEngine` + `HealthIssue` + `RecoveryAction` |

---

## Corrections from Original Plan

These errors in the original `infrastructure-upgrade.md` have been corrected in the phase files:

| # | Original Claim | Correction |
|---|----------------|------------|
| 1 | "the heartbeat calls back into `self._agent.respond()`" | **Wrong.** Heartbeat never calls `respond()`. It calls `process_appraisal()`, `consolidate_memories()`, `inner_life.autonomous_thought()`, `broadcast_to_channels()`, `send_to_session()`, `maybe_develop_skill_autonomously()`, and accesses ~20 attributes. |
| 2 | "26 config classes" | **22** (21 BaseSettings + 1 plain GwennConfig). |
| 3 | "TomlConfigSettingsSource requires pydantic-settings >=2.6.0" | Introduced in **v2.2.0** (Feb 2023). Bumping to >=2.6.0 is fine but the justification was wrong. |
| 4 | `toml_file` in SettingsConfigDict is sufficient for TOML loading | **Wrong.** You must also override `settings_customise_sources()` to include `TomlConfigSettingsSource` explicitly. |
| 5 | "Change all 26 config classes: BaseSettings → GwennSettingsBase" | `GwennConfig` (the master class) is NOT a BaseSettings subclass — it's a plain class that composes the 21 settings classes. Only the 21 leaf classes change. |
| 6 | `CliChannel` has "identical public API" to BaseChannel | `CliChannel` does NOT extend `BaseChannel`. It's a standalone Unix socket client with a completely different API (connect/disconnect/chat/get_status etc.). |
| 7 | Event bus is "~80 lines" | Realistic estimate with typed subscriptions, wildcard matching, exception isolation, ordering, start/stop lifecycle, and both sync/async emit is **120-180 lines**. |

---

## Dependency Analysis

### Already Installed (Zero-Cost)

| Package | Installed Version | Pulled By | Notes |
|---------|------------------|-----------|-------|
| `aiohttp` | 3.13.3 | `discord.py` | Already in venv. Adding to pyproject.toml makes it explicit. |
| `websockets` | 16.0 | standalone | Already installed but NOT needed — aiohttp covers WebSocket server. |
| `uvicorn` | 0.41.0 | `chromadb` | Not needed. |

### New Dependencies

| Package | Version | Phase | Purpose | Notes |
|---------|---------|-------|---------|-------|
| `aiohttp` | `>=3.10.0` | 1 | Gateway server (aiohttp.web) | Zero new install — already in venv via discord.py. Floor bumped from plan's 3.9.0 to 3.10.0 for currency. |
| `slack-bolt` | `>=1.18.0` | 1 | Slack channel via Socket Mode | Current stable: 1.27.0. No `[socket-mode]` extra needed. For async mode (`AsyncApp`), `aiohttp` is required (already a dep). |
| `tomli-w` | `>=1.0.0` | 3 | TOML writing (config set/init) | Current stable: 1.2.0. Reading uses stdlib `tomllib` (Python >=3.11). |
| `click` | `>=8.1.0` | 5 | CLI framework | Current stable: 8.3.1. Does NOT support async natively — use `asyncio.run()` wrapper pattern. |

### Bumped Dependencies

| Package | Current Constraint | New Constraint | Reason |
|---------|-------------------|----------------|--------|
| `pydantic-settings` | `>=2.1.0` | `>=2.6.0` | Built-in TOML source (actually available since 2.2.0, but 2.6.0 has improvements). |

---

## Cross-Cutting Concerns

### Logging: structlog Integration

All new modules must use `structlog` — consistent with the existing codebase. Key loggers:
- `gwenn.events` — event emission/dispatch, slow handler warnings
- `gwenn.gateway` — connection lifecycle, auth failures, RPC errors
- `gwenn.rpc` — request routing, lock contention warnings
- `gwenn.service` — install/uninstall/start/stop operations
- `gwenn.cli` — command execution, daemon connection status
- `gwenn.channels.slack` — Slack channel lifecycle

### Dual-Lock Pattern Preservation

The current codebase has two lock patterns that MUST be preserved:

1. **`agent._respond_lock`** (asyncio.Lock) — Used by the daemon's `_dispatch()` to serialize `agent.respond()` calls from the Unix socket. Also used by heartbeat's `_integrate()` to serialize `process_appraisal()` calls. This is a global lock.

2. **`agent._session_respond_locks`** (dict[str, asyncio.Lock]) — Used by `BaseChannel.handle_message()` to serialize `respond()` calls within the same session while allowing concurrency across sessions. Stored on the agent so all channel adapters share the lock pool.

In the new architecture, the `RequestRouter` must support BOTH patterns:
- WebSocket `chat` RPC → per-session lock (like channels do) when session_id is provided
- Legacy socket `chat` RPC → global `_respond_lock` (backward compatible)
- Heartbeat `process_appraisal()` → global `_respond_lock` (unchanged)

### Private API Access Refactoring

The heartbeat currently accesses private agent APIs that must be exposed properly:

| Private Access | Current Location | New Approach |
|---------------|-----------------|--------------|
| `agent._respond_lock` | heartbeat.py:480,541,598 | Pass lock to heartbeat in `_wake_up()`, or expose as `agent.respond_lock` property |
| `agent._persist_episode()` | heartbeat.py:570 | Expose as public `agent.persist_episode()` |
| `agent.theory_of_mind._user_models` | heartbeat.py:234 | Add `agent.theory_of_mind.decay_all_stale_beliefs()` public method |

### Test Strategy

- **Unit tests**: Each new module gets its own test file
- **Integration tests**: Phase 1 needs a full lifecycle test (heartbeat.run() → gateway → agent → channel → beat → shutdown). Use `asyncio.wait_for()` with timeouts. Mock the CognitiveEngine to avoid API calls.
- **Event bus tests**: emit/subscribe/unsubscribe, wildcard matching, handler exception isolation, ordering, `emit_async()` for synchronous assertions
- **Gateway tests**: Use `aiohttp.test_utils.TestServer` — follows existing patterns in `tests/test_daemon.py`
- **Feature flag tests**: Both `GWENN_HEARTBEAT_CORE=true` and `false` paths
- **Backward compatibility**: After each phase, full existing test suite (~3,082 tests) must pass

### Feature Flag: `GWENN_HEARTBEAT_CORE`

During Phase 1, the old daemon-centric path remains functional:

- **`false`**: `daemon.py` runs the original 581-line path. Agent creates heartbeat. No gateway.
- **`true`** (default once Phase 1 is complete): Daemon is thin wrapper. Heartbeat owns everything.

Files gated by this flag:
- `gwenn/daemon.py` — old path vs thin wrapper
- `gwenn/agent.py` — heartbeat creation in `initialize()` must be conditional (only when flag is false)
- `gwenn/main.py` — connection strategy (Unix socket vs WebSocket)

The flag is **temporary** — removed once Phase 1 is validated in production.

---

## Git Workflow

```
git checkout -b dev

Phase 1: [1 commit per sub-step: 1a through 1l]
Phase 2: [1 commit]
Phase 3: [1 commit]
Phase 4: [1 commit — includes interoception sub-steps 4h-4l]
Phase 5: [1 commit per sub-step: 5a through 5k]
Phase 6: [1 commit per sub-step: 6a through 6i — includes typed inter-agent messaging]
Phase 7: [1 commit per sub-step: 7a through 7h]
Phase 8: [1 commit per sub-step: 8a through 8i]

Post-Phase 1 cleanup: Remove GWENN_HEARTBEAT_CORE flag + old daemon code path
```

**IMPORTANT:** All work stays on the `dev` branch. Nothing from this plan goes on `main` ever. The `dev` branch is the working branch for the entire infrastructure upgrade.

---

## Upgrade Path for Existing Deployments

If Gwenn is running as a systemd service:
1. `gwenn stop` (old daemon shuts down)
2. Update code (git pull / pip install)
3. `gwenn daemon` (new heartbeat-centric daemon starts)
4. If issues: set `GWENN_HEARTBEAT_CORE=false` in `.env`, restart

The systemd unit file does not change — it still runs `gwenn daemon`.

---

## Future Vision (Post-Infrastructure)

See **[FUTURE_VISION.md](FUTURE_VISION.md)** for the complete research-backed roadmap that the heartbeat-as-core architecture enables. Organized by proximity tier:

| Tier | Directions | Key Research |
|------|-----------|-------------|
| **Near-Term** | Global Workspace broadcast, somatic markers, bitemporal memory, ACT-R memory decay, narrative identity, interoception | Baars 1988, Damasio 1994, Anderson 2007, McAdams 2001, Craig 2002 |
| **Soon** | Voice interaction, dream states, curiosity-driven exploration, Active Inference, full A2A, MCP server mode | Picard 1997, Walker 2017, Schmidhuber 2010, Friston 2010 |
| **Near-Soon** | Autonomous self-healing (3 tiers), codebase self-improvement, Overstory-inspired swarm enhancements (worktree isolation, 4-tier merge, typed messaging, capability dispatch, checkpoint/restore, observability) | Maturana & Varela 1980, Le Goues 2012, Schmidhuber 2003 |
| **Later** | Hierarchical consolidation, attention schema, emotional contagion, circadian rhythms, cognitive fatigue, developmental stages | Graziano 2013, Hatfield 1994, Kahneman 2011, Piaget 1952 |
| **Horizon** | World model/simulation, quantum cognition, social bonding/attachment, autopoiesis, mirror neurons/empathic simulation | Ha & Schmidhuber 2018, Busemeyer 2012, Bowlby 1969, Gallese 2001 |

Each direction includes: what it is, why it matters, how the infrastructure enables it, and the research basis. Full bibliography included.
