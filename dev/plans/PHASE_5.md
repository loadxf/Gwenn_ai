# Phase 5: CLI Redesign (Overstory-Inspired)

**Risk: MEDIUM** — Large refactor of main.py, but purely additive. No core logic changes.

**Prerequisites:** Phase 4 complete (event subscriptions available for dashboard/feed).

---

## Goal

Transform `gwenn`'s CLI from a minimal argparse interface into a rich, organized command system with live monitoring, fleet management, and structured output.

---

## Current State

`gwenn/main.py` is **1,935 lines** with:
- argparse with 3 subcommands: `daemon`, `stop`, `status` (plus `install`/`uninstall`/`restart`/`config` from Phases 2-3)
- `GwennSession` class (~1,600 lines) — the interactive REPL with slash commands, onboarding, output styling, terminal management
- The REPL is the **default mode** when no subcommand is given

### GwennSession Migration Strategy

`GwennSession` is large and complex. Rather than rewriting it entirely, the approach is:

1. **Extract** the REPL into `gwenn/cli/repl.py` as-is (minimal changes)
2. **New commands** (status, dashboard, feed, agents, etc.) are pure click commands that connect to the daemon via WebSocket
3. **The REPL** becomes the default click command (invoked when no subcommand is given)
4. GwennSession's internal slash commands (`/status`, `/heartbeat`, `/agents`, etc.) remain working within the REPL — they are convenience shortcuts, not duplicates of CLI commands

This avoids a risky rewrite of the REPL while getting the benefits of click for everything else.

---

## Target: Command Hierarchy

```
gwenn                                    # Interactive REPL (connect to daemon or in-process)
gwenn daemon                             # Start heartbeat daemon (foreground)
gwenn stop                               # Stop daemon gracefully

# --- Monitoring & Observability ---
gwenn status                             # Overview: heartbeat state, channels, connections
  --json                                 # JSON output for scripting
  --verbose                              # Extended details (memory usage, beat history)
gwenn dashboard                          # Live TUI: multi-panel real-time view
  --interval <ms>                        # Refresh rate (default: 1000ms)
gwenn feed                               # Real-time event stream from event bus
  --follow                               # Continuous streaming
  --type <pattern>                       # Filter: "heartbeat.*", "affect.*", "channel.*"
  --json                                 # JSON output
gwenn logs                               # Query structured logs
  --level <level>                        # debug/info/warn/error
  --subsystem <name>                     # Filter by cognitive subsystem
  --since <time>                         # ISO 8601 or relative (1h, 30m, 2d)
  --follow                               # Tail in real time
  --json                                 # JSON output
gwenn costs                              # Token/cost analysis
  --live                                 # Continuous monitoring
  --by-subsystem                         # Group by cognitive subsystem
  --by-channel                           # Group by channel
  --since <time>                         # Time range

# --- Swarm & Subagent Management ---
gwenn agents                             # List active subagents
  --all                                  # Include completed
  --capability <type>                    # Filter: scout, builder, reviewer, merger
  --json                                 # JSON output
gwenn agents inspect <task-id>           # Deep per-agent inspection
  --follow                               # Continuous polling
gwenn agents spawn <description>         # Manually spawn a subagent
  --capability <type>                    # Role assignment
  --persona <name>                       # For Telegram swarm visualization
  --timeout <seconds>                    # Max execution time
gwenn agents stop <task-id>              # Cancel a running subagent
gwenn swarm status                       # Show active swarms
gwenn swarm history                      # Past swarm results
  --last <n>                             # Most recent N swarms

# --- Channel Management ---
gwenn channels                           # List channels and connection status
gwenn channels restart <name>            # Restart a specific channel
gwenn channels test <name>               # Send test message

# --- Service Management (Phase 2) ---
gwenn install                            # Install as system service
gwenn uninstall                          # Remove system service
gwenn restart                            # Restart daemon service

# --- Configuration (Phase 3) ---
gwenn config                             # Show current configuration
gwenn config get <key>                   # Get specific value
gwenn config set <key> <value>           # Set value in gwenn.toml
gwenn config unset <key>                 # Remove override
gwenn config init                        # Generate gwenn.toml template
gwenn config validate                    # Validate configuration

# --- Health & Diagnostics ---
gwenn doctor                             # Run health checks
  --fix                                  # Auto-fix issues where possible
  --json                                 # JSON output
  Categories: python version, dependencies, config validity,
              data directory, API keys, channel connectivity,
              daemon status, disk space, memory usage

# --- Global Flags (all commands) ---
  --json                                 # Structured JSON output
  --quiet / -q                           # Suppress output
  --verbose / -v                         # Extended details
  --no-color                             # Disable ANSI colors
```

### Commands Requiring Running Daemon

Connect to daemon's gateway via WebSocket. Fail gracefully if no daemon:
- `status`, `dashboard`, `feed`, `logs`, `costs`
- `agents` (list/inspect/spawn/stop), `swarm`
- `channels` (list/restart/test)
- `stop`, `restart`

### Command Overlap Resolution

- `gwenn inspect` is **removed** — its functionality is covered by `gwenn status --verbose` (extended heartbeat details) and `gwenn feed --type "heartbeat.*"` (real-time heartbeat events). Having both `inspect` and `status --verbose` would create confusion about which to use.

### Commands That Work Without Daemon

- `gwenn` (interactive REPL — falls back to in-process mode)
- `gwenn daemon` (starts the daemon)
- `gwenn install`, `gwenn uninstall`
- `gwenn config` (all subcommands — reads/writes config files directly)
- `gwenn doctor` (runs local health checks; daemon checks report "not running")

---

## Dashboard Layout (TUI)

```
┌─ Gwenn Dashboard ──────────────────── v0.x.x ── 2026-02-28 14:23:45 ─┐
│                                                                        │
│  HEARTBEAT              │  CHANNELS                                    │
│  Beat #4,217            │  > telegram  connected  3 users              │
│  Phase: THINK (wander)  │  > discord   connected  1 server             │
│  Emotion: curious 0.72  │  > slack     connected  2 workspaces         │
│  Interval: 15s          │  > cli       2 sessions                      │
│  Arousal: 0.45          │  x websocket idle                            │
│  Next beat: 12s         │                                              │
│                         │  SUBAGENTS                                   │
│  AFFECT                 │  > scout-1    researching   task-a3f  42s    │
│  Joy: 0.62              │  > builder-2  implementing  task-b7c  1m23s  │
│  Curiosity: 0.71        │  - reviewer   idle                           │
│  Serenity: 0.55         │                                              │
├─────────────────────────┴──────────────────────────────────────────────┤
│  EVENT FEED                                                            │
│  14:23:42 [beat]    heartbeat #4217 phase=think mode=wander            │
│  14:23:38 [affect]  emotion shift: reflective → curious (arousal +0.2) │
│  14:23:35 [channel] telegram message from user:42981 in thread:topic-3 │
│  14:23:30 [agent]   scout-1 spawned for task-a3f                       │
│  14:23:28 [memory]  consolidated 3 episodes → 1 knowledge entry        │
├────────────────────────────────────────────────────────────────────────┤
│  Tokens: 12,450 in / 8,230 out  │  Cost: $0.34  │  Uptime: 2h 14m    │
└────────────────────────────────────────────────────────────────────────┘
```

**Visual patterns:**
- `>` (green) active, `x` (red) down, `-` (dim) idle, `!` (yellow) warning
- Box-drawing with `rich` (already a dependency via `rich>=13.7.0`)
- `--json` everywhere for scripting/integration

---

## New Files

### `gwenn/cli/` Package

```
gwenn/cli/__init__.py          # Package init
gwenn/cli/app.py               # Main CLI group + global flags (click)
gwenn/cli/repl.py              # GwennSession extracted from main.py
gwenn/cli/monitoring.py        # status, feed, logs, costs
gwenn/cli/dashboard.py         # Live TUI rendering (rich.live)
gwenn/cli/agents.py            # agents, swarm subcommands
gwenn/cli/channels.py          # channels subcommands
gwenn/cli/service.py           # install, uninstall, restart (delegates to gwenn.service)
gwenn/cli/config_cmd.py        # config subcommands (delegates to gwenn.config_file)
gwenn/cli/doctor.py            # Health checks
gwenn/cli/formatters.py        # Color helpers, state indicators, table formatting
gwenn/cli/connection.py        # Shared daemon connection helper (WebSocket client)
```

### `gwenn/cli/app.py` — Main CLI Group

```python
import click

# Async wrapper — click doesn't support async natively
def async_cmd(func):
    import asyncio, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@click.group(invoke_without_command=True)
@click.option("--json", "json_output", is_flag=True, help="Structured JSON output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--verbose", "-v", is_flag=True, help="Extended details")
@click.option("--no-color", is_flag=True, help="Disable ANSI colors")
@click.pass_context
def cli(ctx, json_output, quiet, verbose, no_color):
    """Gwenn - Genesis Woven from Evolved Neural Networks"""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["no_color"] = no_color

    if ctx.invoked_subcommand is None:
        # Default: launch interactive REPL
        from gwenn.cli.repl import run_repl
        run_repl(ctx.obj)
```

**Libraries:**
- `click>=8.1.0` — command framework. Does not support async natively. Use `asyncio.run()` wrapper (see above). Current stable: 8.3.1.
- `rich>=13.7.0` — already a dependency. Tables, panels, live display, color, box-drawing.
- **NOT** using `asyncclick` (community fork) — the wrapper approach is simpler and avoids adding a fork dependency.
- Consider `rich-click` (v1.9.7) for prettier help output — optional, easy to add later.

### `gwenn/cli/connection.py` — Shared Daemon Connection

```python
class DaemonConnection:
    """Shared WebSocket connection to daemon gateway.

    Used by all CLI commands that need daemon state.
    Handles connection, authentication, and error reporting.
    """
    async def connect(self) -> None: ...      # WS connect + auth
    async def rpc(self, method, params) -> dict: ...  # JSON-RPC call
    async def subscribe(self, patterns) -> AsyncIterator: ...  # Event stream
    async def disconnect(self) -> None: ...

def require_daemon(func):
    """Decorator: connects to daemon, passes connection, handles 'not running' error."""
```

### Dashboard Data Source

- `gwenn dashboard` connects via `DaemonConnection`
- Subscribes to `heartbeat.*`, `affect.*`, `channel.*`, `swarm.*` events
- Falls back to HTTP polling (`GET /dashboard` → JSON snapshot) if WS subscription unavailable
- If no daemon running, exits with clear error: "Gwenn daemon is not running. Start with: gwenn daemon"

---

## Modified Files

### `gwenn/main.py` — Simplified

```python
from gwenn.cli.app import cli

def main():
    from gwenn.config import configure_logging
    configure_logging()
    cli()
```

The 1,935 lines reduce to ~10 lines. All logic moves to `gwenn/cli/`.

### `gwenn/gateway.py` — Add `/dashboard` Endpoint

New HTTP endpoint:
```
GET /dashboard → JSON snapshot with all panel data
```

Returns:
```json
{
    "heartbeat": {"beat_count": 4217, "phase": "think", "emotion": "curious", ...},
    "affect": {"joy": 0.62, "curiosity": 0.71, ...},
    "channels": [{"name": "telegram", "status": "connected", "users": 3}, ...],
    "subagents": [{"task_id": "a3f", "role": "scout", "status": "researching", ...}],
    "costs": {"total_tokens_in": 12450, "total_tokens_out": 8230, "total_cost": 0.34},
    "events": [{"timestamp": "...", "type": "heartbeat.beat", "summary": "..."}],
    "uptime_seconds": 8040
}
```

### `pyproject.toml`

Add `"click>=8.1.0"` to dependencies.

---

## Implementation Sub-Steps

```
5a. Create gwenn/cli/ package — app.py, formatters.py, connection.py, global flags
5b. Extract GwennSession from main.py into gwenn/cli/repl.py
5c. Create gwenn/cli/monitoring.py — status, feed, logs, costs
5d. Create gwenn/cli/dashboard.py — Live TUI with rich.live
5e. Create gwenn/cli/agents.py — agents list/inspect/spawn/stop, swarm
5f. Create gwenn/cli/channels.py — channels list/restart/test
5g. Create gwenn/cli/doctor.py — health checks
5h. Move service + config commands into gwenn/cli/service.py, config_cmd.py
5i. Simplify gwenn/main.py — delegate to cli app
5j. Add /dashboard data endpoint to gateway.py
5k. Write tests
```

1 commit per sub-step (5a-5k).

---

## Verification

- `gwenn status` → colored overview of heartbeat, channels, subagents (requires daemon)
- `gwenn status --verbose` → extended details including beat history, memory usage
- `gwenn status --json` → structured JSON output
- `gwenn dashboard` → live TUI with panels updating in real-time (requires daemon)
- `gwenn feed --follow` → streaming event bus output (requires daemon)
- `gwenn agents` → list active subagents with status indicators
- `gwenn logs --level error --since 1h` → filtered log query
- `gwenn costs --by-subsystem` → token/cost breakdown table
- `gwenn doctor` → health checks pass, `--fix` resolves issues (works without daemon)
- `gwenn channels` → shows all channels with connection status
- All commands support `--json`, `--quiet`, `--verbose`, `--no-color`
- Commands requiring daemon show clear error if not running
- `gwenn` (no subcommand) → REPL launches as before, with all slash commands working
- `gwenn --help` → clean help output listing all commands
