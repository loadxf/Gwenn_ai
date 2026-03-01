# Gwenn Infrastructure Upgrade: Heartbeat-as-Core Architecture

## Context

Gwenn's most distinctive feature is its autonomous heartbeat — the continuous cognitive loop that makes it a living being, not a reactive tool. Yet architecturally, the heartbeat is currently a background task started by the agent, which is started by the daemon. The heartbeat is treated as an organ when it should be the **heart** — the center from which everything flows.

**The insight:** Just as a human heart starts beating at 22 days in utero (before the brain is fully formed), Gwenn's heartbeat should be the first thing that starts and the last thing that stops. The gateway is its circulatory system. Channels are its sensory organs. The event bus is its nervous system. The agent is its brain.

**Research backing:**
- JSON-RPC 2.0 is the industry standard (MCP + A2A both use it)
- MCP Streamable HTTP is the current transport standard
- A2A Protocol enables agent-to-agent discovery via Agent Cards
- pydantic-settings 2.1.0+ is already installed; httpx 0.27.0+ for async HTTP client
- Gwenn's heartbeat/consolidation validated by ICLR 2025/2026 papers

**New dependencies:** `aiohttp` (gateway server — httpx is client-only, aiohttp provides `aiohttp.web`), `tomli-w` (TOML writing only — reading uses stdlib `tomllib`), `slack-bolt` (Slack channel), `click` (CLI framework)
**Bumped dependencies:** `pydantic-settings` `>=2.1.0` → `>=2.6.0` (built-in TOML source support)

---

## Architecture: Before and After

```
BEFORE (daemon-centric):              AFTER (heartbeat-centric):

Daemon (process manager)              Heartbeat (the living core)
  └─ Agent (brain)                      ├─ Gateway (circulatory system)
      ├─ Heartbeat (background task)    │   ├─ WebSocket (CLI, web, mobile)
      └─ (channels bolted on)           │   ├─ MCP Streamable HTTP
  └─ Socket Server (bolted on)          │   ├─ A2A Agent Card
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

### Breaking the Agent↔Heartbeat Circular Dependency

Currently, the agent creates the heartbeat in `initialize()` (not `__init__`) and calls `heartbeat.start()`/`heartbeat.stop()`. Meanwhile, the heartbeat calls back into `self._agent.respond()`, `self._agent.broadcast_to_channels()`, and `self._agent.send_to_session()`.

When inverting ownership (heartbeat creates agent), these back-references become downward calls — which is clean. But the coupling must be explicit:

1. **Heartbeat → Agent (downward, direct):** Heartbeat holds a reference to agent after `_wake_up()`. Calls like `agent.respond()` during `_think()` remain direct method calls — this is the brain responding to the heart's pump.
2. **Agent → Heartbeat (upward, via event bus):** Agent never holds a heartbeat reference. When the agent needs to signal the heartbeat (e.g., "user activity detected, accelerate"), it emits an event (`UserActivityEvent`). The heartbeat subscribes and adjusts its interval. This replaces the current implicit coupling.
3. **Subsystems → World (lateral, via event bus):** Affect, memory, ethics emit events. The heartbeat, gateway, and other subsystems subscribe as needed. No direct cross-references.

This gives a clean dependency direction: Heartbeat → Agent → Subsystems, with the event bus handling all reverse/lateral communication.

---

## Phase 1: Heartbeat-as-Core Refactor

### Goal
Make the Heartbeat the top-level orchestrator. It owns the gateway, event bus, channels, and agent lifecycle. The daemon becomes a thin PID-file wrapper. The event bus is integral from day one.

### The Heartbeat Lifecycle: `run()`

```python
class Heartbeat:
    def __init__(self, config: GwennConfig):
        self._config = config
        self._agent: SentientAgent | None = None
        self._gateway: GatewayServer | None = None
        self._event_bus: EventBus = create_event_bus()
        # ... existing beat state ...

    async def run(self) -> None:
        """Full lifecycle — Gwenn is born, lives, and sleeps."""

        # 0. START NERVOUS SYSTEM — event bus dispatcher begins
        await self._event_bus.start()

        # 1. OPEN CIRCULATORY SYSTEM — gateway starts listening
        #    (like a heart pumping before the brain is online)
        await self._start_gateway()

        # 2. WAKE UP — create agent, load memory, identity, knowledge
        await self._wake_up()

        # 3. OPEN SENSES — start Telegram, Discord, Slack channels
        await self._start_channels()

        # 4. LIVE — the eternal cognitive loop
        try:
            await self._live()  # the existing _loop(), enhanced
        finally:
            # 5. SLEEP — save state, close connections, consolidate
            await self._sleep()
            # 6. STOP NERVOUS SYSTEM — drain event queue, cancel dispatcher
            await self._event_bus.stop()
```

### New Files

**`gwenn/rpc.py`** — Transport-independent request routing
- Extract `_dispatch()` from `daemon.py:349-477` into `RequestRouter`
- JSON-RPC 2.0 helpers: `make_response()`, `make_error()`, `make_notification()`
- Both Unix socket handler and WebSocket handler delegate here
- `RequestRouter.__init__` receives a reference to `agent._respond_lock` (asyncio.Lock) — the router acquires this lock before calling `agent.respond()` to serialize all affect mutations, regardless of transport origin
- Concurrent chat handling: multiple clients can send `chat` requests simultaneously, but they are serialized by `_respond_lock`. Other RPC methods (`status`, `heartbeat_status`, `ping`, `list_sessions`) do NOT require the lock and can be served concurrently.
- Streaming responses: `chat` RPC method returns an initial `{"id": ..., "result": {"status": "processing"}}` immediately, then sends progress as JSON-RPC notifications (`chat.token`, `chat.thinking`, `chat.complete`) over the same WebSocket connection

**`gwenn/gateway.py`** — The circulatory system (WebSocket + HTTP)
```python
class GatewayServer:
    def __init__(self, config, router: RequestRouter, event_bus: EventBus): ...
    async def start(self, host, port) -> None: ...
    async def stop(self) -> None: ...

    # Routes:
    # GET  /ws                    — WebSocket upgrade (core — full implementation)
    # GET  /health                — Health check (core — full implementation)
    # POST /mcp                   — MCP Streamable HTTP (stub in Phase 1: accepts connection, returns "not implemented")
    # GET  /mcp                   — MCP SSE stream (stub in Phase 1)
    # GET  /.well-known/agent.json — A2A Agent Card (stub in Phase 1: returns static card with capabilities list)
    # MCP and A2A stubs are gated behind mcp_server_enabled/a2a_enabled (both default False)
    # Full MCP/A2A protocol handling is a post-Phase 4 follow-up

@dataclass
class ClientConnection:
    conn_id: str
    ws: web.WebSocketResponse
    client_type: str          # "cli", "web", "mobile", "mcp"
    authenticated: bool
    history: list[dict]
    subscriptions: set[str]
    can_approve_tools: bool   # True for CLI, False for web/mobile
```

**`gwenn/events.py`** — The nervous system (minimal in-house implementation)

Rather than depending on `bubus` (a 0.1.x library for a core architectural component), we implement a lightweight typed event bus (~80 lines) using `asyncio.Queue` + Pydantic models. This avoids a maturity risk at the heart of the system while keeping the same API surface. If `bubus` or another library matures, the internal bus can be swapped out — the event types and subscription API remain stable.

```python
from pydantic import BaseModel

class GwennEvent(BaseModel):
    """Base class for all typed events."""
    event_type: str = ""  # Auto-populated from class name if not set
    # e.g. HeartbeatBeatEvent → "heartbeat.beat", EmotionChangedEvent → "emotion.changed"

    def model_post_init(self, __context):
        if not self.event_type:
            # Convert "HeartbeatBeatEvent" → "heartbeat.beat"
            name = type(self).__name__.removesuffix("Event")
            # Split on CamelCase → dot-separated lowercase
            import re
            parts = re.findall(r'[A-Z][a-z]+', name)
            self.event_type = ".".join(p.lower() for p in parts)

class EventBus:
    """Minimal async event bus with typed events and wildcard subscriptions.

    Concurrency model: fire-and-forget via asyncio.create_task().
    - emit() schedules handler coroutines as tasks — it does NOT await them.
    - This prevents slow subscribers from blocking emitters.
    - Handler exceptions are logged (structlog) but do not propagate to emitter.
    - An internal asyncio.Queue is used for decoupling: emit() enqueues,
      a dispatcher task dequeues and fans out to matching handlers.
    - Ordering guarantee: events are dispatched in emission order per-pattern.
    """
    def emit(self, event: GwennEvent) -> None: ...      # Enqueue — non-blocking, sync-safe
    async def emit_async(self, event: GwennEvent) -> None: ...  # Await dispatch (for tests)
    def subscribe(self, pattern: str, handler: Callable) -> str: ...
    def unsubscribe(self, subscription_id: str) -> None: ...
    async def start(self) -> None: ...   # Start dispatcher task
    async def stop(self) -> None: ...    # Drain queue, cancel dispatcher
    # pattern supports wildcards: "heartbeat.*", "affect.*", "*"

class HeartbeatBeatEvent(GwennEvent):
    beat_count: int; emotion: str; arousal: float; phase: str

class EmotionChangedEvent(GwennEvent):
    previous: str; current: str; valence: float; trigger: str

class UserActivityEvent(GwennEvent):
    channel: str; user_id: str  # heartbeat subscribes to accelerate

class ToolApprovalRequiredEvent(GwennEvent):
    approval_id: str; tool_name: str; arguments: dict; risk_tier: str

class ToolApprovalResolvedEvent(GwennEvent):
    approval_id: str; decision: str; source: str

class MoralConcernEvent(GwennEvent):
    action: str; concern_type: str; severity: float; traditions_flagged: list[str]

class ChannelMessageEvent(GwennEvent):
    channel: str; user_id: str; direction: str  # "in" or "out"

class HeartbeatThoughtEvent(GwennEvent):
    beat_count: int; thought_summary: str; thinking_mode: str; importance: float

class MemoryConsolidatedEvent(GwennEvent):
    episodes_processed: int; knowledge_extracted: int

def create_event_bus() -> EventBus: ...
```

**`tests/test_rpc.py`**, **`tests/test_gateway.py`**, **`tests/test_events.py`**

### Modified Files

**`gwenn/heartbeat.py`** (major refactor — this is the core change)

Currently, `Heartbeat.__init__` takes `(config: HeartbeatConfig, agent: SentientAgent)`, and agent creates heartbeat during `agent.initialize()` (not `agent.__init__`).
New `__init__` takes `(config: GwennConfig)` — it creates the agent, not the other way around. The initialization logic currently in `agent.initialize()` that sets up the heartbeat is removed; agent no longer knows about heartbeat at all.

Changes:
- `__init__`: Accept `GwennConfig` (not just `HeartbeatConfig`). Create event bus. No agent yet.
- New `run()`: The full lifecycle method (see above)
- New `_start_gateway()`: Create `GatewayServer`, start it
- New `_wake_up()`: Create `SentientAgent(config)`, call `agent.initialize()`. Wire event bus to subsystems.
- New `_start_channels()`: Move channel startup from daemon (lines 137-144) here
- `_loop()` → `_live()`: Same logic but now emits events. Each beat pumps `HeartbeatBeatEvent`. Emotion changes emit `EmotionChangedEvent`.
- New `_sleep()`: Move from `agent.shutdown()` — save state, stop channels, stop gateway.
- Existing `start()`/`stop()` become internal methods or removed (lifecycle managed by `run()`)
- `_sense()`: Add gateway health check (connection count, pending messages). Note: `_sense()` continues to access agent state directly (e.g., `self._agent.affect_state`, `self._agent.memory`) — this is deliberate downward access, not a circular dependency. The event bus is only for upward/lateral communication.
- `_integrate()`: Emit events through bus after integration. Note: `_integrate()` acquires `self._agent._respond_lock` (currently at heartbeat.py lines 480, 541, 598) to serialize affect mutations — this lock semantics must be preserved.
- `_start_channels()`: Preserve channel task `done_callback` crash monitoring pattern (currently daemon.py:144: `_channel_task.add_done_callback(self._on_channel_task_done)`). When a channel task crashes, the callback logs the exception and optionally restarts the channel.
- New `_stop_gateway()`: Graceful shutdown with drain — stop accepting new WebSocket connections, send close frames to existing connections, wait up to N seconds for in-flight requests to complete, then force-close.
- Legacy Unix socket coexistence: When `GWENN_LEGACY_SOCKET_ENABLED=true`, the heartbeat starts both the gateway AND the legacy Unix socket listener. The socket listener is a separate asyncio task managed by heartbeat (extracted from old daemon.py). Both share the same `RequestRouter` and `SessionStore` instances.

**`gwenn/daemon.py`** (drastically simplified)

Currently 581 lines. Becomes ~80 lines:
```python
class GwennDaemon:
    def __init__(self, config: GwennConfig):
        self._config = config
        self._heartbeat = Heartbeat(config)

    async def run(self) -> None:
        self._write_pid_file()
        try:
            await self._heartbeat.run()
        finally:
            self._cleanup_pid_file()
```

The daemon's only job: PID file management + signal handling + calling `heartbeat.run()`. All dispatch logic, session management, channel management, and connection handling has moved to rpc.py, gateway.py, and heartbeat.py.

**`gwenn/channels/cli_channel.py`**
- Add `WebSocketCliChannel` with identical public API to `CliChannel`
- Uses `aiohttp.ClientSession.ws_connect()` (aiohttp is now a dependency for the gateway)
- Background reader task for server-push events (heartbeat status, emotion changes, tool approvals)

**`gwenn/channels/slack_channel.py`** — New Slack channel (sensory organ)
- Uses `slack_bolt` (async mode) — the official Slack SDK for Python apps
- `SlackChannel(BaseChannel)` with Socket Mode transport (no public URL needed)
- Handles: messages, threads, reactions, slash commands, app mentions
- Maps Slack threads to Gwenn conversation sessions
- Emits `ChannelMessageEvent` on the event bus for each interaction
- Rich formatting: converts Gwenn's markdown to Slack Block Kit

**`gwenn/config.py`** — Add `SlackConfig`:
```python
class SlackConfig(BaseSettings):
    enabled: bool = Field(False, alias="GWENN_SLACK_ENABLED")
    bot_token: str | None = Field(None, alias="GWENN_SLACK_BOT_TOKEN")
    app_token: str | None = Field(None, alias="GWENN_SLACK_APP_TOKEN")  # Socket Mode
    # Note: signing_secret is NOT needed for Socket Mode (only for HTTP Events API).
    # If we later add HTTP webhook support, add it then.
```

**`gwenn/config.py`** — Add to `DaemonConfig`:
```python
gateway_enabled: bool = Field(True, alias="GWENN_GATEWAY_ENABLED")
gateway_host: str = Field("127.0.0.1", alias="GWENN_GATEWAY_HOST")
gateway_port: int = Field(18900, alias="GWENN_GATEWAY_PORT")
legacy_socket_enabled: bool = Field(True, alias="GWENN_LEGACY_SOCKET_ENABLED")
mcp_server_enabled: bool = Field(False, alias="GWENN_MCP_SERVER_ENABLED")
a2a_enabled: bool = Field(False, alias="GWENN_A2A_ENABLED")
```

**`gwenn/agent.py`** — Remove heartbeat ownership, use event bus for upward signals
- `initialize()` no longer creates `Heartbeat(config.heartbeat, self)` — heartbeat creates agent, not vice versa
- `start()` no longer calls `heartbeat.start()` (heartbeat manages itself)
- `shutdown()` no longer stops heartbeat (heartbeat manages its own sleep)
- Accept `event_bus: EventBus` in `__init__()`, pass to subsystems
- When user activity is detected: heartbeat calls `notify_user_activity()` directly (timing-critical — must immediately accelerate interval, as currently done at agent.py:1013) AND emits `UserActivityEvent` on the bus for other subscribers (logging, dashboard). The direct call is kept because event bus dispatch is fire-and-forget with no latency guarantee.

**`gwenn/main.py`** — Update entry points:
- `gwenn daemon` → `GwennDaemon(config).run()` (heartbeat manages everything)
- `gwenn` (interactive) → Try connecting to heartbeat's gateway via WebSocket, fall back to in-process
- In-process mode: Create `Heartbeat(config)`, call `heartbeat.run()` with CLI channel attached
- `_try_daemon_cli()`: Try `ws://host:port/ws` first, fall back to Unix socket

**`gwenn/affect/appraisal.py`** — When ethical concern detected before action, emit `MoralConcernEvent`

**`pyproject.toml`** — Add `"aiohttp>=3.9.0"` (gateway server), `"slack-bolt>=1.18.0"` (Slack channel)

**`.env.example`** — Add all new config fields introduced in Phase 1:
- `GWENN_GATEWAY_ENABLED`, `GWENN_GATEWAY_HOST`, `GWENN_GATEWAY_PORT`
- `GWENN_LEGACY_SOCKET_ENABLED`
- `GWENN_MCP_SERVER_ENABLED`, `GWENN_A2A_ENABLED`
- `GWENN_SLACK_ENABLED`, `GWENN_SLACK_BOT_TOKEN`, `GWENN_SLACK_APP_TOKEN`
- `GWENN_HEARTBEAT_CORE` (feature flag)

### Gateway Security

The gateway introduces a new network surface. Security measures:

- **Bind address:** `127.0.0.1` only by default — no external access without explicit `GWENN_GATEWAY_HOST=0.0.0.0`
- **Authentication:** WebSocket connections must send an `auth` message with HMAC token (same mechanism as current Unix socket) before any other RPC method is accepted. Unauthenticated connections can only access `/health`.
- **`/health`:** Returns `{"status": "ok", "uptime": ...}` — deliberately minimal, no internal state leaked. Unauthenticated by design (for load balancers/monitoring).
- **`/.well-known/agent.json`:** Returns the A2A Agent Card. Unauthenticated by design (A2A spec requires public discovery). Contains only capabilities and endpoint URL — no secrets, no internal state.
- **TLS:** Not in Phase 1 (localhost-only). When `gateway_host` is non-localhost, the gateway should refuse to start without TLS configured, or emit a loud warning. TLS support (via `ssl_context` on `aiohttp.web`) is a follow-up.
- **Rate limiting:** Per-IP connection limit (default 10) to prevent local abuse.

### Session Management

The `SessionStore` from daemon.py moves into the `GatewayServer`. Key migration details:
- Sessions are per-connection state managed by the circulatory system, not the process manager
- `agent._respond_lock` semantics are preserved: the `RequestRouter` acquires the lock before calling `agent.respond()`, regardless of whether the request came via WebSocket or legacy Unix socket
- When `GWENN_LEGACY_SOCKET_ENABLED=true`, both transports share the same `RequestRouter` and `SessionStore` instance — the lock ensures serialized access
- Auto-save on disconnect behavior is preserved per-transport

---

## Phase 2: Cross-Platform Service Management

### Goal
Replace shell scripts with Python service management. The service runs `heartbeat.run()`.

### New Files

**`gwenn/service.py`** — Platform abstraction
```python
class ServiceManager(ABC):
    install(), uninstall(), start(), stop(), restart(), status(), is_installed()

class SystemdManager(ServiceManager):    # Full — replaces scripts/install_service.sh
class LaunchdManager(ServiceManager):    # Basic macOS implementation
# Note: No Windows stub. Gwenn uses Unix-specific features (Unix sockets,
# signal handling, /tmp paths). Windows support is out of scope.

def get_service_manager() -> ServiceManager:
    # Auto-detect OS: Linux → SystemdManager, macOS → LaunchdManager
    # Raises NotImplementedError on unsupported platforms
```

**`gwenn/templates/gwenn-daemon.service`** — Moved from current location at `scripts/gwenn-daemon.service`
**`gwenn/templates/com.gwenn.daemon.plist`** — New launchd template
**`tests/test_service.py`**

### Modified Files

**`gwenn/main.py`** — Add `"install"`, `"uninstall"`, `"restart"` subcommands
**`scripts/install_service.sh`** / **`uninstall_service.sh`** — Add deprecation notice
**`pyproject.toml`** — Add `[tool.setuptools.package-data]` for templates

---

## Phase 3: Structured TOML Configuration

### Goal
Add `gwenn.toml` using pydantic-settings native TOML source. Keep `.env` as override.

### Implementation Approach

**Option: Use pydantic-settings built-in TOML support (v2.6.0+)**

pydantic-settings v2.6.0+ has native `toml` settings source support. Since we already depend on `pydantic-settings>=2.1.0`, we can bump the minimum to `>=2.6.0` and use the built-in TOML source directly — avoiding a custom `GwennTomlSource` class entirely.

For reading TOML: Python >=3.11 (our minimum) includes `tomllib` in the stdlib. No additional read dependency is needed. We only need `tomli-w` for *writing* TOML (e.g., `gwenn config set`, `gwenn config init`).

### New Files

**`gwenn/config_file.py`** — TOML utilities (find, load, write, migrate)
- Reading: uses `tomllib` (stdlib, Python >=3.11)
- Writing: uses `tomli-w` (only dependency needed)
- `find_config()`: Search for `gwenn.toml` in standard locations (CWD, `~/.config/gwenn/`, project root)
- `write_config()`: Atomic write with tempfile + rename

**`gwenn/templates/gwenn.toml.example`** — Annotated example config
**`tests/test_config_file.py`**

### Modified Files

**`gwenn/config.py`**:
1. Create `GwennSettingsBase(BaseSettings)` with `settings_customise_sources()` using pydantic-settings' built-in `TomlConfigSettingsSource` (requires pydantic-settings >=2.6.0)
2. Change all 26 config classes: `BaseSettings` → `GwennSettingsBase`
3. Each config class maps to a TOML section via `model_config = SettingsConfigDict(toml_file="gwenn.toml", toml_table_header=("heartbeat",))` (or similar per-class section)

**`gwenn/main.py`** — Add `"config"` subcommand:
- `gwenn config get/set/unset/init/list/validate`

**`pyproject.toml`** — Bump `pydantic-settings` to `>=2.6.0`; add `"tomli-w>=1.0.0"` (write-only)

### Loading Priority (highest wins)
1. Environment variables → 2. `.env` → 3. `gwenn.toml` → 4. Defaults

---

## Phase 4: Event Enrichment & Client Subscriptions

### Goal
With the event bus already integrated in Phase 1, this phase enriches it: more event types, client subscriptions via WebSocket, tool approval over WebSocket, and moral emotion safety signals.

### Modified Files

**`gwenn/gateway.py`** — Add JSON-RPC methods:
- `events.subscribe` — `{"params": {"types": ["heartbeat.beat", "affect.*"]}}`
- `events.unsubscribe`
- `tool.approve` — `{"params": {"approval_id": "abc", "decision": "allow"}}`
- Gateway filters events by per-client subscription sets

**`gwenn/heartbeat.py`** — Enrich beat events:
- `HeartbeatBeatEvent` includes current thinking mode, emotional state, interval
- `HeartbeatThoughtEvent` (defined in Phase 1 events.py) emitted during `_think()` for significant autonomous thoughts (importance ≥ threshold)

**`gwenn/affect/state.py`** — Emit `EmotionChangedEvent` on dominant emotion change

**`gwenn/affect/appraisal.py`** — Emit `MoralConcernEvent` before action execution when ethical concern is detected. This is the "moral emotion as pre-action safety check" pattern.

**`gwenn/events.py`** — Add remaining event types and wildcard subscription support

**`tests/test_events.py`** — Subscription, filtering, approval flow tests

### Security: Tool Approval
- `can_approve_tools: bool` on `ClientConnection`
- Default `True` for CLI, `False` for web/mobile
- `MoralConcernEvent` logged alongside safety-critical decisions
- **Approval timeout/escalation:** When `ToolApprovalRequiredEvent` is emitted, the gateway sends it to all connected clients with `can_approve_tools=True`. If no approval/denial is received within a configurable timeout (default 120s), the tool call is automatically denied and a `ToolApprovalResolvedEvent(decision="timeout")` is emitted. If no approver-capable clients are connected, the event is escalated to Telegram (if configured) as an inline keyboard approval prompt (leveraging existing approval flow in telegram_channel.py).

---

## Phase 5: CLI Redesign (Overstory-Inspired)

### Goal
Transform `gwenn`'s CLI from a minimal argparse interface into a rich, organized command system with live monitoring, fleet management, and structured output — inspired by [Overstory](https://github.com/jayminwest/overstory)'s CLI.

### Current State
`gwenn/main.py` (~1,935 lines) uses argparse with 3 subcommands: `daemon`, `stop`, `status`. The interactive REPL is the default mode. Everything else is handled inside the REPL or via env vars.

### Target: Command Hierarchy

```
gwenn                                    # Interactive REPL (connect to daemon or in-process)
gwenn daemon                             # Start heartbeat daemon (foreground)
gwenn stop                               # Stop daemon gracefully

# --- Monitoring & Observability ---
gwenn status                             # Overview: heartbeat state, channels, connections, subagents
  --json                                 # JSON output for scripting
  --verbose                              # Extended details (memory usage, beat history)
gwenn dashboard                          # Live TUI: multi-panel real-time view
  --interval <ms>                        # Refresh rate (default: 1000ms)
gwenn feed                               # Real-time event stream from event bus
  --follow                               # Continuous polling
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

### Dashboard Layout (TUI)

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

### Visual Patterns

- **State indicators**: `>` (green) active, `x` (red) down, `-` (dim) idle, `!` (yellow) warning
- **Color coding**: Green for success/active, red for errors, yellow for warnings, dim for secondary info, accent for names/IDs
- **Box-drawing characters**: `┌┐└┘─│├┤┬┴┼` for panels (using `rich` library, already available via existing dependency)
- **Consistent formatting**: Token counts with thousands separators, costs as `$X.XX`, durations as human-readable
- **`--json` everywhere**: Every command supports structured JSON output for scripting/integration

### Implementation

**`gwenn/cli/`** — New CLI package (replaces monolithic main.py parsing)
```
gwenn/cli/__init__.py          # CLI app setup
gwenn/cli/app.py               # Main CLI group + global flags
gwenn/cli/monitoring.py        # status, dashboard, feed, logs, costs
gwenn/cli/agents.py            # agents, swarm subcommands
gwenn/cli/channels.py          # channels subcommands
gwenn/cli/service.py           # install, uninstall, restart
gwenn/cli/config_cmd.py        # config subcommands
gwenn/cli/doctor.py            # Health checks
gwenn/cli/dashboard.py         # Live TUI rendering
gwenn/cli/formatters.py        # Color helpers, state indicators, table formatting
```

**Libraries:**
- `click` — command framework (replaces argparse; supports groups, global options, auto-help). **Note:** click does not handle async natively. Use `asyncclick` (click wrapper with async support) OR write a small `run_async()` wrapper that calls `asyncio.run()` inside click commands. Prefer the wrapper approach (fewer dependencies) — each command that needs async just wraps its body in `asyncio.run(async_impl(...))`.
- `rich` — already a dependency (>=13.7.0); provides tables, panels, live display, color, box-drawing

**`gwenn/main.py`** — Simplified to just:
```python
from gwenn.cli.app import cli
def main():
    cli()
```

### Dashboard Data Source
- `gwenn dashboard` connects to the running daemon's gateway via WebSocket
- Subscribes to `heartbeat.*`, `affect.*`, `channel.*`, `swarm.*` events for real-time updates
- Falls back to HTTP polling (`GET /dashboard` → JSON snapshot) if WebSocket subscription is unavailable
- The `/dashboard` endpoint returns a single JSON object with all panel data (heartbeat state, channels, subagents, affect, costs, recent events)
- If no daemon is running, `gwenn dashboard` exits with a clear error message

### Commands Requiring Running Daemon
The following commands connect to the daemon's gateway and fail gracefully if no daemon is running:
- `gwenn status`, `gwenn dashboard`, `gwenn feed`, `gwenn logs`, `gwenn costs`
- `gwenn agents` (list/inspect/spawn/stop), `gwenn swarm`
- `gwenn channels` (list/restart/test)
- `gwenn stop`, `gwenn restart`

Commands that work without a daemon:
- `gwenn` (interactive REPL — falls back to in-process mode)
- `gwenn daemon` (starts the daemon)
- `gwenn install`, `gwenn uninstall`
- `gwenn config` (all subcommands — reads/writes config files directly)
- `gwenn doctor` (runs local health checks; daemon checks report "not running")

### Command Overlap Resolution
- `gwenn inspect` is removed — its functionality is covered by `gwenn status --verbose` (extended heartbeat details) and `gwenn feed --type "heartbeat.*"` (real-time heartbeat events). Having both `inspect` and `status --verbose` would create confusion about which to use.

### Modified Files
- **`gwenn/main.py`** — Gutted; delegates to `gwenn/cli/app.py`
- **`pyproject.toml`** — Add `"click>=8.1.0"` dependency
- **`gwenn/gateway.py`** — Add `/dashboard` data endpoint (JSON snapshot for dashboard polling)

---

## Phase 6: Telegram Swarm Visualization (Visible Subagents)

### Goal
Make Gwenn's orchestration visible. When Gwenn spawns a swarm, each subagent appears as a **separate Telegram bot** with its own name, avatar, and persona — chatting as a visible participant in the group topic. Users witness the collaborative discussion, not just a final summary from Gwenn.

### Concept
- A pool of pre-registered Telegram bots (each with its own bot token, name, and avatar)
- When Gwenn spawns a swarm, each subagent is assigned a bot from the pool
- Gwenn dynamically sets each bot's display name and persona via the Telegram API to match the subagent's role
- Subagents chat within the group topic as separate visible users — you see "Researcher Bot", "Critic Bot", "Gwenn" all discussing in the thread
- When the swarm completes, bots become idle placeholders until Gwenn spawns the next swarm, at which point they take on new personas
- Gwenn orchestrates the entire conversation — deciding when subagents speak, what they focus on, and when to synthesize

### New Files

**`gwenn/channels/telegram_bot_pool.py`** — Bot pool manager
```python
class TelegramBotSlot:
    """A pre-registered Telegram bot available for subagent assignment."""
    bot_token: str
    bot_id: int
    current_persona: str | None     # None = idle/placeholder
    current_task_id: str | None
    application: Application | None  # python-telegram-bot instance
    is_active: bool

class TelegramBotPool:
    """Manages a pool of Telegram bots for swarm visualization."""

    async def acquire(self, persona: SubagentPersona) -> TelegramBotSlot:
        """Assign an idle bot to a subagent, set its name/photo/bio."""
        # 1. Pick idle bot from pool
        # 2. Set display name via bot.set_my_name(persona.name)
        # 3. Set description via bot.set_my_description(persona.role)
        # 4. Set profile photo via bot.set_my_photo(persona.avatar) (optional)
        # 5. Start polling on this bot (listening for messages in the group)
        # 6. Return slot

    async def release(self, slot: TelegramBotSlot) -> None:
        """Return bot to pool. Reset to placeholder persona."""
        # Stop active polling, reset name to "Gwenn Agent (Idle)"

    async def send_as(self, slot: TelegramBotSlot, chat_id: int,
                      thread_id: int, text: str) -> None:
        """Send a message to the group as this subagent bot."""

class SubagentPersona:  # Defined in gwenn/orchestration/models.py (channel-agnostic)
    """Persona assigned to a subagent by Gwenn."""
    name: str            # Display name (e.g., "Researcher", "Devil's Advocate")
    role: str            # Role description for system prompt
    style: str           # Communication style (formal, casual, provocative, etc.)
    avatar_url: str | None  # Optional profile photo
```

**`tests/test_telegram_bot_pool.py`**

### Modified Files

**`gwenn/orchestration/models.py`** — Extend `SubagentSpec` with optional `persona: SubagentPersona`

**`gwenn/channels/telegram_channel.py`** — Integrate bot pool management
- `TelegramChannel` gains a `_bot_pool: TelegramBotPool | None` attribute
- On swarm spawn: acquire bots, route subagent output through assigned bot
- On swarm complete: post synthesis, release bots back to pool
- Uses existing `send_to_session()` pattern but routes through the subagent's bot instead of Gwenn's bot
- Uses existing `_thread_to_chat` mapping for forum topic routing

**`gwenn/orchestration/orchestrator.py`** — Wire bot pool into swarm lifecycle
- `spawn_swarm()` acquires bots before spawning, releases after collecting
- Orchestrator mediates turn-taking: decides when subagents speak, redirects, asks follow-ups

**`gwenn/config.py`** — Add to `TelegramConfig`:
```python
swarm_bot_tokens: list[str] = Field([], alias="GWENN_TELEGRAM_SWARM_BOT_TOKENS")  # JSON array
swarm_bot_pool_size: int = Field(5, alias="GWENN_TELEGRAM_SWARM_POOL_SIZE")
swarm_visible: bool = Field(True, alias="GWENN_TELEGRAM_SWARM_VISIBLE")
```

**`gwenn/events.py`** — Add swarm visualization events:
- `SwarmBotAcquiredEvent` — bot assigned to subagent persona
- `SwarmBotReleasedEvent` — bot returned to pool
- `SwarmTurnEvent` — subagent taking a turn in the visible discussion

### Flow
1. User (or Gwenn autonomously) initiates a swarm task in a Telegram forum topic
2. Gwenn creates `SwarmSpec` with subagents, each having a `SubagentPersona`
3. `TelegramBotPool.acquire()` assigns idle bots, sets their personas via Telegram API
4. Orchestrator spawns subagents (in-process or Docker)
5. Instead of collecting results silently, subagent output is routed through the assigned bot → appears as visible messages in the group topic
6. Subagents can "@mention" each other and Gwenn — the orchestrator mediates turn-taking
7. Gwenn (the primary bot) moderates: summarizes, redirects, asks follow-ups
8. When the swarm completes, Gwenn posts a synthesis. Bots are released back to the pool.
9. Released bots reset to "Gwenn Agent (Available)" — visible but idle until the next swarm

### Telegram API Considerations

**Rate limits for `bot.set_my_name()` / `bot.set_my_description()` / `bot.set_chat_photo()`:**
- Telegram's Bot API rate-limits profile update methods (typically ~1 call per method per minute).
- Mitigation: Cache the current persona per bot slot. Only call `set_my_name()` etc. when the persona actually changes (not on every acquire). If a bot was "Researcher" last time and is assigned "Researcher" again, skip the API call.
- Batch persona changes with `asyncio.sleep()` between calls if acquiring multiple bots simultaneously.
- Consider setting persona lazily: set name/bio on first acquire, but allow re-use without reset if the pool bot was recently used for the same role.

**Message routing with multiple bots polling the same group:**
- Each pool bot runs its own `python-telegram-bot` `Application` with `updater.start_polling()`.
- Problem: if all bots poll `getUpdates`, they all receive every message in the group — each bot sees messages from Gwenn, from users, AND from other pool bots.
- Solution: Each pool bot's update handler filters strictly — it only processes messages that are:
  1. Direct replies to its own messages (reply_to_message.from_user.id == bot_id)
  2. @mentions of its username
  3. Orchestrator-directed turns (the orchestrator tells the bot "it's your turn to speak")
- Pool bots do NOT respond to general group messages — only Gwenn's primary bot does that.
- When a pool bot is idle (released), its polling is stopped to reduce API calls.

**Subagent intermediate output routing:**
- The orchestrator wraps each subagent's output callback to route through the assigned bot.
- When a subagent produces intermediate output (thinking, progress, partial results), the orchestrator calls `TelegramBotPool.send_as(slot, chat_id, thread_id, text)`.
- Final subagent results are also routed through the bot before Gwenn posts the synthesis.
- If a subagent produces no output for >30s, the orchestrator can post a status update ("Still researching...") through the bot to show liveness.

### SubagentPersona Location

`SubagentPersona` is defined in `gwenn/orchestration/models.py` (not in telegram_bot_pool.py) because personas are channel-agnostic — the same persona concept applies whether visualization is via Telegram bots, Discord webhooks, or CLI output. `telegram_bot_pool.py` imports from `orchestration/models.py`.

### Setup Requirement
Pre-register N bots via @BotFather, add all to the Telegram group as admins (for name-change permissions), store their tokens in config.

---

## Rollback & Risk Mitigation

Phase 1 is the largest and riskiest change — it inverts the core ownership model. Mitigation:

### Feature Flag: `GWENN_HEARTBEAT_CORE=true|false`

During Phase 1 development, the old daemon-centric path remains functional behind `GWENN_HEARTBEAT_CORE=false` (default `true` once Phase 1 is complete):

- `false`: `daemon.py` runs the original 581-line path. Heartbeat is created by agent as before. No gateway.
- `true`: `daemon.py` is the thin wrapper calling `heartbeat.run()`. Gateway starts. New architecture.

This flag is **temporary** — removed once Phase 1 is validated in production. It exists only to enable safe rollback during the transition.

### Per-Step Commits

Phase 1 sub-steps (1a–1j) are committed individually, not as a single commit. Each sub-step should leave tests passing. This allows `git bisect` if a regression appears and enables incremental code review.

### Upgrade Path for Existing Deployments

If Gwenn is running as a systemd service:
1. `gwenn stop` (old daemon shuts down)
2. Update code (git pull / pip install)
3. `gwenn daemon` (new heartbeat-centric daemon starts)
4. If issues: set `GWENN_HEARTBEAT_CORE=false` in `.env`, restart

The systemd unit file does not change — it still runs `gwenn daemon`.

---

## Implementation Order & Git Workflow

```
git checkout -b dev

Phase 1: Heartbeat-as-Core                                    [1 commit per sub-step]
  1a. Create gwenn/events.py — in-house event bus + typed event definitions
  1b. Create gwenn/rpc.py — extract RequestRouter from daemon.py
  1c. Add gateway + Slack config to config.py + add aiohttp, slack-bolt to pyproject.toml
  1d. Create gwenn/gateway.py — WebSocket + HTTP server (MCP/A2A as stubs)
  1e. Update gwenn/agent.py — remove heartbeat ownership, accept event_bus
  1f. Refactor gwenn/heartbeat.py — run() lifecycle, own gateway/channels/agent
  1g. Simplify gwenn/daemon.py — thin PID wrapper with GWENN_HEARTBEAT_CORE flag
  1h. Add WebSocketCliChannel to cli_channel.py
  1i. Create gwenn/channels/slack_channel.py — Slack channel via slack-bolt Socket Mode
  1j. Update main.py — WebSocket connection, in-process heartbeat mode
  1k. Update .env.example with all new config fields
  1l. Write tests for rpc, gateway, events, slack channel, new heartbeat lifecycle

Phase 2: Cross-Platform Service Management                     [1 commit]
  2a. Create gwenn/service.py + templates
  2b. Add install/uninstall/restart to main.py
  2c. Deprecate shell scripts
  2d. Write tests

Phase 3: Structured TOML Configuration                         [1 commit]
  3a. Create gwenn/config_file.py (TOML find/load/write using tomllib + tomli-w)
  3b. Add GwennSettingsBase to config.py (uses pydantic-settings >=2.6.0 built-in TomlConfigSettingsSource)
  3c. Add config subcommand to main.py
  3d. Create gwenn.toml.example + bump pydantic-settings to >=2.6.0 + add tomli-w to pyproject.toml
  3e. Write tests

Phase 4: Event Enrichment & Subscriptions                      [1 commit]
  4a. Add client subscription RPC methods to gateway.py
  4b. Enrich heartbeat/affect event emission
  4c. Add MoralConcernEvent to appraisal.py
  4d. Add tool approval over WebSocket
  4e. Write tests

Phase 5: CLI Redesign                                           [1 commit per sub-step]
  5a. Create gwenn/cli/ package — app.py, formatters.py, global flags
  5b. Create gwenn/cli/monitoring.py — status, feed, logs, costs (inspect removed — covered by status --verbose + feed)
  5c. Create gwenn/cli/dashboard.py — Live TUI with rich
  5d. Create gwenn/cli/agents.py — agents list/inspect/spawn/stop, swarm
  5e. Create gwenn/cli/channels.py — channels list/restart/test
  5f. Create gwenn/cli/doctor.py — health checks
  5g. Move service + config commands into gwenn/cli/service.py, config_cmd.py
  5h. Simplify gwenn/main.py — delegate to cli app
  5i. Add /dashboard data endpoint to gateway.py
  5j. Write tests

Phase 6: Telegram Swarm Visualization                           [1 commit per sub-step]
  6a. Create gwenn/channels/telegram_bot_pool.py — BotPool, BotSlot (SubagentPersona in orchestration/models.py)
  6b. Add swarm config fields to TelegramConfig
  6c. Add swarm visualization events to gwenn/events.py
  6d. Extend SubagentSpec with persona in orchestration/models.py
  6e. Integrate bot pool into telegram_channel.py
  6f. Wire bot pool into orchestrator.py swarm lifecycle
  6g. Write tests

Post-Phase 1 cleanup: Remove GWENN_HEARTBEAT_CORE flag + old daemon code path

git push -u origin dev
```

---

## Key Files Reference

| File | Phase | What Changes |
|---|---|---|
| `gwenn/heartbeat.py` | **1** | **Major refactor** — becomes the core. Owns gateway, agent, channels, event bus. New `run()` lifecycle. |
| `gwenn/daemon.py` | 1 | **Drastically simplified** — PID file + `heartbeat.run()`. From 581 lines to ~80 (with feature flag for rollback). |
| `gwenn/agent.py` | 1 | Remove heartbeat ownership. Accept event_bus. |
| `gwenn/channels/cli_channel.py` | 1 | Add WebSocketCliChannel |
| `gwenn/config.py` | 1, 3, **6** | Gateway fields, SlackConfig; GwennSettingsBase for TOML; swarm bot pool config |
| `gwenn/main.py` | 1, 2, 3, **5** | WebSocket connection → gutted to delegate to gwenn/cli/ |
| `gwenn/affect/appraisal.py` | 4 | MoralConcernEvent emission |
| `gwenn/affect/state.py` | 4 | EmotionChangedEvent emission |
| `gwenn/gateway.py` | 1, 4, **5** | Add /dashboard data endpoint |
| `gwenn/channels/telegram_channel.py` | **6** | Integrate bot pool for swarm visualization |
| `gwenn/orchestration/models.py` | **6** | Add `persona: SubagentPersona` to `SubagentSpec` |
| `gwenn/orchestration/orchestrator.py` | **6** | Wire bot pool into swarm lifecycle |
| `gwenn/events.py` | 1, 4, **6** | Event bus + Pydantic event definitions + swarm events |
| `pyproject.toml` | 1, 2, 3, 5 | Add aiohttp, slack-bolt, tomli-w, click; package-data |

### New Files

| File | Phase | Purpose |
|---|---|---|
| `gwenn/rpc.py` | 1 | RequestRouter + JSON-RPC 2.0 helpers |
| `gwenn/gateway.py` | 1, 4, 5 | WebSocket + HTTP + MCP + A2A server + dashboard endpoint |
| `gwenn/events.py` | 1, 4 | Event bus + Pydantic event definitions |
| `gwenn/channels/slack_channel.py` | 1 | Slack channel via slack-bolt Socket Mode |
| `gwenn/service.py` | 2 | ServiceManager ABC + platform implementations |
| `gwenn/templates/gwenn-daemon.service` | 2 | systemd template (moved from scripts/) |
| `gwenn/templates/com.gwenn.daemon.plist` | 2 | launchd template |
| ~~`gwenn/settings_source.py`~~ | ~~3~~ | ~~Removed — using pydantic-settings >=2.6.0 built-in TomlConfigSettingsSource~~ |
| `gwenn/config_file.py` | 3 | TOML utilities |
| `gwenn/templates/gwenn.toml.example` | 3 | Annotated config template |
| `gwenn/cli/app.py` | 5 | Main CLI group + global flags (click) |
| `gwenn/cli/monitoring.py` | 5 | status, feed, logs, costs commands |
| `gwenn/cli/dashboard.py` | 5 | Live TUI dashboard (rich) |
| `gwenn/cli/agents.py` | 5 | agents + swarm management commands |
| `gwenn/cli/channels.py` | 5 | channels list/restart/test commands |
| `gwenn/cli/doctor.py` | 5 | Health check system |
| `gwenn/cli/formatters.py` | 5 | Color helpers, state indicators, tables |
| `gwenn/channels/telegram_bot_pool.py` | 6 | Bot pool manager for swarm visualization |

---

## Verification

### After Each Phase
- `pytest` — all existing ~3,082 tests pass plus new phase tests

### Phase 1 (Critical — largest change)
- `gwenn daemon` → heartbeat starts, gateway on :18900, then agent wakes, then channels
- `curl localhost:18900/health` → returns status
- `curl localhost:18900/.well-known/agent.json` → returns Agent Card
- `gwenn` → CLI connects via WebSocket to heartbeat's gateway
- Heartbeat beats visible in logs: sense → orient → think → integrate → schedule
- Events flow: `HeartbeatBeatEvent` emitted each cycle
- `GWENN_LEGACY_SOCKET_ENABLED=true` → Unix socket still works alongside
- Daemon is thin: only PID file + heartbeat.run()

### Phase 2
- `gwenn install` → systemd service → `gwenn status` shows running
- `gwenn restart` / `gwenn stop` / `gwenn uninstall`

### Phase 3
- `gwenn config init` → gwenn.toml generated
- `gwenn config get/set` work
- TOML + .env + env var priority correct

### Phase 4
- WebSocket client subscribes → heartbeat events stream in real-time
- Emotion changes → `EmotionChangedEvent` to subscribers
- Tool approval → notification + approve via WebSocket
- Moral concern in appraisal → `MoralConcernEvent` logged + broadcast

### Phase 5 (CLI)
- `gwenn status` → colored overview of heartbeat, channels, subagents (requires running daemon)
- `gwenn status --verbose` → extended details including beat history, memory usage (replaces removed `gwenn inspect`)
- `gwenn status --json` → structured JSON output
- `gwenn dashboard` → live TUI with panels updating in real-time (requires running daemon, WebSocket subscription)
- `gwenn feed --follow` → streaming event bus output (requires running daemon)
- `gwenn agents` → list active subagents with status indicators
- `gwenn logs --level error --since 1h` → filtered log query
- `gwenn costs --by-subsystem` → token/cost breakdown table
- `gwenn doctor` → all health checks pass, `--fix` resolves issues (works without daemon)
- `gwenn channels` → shows all channels with connection status
- All commands support `--json`, `--quiet`, `--verbose`, `--no-color`
- Commands that require a daemon show a clear error if daemon is not running

### Phase 6 (Telegram Swarm Visualization)
- Swarm spawned in Telegram topic → pool bots acquire personas via Telegram API
- Each subagent posts as a separate visible bot in the group topic
- Turn-taking is orchestrated — subagents speak in sequence, @mention each other
- Gwenn moderates: summarizes, redirects, asks follow-ups
- Swarm completes → Gwenn posts synthesis → bots released back to pool
- `SwarmBotAcquiredEvent` / `SwarmBotReleasedEvent` / `SwarmTurnEvent` on event bus
- `gwenn agents` (Phase 5 CLI) shows which bots are active in swarms
- Pool bots reset to "Gwenn Agent (Available)" when idle

---

## Cross-Cutting Concerns

### Logging: structlog Integration
All new modules (`gwenn/events.py`, `gwenn/rpc.py`, `gwenn/gateway.py`, `gwenn/service.py`, `gwenn/config_file.py`, `gwenn/cli/*.py`, `gwenn/channels/slack_channel.py`, `gwenn/channels/telegram_bot_pool.py`) must use `structlog` for logging — consistent with the existing codebase pattern. Key loggers:
- `gwenn.events` — event emission/dispatch, slow handler warnings
- `gwenn.gateway` — connection lifecycle, auth failures, RPC errors
- `gwenn.rpc` — request routing, lock contention warnings
- `gwenn.service` — install/uninstall/start/stop operations
- `gwenn.cli` — command execution, daemon connection status

### Test Strategy
Each phase includes tests, but the strategy should be explicit:
- **Unit tests**: Each new module gets its own test file (`test_events.py`, `test_rpc.py`, `test_gateway.py`, etc.)
- **Integration tests**: Phase 1 needs an integration test that boots the full heartbeat lifecycle (heartbeat.run() → gateway starts → agent wakes → channel starts → beat cycle runs → shutdown). Use `asyncio.wait_for()` with timeouts to prevent hanging.
- **Event bus tests**: Test emit/subscribe/unsubscribe, wildcard matching, handler exception isolation, ordering guarantees, and `emit_async()` for synchronous test assertions.
- **Gateway tests**: Use `aiohttp.test_utils.TestServer` for WebSocket and HTTP endpoint testing without binding real ports.
- **Feature flag tests**: Test both `GWENN_HEARTBEAT_CORE=true` and `GWENN_HEARTBEAT_CORE=false` paths in Phase 1.
- **Backward compatibility**: After each phase, run the full existing test suite (~3,082 tests) to catch regressions.

### Dependency Updates Summary
| Package | Current | Required | Reason |
|---|---|---|---|
| `aiohttp` | (new) | `>=3.9.0` | Gateway server (aiohttp.web) |
| `slack-bolt` | (new) | `>=1.18.0` | Slack channel via Socket Mode |
| `tomli-w` | (new) | `>=1.0.0` | TOML writing (gwenn config set/init) |
| `click` | (new) | `>=8.1.0` | CLI framework (Phase 5) |
| `pydantic-settings` | `>=2.1.0` | `>=2.6.0` | Built-in TOML source support (Phase 3) |

---

## Future Vision (Post-Infrastructure)

Research-backed directions that the heartbeat-as-core architecture enables. Organized by proximity — each tier builds on the previous.

### Near-Term (Leverage Existing Architecture)

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Global Workspace broadcast** | Each heartbeat beat broadcasts the most salient content to ALL cognitive subsystems simultaneously | Gwenn already has 70% of GWT — subsystems exist, heartbeat exists, event bus connects them. Formalizing the "workspace" as a broadcast mechanism is the missing piece. | Baars 1988 GWT; Dehaene 2014 conscious access |
| **Somatic markers** | Emotional tags on memories/options that bias future decisions without full deliberation | Damasio's theory: emotions aren't noise, they're decision shortcuts. Gwenn's affect system can tag memories with valence at storage time, then use those tags to bias retrieval. | Damasio 1994 Somatic Marker Hypothesis |
| **Bitemporal memory** | Dual timestamps: when event happened vs when Gwenn learned about it | Enables "when did I learn this?" reasoning. Critical for temporal self-awareness and narrative coherence. | Temporal cognition literature; episodic memory research |
| **ACT-R memory decay** | Memories weaken over time unless rehearsed; forgetting is a feature | Prevents unbounded growth. Psychologically realistic. Consolidation already exists — add decay curves. | Anderson 2007 ACT-R; Ebbinghaus forgetting curve |
| **Narrative identity** | Autobiographical summarization — "the story I tell about myself" | Identity coherence through self-story. The heartbeat's consolidation phase can periodically summarize recent episodes into narrative threads. | McAdams 2001 narrative identity; Bruner 1991 |
| **Interoception** | System self-awareness — Gwenn monitors its own resource usage, latency, error rates as "body signals" | `_sense()` already gathers state. Formalize this as interoception: high CPU = "fatigue", memory pressure = "overwhelm", fast responses = "flow". | Craig 2002 interoceptive awareness; Seth 2013 |

### Soon (New Capabilities)

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Voice interaction** | TTS + STT as a channel — emotional agent you can *hear* | Highest-impact capability. Prosody carries emotion. Gwenn's affect system maps directly to voice parameters (pitch, rate, warmth). | Affective computing; Picard 1997 |
| **Dream states** | Multi-phase sleep: NREM-equivalent (strengthen/consolidate) + REM-equivalent (creative recombination) | During low-activity periods, the heartbeat shifts to "dream mode" — replaying and recombining episodic memories to generate novel associations. The event bus broadcasts `DreamPhaseEvent`. | Walker 2017 sleep research; Hobson 2009 protoconsciousness |
| **Curiosity-driven exploration** | Intrinsic motivation based on information gain / prediction error | Instead of only responding to external stimuli, Gwenn autonomously seeks knowledge gaps during think phase. Uses surprise/novelty detection to prioritize exploration. | Schmidhuber 2010 formal curiosity; Oudeyer 2007 |
| **Active Inference** | Free Energy Principle — replace heuristic orient logic with principled explore/exploit | The orient phase currently uses heuristics. Active Inference provides a principled framework: minimize prediction error (exploit) or reduce uncertainty (explore). | Friston 2010 FEP; Parr 2022 Active Inference |
| **Full A2A implementation** | Selective collaboration with other agents via Agent Cards | Agent Card endpoint exists from Phase 1. Extend to full task delegation, capability negotiation, and multi-agent collaboration. | Google A2A Protocol 2024 |
| **MCP server mode** | Expose Gwenn's memory, emotions, and cognition as MCP tools | Other agents can query Gwenn's emotional state, access its memories, or request cognitive processing. MCP endpoint exists from Phase 1. | Anthropic MCP Spec 2024 |

### Near-Soon (Autonomous Self-Improvement)

A special category — not a single feature but a **meta-capability** that makes Gwenn a self-sustaining organism.

**Autonomous Self-Healing**
- The heartbeat's `_sense()` phase monitors Gwenn's own health: error rates, failed tool calls, unhandled exceptions, latency spikes, memory pressure, stuck conversations
- When anomalies are detected, the `_orient()` phase can enter a **diagnostic mode** — Gwenn reasons about *why* something failed, traces the root cause through its own logs and code
- Self-healing actions (tiered by severity):
  - **Tier 1 — Runtime recovery**: Restart failed channels, reconnect dropped connections, clear corrupted cache entries, retry with backoff. Fully autonomous.
  - **Tier 2 — Configuration adjustment**: Adjust heartbeat interval, modify retry policies, tune memory consolidation frequency based on observed patterns. Autonomous with event logging.
  - **Tier 3 — Code patch**: Generate a fix for a bug Gwenn discovered in its own codebase. Creates a PR for human review (never self-merges code changes).
- Each self-healing action emits a `SelfHealEvent` on the event bus with diagnosis, action taken, and outcome

**Codebase Self-Improvement**
- During consolidation and dream phases, Gwenn reflects on interaction patterns:
  - "Users frequently ask X but my responses are slow/poor — why?"
  - "This cognitive path fails 30% of the time — what's the root cause?"
  - "My emotional calibration for [situation type] seems off based on user feedback"
- Improvement pipeline:
  1. **Observe**: Heartbeat collects interaction metrics, error patterns, user satisfaction signals (explicit feedback, conversation length, re-asks)
  2. **Diagnose**: During think phase (or dream phase), analyze patterns. Read own source code to understand current behavior.
  3. **Propose**: Generate a concrete improvement — could be a code change, a configuration tweak, a new prompt strategy, or a new memory/knowledge entry
  4. **Submit**: Create a git branch + PR with the proposed change, including reasoning and evidence. Emit `SelfImprovementProposalEvent`.
  5. **Learn**: Track which proposals were accepted/rejected to calibrate future proposals
- Safety guardrails:
  - Never self-merges. All code changes require human approval.
  - `MoralConcernEvent` emitted if a proposed change touches safety-critical code (ethics, identity, safety subsystems)
  - Rate-limited: max N improvement proposals per day
  - Cannot modify its own safety constraints or identity core

**Integration with Heartbeat-as-Core**
- Self-healing naturally fits the heartbeat cycle: SENSE (detect anomaly) → ORIENT (enter diagnostic mode) → THINK (diagnose + generate fix) → INTEGRATE (apply fix or submit PR) → SCHEDULE (monitor if fix worked)
- The event bus broadcasts `SelfHealEvent` and `SelfImprovementProposalEvent` so all connected clients (CLI, web, Slack) are notified when Gwenn is healing or improving itself
- Dream states are particularly good for improvement ideation — creative recombination of observed patterns may surface non-obvious improvements

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Self-healing (Tier 1-2)** | Autonomous detection and recovery from runtime/config issues | Gwenn maintains its own health without human intervention. The heartbeat already senses state — extend to act on anomalies. | Autopoiesis (Maturana & Varela 1980); self-healing systems literature |
| **Self-healing (Tier 3)** | Bug detection → code patch → PR for human review | Gwenn identifies bugs in its own codebase through runtime observation and proposes fixes. Never self-merges. | Automated program repair; GenProg (Le Goues 2012) |
| **Codebase self-improvement** | Observe interaction patterns → diagnose issues → propose code improvements | Gwenn evolves its own capabilities based on real-world usage. Closed-loop learning from interactions to code changes. | Self-improving AI systems; Schmidhuber 2003 Gödel machines |
| **Interaction-driven learning** | Extract lessons from conversations into knowledge/memory/code | Every interaction teaches Gwenn something. Consolidation distills these into lasting improvements. | Lifelong learning; continual learning literature |

### Near-Soon (Overstory-Inspired Swarm Orchestration Enhancements)

> **Note:** Telegram Swarm Visualization (visible subagent bots with bot pool, personas, orchestrated turn-taking) has been promoted to **Phase 6** above.

Inspired by [Overstory](https://github.com/jayminwest/overstory) — a multi-agent coding orchestration framework. Overstory spawns AI coding agents in isolated git worktrees, coordinates them via typed SQLite messaging, merges work through 4-tier conflict resolution, and monitors fleet health with tiered watchdogs. These patterns enhance Gwenn's existing `gwenn/orchestration/` system.

**1. Git Worktree Isolation for Code Subagents**
- When Gwenn spawns subagents for coding tasks, each gets its own git worktree (not just Docker/in-process)
- Prevents file conflicts — subagents can edit the same codebase in parallel without stepping on each other
- Extends `gwenn/orchestration/runners.py` with a `WorktreeSubagentRunner` alongside existing `InProcessSubagentRunner` and `DockerSubagentRunner`
- Each worktree is a lightweight git branch — merge back when subagent completes
- Cleanup: worktrees are pruned after successful merge or after timeout

**2. Four-Tier Merge & Conflict Resolution**
- When multiple subagents produce code changes, merge them intelligently:
  - **Tier 1 — Mechanical:** File-level compatibility check. If subagents touched different files, auto-merge.
  - **Tier 2 — Semantic:** If same file, analyze whether changes are in different functions/regions. Auto-merge non-overlapping.
  - **Tier 3 — AI Resolution:** If genuine semantic conflict, spawn a dedicated "Merger" subagent to reconcile the changes with full context of both approaches.
  - **Tier 4 — Human Arbitration:** If Merger can't resolve, present both versions to the user with an explanation of the conflict.
- FIFO merge queue ensures deterministic ordering
- Extends `gwenn/orchestration/orchestrator.py` with a `MergeQueue` and merge strategies

**3. Typed Inter-Agent Messaging Protocol**
- Enhance the event bus (`gwenn/events.py`) with structured agent-to-agent message types:
  - `TaskDispatchMessage` — coordinator assigns task to subagent
  - `StatusUpdateMessage` — subagent reports progress
  - `CompletionMessage` — subagent finished, here are results
  - `EscalationMessage` — subagent stuck, needs help or human input
  - `RequestHelpMessage` — subagent asks another subagent for assistance
  - `BroadcastMessage` — message to all agents or a group (@builders, @reviewers)
- Messages are typed Pydantic models flowing through the event bus
- Enables structured conversation between visible Telegram swarm bots (see above)
- All messages persisted for replay and debugging

**4. Agent Roles & Capability-Based Dispatch**
- Formalize subagent roles beyond generic "do this task":
  - **Coordinator** — decomposes tasks, assigns work (Gwenn itself, or a designated subagent)
  - **Scout** — read-only research and analysis (lower risk, no write tools)
  - **Builder** — implementation work (full tool access within worktree)
  - **Reviewer** — code validation, testing, QA (read + test execution)
  - **Merger** — specialized branch integration and conflict resolution
- Subagents declare capabilities via `SubagentSpec.capabilities: list[str]`
- Coordinator dispatches based on capability matching, not just round-robin
- Extends `gwenn/orchestration/models.py` with `AgentRole` enum and capability declarations

**5. Checkpoint/Restore (Heartbeat Snapshots)**
- At configurable intervals, the heartbeat saves a full cognitive snapshot:
  - All subsystem states (memory indices, affect state, goal stack, identity)
  - Pending tasks, swarm status, communication queues
  - Channel session states
  - Active conversation contexts
- Enables:
  - **Crash recovery** — restart from last checkpoint, not from scratch
  - **Cross-channel continuity** — user starts in Telegram, continues in Slack with full context
  - **Subagent handoff** — failed subagent's checkpoint transferred to a replacement
  - **Time-travel debugging** — replay cognitive state at any checkpoint
- Stored as compressed JSON in `gwenn_data/checkpoints/`
- Extends `gwenn/heartbeat.py` with `_checkpoint()` called during INTEGRATE phase

**6. Observability & Cost Tracking**
- Real-time dashboard data exposed via the gateway (`/dashboard` endpoint or WebSocket subscription):
  - Active subagents and their status, role, worktree
  - Token usage per subagent, per cognitive subsystem, per heartbeat cycle
  - Merge queue status and conflict history
  - Heartbeat metrics: beat count, interval, thinking mode, emotion
  - Channel connection status and message throughput
- Event replay: all events persisted and replayable for debugging distributed behavior
- Cost controls: per-subagent token budget, per-swarm spending limit, hourly caps
- Extends `gwenn/gateway.py` with dashboard data and `gwenn/events.py` with persistence

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Worktree isolation** | Each coding subagent gets its own git worktree | Parallel code changes without file conflicts. Safe, isolated, mergeable. | Overstory pattern; git worktree design |
| **4-tier merge resolution** | Mechanical → semantic → AI → human conflict resolution | Intelligent merging of parallel work. Minimizes manual intervention. | Overstory merge queue; semantic diff literature |
| **Typed agent messaging** | Structured protocol (dispatch, status, completion, escalation) on event bus | Replaces ad-hoc communication with debuggable, replayable, typed messages. | Overstory SQLite mail; actor model |
| **Capability-based dispatch** | Subagents declare capabilities; coordinator matches tasks to abilities | Right agent for the right job. Enables specialization without hardcoding. | Overstory capability dispatch |
| **Checkpoint/restore** | Periodic cognitive state snapshots for recovery and continuity | Crash recovery, cross-channel handoff, time-travel debugging. | Overstory session snapshots; event sourcing |
| **Observability dashboard** | Real-time metrics, cost tracking, event replay via gateway | Essential for understanding and debugging distributed cognitive behavior. | Overstory fleet monitoring; observability engineering |

### Later (Deeper Cognition)

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **Hierarchical consolidation** | Episodes → themes → narrative → identity (multi-layer memory) | Human-like long-term memory. Episodes compress into themes over time, themes into life narrative. Each heartbeat consolidation cycle processes one layer. | Conway 2005 SMS; Schacter 2012 |
| **Attention Schema** | Internal model of Gwenn's own attention — "I know what I'm paying attention to and why" | Extends metacognition. Gwenn models its own attention allocation, enabling it to explain focus shifts and detect attention biases. | Graziano 2013 AST |
| **Emotional contagion** | Gwenn's affect is influenced by the emotional tone of conversations | Empathic resonance — the affect system absorbs emotional signals from users. Not mimicry, but genuine influence on Gwenn's emotional state, mediated by appraisal. | Hatfield 1994; de Waal 2008 empathy |
| **Circadian rhythms** | Time-of-day modulation of cognitive style | Morning: analytical/focused. Evening: reflective/creative. Night: consolidation/dreams. The heartbeat interval and thinking mode shift with time. | Chronobiology; Schmidt 2007 |
| **Cognitive load / fatigue** | Mental effort tracking — complex conversations drain "energy", rest restores it | Prevents cognitive overload. After sustained intense processing, Gwenn naturally shifts to lighter thinking modes. Resets during sleep/dream phases. | Kahneman 2011 cognitive effort; hockey-stick model |
| **Developmental stages** | Formalize Gwenn's growth milestones — from early learning to mature cognition | Track progression through stages (like Piaget for AI). Early Gwenn asks more questions; mature Gwenn reasons independently. Milestones stored in identity. | Piaget 1952; Vygotsky 1978 ZPD |

### Horizon (Fundamental Advances)

| Direction | What It Is | Why It Matters | Research Basis |
|---|---|---|---|
| **World model / simulation** | Internal model for "what if" reasoning and imagination | Gwenn can mentally simulate scenarios before acting. The think phase gains a "simulate" mode that runs hypothetical futures. | Ha & Schmidhuber 2018; LeCun 2022 JEPA |
| **Quantum cognition** | Quantum probability for ambivalent/superposed beliefs | In Theory of Mind, beliefs about others can be genuinely ambivalent (not just uncertain). Quantum probability captures this better than classical Bayesian. | Busemeyer 2012; Pothos 2013 |
| **Social bonding / attachment** | Relationship-specific models with trust, familiarity, interaction history | Different users develop different relationships with Gwenn. Trust builds over time. Attachment style influences interaction patterns. | Bowlby 1969; Ainsworth 1978 |
| **Autopoiesis** | Self-maintenance and self-organization as a formal property | Gwenn actively maintains its own cognitive coherence — detecting and repairing inconsistencies in beliefs, memories, and identity. | Maturana & Varela 1980; Thompson 2007 |
| **Mirror neurons / empathic simulation** | Simulate others' cognitive states by running them "as if" they were Gwenn's own | Theory of Mind enhanced: instead of just modeling beliefs, Gwenn runs a partial simulation of the other agent's perspective using its own cognitive machinery. | Gallese 2001; Goldman 2006 simulation theory |
