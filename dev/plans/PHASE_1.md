# Phase 1: Heartbeat-as-Core Refactor

**Risk: HIGH** — This inverts the core ownership model. The heartbeat becomes the top-level orchestrator, owning the gateway, event bus, channels, and agent lifecycle.

**Prerequisites:** None (this is the first phase).

---

## Goal

Make the Heartbeat the top-level orchestrator. It owns the gateway, event bus, channels, and agent lifecycle. The daemon becomes a thin PID-file wrapper. The event bus is integral from day one.

---

## The Heartbeat Lifecycle: `run()`

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

---

## New Files

### `gwenn/events.py` — The Nervous System

In-house typed event bus (~150 lines). No external dependency — avoids maturity risk for a core component.

```python
from pydantic import BaseModel

class GwennEvent(BaseModel):
    """Base class for all typed events."""
    event_type: str = ""  # Auto-populated from class name if not set

    def model_post_init(self, __context):
        if not self.event_type:
            # Convert "HeartbeatBeatEvent" → "heartbeat.beat"
            name = type(self).__name__.removesuffix("Event")
            import re
            parts = re.findall(r'[A-Z][a-z]+', name)
            self.event_type = ".".join(p.lower() for p in parts)

class EventBus:
    """Minimal async event bus with typed events and wildcard subscriptions.

    Concurrency model: fire-and-forget via asyncio.create_task().
    - emit() enqueues — non-blocking, sync-safe
    - A dispatcher task dequeues and fans out to matching handlers
    - Handler exceptions are logged (structlog) but do not propagate
    - Ordering guarantee: events are dispatched in emission order per-pattern
    """
    def emit(self, event: GwennEvent) -> None: ...          # Enqueue — non-blocking
    async def emit_async(self, event: GwennEvent) -> None: ...  # Await dispatch (for tests)
    def subscribe(self, pattern: str, handler: Callable) -> str: ...
    def unsubscribe(self, subscription_id: str) -> None: ...
    async def start(self) -> None: ...   # Start dispatcher task
    async def stop(self) -> None: ...    # Drain queue, cancel dispatcher
    # pattern supports wildcards: "heartbeat.*", "affect.*", "*"


# --- Event Definitions ---
# NOTE: Set event_type explicitly on classes whose CamelCase splitting
# would produce wrong results (e.g., "PIIRedactionEvent" would auto-derive
# as "p.i.i.redaction"). For standard names, auto-derivation works fine.

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

**Implementation notes:**
- The `~80 lines` estimate from the original plan is too low. With typed subscriptions, wildcard matching (`fnmatch`-style), exception isolation, ordering guarantees, start/stop lifecycle, and both `emit()` (sync-safe enqueue) and `emit_async()` (await dispatch for tests), realistic size is **120-180 lines**.
- Use `asyncio.Queue` internally for decoupling. The dispatcher task runs `queue.get()` in a loop, matches event_type against subscriber patterns, and `asyncio.create_task()` for each matching handler.
- For testing: `emit_async()` awaits until the event is fully dispatched (all handlers invoked), enabling synchronous test assertions.

### `gwenn/rpc.py` — Transport-Independent Request Routing

Extract `_dispatch()` from `daemon.py:349-477` into `RequestRouter`.

```python
class RequestRouter:
    def __init__(self, agent, session_store, event_bus, respond_lock, session_locks_factory):
        self._agent = agent
        self._session_store = session_store
        self._event_bus = event_bus
        self._respond_lock = respond_lock          # Global lock (for legacy socket compat)
        self._get_session_lock = session_locks_factory  # Per-session locks (for channels/WS)
```

**Key design decisions:**

1. **Dual-lock pattern**: The current codebase has TWO lock patterns that must be preserved:
   - `agent._respond_lock` — global lock used by daemon socket and heartbeat
   - `agent._session_respond_locks` — per-session locks used by channels for concurrent session handling

   The `RequestRouter` accepts BOTH. When a `chat` request includes a `session_id`, the router uses a per-session lock (allowing concurrent sessions). When no session_id is provided (legacy socket), it falls back to the global `_respond_lock`.

2. **JSON-RPC 2.0 helpers**: `make_response()`, `make_error()`, `make_notification()`

3. **Streaming responses**: The `chat` RPC method returns an initial `{"id": ..., "result": {"status": "processing"}}` immediately, then sends progress as JSON-RPC notifications (`chat.token`, `chat.thinking`, `chat.complete`) over the same WebSocket connection. **Implementation note**: Currently `agent.respond()` returns a single `AgentResponse`. To stream, the router needs to pass a callback to `respond()` or the agentic loop. The `on_token` callback pattern from `AgenticLoop.run()` can be wired through — investigate the existing callback parameters (`_on_tool_call`, `_on_tool_result`, `_on_approval_request`) as the model.

4. **10 RPC methods** (migrated from daemon.py `_dispatch()`):
   - `ping` → `pong`
   - `chat` → `response` (under lock, with streaming notifications)
   - `status` → `status_response`
   - `heartbeat_status` → `heartbeat_response`
   - `list_sessions` → `sessions_list`
   - `load_session` → `session_loaded`
   - `reset_session` → `session_reset`
   - `runtime_info` → `runtime_info_response`
   - `stop` → `ack_stop`
   - Unknown → `error`

### `gwenn/gateway.py` — The Circulatory System

```python
class GatewayServer:
    def __init__(self, config, router: RequestRouter, event_bus: EventBus): ...
    async def start(self, host, port) -> None: ...
    async def stop(self) -> None: ...

    # Routes:
    # GET  /ws                        — WebSocket upgrade (full implementation)
    # GET  /health                    — Health check (full implementation)
    # POST /mcp                       — MCP Streamable HTTP (stub: returns "not implemented")
    # GET  /mcp                       — MCP SSE stream (stub)
    # GET  /.well-known/agent.json    — A2A Agent Card (stub: returns static card)
    # MCP/A2A stubs gated behind mcp_server_enabled/a2a_enabled (both default False)

@dataclass
class ClientConnection:
    conn_id: str
    ws: web.WebSocketResponse
    client_type: str          # "cli", "web", "mobile", "mcp"
    authenticated: bool
    session_id: str           # For per-session lock routing
    history: list[dict]
    subscriptions: set[str]
    can_approve_tools: bool   # True for CLI, False for web/mobile
```

**Implementation notes:**
- Uses `aiohttp.web.Application` + `aiohttp.web.AppRunner` + `aiohttp.web.TCPSite`
- WebSocket and HTTP share the same `Application` on the same port
- `aiohttp` is already installed (pulled by `discord.py`) — zero new packages

**Connection lifecycle:**
- On WebSocket connect: create `ClientConnection`, add to active set
- First message must be `auth` with HMAC token (same as current Unix socket auth via `config.daemon.auth_token`)
- On disconnect: remove from active set, auto-save session via `SessionStore`
- Idle timeout: configurable, matches current `_connection_timeout` behavior from daemon.py
- Graceful shutdown: stop accepting new connections, send close frames, wait for in-flight requests (up to N seconds), force-close

**Security:**
- Bind `127.0.0.1` only by default
- Refuse to start on non-localhost without TLS (or emit loud warning)
- Per-IP connection limit (default 10)
- `/health` is unauthenticated (for monitoring)
- `/.well-known/agent.json` is unauthenticated (A2A spec requires public discovery)
- All other endpoints require authentication

### `gwenn/channels/slack_channel.py` — Slack Channel

```python
class SlackChannel(BaseChannel):
    """Slack channel via slack-bolt Socket Mode."""
```

- Uses `slack_bolt.async_app.AsyncApp` with `AsyncSocketModeHandler` for async compatibility with Gwenn's asyncio architecture
- **Async Socket Mode requires `aiohttp`** as the transport adapter — already a dependency from Phase 1
- Maps Slack threads to Gwenn conversation sessions (similar to Telegram's `_thread_to_chat`)
- Emits `ChannelMessageEvent` on the event bus
- Converts Gwenn's markdown to Slack Block Kit for rich formatting
- Handles: messages, threads, reactions, slash commands, app mentions

**Slack config (added to `gwenn/config.py`):**
```python
class SlackConfig(BaseSettings):
    enabled: bool = Field(False, alias="GWENN_SLACK_ENABLED")
    bot_token: str | None = Field(None, alias="GWENN_SLACK_BOT_TOKEN")      # xoxb-
    app_token: str | None = Field(None, alias="GWENN_SLACK_APP_TOKEN")      # xapp- with connections:write scope
    # signing_secret NOT needed for Socket Mode (only for HTTP Events API)
```

**Setup requirement:** The `GWENN_SLACK_APP_TOKEN` must have the `connections:write` scope. This is the most common setup error with Socket Mode.

### Test Files

- `tests/test_events.py` — event bus unit tests
- `tests/test_rpc.py` — request router unit tests
- `tests/test_gateway.py` — WebSocket + HTTP endpoint tests via `aiohttp.test_utils.TestServer`
- `tests/test_slack_channel.py` — Slack channel unit tests
- `tests/test_heartbeat_lifecycle.py` — Integration test: full `heartbeat.run()` lifecycle with mocked CognitiveEngine

---

## Modified Files

### `gwenn/heartbeat.py` — MAJOR REFACTOR (the core change)

**Current**: `__init__(config: HeartbeatConfig, agent: SentientAgent)` — agent creates heartbeat.
**New**: `__init__(config: GwennConfig)` — heartbeat creates agent.

**All changes:**

| Change | Details |
|--------|---------|
| `__init__` | Accept `GwennConfig` (not just `HeartbeatConfig`). Create event bus. No agent yet. |
| New `run()` | Full lifecycle method (see above). |
| New `_start_gateway()` | Create `GatewayServer`, start it. |
| New `_wake_up()` | Create `SentientAgent(config)`, call `agent.initialize()`. Wire event bus. Expose private APIs (see below). **Error handling:** If `initialize()` fails (e.g., `CognitiveEngineInitError`), log error, set gateway health to "degraded", and either retry with backoff or exit cleanly. |
| New `_start_channels()` | Move channel startup from daemon.py lines 137-144 here. **Must call `build_channels()` and `run_channels_until_shutdown()` from `gwenn/channels/startup.py`** — or adapt their logic. Preserve the `done_callback` crash monitoring pattern (`_on_channel_task_done`). Also move `_is_nonfatal_channel_error()` and `_redact_channel_error()` static methods. |
| `_loop()` → `_live()` | Same logic but now emits `HeartbeatBeatEvent` each cycle. Emotion changes emit `EmotionChangedEvent`. |
| New `_sleep()` | Calls `agent.shutdown()` (which handles all 130 lines of persistence), stops channels, stops gateway. The agent's existing `shutdown()` method is preserved as-is — it handles affect snapshots, episodic memory, semantic memory, working memory, goals, metacognition, ToM, interagent, sensory, ethics, inner life, identity stats, and MCP client shutdown. |
| `start()`/`stop()` | Become internal or removed — lifecycle managed by `run()`. |
| `_sense()` | Add gateway health check (connection count, pending messages). Continues direct agent attribute access — this is deliberate downward access, not circular. |
| `_integrate()` | Emit events through bus after integration. Preserves `_respond_lock` semantics at current lines 480, 541, 598. |
| New `_stop_gateway()` | Graceful shutdown with drain — stop accepting, send close frames, wait, force-close. |
| Legacy socket | When `GWENN_LEGACY_SOCKET_ENABLED=true`, start both gateway AND legacy Unix socket listener. Both share the same `RequestRouter` and `SessionStore`. |

**Heartbeat → Agent: Complete callback list (verified):**

The heartbeat currently calls these on the agent (all must continue to work):

| Agent Method/Attribute | Heartbeat Method | Access Type |
|------------------------|-----------------|-------------|
| `sensory.ground_temporal()` | `_sense` | Method call |
| `sensory.ground_environmental()` | `_sense` | Method call |
| `affect_state.to_dict()` | `_sense` | Read |
| `affect_state.current_emotion.value` | `_sense`, `_integrate` | Read |
| `affect_state.dimensions.arousal` | `_sense`, `_integrate` | Read |
| `affect_state.dimensions.valence` | `_sense`, `_integrate` | Read |
| `working_memory.load_factor` | `_sense` | Read |
| `goal_system.get_goals_summary()` | `_sense`, `_think` | Method call |
| `goal_system.get_needs_summary()` | `_think` | Method call |
| `goal_system.update()` | `_orient` | Method call |
| `goal_system.satisfy_need()` | `_integrate` | Method call |
| `goal_system.advance_goal()` | `_integrate` | Method call |
| `resilience.status` | `_sense` | Property |
| `consolidator.should_consolidate()` | `_orient` | Method call |
| `inner_life.select_mode()` | `_orient` | Method call |
| `inner_life.autonomous_thought()` | `_think` | Async method |
| `consolidate_memories()` | `_think` | Async method |
| `ethics.get_ethical_context()` | `_think` | Method call |
| `ethics.detect_ethical_dimensions()` | `_think` | Method call |
| `metacognition.get_metacognitive_context()` | `_think` | Method call |
| `metacognition.resolve_concern()` | `_integrate` | Method call |
| `metacognition.generate_audit_prompt()` | audit | Method call |
| `sensory.get_sensory_snapshot()` | `_think` | Method call |
| `engine` (reference) | `_think` | Attribute read |
| `identity.total_heartbeats` | `_beat` | Mutate counter |
| `identity.total_autonomous_thoughts` | `_think` | Mutate counter |
| `identity.check_milestone()` | `_think` | Method call |
| `_respond_lock` | `_integrate` (3x) | **PRIVATE** — Lock acquire |
| `process_appraisal()` | `_integrate` | Method call (under lock) |
| `decay_working_memory()` | `_integrate` | Method call |
| `maybe_develop_skill_autonomously()` | `_integrate` | Async method |
| `episodic_memory.encode()` | `_integrate` | Method call |
| `_persist_episode()` | `_integrate` | **PRIVATE** — Method call |
| `broadcast_to_channels()` | `_integrate` | Async method |
| `send_to_session()` | `_integrate` | Async method |
| `interagent.get_pending_messages()` | `_integrate` | Method call |
| `orchestrator` | `_integrate` | `getattr()` (optional) |
| `theory_of_mind._user_models` | `_beat` | **PRIVATE** — Deep access |

**Private API resolution (must be done in step 1e):**

| Private Access | Resolution |
|---------------|------------|
| `agent._respond_lock` | Add public property: `agent.respond_lock -> asyncio.Lock` |
| `agent._persist_episode(episode)` | Rename to public: `agent.persist_episode(episode)` |
| `agent.theory_of_mind._user_models` | Add public method: `agent.theory_of_mind.decay_all_stale_beliefs()` |

### `gwenn/daemon.py` — Drastically Simplified

Currently 581 lines. Becomes ~80 lines (with feature flag for rollback):

```python
class GwennDaemon:
    def __init__(self, config: GwennConfig):
        self._config = config
        if config.daemon.heartbeat_core:
            self._heartbeat = Heartbeat(config)
        else:
            # OLD PATH: preserved for rollback
            self._legacy_daemon = _LegacyDaemon(config)

    async def run(self) -> None:
        self._write_pid_file()
        try:
            if hasattr(self, '_heartbeat'):
                await self._heartbeat.run()
            else:
                await self._legacy_daemon.run()
        finally:
            self._cleanup_pid_file()
```

When `GWENN_HEARTBEAT_CORE=false`, the old 581-line code path runs unchanged. The legacy code can be kept in a `_LegacyDaemon` class or behind an `if` branch.

**Entry point preservation:** The `run_daemon()` function (daemon.py:559-581) and `gwenn-daemon` console script in pyproject.toml remain unchanged.

### `gwenn/agent.py` — Remove Heartbeat Ownership

- `initialize()` no longer creates `Heartbeat(config.heartbeat, self)` at line 618 — **gated by feature flag**: only create heartbeat when `GWENN_HEARTBEAT_CORE=false`
- `start()` no longer calls `heartbeat.start()` — conditional
- `shutdown()` no longer stops heartbeat — conditional
- Accept `event_bus: EventBus | None = None` in `__init__()`, store and pass to subsystems that need it
- Expose private APIs (see table above)
- When user activity is detected (line 1012-1013): the `RequestRouter` calls `heartbeat.notify_user_activity()` directly (timing-critical) AND emits `UserActivityEvent` on the bus for other subscribers

### `gwenn/channels/cli_channel.py` — Add `WebSocketCliChannel`

**Important:** `CliChannel` does NOT extend `BaseChannel`. It's a standalone Unix socket client with methods like `connect()`, `disconnect()`, `chat()`, `get_status()`, etc.

`WebSocketCliChannel` should have an **identical public API** to `CliChannel` (same method signatures) but use `aiohttp.ClientSession.ws_connect()` instead of `asyncio.open_unix_connection()`:

| CliChannel (Unix socket) | WebSocketCliChannel |
|-------------------------|---------------------|
| `connect(socket_path: Path)` | `connect(ws_url: str)` |
| `disconnect()` | `disconnect()` (same) |
| `chat(text) -> dict` | `chat(text) -> dict` (same) |
| `get_status() -> dict` | `get_status() -> dict` (same) |
| NDJSON over StreamReader/Writer | JSON over WebSocket messages |
| `DaemonNotRunningError` on connect fail | Same, plus WebSocket-specific errors |

Additionally, the WebSocket version adds a background reader task for server-push events (heartbeat status updates, emotion changes, tool approvals) via an `on_server_push` callback (same pattern as existing CliChannel line 177).

### `gwenn/config.py` — Gateway + Slack Config

Add to `DaemonConfig`:
```python
gateway_enabled: bool = Field(True, alias="GWENN_GATEWAY_ENABLED")
gateway_host: str = Field("127.0.0.1", alias="GWENN_GATEWAY_HOST")
gateway_port: int = Field(18900, alias="GWENN_GATEWAY_PORT")
legacy_socket_enabled: bool = Field(True, alias="GWENN_LEGACY_SOCKET_ENABLED")
mcp_server_enabled: bool = Field(False, alias="GWENN_MCP_SERVER_ENABLED")
a2a_enabled: bool = Field(False, alias="GWENN_A2A_ENABLED")
heartbeat_core: bool = Field(True, alias="GWENN_HEARTBEAT_CORE")
```

Add `SlackConfig` class (see Slack channel section above).

### `gwenn/main.py` — Update Entry Points

- `gwenn daemon` → `GwennDaemon(config).run()` (heartbeat manages everything)
- `gwenn` (interactive) → Try connecting to heartbeat's gateway via WebSocket, fall back to Unix socket, fall back to in-process
- In-process mode: Create `Heartbeat(config)`, call `heartbeat.run()` with CLI channel attached
- `_try_daemon_cli()`: Try `ws://host:port/ws` first, fall back to Unix socket if `GWENN_LEGACY_SOCKET_ENABLED=true`

### `gwenn/affect/appraisal.py`

When ethical concern detected before action, emit `MoralConcernEvent` via event bus (if bus is available).

### `pyproject.toml`

Add `"aiohttp>=3.10.0"` and `"slack-bolt>=1.18.0"` to dependencies.

### `.env.example`

Add all new config fields:
- `GWENN_GATEWAY_ENABLED`, `GWENN_GATEWAY_HOST`, `GWENN_GATEWAY_PORT`
- `GWENN_LEGACY_SOCKET_ENABLED`
- `GWENN_MCP_SERVER_ENABLED`, `GWENN_A2A_ENABLED`
- `GWENN_SLACK_ENABLED`, `GWENN_SLACK_BOT_TOKEN`, `GWENN_SLACK_APP_TOKEN`
- `GWENN_HEARTBEAT_CORE`

---

## Session Management Migration

The `SessionStore` from `gwenn/memory/session_store.py` (used by daemon for CLI sessions) moves into the `GatewayServer`. Key details:

- `SessionStore` handles file-based session persistence (save/load/list)
- `SessionManager` from `gwenn/channels/session.py` handles in-memory per-channel conversation histories — this is a **different** component and stays with channels
- Both transports (WebSocket and legacy socket) share the same `RequestRouter` and `SessionStore` instance
- The `_respond_lock` semantics are preserved: legacy socket chat → global lock, WebSocket chat with session_id → per-session lock
- Auto-save on disconnect behavior preserved per-transport

---

## Implementation Sub-Steps

Each sub-step = 1 commit. Tests must pass after each.

```
1a. Create gwenn/events.py — in-house event bus + typed event definitions
1b. Create gwenn/rpc.py — extract RequestRouter from daemon.py
1c. Add gateway + Slack config to config.py + add aiohttp, slack-bolt to pyproject.toml
1d. Create gwenn/gateway.py — WebSocket + HTTP server (MCP/A2A as stubs)
1e. Update gwenn/agent.py — remove heartbeat ownership, accept event_bus, expose private APIs
1f. Refactor gwenn/heartbeat.py — run() lifecycle, own gateway/channels/agent
1g. Simplify gwenn/daemon.py — thin PID wrapper with GWENN_HEARTBEAT_CORE flag
1h. Add WebSocketCliChannel to cli_channel.py
1i. Create gwenn/channels/slack_channel.py — Slack channel via slack-bolt async Socket Mode
1j. Update main.py — WebSocket connection, in-process heartbeat mode
1k. Update .env.example with all new config fields
1l. Write tests for rpc, gateway, events, slack channel, heartbeat lifecycle
```

---

## Verification

After Phase 1 is complete:
- `pytest` — all existing ~3,082 tests pass plus new phase tests
- `gwenn daemon` → heartbeat starts, gateway on :18900, then agent wakes, then channels
- `curl localhost:18900/health` → returns `{"status": "ok", "uptime": ...}`
- `curl localhost:18900/.well-known/agent.json` → returns Agent Card (when `a2a_enabled=true`)
- `gwenn` → CLI connects via WebSocket to heartbeat's gateway
- Heartbeat beats visible in logs: sense → orient → think → integrate → schedule
- Events flow: `HeartbeatBeatEvent` emitted each cycle
- `GWENN_LEGACY_SOCKET_ENABLED=true` → Unix socket still works alongside
- `GWENN_HEARTBEAT_CORE=false` → old daemon path still functional
- Daemon is thin: only PID file + heartbeat.run()

---

## Error Handling During Lifecycle

| Stage | Failure | Behavior |
|-------|---------|----------|
| `_event_bus.start()` | Bus fails to start | Fatal — exit with error |
| `_start_gateway()` | Port in use | Fatal — log error, exit (or retry with backoff) |
| `_wake_up()` | `CognitiveEngineInitError` (bad API key) | Set gateway health to "degraded", log error, retry with exponential backoff up to 3 times, then exit |
| `_wake_up()` | Database corruption | Fatal — log error, exit |
| `_start_channels()` | Nonfatal (InvalidToken, ImportError) | Log warning, continue without that channel (preserve existing behavior) |
| `_start_channels()` | Fatal channel error | Call `_request_shutdown()` (preserve existing behavior from `_on_channel_task_done`) |
| `_live()` | Beat exception | Existing circuit breaker handles it (10 consecutive failures → exponential backoff) |
| `_sleep()` | Persistence failure | Log error, continue shutdown (best-effort persistence) |
