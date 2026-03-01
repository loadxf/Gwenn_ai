# Phase 4: Event Enrichment & Client Subscriptions

**Risk: MEDIUM** — Enriches the event bus from Phase 1. Adds WebSocket subscriptions and tool approval routing.

**Prerequisites:** Phase 3 complete (all infrastructure in place).

---

## Goal

With the event bus already integrated in Phase 1, this phase enriches it: more event types, client subscriptions via WebSocket, tool approval over WebSocket, and moral emotion safety signals.

---

## Interoception: System Self-Awareness

Interoception formalizes what `_sense()` already partially does — monitoring Gwenn's own resource usage, latency, and error rates as "body signals." This maps system metrics to cognitive/emotional states the heartbeat can reason about.

### New File: `gwenn/interoception.py`

```python
class InteroceptiveState(BaseModel):
    """System-level 'body' signals mapped to cognitive states."""
    cpu_percent: float = 0.0          # 0-100
    memory_percent: float = 0.0       # 0-100
    response_latency_ms: float = 0.0  # Rolling average of recent respond() calls
    error_rate: float = 0.0           # Errors per minute (rolling window)
    active_connections: int = 0       # Current gateway connections
    beat_drift_ms: float = 0.0        # How much the actual beat interval drifts from target

    @property
    def fatigue(self) -> float:
        """0.0-1.0: high CPU + high memory + high latency = fatigue."""
        return min(1.0, (self.cpu_percent / 100 * 0.3)
                      + (self.memory_percent / 100 * 0.3)
                      + min(self.response_latency_ms / 10000, 1.0) * 0.4)

    @property
    def flow(self) -> float:
        """0.0-1.0: low latency + low errors + moderate load = flow state."""
        latency_score = max(0, 1.0 - self.response_latency_ms / 5000)
        error_score = max(0, 1.0 - self.error_rate / 10)
        return min(1.0, latency_score * 0.5 + error_score * 0.5)

    @property
    def overwhelm(self) -> float:
        """0.0-1.0: high connections + high error rate + memory pressure."""
        conn_score = min(self.active_connections / 20, 1.0)
        return min(1.0, (self.memory_percent / 100 * 0.3)
                      + (self.error_rate / 10 * 0.4)
                      + conn_score * 0.3)

class InteroceptiveMonitor:
    """Gathers system metrics and produces InteroceptiveState snapshots."""

    def __init__(self):
        self._latency_samples: deque[float] = deque(maxlen=50)
        self._error_timestamps: deque[float] = deque(maxlen=100)
        self._last_beat_time: float | None = None
        self._target_interval: float = 30.0

    def record_response_latency(self, ms: float) -> None: ...
    def record_error(self) -> None: ...
    def record_beat(self, target_interval: float) -> None: ...

    def snapshot(self, active_connections: int = 0) -> InteroceptiveState:
        """Gather current system metrics into a snapshot.

        Uses psutil-free approach:
        - CPU: read /proc/stat (Linux) or fallback to os.getloadavg()
        - Memory: read /proc/meminfo (Linux) or resource.getrusage()
        - Latency/errors: from internal rolling buffers
        """
```

**No new dependency.** CPU and memory are read from `/proc/` (Linux) or `os.getloadavg()` fallback. No `psutil` needed.

### Event

```python
class InteroceptiveSnapshotEvent(GwennEvent):
    event_type: str = "interoceptive.snapshot"
    fatigue: float; flow: float; overwhelm: float
    cpu_percent: float; memory_percent: float; response_latency_ms: float
```

Emitted each heartbeat beat during `_sense()`. Subscribers (dashboard, CLI feed) can visualize Gwenn's "body state."

### Integration with `_sense()` and `_orient()`

In `heartbeat.py`:
- `_sense()` calls `self._interoceptive_monitor.snapshot()` and includes it in the state dict
- `_orient()` can use `fatigue > 0.7` to bias toward lighter thinking modes (WANDER instead of PLAN)
- `_schedule()` can factor in `overwhelm` — when overwhelmed, slow the heartbeat to reduce load

### Integration with Affect

The `AppraisalEngine` can optionally receive interoceptive signals:
- High fatigue → slight negative valence drift (Gwenn "feels tired")
- Flow state → slight positive valence + arousal boost (Gwenn "feels in the zone")
- This is NOT direct mapping — it goes through appraisal, so the emotional response is mediated and can be overridden by stronger stimuli

### Tests: `tests/test_interoception.py`

- Test `InteroceptiveState` property calculations (fatigue, flow, overwhelm)
- Test `InteroceptiveMonitor.snapshot()` with mocked `/proc/` reads
- Test rolling latency/error buffers
- Test event emission during `_sense()`

---

## Modified Files

### `gwenn/gateway.py` — Add JSON-RPC Subscription Methods

New RPC methods handled by `RequestRouter`:

| Method | Params | Description |
|--------|--------|-------------|
| `events.subscribe` | `{"types": ["heartbeat.beat", "affect.*"]}` | Client subscribes to event patterns. Events matching any subscribed pattern are forwarded as JSON-RPC notifications. |
| `events.unsubscribe` | `{"types": ["heartbeat.beat"]}` | Remove subscriptions. |
| `tool.approve` | `{"approval_id": "abc", "decision": "allow"}` | Resolve a pending tool approval. Only accepted from clients with `can_approve_tools=True`. |

**Event forwarding pipeline:**

1. When a `GwennEvent` is emitted on the event bus, the gateway has a subscription handler that receives ALL events.
2. For each connected `ClientConnection`, the gateway checks the event's `event_type` against the client's `subscriptions` set (using the same wildcard matching as the event bus).
3. Matching events are sent as JSON-RPC notifications: `{"jsonrpc": "2.0", "method": "event", "params": {"type": "heartbeat.beat", "data": {...}}}`.
4. Non-subscribed events are silently dropped for that client.
5. Backpressure: if a client's WebSocket send buffer exceeds a threshold, skip events for that client (log warning).

### `gwenn/heartbeat.py` — Enrich Beat Events

In `_live()` (formerly `_loop()`):

**Each beat emits `HeartbeatBeatEvent`:**
```python
self._event_bus.emit(HeartbeatBeatEvent(
    beat_count=self._beat_count,
    emotion=self._agent.affect_state.current_emotion.value,
    arousal=self._agent.affect_state.dimensions.arousal,
    phase=current_phase.value,
))
```

**Significant autonomous thoughts emit `HeartbeatThoughtEvent`:**
```python
# In _think(), when thought is non-empty and importance >= threshold:
if thought and importance >= 0.4:
    self._event_bus.emit(HeartbeatThoughtEvent(
        beat_count=self._beat_count,
        thought_summary=thought[:200],  # Truncate for bus
        thinking_mode=mode.value,
        importance=importance,
    ))
```

### `gwenn/affect/state.py` — Emit Emotion Change Events

In `AffectiveState.classify_emotion()` (or wherever the named emotion transitions):

```python
# After emotion reclassification:
if new_emotion != previous_emotion and self._event_bus:
    self._event_bus.emit(EmotionChangedEvent(
        previous=previous_emotion.value,
        current=new_emotion.value,
        valence=self.dimensions.valence,
        trigger="appraisal",  # or "baseline_pull", "resilience", etc.
    ))
```

**Note:** `AffectiveState` currently has no reference to the event bus. The event bus reference must be injected — either by passing it in the constructor or by having the `AppraisalEngine` emit the event after calling `classify_emotion()`. The latter is cleaner since the `AppraisalEngine` already orchestrates the full appraisal cycle and knows the trigger context.

### `gwenn/affect/appraisal.py` — Emit Moral Concern Events

In the `appraise()` method, after ethical concern detection:

```python
# When ethical concern is detected before action:
if concern and self._event_bus:
    self._event_bus.emit(MoralConcernEvent(
        action=action_description,
        concern_type=concern.type,
        severity=concern.severity,
        traditions_flagged=concern.traditions,
    ))
```

This is the "moral emotion as pre-action safety check" pattern. The event is logged AND broadcast to all subscribed clients.

**Note:** The `AppraisalEngine` needs an `event_bus` reference. Add `event_bus: EventBus | None = None` to its constructor (passed from agent.py during wiring in Phase 1 step 1e).

### `gwenn/events.py` — Add Remaining Event Types

Ensure wildcard subscription support works correctly. Add any event types not already defined in Phase 1 that are needed for the enrichment.

### `tests/test_events.py` — Extended Tests

- Subscription filtering: client subscribes to `"heartbeat.*"`, receives `HeartbeatBeatEvent` but not `EmotionChangedEvent`
- Wildcard patterns: `"*"` receives everything, `"affect.*"` receives emotion changes
- Approval flow end-to-end: `ToolApprovalRequiredEvent` → client receives → client sends `tool.approve` → `ToolApprovalResolvedEvent`
- Backpressure: slow client doesn't block event emission
- `MoralConcernEvent` emission during appraisal

---

## Tool Approval Over WebSocket

### Flow

1. Agent's agentic loop encounters a tool call requiring approval (based on `require_approval_for` config).
2. The existing `_on_approval_request` callback (agent.py:1184) is invoked.
3. Instead of only routing to Telegram (current behavior), the approval request is emitted as `ToolApprovalRequiredEvent` on the event bus.
4. The gateway receives the event and sends it to all connected clients with `can_approve_tools=True`.
5. The first client to respond with `tool.approve` resolves the approval.
6. `ToolApprovalResolvedEvent` is emitted with the decision and source.

### Security

- `can_approve_tools: bool` on `ClientConnection` — default `True` for CLI, `False` for web/mobile
- Only clients with `can_approve_tools=True` can send `tool.approve` RPC
- `MoralConcernEvent` is logged alongside safety-critical decisions (cannot be suppressed)

### Timeout/Escalation

When `ToolApprovalRequiredEvent` is emitted:
1. Gateway sends to all connected approver-capable clients.
2. If no approval/denial within configurable timeout (default 120s):
   - Tool call is automatically denied
   - `ToolApprovalResolvedEvent(decision="timeout")` is emitted
3. If no approver-capable clients are connected:
   - Escalate to Telegram (if configured) as inline keyboard approval prompt (leveraging existing flow in `telegram_channel.py:581-704`)
   - If no Telegram either: auto-deny after timeout

---

## Implementation Sub-Steps

```
4a. Add events.subscribe / events.unsubscribe RPC methods to gateway.py + rpc.py
4b. Add event forwarding pipeline in gateway (bus subscription → per-client filter → WS send)
4c. Enrich heartbeat beat/thought event emission
4d. Add EmotionChangedEvent emission to appraisal engine
4e. Add MoralConcernEvent emission to appraisal engine
4f. Add tool.approve RPC method + approval timeout/escalation logic
4g. Write tests
```

All in a single commit.

### Interoception Sub-Steps (included in Phase 4)

```
4h. Create gwenn/interoception.py — InteroceptiveState, InteroceptiveMonitor
4i. Add InteroceptiveSnapshotEvent to events.py
4j. Wire interoceptive monitor into heartbeat _sense(), _orient(), _schedule()
4k. Optional: wire fatigue/flow signals into AppraisalEngine
4l. Write tests/test_interoception.py
```

---

## Verification

- WebSocket client subscribes to `heartbeat.*` → receives beat events in real-time
- Subscribe to `affect.*` → receives `EmotionChangedEvent` on emotion shifts
- Subscribe to `*` → receives all events
- Tool requiring approval → `ToolApprovalRequiredEvent` arrives at CLI client → approve via `tool.approve` → tool executes
- No approver connected → escalates to Telegram → approve via inline keyboard
- Timeout → tool auto-denied → `ToolApprovalResolvedEvent(decision="timeout")`
- Moral concern in appraisal → `MoralConcernEvent` logged + broadcast to subscribers
- `InteroceptiveSnapshotEvent` emitted each beat with fatigue/flow/overwhelm scores
- `gwenn dashboard` (Phase 5) shows interoceptive state in heartbeat panel
- High fatigue biases `_orient()` toward lighter thinking modes
