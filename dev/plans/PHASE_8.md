# Phase 8: Self-Healing Tier 1 (Autonomous Runtime Recovery)

**Risk: LOW-MEDIUM** — Adds recovery behaviors to existing error paths. Does not change core cognitive logic.

**Prerequisites:** Phase 7 complete (checkpoints provide the safety net for recovery actions).

---

## Goal

Gwenn autonomously detects and recovers from runtime failures: crashed channels, dropped connections, corrupted caches, and stuck operations. This is Tier 1 — fully autonomous recovery that requires no human intervention and no code changes.

---

## Why Now

The infrastructure from Phases 1-7 already detects most of these failures — the heartbeat's `_sense()` gathers state, the circuit breaker handles beat failures, and `_on_channel_task_done` catches channel crashes. But currently the response is limited to logging + stopping. Self-healing Tier 1 closes the loop: detect → diagnose → recover → verify.

---

## Architecture

Self-healing maps naturally to the heartbeat cycle:

```
SENSE:    Detect anomaly (error rates, stale channels, high latency, stuck tasks)
ORIENT:   Enter diagnostic mode (classify the failure, determine recovery action)
THINK:    Not needed for Tier 1 (recovery actions are deterministic, no LLM reasoning)
INTEGRATE: Apply recovery action
SCHEDULE:  Monitor if fix worked (accelerate next beat to check sooner)
```

---

## New Files

### `gwenn/healing.py` — Self-Healing Engine

```python
class HealthIssue(BaseModel):
    """A detected health problem."""
    issue_id: str              # UUID
    category: str              # "channel_crash", "connection_lost", "high_error_rate",
                               # "stuck_subagent", "memory_pressure", "stale_session"
    severity: float            # 0.0-1.0
    detected_at: float         # time.time()
    component: str             # "telegram", "discord", "slack", "gateway", "orchestrator"
    detail: str                # Human-readable description
    suggested_action: str      # "restart_channel", "reconnect", "clear_cache", "cancel_task"

class RecoveryAction(BaseModel):
    """A recovery action taken."""
    action_id: str
    issue_id: str              # Which issue this addresses
    action_type: str           # "restart_channel", "reconnect_gateway", "clear_stale_sessions",
                               # "cancel_stuck_subagent", "force_consolidation"
    started_at: float
    completed_at: float | None = None
    success: bool | None = None
    detail: str = ""

class SelfHealingEngine:
    """Detects health issues and applies autonomous recovery actions.

    Tier 1 only — deterministic recovery, no LLM reasoning needed.
    All actions are safe, reversible, and logged via SelfHealEvent.
    """

    def __init__(self, event_bus: EventBus, config: SelfHealingConfig):
        self._event_bus = event_bus
        self._config = config
        self._recent_issues: deque[HealthIssue] = deque(maxlen=100)
        self._recent_actions: deque[RecoveryAction] = deque(maxlen=100)
        self._cooldowns: dict[str, float] = {}  # category → earliest_retry_time

    def diagnose(self, sense_state: dict, interoceptive: InteroceptiveState,
                 channel_statuses: dict, subagent_statuses: dict) -> list[HealthIssue]:
        """Analyze current state for health issues.

        Called during _sense() phase of heartbeat. Returns list of detected issues.

        Detection rules:
        - Channel crash: channel task completed with exception
        - Connection lost: gateway active_connections dropped to 0 unexpectedly
        - High error rate: interoceptive.error_rate > threshold
        - Stuck subagent: subagent running > 2x its timeout with no progress
        - Memory pressure: interoceptive.memory_percent > threshold
        - Stale sessions: sessions idle > 2x TTL still in memory
        """

    async def heal(self, issues: list[HealthIssue], heartbeat, agent) -> list[RecoveryAction]:
        """Apply recovery actions for detected issues.

        Called during _integrate() phase. Returns list of actions taken.

        Each action:
        1. Checks cooldown (don't retry same action within cooldown window)
        2. Applies the recovery
        3. Records the action
        4. Emits SelfHealEvent

        Recovery actions (all Tier 1 — safe, autonomous):
        """

    async def _restart_channel(self, component: str, heartbeat) -> RecoveryAction:
        """Restart a crashed channel.

        Uses the same startup logic as _start_channels() but for a single channel.
        Re-calls build_channels() for the specific channel, starts it,
        re-registers the done_callback.
        """

    async def _reconnect_gateway_client(self, heartbeat) -> RecoveryAction:
        """Attempt to re-establish dropped gateway connections.

        Not applicable for server-side (server doesn't reconnect to clients).
        This handles reconnection of Gwenn-initiated connections (e.g., if
        Gwenn connects to an external service via the gateway).
        """

    async def _clear_stale_sessions(self, agent) -> RecoveryAction:
        """Force-expire sessions that are well past their TTL.

        Calls SessionManager.expire_stale_sessions() on each channel's
        session manager. Logs how many sessions were cleaned.
        """

    async def _cancel_stuck_subagent(self, task_id: str, agent) -> RecoveryAction:
        """Cancel a subagent that has exceeded 2x its timeout.

        Calls orchestrator.cancel(task_id). The subagent's result will be
        recorded as 'cancelled' with a note about self-healing.
        """

    async def _force_consolidation(self, agent) -> RecoveryAction:
        """Trigger memory consolidation if memory pressure is high.

        Calls agent.consolidate_memories() to compress episodic memory
        and free working memory slots. Only triggered when
        interoceptive.memory_percent > threshold.
        """

    async def _reduce_heartbeat_load(self, heartbeat) -> RecoveryAction:
        """Temporarily increase heartbeat interval when system is overwhelmed.

        Sets heartbeat._interval to max_interval for N beats, then
        lets _schedule() resume adaptive timing. Gives the system
        breathing room.
        """

    def verify_recovery(self, action: RecoveryAction,
                        new_state: dict) -> bool:
        """Check if a previous recovery action was successful.

        Called on the next beat after a recovery action. Checks if the
        issue that triggered the action is still present.
        Returns True if the issue is resolved.
        """
```

### `gwenn/events.py` — Self-Heal Events

```python
class SelfHealEvent(GwennEvent):
    event_type: str = "self.heal"
    issue_id: str
    action_id: str
    category: str             # "channel_crash", "stuck_subagent", etc.
    action_type: str          # "restart_channel", "cancel_stuck_subagent", etc.
    component: str            # Which component was healed
    success: bool | None      # None if not yet verified
    detail: str               # Human-readable summary
    diagnosis: str            # What was detected
    recovery: str             # What was done

class HealthCheckEvent(GwennEvent):
    event_type: str = "health.check"
    issues_found: int
    actions_taken: int
    all_healthy: bool
```

### `tests/test_healing.py`

- Test `diagnose()` detects each issue category with mocked state
- Test cooldown: same issue category not acted on within cooldown window
- Test `_restart_channel()` restarts a mocked crashed channel
- Test `_cancel_stuck_subagent()` cancels via orchestrator
- Test `_clear_stale_sessions()` cleans up expired sessions
- Test `_force_consolidation()` triggers consolidation under memory pressure
- Test `_reduce_heartbeat_load()` temporarily increases interval
- Test `verify_recovery()` confirms issue resolution on next beat
- Test `SelfHealEvent` emission with correct fields
- Test no action taken when no issues detected
- Test multiple simultaneous issues: each gets appropriate action
- Test recovery failure: action taken but issue persists → re-diagnosed on next beat

---

## Modified Files

### `gwenn/config.py` — Self-Healing Config

```python
class SelfHealingConfig(BaseSettings):
    enabled: bool = Field(True, alias="GWENN_SELF_HEALING_ENABLED")
    cooldown_seconds: float = Field(300.0, alias="GWENN_SELF_HEALING_COOLDOWN")
    max_actions_per_hour: int = Field(20, alias="GWENN_SELF_HEALING_MAX_ACTIONS_HOUR")
    channel_restart_enabled: bool = Field(True, alias="GWENN_SELF_HEALING_CHANNEL_RESTART")
    stuck_subagent_timeout_multiplier: float = Field(2.0, alias="GWENN_SELF_HEALING_STUCK_MULTIPLIER")
    memory_pressure_threshold: float = Field(85.0, alias="GWENN_SELF_HEALING_MEMORY_THRESHOLD")
    error_rate_threshold: float = Field(5.0, alias="GWENN_SELF_HEALING_ERROR_RATE_THRESHOLD")
```

Add `self_healing: SelfHealingConfig` to `GwennConfig.__init__`.

### `gwenn/heartbeat.py` — Wire Self-Healing into Beat Cycle

**In `__init__`:**
```python
self._healing_engine = SelfHealingEngine(
    event_bus=self._event_bus,
    config=config.self_healing,
) if config.self_healing.enabled else None
```

**In `_sense()`** — add health diagnosis:
```python
if self._healing_engine:
    health_issues = self._healing_engine.diagnose(
        sense_state=state_snapshot,
        interoceptive=interoceptive_state,
        channel_statuses=self._get_channel_statuses(),
        subagent_statuses=self._get_subagent_statuses(),
    )
    state_snapshot["health_issues"] = health_issues
```

**In `_integrate()`** — apply recovery after other integration:
```python
if self._healing_engine and state_snapshot.get("health_issues"):
    actions = await self._healing_engine.heal(
        state_snapshot["health_issues"], self, self._agent
    )
    if actions:
        self._event_bus.emit(HealthCheckEvent(
            issues_found=len(state_snapshot["health_issues"]),
            actions_taken=len(actions),
            all_healthy=False,
        ))
```

**In `_schedule()`** — accelerate after recovery:
```python
# If recovery actions were taken, accelerate next beat to verify
if state_snapshot.get("health_issues"):
    self._interval = self._config.heartbeat.min_interval  # Check sooner
```

### `gwenn/gateway.py` — Health Status in Dashboard

The `/dashboard` endpoint (from Phase 5) includes self-healing status:
```json
{
    "health": {
        "status": "healthy" | "recovering" | "degraded",
        "recent_issues": [...],
        "recent_actions": [...],
        "last_check_at": "..."
    }
}
```

### `.env.example`

Add all new config fields:
```
GWENN_SELF_HEALING_ENABLED=true
GWENN_SELF_HEALING_COOLDOWN=300
GWENN_SELF_HEALING_MAX_ACTIONS_HOUR=20
GWENN_SELF_HEALING_CHANNEL_RESTART=true
GWENN_SELF_HEALING_STUCK_MULTIPLIER=2.0
GWENN_SELF_HEALING_MEMORY_THRESHOLD=85.0
GWENN_SELF_HEALING_ERROR_RATE_THRESHOLD=5.0
```

---

## Recovery Actions Reference

| Issue Category | Detection Signal | Recovery Action | Cooldown |
|---------------|-----------------|----------------|----------|
| `channel_crash` | Channel task completed with exception | `restart_channel` — re-build and re-start the specific channel | 5 min per channel |
| `high_error_rate` | `interoceptive.error_rate > threshold` | `reduce_heartbeat_load` — temporarily set interval to max | 10 min |
| `stuck_subagent` | Subagent running > `2x timeout` with no progress update | `cancel_stuck_subagent` — cancel via orchestrator | 5 min per task |
| `memory_pressure` | `interoceptive.memory_percent > threshold` | `force_consolidation` — trigger memory consolidation | 15 min |
| `stale_sessions` | Sessions idle > `2x TTL` still in memory | `clear_stale_sessions` — expire via SessionManager | 30 min |

**Not in Tier 1 (future Tier 2-3):**
- Configuration auto-adjustment (Tier 2)
- Code patching (Tier 3)
- LLM-based diagnosis (Tier 2-3)

---

## Safety Constraints

- **Rate limited**: max `max_actions_per_hour` (default 20) recovery actions per hour
- **Cooldown per category**: same issue type not retried within `cooldown_seconds`
- **No cascading recovery**: if a recovery action itself fails, log and wait for next beat cycle — do not retry immediately
- **All actions are reversible**: restart a channel (can stop again), cancel a subagent (can re-spawn), force consolidation (normal operation)
- **Logged unconditionally**: every `SelfHealEvent` is emitted regardless of success/failure. All actions are visible in the event feed and dashboard.
- **Checkpoint before risky recovery**: before `restart_channel` or `force_consolidation`, trigger a checkpoint (Phase 7) if one hasn't been taken recently. This ensures recovery can be rolled back.

---

## Implementation Sub-Steps

```
8a. Add SelfHealingConfig to config.py + GwennConfig
8b. Create gwenn/healing.py — HealthIssue, RecoveryAction, SelfHealingEngine
8c. Add SelfHealEvent, HealthCheckEvent to events.py
8d. Wire diagnose() into heartbeat _sense()
8e. Wire heal() into heartbeat _integrate()
8f. Wire verification into _schedule() (accelerate after recovery)
8g. Add health status to gateway /dashboard endpoint
8h. Update .env.example
8i. Write tests
```

1 commit per sub-step.

---

## Verification

- Crash a channel (e.g., revoke Telegram bot token) → `SelfHealEvent` emitted → channel restarted automatically
- Spawn a subagent with a very long task + short timeout → stuck detection → auto-cancel
- Simulate high memory usage → consolidation triggered → memory pressure reduced
- Check `gwenn feed --type "self.*"` → shows all heal events
- Check `gwenn dashboard` → health panel shows "recovering" during actions, "healthy" after
- Verify cooldown: crash same channel twice within 5 minutes → second crash logged but not acted on until cooldown expires
- Verify rate limit: trigger many issues → max 20 actions/hour enforced
- Verify checkpoint before risky recovery: checkpoint created before channel restart
- No recovery actions when everything is healthy: `HealthCheckEvent(all_healthy=True)` emitted
