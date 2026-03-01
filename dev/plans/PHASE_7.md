# Phase 7: Checkpoint/Restore (Heartbeat Snapshots)

**Risk: MEDIUM** — Touches heartbeat persistence but does not change core cognitive logic.

**Prerequisites:** Phase 6 complete (full infrastructure in place, event bus mature).

---

## Goal

Periodic cognitive state snapshots for crash recovery, cross-channel continuity, and debugging. The heartbeat saves full snapshots at configurable intervals. On restart, Gwenn resumes from the last checkpoint instead of cold-starting.

---

## Why This Is Critical

Phase 1 makes the heartbeat the single lifecycle owner — if `heartbeat.run()` crashes, ALL state is lost unless it was persisted during `_sleep()`. But crashes don't call `_sleep()`. Without checkpoints:
- A crash after 1000 beats loses all in-memory working memory, affect state, goal progress, and conversation contexts
- The agent restarts cold, losing emotional continuity and in-progress work
- Swarm tasks in flight are orphaned with no recovery path

Checkpoints provide a safety net: restart from the last snapshot, not from scratch.

---

## New Files

### `gwenn/checkpoint.py` — Checkpoint Manager

```python
class CognitiveCheckpoint(BaseModel):
    """Full cognitive state snapshot."""
    checkpoint_id: str              # UUID
    timestamp: float                # time.time()
    beat_count: int

    # Subsystem snapshots (serialized dicts)
    affect_state: dict              # AffectiveState.to_dict()
    working_memory: list[dict]      # Current WM items
    goal_state: dict                # GoalSystem state
    metacognition_state: dict       # MetacognitionEngine state
    theory_of_mind_state: dict      # TheoryOfMind state
    interagent_state: dict          # InterAgentBridge state
    sensory_state: dict             # SensoryIntegrator state
    ethics_state: dict              # EthicalReasoner state
    inner_life_state: dict          # InnerLife state
    identity_stats: dict            # Identity counters (total_heartbeats, etc.)

    # Runtime state
    heartbeat_interval: float       # Current adaptive interval
    heartbeat_phase: str            # Last phase completed
    active_swarms: list[dict]       # SwarmSpec snapshots for in-flight swarms
    active_subagents: list[dict]    # SubagentProgress snapshots
    channel_sessions: dict          # Per-channel session state summaries

    # Metadata
    gwenn_version: str
    checkpoint_format_version: int = 1

class CheckpointManager:
    """Manages periodic checkpoint creation, storage, and restoration."""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 10,
                 interval_beats: int = 50):
        self._checkpoint_dir = checkpoint_dir
        self._max_checkpoints = max_checkpoints
        self._interval_beats = interval_beats  # Checkpoint every N beats
        self._beats_since_checkpoint = 0

    def should_checkpoint(self, beat_count: int) -> bool:
        """True if enough beats have passed since last checkpoint."""

    async def create_checkpoint(self, agent, heartbeat) -> CognitiveCheckpoint:
        """Gather full cognitive state from agent and heartbeat.

        Reads subsystem state using the same accessors as agent.shutdown(),
        but WITHOUT stopping anything. This is a snapshot, not a shutdown.

        Important: must NOT acquire _respond_lock — the checkpoint runs
        during _integrate() which may already hold the lock. Use best-effort
        reads of subsystem state (acceptable since asyncio is cooperative
        and no await points during the snapshot gathering).
        """

    async def save_checkpoint(self, checkpoint: CognitiveCheckpoint) -> Path:
        """Write checkpoint to disk as compressed JSON.

        Storage: gwenn_data/checkpoints/{checkpoint_id}.json.gz
        Atomic write: tempfile + rename.
        Prunes old checkpoints beyond max_checkpoints (keep newest).
        """

    async def load_latest_checkpoint(self) -> CognitiveCheckpoint | None:
        """Load the most recent checkpoint, or None if no checkpoints exist."""

    async def load_checkpoint(self, checkpoint_id: str) -> CognitiveCheckpoint | None:
        """Load a specific checkpoint by ID."""

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints with metadata (id, timestamp, beat_count)."""

    async def restore_from_checkpoint(self, agent, checkpoint: CognitiveCheckpoint) -> None:
        """Restore agent subsystem state from a checkpoint.

        Uses the same restore methods as agent.initialize() but with
        checkpoint data instead of database data. Called during _wake_up()
        when a checkpoint is available and more recent than DB state.

        Restore order (same as initialize):
        1. Affect state (dimensions + baseline)
        2. Goal state
        3. Metacognition state
        4. Theory of Mind state
        5. Interagent state
        6. Sensory state
        7. Ethics state
        8. Inner life state
        9. Identity stats
        10. Working memory items

        Does NOT restore: episodic/semantic memory (these are DB-backed and
        already loaded during initialize), swarm/subagent state (these must
        be re-evaluated on restart), channel sessions (channels reconnect fresh).
        """
```

### `gwenn/events.py` — Checkpoint Events

```python
class CheckpointCreatedEvent(GwennEvent):
    event_type: str = "checkpoint.created"
    checkpoint_id: str
    beat_count: int
    size_bytes: int

class CheckpointRestoredEvent(GwennEvent):
    event_type: str = "checkpoint.restored"
    checkpoint_id: str
    beats_recovered: int  # beat_count at checkpoint
    age_seconds: float    # How old the checkpoint was
```

### `tests/test_checkpoint.py`

- Test `CognitiveCheckpoint` serialization roundtrip (create → save → load → compare)
- Test `should_checkpoint()` interval logic
- Test `create_checkpoint()` gathers all subsystem state (mock agent with known state)
- Test `save_checkpoint()` writes compressed JSON atomically
- Test `load_latest_checkpoint()` finds most recent
- Test checkpoint pruning (max_checkpoints honored, oldest deleted)
- Test `restore_from_checkpoint()` applies state correctly to agent subsystems
- Test restore order matches initialize order
- Test checkpoint with no prior checkpoints → `None` returned
- Test corrupted checkpoint file → graceful error, falls back to DB restore

---

## Modified Files

### `gwenn/config.py` — Checkpoint Config

Add to `HeartbeatConfig`:
```python
checkpoint_enabled: bool = Field(True, alias="GWENN_CHECKPOINT_ENABLED")
checkpoint_interval_beats: int = Field(50, alias="GWENN_CHECKPOINT_INTERVAL_BEATS")
checkpoint_max_count: int = Field(10, alias="GWENN_CHECKPOINT_MAX_COUNT")
```

### `gwenn/heartbeat.py` — Wire Checkpoints into Lifecycle

**In `__init__`:**
```python
self._checkpoint_manager = CheckpointManager(
    checkpoint_dir=config.memory.data_dir / "checkpoints",
    max_checkpoints=config.heartbeat.checkpoint_max_count,
    interval_beats=config.heartbeat.checkpoint_interval_beats,
) if config.heartbeat.checkpoint_enabled else None
```

**In `_integrate()`** (at the end, after all other integration work):
```python
if self._checkpoint_manager and self._checkpoint_manager.should_checkpoint(self._beat_count):
    checkpoint = await self._checkpoint_manager.create_checkpoint(self._agent, self)
    path = await self._checkpoint_manager.save_checkpoint(checkpoint)
    self._event_bus.emit(CheckpointCreatedEvent(
        checkpoint_id=checkpoint.checkpoint_id,
        beat_count=checkpoint.beat_count,
        size_bytes=path.stat().st_size,
    ))
```

**In `_wake_up()`** (after `agent.initialize()`, before returning):
```python
if self._checkpoint_manager:
    checkpoint = await self._checkpoint_manager.load_latest_checkpoint()
    if checkpoint:
        # Only restore if checkpoint is more recent than DB state
        await self._checkpoint_manager.restore_from_checkpoint(self._agent, checkpoint)
        self._beat_count = checkpoint.beat_count  # Resume beat counter
        self._event_bus.emit(CheckpointRestoredEvent(
            checkpoint_id=checkpoint.checkpoint_id,
            beats_recovered=checkpoint.beat_count,
            age_seconds=time.time() - checkpoint.timestamp,
        ))
```

### `gwenn/gateway.py` — Checkpoint RPC Methods

Add to `RequestRouter`:
- `checkpoint.list` → returns list of available checkpoints
- `checkpoint.create` → force an immediate checkpoint (for manual use)
- `checkpoint.restore <id>` → restore a specific checkpoint (requires confirmation)

### `.env.example`

Add:
```
GWENN_CHECKPOINT_ENABLED=true
GWENN_CHECKPOINT_INTERVAL_BEATS=50
GWENN_CHECKPOINT_MAX_COUNT=10
```

---

## Storage Format

Checkpoints stored as gzip-compressed JSON:
```
gwenn_data/checkpoints/
  ├── 20260228-143245-a1b2.json.gz   (newest)
  ├── 20260228-141530-c3d4.json.gz
  ├── 20260228-135812-e5f6.json.gz
  └── ...                             (max 10, oldest pruned)
```

File naming: `{YYYYMMDD}-{HHMMSS}-{4-char-hex}.json.gz`

Typical checkpoint size estimate: ~50-200 KB compressed (depending on working memory and theory of mind model count).

---

## Implementation Sub-Steps

```
7a. Add checkpoint config fields to HeartbeatConfig in config.py
7b. Create gwenn/checkpoint.py — CognitiveCheckpoint model, CheckpointManager
7c. Add CheckpointCreatedEvent, CheckpointRestoredEvent to events.py
7d. Wire checkpoint creation into heartbeat _integrate()
7e. Wire checkpoint restoration into heartbeat _wake_up()
7f. Add checkpoint RPC methods to gateway (list, create, restore)
7g. Update .env.example
7h. Write tests
```

1 commit per sub-step.

---

## Verification

- After 50 beats, a checkpoint file appears in `gwenn_data/checkpoints/`
- `gwenn` CLI (or direct RPC): `checkpoint.list` → shows available checkpoints
- Kill daemon with `SIGKILL` (no graceful shutdown) → restart → Gwenn resumes from checkpoint
- Affect state, working memory, goal progress restored correctly
- Beat counter resumes from checkpoint value
- `CheckpointCreatedEvent` appears in event feed every 50 beats
- `CheckpointRestoredEvent` appears on startup when checkpoint is used
- Old checkpoints pruned: never more than `max_count` files
- Corrupted checkpoint: graceful degradation to cold start with DB-only restore
