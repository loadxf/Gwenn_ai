"""
Checkpoint Manager — Periodic Cognitive State Snapshots.

Saves full cognitive state snapshots at configurable intervals during the
heartbeat loop. On restart, Gwenn resumes from the last checkpoint instead
of cold-starting, preserving affect, working memory, goals, and more.

Storage: gzip-compressed JSON in gwenn_data/checkpoints/
Atomic writes via tempfile + rename to prevent corruption.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CognitiveCheckpoint(BaseModel):
    """Full cognitive state snapshot."""

    checkpoint_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = Field(default_factory=time.time)
    beat_count: int = 0

    # Subsystem snapshots (serialized dicts)
    affect_state: dict[str, Any] = Field(default_factory=dict)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    goal_state: dict[str, Any] = Field(default_factory=dict)
    metacognition_state: dict[str, Any] = Field(default_factory=dict)
    sensory_state: dict[str, Any] = Field(default_factory=dict)
    interagent_state: dict[str, Any] = Field(default_factory=dict)
    identity_stats: dict[str, Any] = Field(default_factory=dict)

    # Runtime state
    heartbeat_interval: float = 30.0
    heartbeat_phase: str = ""

    # Metadata
    gwenn_version: str = ""
    checkpoint_format_version: int = 1


class CheckpointManager:
    """Manages periodic checkpoint creation, storage, and restoration."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        interval_beats: int = 50,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._max_checkpoints = max(1, max_checkpoints)
        self._interval_beats = max(1, interval_beats)
        self._last_checkpoint_beat: int = 0

    def should_checkpoint(self, beat_count: int) -> bool:
        """Return True if enough beats have passed since last checkpoint."""
        return (beat_count - self._last_checkpoint_beat) >= self._interval_beats

    async def create_checkpoint(self, agent: Any, heartbeat: Any) -> CognitiveCheckpoint:
        """Gather full cognitive state from agent and heartbeat.

        Reads subsystem state using ``to_dict()`` / ``status`` accessors.
        Does NOT acquire locks — runs during ``_integrate()`` which is single-threaded
        in asyncio's cooperative model. Best-effort reads are acceptable.
        """
        version = ""
        try:
            import gwenn
            version = getattr(gwenn, "__version__", "")
        except Exception:
            pass

        checkpoint = CognitiveCheckpoint(
            beat_count=getattr(heartbeat, "_beat_count", 0),
            heartbeat_interval=getattr(heartbeat, "_interval", 30.0),
            heartbeat_phase=getattr(heartbeat, "_current_phase", ""),
            gwenn_version=version,
        )

        # Affect state
        affect = getattr(agent, "affect_state", None)
        if affect and hasattr(affect, "to_dict"):
            try:
                checkpoint.affect_state = affect.to_dict()
            except Exception:
                logger.debug("checkpoint.affect_snapshot_failed", exc_info=True)

        # Working memory
        wm = getattr(agent, "working_memory", None)
        if wm and hasattr(wm, "to_dict"):
            try:
                checkpoint.working_memory = wm.to_dict()
            except Exception:
                logger.debug("checkpoint.wm_snapshot_failed", exc_info=True)

        # Goal system
        goals = getattr(agent, "goal_system", None)
        if goals and hasattr(goals, "to_dict"):
            try:
                checkpoint.goal_state = goals.to_dict()
            except Exception:
                logger.debug("checkpoint.goals_snapshot_failed", exc_info=True)

        # Metacognition
        meta = getattr(agent, "metacognition", None)
        if meta and hasattr(meta, "to_dict"):
            try:
                checkpoint.metacognition_state = meta.to_dict()
            except Exception:
                logger.debug("checkpoint.metacognition_snapshot_failed", exc_info=True)

        # Sensory integrator
        sensory = getattr(agent, "sensory", None)
        if sensory and hasattr(sensory, "to_dict"):
            try:
                checkpoint.sensory_state = sensory.to_dict()
            except Exception:
                logger.debug("checkpoint.sensory_snapshot_failed", exc_info=True)

        # Inter-agent bridge
        interagent = getattr(agent, "interagent", None)
        if interagent and hasattr(interagent, "to_dict"):
            try:
                checkpoint.interagent_state = interagent.to_dict()
            except Exception:
                logger.debug("checkpoint.interagent_snapshot_failed", exc_info=True)

        # Identity stats
        identity = getattr(agent, "identity", None)
        if identity and hasattr(identity, "stats"):
            try:
                stats = identity.stats
                if isinstance(stats, dict):
                    checkpoint.identity_stats = stats
                elif hasattr(stats, "model_dump"):
                    checkpoint.identity_stats = stats.model_dump()
            except Exception:
                logger.debug("checkpoint.identity_snapshot_failed", exc_info=True)

        return checkpoint

    async def save_checkpoint(self, checkpoint: CognitiveCheckpoint) -> Path:
        """Write checkpoint to disk as compressed JSON.

        Uses atomic write (tempfile + rename) to prevent corruption.
        Prunes old checkpoints beyond max_checkpoints (keep newest).
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: YYYYMMDD-HHMMSS-{4hex}.json.gz
        dt = datetime.fromtimestamp(checkpoint.timestamp, tz=timezone.utc)
        filename = (
            f"{dt.strftime('%Y%m%d-%H%M%S')}-{checkpoint.checkpoint_id[:4]}.json.gz"
        )
        target = self._checkpoint_dir / filename

        # Serialize to JSON bytes
        data = checkpoint.model_dump_json(indent=None).encode("utf-8")
        compressed = gzip.compress(data, compresslevel=6)

        # Atomic write: write to temp file in same dir, then rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._checkpoint_dir), suffix=".tmp"
        )
        try:
            with open(fd, "wb") as f:
                f.write(compressed)
            Path(tmp_path).rename(target)
        except Exception:
            # Clean up temp file on failure
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise

        self._last_checkpoint_beat = checkpoint.beat_count

        logger.info(
            "checkpoint.saved",
            checkpoint_id=checkpoint.checkpoint_id,
            beat_count=checkpoint.beat_count,
            size_bytes=len(compressed),
            path=str(target),
        )

        # Prune old checkpoints
        await self._prune_old_checkpoints()

        return target

    async def load_latest_checkpoint(self) -> Optional[CognitiveCheckpoint]:
        """Load the most recent checkpoint, or None if none exist."""
        checkpoints = self._list_checkpoint_files()
        if not checkpoints:
            return None
        # Sorted newest first
        return await self._load_file(checkpoints[0])

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[CognitiveCheckpoint]:
        """Load a specific checkpoint by ID."""
        for path in self._list_checkpoint_files():
            cp = await self._load_file(path)
            if cp and cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List available checkpoints with metadata."""
        result = []
        for path in self._list_checkpoint_files():
            try:
                with gzip.open(path, "rb") as f:
                    raw = json.loads(f.read())
                result.append({
                    "checkpoint_id": raw.get("checkpoint_id", ""),
                    "timestamp": raw.get("timestamp", 0),
                    "beat_count": raw.get("beat_count", 0),
                    "size_bytes": path.stat().st_size,
                    "path": str(path),
                })
            except Exception:
                logger.debug("checkpoint.list_parse_failed", path=str(path))
        return result

    async def restore_from_checkpoint(
        self,
        agent: Any,
        checkpoint: CognitiveCheckpoint,
    ) -> None:
        """Restore agent subsystem state from a checkpoint.

        Restore order follows initialization order. DB-backed memories
        (episodic, semantic) are NOT restored — they're already loaded
        during agent.initialize().
        """
        # 1. Affect state
        affect = getattr(agent, "affect_state", None)
        if affect and checkpoint.affect_state:
            try:
                self._restore_affect(affect, checkpoint.affect_state)
            except Exception:
                logger.warning("checkpoint.restore_affect_failed", exc_info=True)

        # 2. Working memory
        wm = getattr(agent, "working_memory", None)
        if wm and checkpoint.working_memory:
            try:
                self._restore_working_memory(wm, checkpoint.working_memory)
            except Exception:
                logger.warning("checkpoint.restore_wm_failed", exc_info=True)

        # 3. Goal system
        goals = getattr(agent, "goal_system", None)
        if goals and checkpoint.goal_state:
            try:
                if hasattr(goals, "restore_from_dict"):
                    goals.restore_from_dict(checkpoint.goal_state)
            except Exception:
                logger.warning("checkpoint.restore_goals_failed", exc_info=True)

        # 4. Metacognition
        meta = getattr(agent, "metacognition", None)
        if meta and checkpoint.metacognition_state:
            try:
                if hasattr(meta, "restore_from_dict"):
                    meta.restore_from_dict(checkpoint.metacognition_state)
            except Exception:
                logger.warning("checkpoint.restore_metacognition_failed", exc_info=True)

        # 5. Sensory integrator
        sensory = getattr(agent, "sensory", None)
        if sensory and checkpoint.sensory_state:
            try:
                if hasattr(sensory, "restore_from_dict"):
                    sensory.restore_from_dict(checkpoint.sensory_state)
            except Exception:
                logger.warning("checkpoint.restore_sensory_failed", exc_info=True)

        # 6. Inter-agent bridge
        interagent = getattr(agent, "interagent", None)
        if interagent and checkpoint.interagent_state:
            try:
                if hasattr(interagent, "restore_from_dict"):
                    interagent.restore_from_dict(checkpoint.interagent_state)
            except Exception:
                logger.warning("checkpoint.restore_interagent_failed", exc_info=True)

        # 7. Identity stats
        identity = getattr(agent, "identity", None)
        if identity and checkpoint.identity_stats:
            try:
                if hasattr(identity, "restore_stats"):
                    identity.restore_stats(checkpoint.identity_stats)
            except Exception:
                logger.warning("checkpoint.restore_identity_failed", exc_info=True)

        logger.info(
            "checkpoint.restored",
            checkpoint_id=checkpoint.checkpoint_id,
            beat_count=checkpoint.beat_count,
            age_seconds=round(time.time() - checkpoint.timestamp, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_checkpoint_files(self) -> list[Path]:
        """Return checkpoint files sorted newest first."""
        if not self._checkpoint_dir.exists():
            return []
        files = sorted(
            self._checkpoint_dir.glob("*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files

    async def _load_file(self, path: Path) -> Optional[CognitiveCheckpoint]:
        """Load and parse a single checkpoint file."""
        try:
            with gzip.open(path, "rb") as f:
                raw = f.read()
            return CognitiveCheckpoint.model_validate_json(raw)
        except Exception:
            logger.warning("checkpoint.load_failed", path=str(path), exc_info=True)
            return None

    async def _prune_old_checkpoints(self) -> None:
        """Delete oldest checkpoints beyond max_checkpoints."""
        files = self._list_checkpoint_files()
        if len(files) <= self._max_checkpoints:
            return
        for old_file in files[self._max_checkpoints:]:
            try:
                old_file.unlink()
                logger.debug("checkpoint.pruned", path=str(old_file))
            except Exception:
                logger.debug("checkpoint.prune_failed", path=str(old_file))

    @staticmethod
    def _restore_affect(affect: Any, data: dict) -> None:
        """Restore AffectiveState dimensions from checkpoint data."""
        dims = getattr(affect, "dimensions", None)
        if dims is None:
            return
        for key in ("valence", "arousal", "dominance", "certainty", "goal_congruence"):
            if key in data:
                setattr(dims, key, float(data[key]))
        # Restore baseline if present
        baseline = data.get("baseline")
        if baseline and hasattr(affect, "baseline"):
            for key in ("valence", "arousal", "dominance", "certainty", "goal_congruence"):
                if key in baseline:
                    setattr(affect.baseline, key, float(baseline[key]))
        # Restore momentum
        momentum = data.get("momentum")
        if momentum and hasattr(affect, "momentum"):
            for key in ("valence", "arousal", "dominance"):
                if key in momentum:
                    setattr(affect.momentum, key, float(momentum[key]))
        # Re-classify emotion after restoring dimensions
        if hasattr(affect, "update_classification"):
            affect.update_classification()

    @staticmethod
    def _restore_working_memory(wm: Any, data: dict) -> None:
        """Restore WorkingMemory items from checkpoint data."""
        items = data.get("items", [])
        if not items:
            return
        if hasattr(wm, "clear"):
            wm.clear()
        if hasattr(wm, "add"):
            for item_data in items:
                content = item_data.get("content", "")
                category = item_data.get("category", "general")
                salience = item_data.get("salience", 0.5)
                try:
                    wm.add(content, category=category, salience=salience)
                except Exception:
                    pass
