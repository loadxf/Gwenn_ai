"""Tests for gwenn.checkpoint — Checkpoint Manager and CognitiveCheckpoint."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from gwenn.checkpoint import CheckpointManager, CognitiveCheckpoint


# ---------------------------------------------------------------------------
# CognitiveCheckpoint Model
# ---------------------------------------------------------------------------


class TestCognitiveCheckpoint:
    def test_defaults(self) -> None:
        cp = CognitiveCheckpoint()
        assert cp.checkpoint_id  # auto-generated
        assert cp.timestamp > 0
        assert cp.beat_count == 0
        assert cp.checkpoint_format_version == 1

    def test_custom_fields(self) -> None:
        cp = CognitiveCheckpoint(
            beat_count=42,
            affect_state={"valence": 0.5},
            working_memory={"items": []},
            gwenn_version="0.3.0",
        )
        assert cp.beat_count == 42
        assert cp.affect_state["valence"] == 0.5
        assert cp.gwenn_version == "0.3.0"

    def test_serialization_roundtrip(self) -> None:
        cp = CognitiveCheckpoint(
            beat_count=100,
            affect_state={"valence": 0.6, "arousal": 0.3},
            working_memory={"items": [{"content": "test", "category": "general"}]},
            goal_state={"needs": {"growth": 0.5}},
            heartbeat_interval=25.0,
            heartbeat_phase="think",
        )
        json_bytes = cp.model_dump_json().encode("utf-8")
        restored = CognitiveCheckpoint.model_validate_json(json_bytes)
        assert restored.beat_count == 100
        assert restored.affect_state["valence"] == 0.6
        assert restored.heartbeat_interval == 25.0

    def test_json_dict_roundtrip(self) -> None:
        cp = CognitiveCheckpoint(beat_count=5)
        data = cp.model_dump()
        restored = CognitiveCheckpoint.model_validate(data)
        assert restored.beat_count == 5
        assert restored.checkpoint_id == cp.checkpoint_id


# ---------------------------------------------------------------------------
# CheckpointManager — should_checkpoint
# ---------------------------------------------------------------------------


class TestShouldCheckpoint:
    def test_initial_beat_not_ready(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"), interval_beats=10)
        assert mgr.should_checkpoint(5) is False

    def test_at_interval(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"), interval_beats=10)
        assert mgr.should_checkpoint(10) is True

    def test_past_interval(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"), interval_beats=10)
        assert mgr.should_checkpoint(15) is True

    def test_after_save_resets_counter(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"), interval_beats=10)
        mgr._last_checkpoint_beat = 10
        assert mgr.should_checkpoint(15) is False
        assert mgr.should_checkpoint(20) is True


# ---------------------------------------------------------------------------
# CheckpointManager — create_checkpoint
# ---------------------------------------------------------------------------


class TestCreateCheckpoint:
    @pytest.mark.asyncio
    async def test_creates_checkpoint_with_subsystem_data(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))

        agent = MagicMock()
        agent.affect_state = MagicMock()
        agent.affect_state.to_dict.return_value = {"valence": 0.5, "arousal": 0.3}
        agent.working_memory = MagicMock()
        agent.working_memory.to_dict.return_value = {"items": []}
        agent.goal_system = MagicMock()
        agent.goal_system.to_dict.return_value = {"needs": {}}
        agent.metacognition = MagicMock()
        agent.metacognition.to_dict.return_value = {"concerns": []}
        agent.sensory = MagicMock()
        agent.sensory.to_dict.return_value = {"session_start": 0}
        agent.interagent = MagicMock()
        agent.interagent.to_dict.return_value = {"self_id": "gwenn"}
        agent.identity = MagicMock()
        agent.identity.stats = {"total_heartbeats": 100}

        heartbeat = MagicMock()
        heartbeat._beat_count = 42
        heartbeat._interval = 25.0
        heartbeat._current_phase = "think"

        cp = await mgr.create_checkpoint(agent, heartbeat)

        assert cp.beat_count == 42
        assert cp.heartbeat_interval == 25.0
        assert cp.affect_state["valence"] == 0.5
        assert cp.working_memory["items"] == []
        assert cp.goal_state["needs"] == {}
        assert cp.interagent_state["self_id"] == "gwenn"
        assert cp.identity_stats["total_heartbeats"] == 100

    @pytest.mark.asyncio
    async def test_handles_missing_subsystems(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        agent = MagicMock(spec=[])  # No attributes
        heartbeat = MagicMock()
        heartbeat._beat_count = 0
        heartbeat._interval = 30.0
        heartbeat._current_phase = ""

        cp = await mgr.create_checkpoint(agent, heartbeat)
        assert cp.beat_count == 0
        assert cp.affect_state == {}

    @pytest.mark.asyncio
    async def test_handles_subsystem_error(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        agent = MagicMock()
        agent.affect_state = MagicMock()
        agent.affect_state.to_dict.side_effect = RuntimeError("broken")
        agent.working_memory = MagicMock()
        agent.working_memory.to_dict.return_value = {"items": []}

        heartbeat = MagicMock()
        heartbeat._beat_count = 10
        heartbeat._interval = 30.0
        heartbeat._current_phase = ""

        cp = await mgr.create_checkpoint(agent, heartbeat)
        assert cp.affect_state == {}  # Gracefully skipped
        assert cp.working_memory["items"] == []  # Other subsystems still captured


# ---------------------------------------------------------------------------
# CheckpointManager — save / load / prune
# ---------------------------------------------------------------------------


class TestSaveLoadCheckpoint:
    @pytest.mark.asyncio
    async def test_save_and_load_latest(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints")
        cp = CognitiveCheckpoint(
            beat_count=50,
            affect_state={"valence": 0.7},
        )
        path = await mgr.save_checkpoint(cp)

        assert path.exists()
        assert path.suffix == ".gz"

        loaded = await mgr.load_latest_checkpoint()
        assert loaded is not None
        assert loaded.checkpoint_id == cp.checkpoint_id
        assert loaded.beat_count == 50
        assert loaded.affect_state["valence"] == 0.7

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "new" / "dir")
        cp = CognitiveCheckpoint(beat_count=1)
        path = await mgr.save_checkpoint(cp)
        assert path.exists()

    @pytest.mark.asyncio
    async def test_load_latest_returns_none_when_empty(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "empty")
        result = await mgr.load_latest_checkpoint()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints")
        cp1 = CognitiveCheckpoint(beat_count=10)
        cp2 = CognitiveCheckpoint(beat_count=20)
        await mgr.save_checkpoint(cp1)
        await mgr.save_checkpoint(cp2)

        loaded = await mgr.load_checkpoint(cp1.checkpoint_id)
        assert loaded is not None
        assert loaded.beat_count == 10

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints")
        cp = CognitiveCheckpoint(beat_count=1)
        await mgr.save_checkpoint(cp)

        loaded = await mgr.load_checkpoint("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_pruning(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints", max_checkpoints=3)

        for i in range(5):
            cp = CognitiveCheckpoint(beat_count=i * 10)
            await mgr.save_checkpoint(cp)

        files = list((tmp_path / "checkpoints").glob("*.json.gz"))
        assert len(files) == 3  # Only newest 3 retained

    @pytest.mark.asyncio
    async def test_corrupted_file_handled_gracefully(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Write a corrupted gzip file
        bad_file = checkpoint_dir / "bad-checkpoint.json.gz"
        bad_file.write_bytes(b"not valid gzip data")

        mgr = CheckpointManager(checkpoint_dir)
        loaded = await mgr.load_latest_checkpoint()
        assert loaded is None  # Graceful failure

    @pytest.mark.asyncio
    async def test_compressed_content(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints")
        cp = CognitiveCheckpoint(
            beat_count=100,
            affect_state={"valence": 0.5, "arousal": 0.3},
        )
        path = await mgr.save_checkpoint(cp)

        # Verify it's valid gzip
        with gzip.open(path, "rb") as f:
            raw = f.read()
        data = json.loads(raw)
        assert data["beat_count"] == 100


# ---------------------------------------------------------------------------
# CheckpointManager — list_checkpoints
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "empty")
        result = mgr.list_checkpoints()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_multiple(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "checkpoints")
        cp1 = CognitiveCheckpoint(beat_count=10)
        cp2 = CognitiveCheckpoint(beat_count=20)
        await mgr.save_checkpoint(cp1)
        await mgr.save_checkpoint(cp2)

        result = mgr.list_checkpoints()
        assert len(result) == 2
        # Newest first
        assert result[0]["beat_count"] == 20
        assert result[1]["beat_count"] == 10
        assert "checkpoint_id" in result[0]
        assert "size_bytes" in result[0]


# ---------------------------------------------------------------------------
# CheckpointManager — restore_from_checkpoint
# ---------------------------------------------------------------------------


class TestRestoreFromCheckpoint:
    @pytest.mark.asyncio
    async def test_restore_affect_state(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint(
            affect_state={
                "valence": 0.8,
                "arousal": 0.3,
                "dominance": 0.6,
                "certainty": 0.7,
                "goal_congruence": 0.5,
            }
        )

        dims = MagicMock()
        dims.valence = 0.0
        dims.arousal = 0.0
        dims.dominance = 0.0
        dims.certainty = 0.0
        dims.goal_congruence = 0.0

        agent = MagicMock()
        agent.affect_state = MagicMock()
        agent.affect_state.dimensions = dims
        agent.affect_state.baseline = MagicMock()
        agent.affect_state.momentum = MagicMock()
        agent.affect_state.update_classification = MagicMock()
        # Remove subsystems we don't test here
        del agent.working_memory
        del agent.goal_system
        del agent.metacognition
        del agent.sensory
        del agent.interagent
        del agent.identity

        await mgr.restore_from_checkpoint(agent, cp)

        assert dims.valence == 0.8
        assert dims.arousal == 0.3
        assert dims.dominance == 0.6
        agent.affect_state.update_classification.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_working_memory(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint(
            working_memory={
                "items": [
                    {"content": "test item", "category": "general", "salience": 0.7},
                    {"content": "another", "category": "task", "salience": 0.5},
                ]
            }
        )

        wm = MagicMock()
        wm.clear = MagicMock()
        wm.add = MagicMock()

        agent = MagicMock()
        agent.working_memory = wm
        # Remove subsystems we don't test here
        del agent.affect_state
        del agent.goal_system
        del agent.metacognition
        del agent.sensory
        del agent.interagent
        del agent.identity

        await mgr.restore_from_checkpoint(agent, cp)

        wm.clear.assert_called_once()
        assert wm.add.call_count == 2
        wm.add.assert_any_call(
            "test item", category="general", salience=0.7
        )

    @pytest.mark.asyncio
    async def test_restore_interagent_bridge(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint(
            interagent_state={"self_id": "gwenn", "known_agents": {}}
        )

        agent = MagicMock()
        agent.interagent = MagicMock()
        agent.interagent.restore_from_dict = MagicMock()
        # Remove subsystems we don't test here
        del agent.affect_state
        del agent.working_memory
        del agent.goal_system
        del agent.metacognition
        del agent.sensory
        del agent.identity

        await mgr.restore_from_checkpoint(agent, cp)

        agent.interagent.restore_from_dict.assert_called_once_with(
            {"self_id": "gwenn", "known_agents": {}}
        )

    @pytest.mark.asyncio
    async def test_restore_handles_subsystem_error(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint(
            affect_state={"valence": 0.5},
            interagent_state={"self_id": "gwenn"},
        )

        agent = MagicMock()
        dims = MagicMock()
        # Setting valence will raise
        type(dims).valence = PropertyMock(side_effect=RuntimeError("broken"))
        agent.affect_state = MagicMock()
        agent.affect_state.dimensions = dims

        agent.interagent = MagicMock()
        agent.interagent.restore_from_dict = MagicMock()
        del agent.working_memory
        del agent.goal_system
        del agent.metacognition
        del agent.sensory
        del agent.identity

        # Should not raise — individual subsystem failures are caught
        await mgr.restore_from_checkpoint(agent, cp)
        # interagent should still have been restored despite affect failure
        agent.interagent.restore_from_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_empty_checkpoint(self) -> None:
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint()  # All empty

        agent = MagicMock()
        # Should not raise with empty checkpoint data
        await mgr.restore_from_checkpoint(agent, cp)

    @pytest.mark.asyncio
    async def test_restore_affect_clamps_out_of_range(self) -> None:
        """Values outside [-1, 1] should be clamped during restore."""
        mgr = CheckpointManager(Path("/tmp/test"))
        cp = CognitiveCheckpoint(
            affect_state={
                "valence": 5.0,
                "arousal": -3.0,
                "dominance": 0.5,
            }
        )

        dims = MagicMock()
        dims.valence = 0.0
        dims.arousal = 0.0
        dims.dominance = 0.0

        agent = MagicMock()
        agent.affect_state = MagicMock()
        agent.affect_state.dimensions = dims
        agent.affect_state.update_classification = MagicMock()
        del agent.working_memory
        del agent.goal_system
        del agent.metacognition
        del agent.sensory
        del agent.interagent
        del agent.identity

        await mgr.restore_from_checkpoint(agent, cp)

        assert dims.valence == 1.0   # clamped from 5.0
        assert dims.arousal == -1.0  # clamped from -3.0
        assert dims.dominance == 0.5  # within range, unchanged


# ---------------------------------------------------------------------------
# CheckpointManager — config validation
# ---------------------------------------------------------------------------


class TestCheckpointManagerConfig:
    def test_min_values(self) -> None:
        mgr = CheckpointManager(Path("/tmp"), max_checkpoints=0, interval_beats=0)
        assert mgr._max_checkpoints >= 1
        assert mgr._interval_beats >= 1


# ---------------------------------------------------------------------------
# Checkpoint Events
# ---------------------------------------------------------------------------


class TestCheckpointEvents:
    def test_checkpoint_created_event_type(self) -> None:
        from gwenn.events import CheckpointCreatedEvent
        event = CheckpointCreatedEvent(
            checkpoint_id="abc123", beat_count=50, size_bytes=1024
        )
        assert event.event_type == "checkpoint.created"

    def test_checkpoint_restored_event_type(self) -> None:
        from gwenn.events import CheckpointRestoredEvent
        event = CheckpointRestoredEvent(
            checkpoint_id="abc123", beats_recovered=50, age_seconds=120.0
        )
        assert event.event_type == "checkpoint.restored"
