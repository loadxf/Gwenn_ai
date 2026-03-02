"""Tests for gwenn.healing — Self-Healing Engine (Tier 1)."""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.config import SelfHealingConfig
from gwenn.events import HealthCheckEvent, SelfHealEvent
from gwenn.healing import HealthIssue, RecoveryAction, SelfHealingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> SelfHealingConfig:
    defaults = {
        "GWENN_SELF_HEALING_ENABLED": True,
        "GWENN_SELF_HEALING_COOLDOWN": 300.0,
        "GWENN_SELF_HEALING_MAX_ACTIONS_HOUR": 20,
        "GWENN_SELF_HEALING_CHANNEL_RESTART": True,
        "GWENN_SELF_HEALING_STUCK_MULTIPLIER": 2.0,
        "GWENN_SELF_HEALING_MEMORY_THRESHOLD": 85.0,
        "GWENN_SELF_HEALING_ERROR_RATE_THRESHOLD": 5.0,
    }
    defaults.update(overrides)
    return SelfHealingConfig(**defaults)


def _make_engine(**config_overrides) -> SelfHealingEngine:
    config = _make_config(**config_overrides)
    event_bus = MagicMock()
    return SelfHealingEngine(event_bus=event_bus, config=config)


def _make_interoceptive(
    error_rate: float = 0.0,
    memory_percent: float = 50.0,
    cpu_percent: float = 30.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        error_rate=error_rate,
        memory_percent=memory_percent,
        cpu_percent=cpu_percent,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestHealthIssue:
    def test_defaults(self) -> None:
        issue = HealthIssue(category="test", component="x")
        assert issue.issue_id  # auto-generated
        assert issue.detected_at > 0
        assert issue.severity == 0.5

    def test_custom_fields(self) -> None:
        issue = HealthIssue(
            category="channel_crash",
            severity=0.9,
            component="telegram",
            detail="connection lost",
            suggested_action="restart_channel",
        )
        assert issue.category == "channel_crash"
        assert issue.component == "telegram"


class TestRecoveryAction:
    def test_defaults(self) -> None:
        action = RecoveryAction()
        assert action.action_id  # auto-generated
        assert action.success is None
        assert action.completed_at is None

    def test_completed_action(self) -> None:
        action = RecoveryAction(
            issue_id="abc",
            action_type="restart_channel",
            completed_at=time.time(),
            success=True,
            detail="restarted telegram",
        )
        assert action.success is True


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_no_issues_healthy(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={"telegram": {"crashed": False}},
            subagent_statuses={},
        )
        assert issues == []

    def test_detects_channel_crash(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={
                "telegram": {"crashed": True, "error": "connection reset"},
            },
            subagent_statuses={},
        )
        assert len(issues) == 1
        assert issues[0].category == "channel_crash"
        assert issues[0].component == "telegram"
        assert issues[0].suggested_action == "restart_channel"

    def test_detects_high_error_rate(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(error_rate=10.0),
            channel_statuses={},
            subagent_statuses={},
        )
        assert len(issues) == 1
        assert issues[0].category == "high_error_rate"
        assert issues[0].suggested_action == "reduce_heartbeat_load"

    def test_error_rate_below_threshold_no_issue(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(error_rate=3.0),
            channel_statuses={},
            subagent_statuses={},
        )
        assert len(issues) == 0

    def test_detects_stuck_subagent(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={
                "task-abc": {
                    "elapsed_seconds": 300.0,
                    "timeout_seconds": 120.0,
                    "is_stuck": True,
                },
            },
        )
        assert len(issues) == 1
        assert issues[0].category == "stuck_subagent"
        assert issues[0].suggested_action == "cancel_stuck_subagent"

    def test_non_stuck_subagent_no_issue(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={
                "task-abc": {
                    "elapsed_seconds": 50.0,
                    "timeout_seconds": 120.0,
                    "is_stuck": False,
                },
            },
        )
        assert len(issues) == 0

    def test_detects_memory_pressure(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(memory_percent=92.0),
            channel_statuses={},
            subagent_statuses={},
        )
        assert len(issues) == 1
        assert issues[0].category == "memory_pressure"
        assert issues[0].suggested_action == "force_consolidation"

    def test_memory_below_threshold_no_issue(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(memory_percent=60.0),
            channel_statuses={},
            subagent_statuses={},
        )
        assert len(issues) == 0

    def test_detects_stale_sessions(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={
                "telegram": {"crashed": False, "stale_sessions": 5},
            },
            subagent_statuses={},
        )
        assert len(issues) == 1
        assert issues[0].category == "stale_sessions"
        assert issues[0].suggested_action == "clear_stale_sessions"

    def test_multiple_issues_detected(self) -> None:
        engine = _make_engine()
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(error_rate=10.0, memory_percent=90.0),
            channel_statuses={
                "telegram": {"crashed": True, "error": "dead"},
            },
            subagent_statuses={
                "task-x": {"elapsed_seconds": 500.0, "timeout_seconds": 100.0, "is_stuck": True},
            },
        )
        categories = {i.category for i in issues}
        assert "channel_crash" in categories
        assert "high_error_rate" in categories
        assert "stuck_subagent" in categories
        assert "memory_pressure" in categories

    def test_issues_stored_in_recent(self) -> None:
        engine = _make_engine()
        engine.diagnose(
            interoceptive=_make_interoceptive(error_rate=10.0),
            channel_statuses={},
            subagent_statuses={},
        )
        assert len(engine.recent_issues) == 1


# ---------------------------------------------------------------------------
# Healing
# ---------------------------------------------------------------------------


class TestHeal:
    @pytest.mark.asyncio
    async def test_no_issues_no_actions(self) -> None:
        engine = _make_engine()
        actions = await engine.heal([], MagicMock(), MagicMock())
        assert actions == []

    @pytest.mark.asyncio
    async def test_reduce_heartbeat_load(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="high_error_rate",
            component="heartbeat",
            suggested_action="reduce_heartbeat_load",
        )
        heartbeat = MagicMock()
        heartbeat._interval = 30.0
        heartbeat._config = SimpleNamespace(
            heartbeat=SimpleNamespace(max_interval=120.0)
        )
        heartbeat._checkpoint_manager = None

        actions = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions) == 1
        assert actions[0].action_type == "reduce_heartbeat_load"
        assert actions[0].success is True
        assert heartbeat._interval == 120.0

    @pytest.mark.asyncio
    async def test_cancel_stuck_subagent(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="stuck_subagent",
            component="orchestrator",
            detail="task task-abc running 300s (timeout 120s)",
            suggested_action="cancel_stuck_subagent",
            target_id="task-abc",
        )
        agent = MagicMock()
        agent.orchestrator = MagicMock()
        agent.orchestrator.cancel = AsyncMock(return_value=True)

        actions = await engine.heal([issue], MagicMock(_checkpoint_manager=None), agent)
        assert len(actions) == 1
        assert actions[0].action_type == "cancel_stuck_subagent"
        assert actions[0].success is True
        agent.orchestrator.cancel.assert_called_once_with("task-abc")

    @pytest.mark.asyncio
    async def test_clear_stale_sessions(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="stale_sessions",
            component="sessions",
            suggested_action="clear_stale_sessions",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._session_manager = MagicMock()
        heartbeat._session_manager.expire_stale_sessions = MagicMock(return_value=3)

        actions = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions) == 1
        assert actions[0].action_type == "clear_stale_sessions"
        assert actions[0].success is True
        assert "3" in actions[0].detail

    @pytest.mark.asyncio
    async def test_force_consolidation(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="memory_pressure",
            component="system",
            suggested_action="force_consolidation",
        )
        agent = MagicMock()
        agent.consolidate_memories = AsyncMock()
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None

        actions = await engine.heal([issue], heartbeat, agent)
        assert len(actions) == 1
        assert actions[0].action_type == "force_consolidation"
        assert actions[0].success is True
        agent.consolidate_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_emits_self_heal_event(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="high_error_rate",
            component="heartbeat",
            suggested_action="reduce_heartbeat_load",
        )
        heartbeat = MagicMock()
        heartbeat._interval = 30.0
        heartbeat._config = SimpleNamespace(
            heartbeat=SimpleNamespace(max_interval=120.0)
        )
        heartbeat._checkpoint_manager = None

        await engine.heal([issue], heartbeat, MagicMock())
        engine._event_bus.emit.assert_called_once()
        event = engine._event_bus.emit.call_args[0][0]
        assert isinstance(event, SelfHealEvent)
        assert event.category == "high_error_rate"
        assert event.action_type == "reduce_heartbeat_load"

    @pytest.mark.asyncio
    async def test_action_stored_in_recent(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="stale_sessions",
            component="sessions",
            suggested_action="clear_stale_sessions",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._session_manager = MagicMock()
        heartbeat._session_manager.expire_stale_sessions = MagicMock(return_value=0)

        await engine.heal([issue], heartbeat, MagicMock())
        assert len(engine.recent_actions) == 1


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_prevents_retry(self) -> None:
        engine = _make_engine(**{"GWENN_SELF_HEALING_COOLDOWN": 300.0})
        issue = HealthIssue(
            category="stale_sessions",
            component="sessions",
            suggested_action="clear_stale_sessions",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._session_manager = MagicMock()
        heartbeat._session_manager.expire_stale_sessions = MagicMock(return_value=0)

        # First heal succeeds
        actions1 = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions1) == 1

        # Second heal within cooldown → skipped
        actions2 = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions2) == 0

    @pytest.mark.asyncio
    async def test_different_components_have_separate_cooldowns(self) -> None:
        engine = _make_engine()
        issue1 = HealthIssue(
            category="channel_crash",
            component="telegram",
            suggested_action="restart_channel",
        )
        issue2 = HealthIssue(
            category="channel_crash",
            component="discord",
            suggested_action="restart_channel",
        )

        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._agent = MagicMock()
        heartbeat._agent._platform_channels = []

        # First issue
        actions1 = await engine.heal([issue1], heartbeat, heartbeat._agent)
        assert len(actions1) == 1

        # Different component — not blocked by cooldown
        actions2 = await engine.heal([issue2], heartbeat, heartbeat._agent)
        assert len(actions2) == 1


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_stops_actions(self) -> None:
        engine = _make_engine(**{"GWENN_SELF_HEALING_MAX_ACTIONS_HOUR": 2})
        # Fill action timestamps (monotonic)
        now = time.monotonic()
        engine._action_timestamps = deque([now, now], maxlen=200)

        issue = HealthIssue(
            category="stale_sessions",
            component="sessions",
            suggested_action="clear_stale_sessions",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._session_manager = MagicMock()
        heartbeat._session_manager.expire_stale_sessions = MagicMock(return_value=0)

        actions = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions) == 0  # Rate limited

    def test_old_timestamps_expire(self) -> None:
        engine = _make_engine(**{"GWENN_SELF_HEALING_MAX_ACTIONS_HOUR": 5})
        old = time.monotonic() - 7200  # 2 hours ago
        engine._action_timestamps = deque([old, old, old], maxlen=200)
        assert engine._check_rate_limit() is True  # Old ones don't count


# ---------------------------------------------------------------------------
# Recovery Verification
# ---------------------------------------------------------------------------


class TestVerifyRecovery:
    def test_issue_resolved(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(category="channel_crash", component="telegram")
        engine._recent_issues.append(issue)
        action = RecoveryAction(issue_id=issue.issue_id, action_type="restart_channel")

        # No matching issue in current check
        resolved = engine.verify_recovery(action, [])
        assert resolved is True

    def test_issue_persists(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(category="channel_crash", component="telegram")
        engine._recent_issues.append(issue)
        action = RecoveryAction(issue_id=issue.issue_id, action_type="restart_channel")

        # Same issue still present
        current = [HealthIssue(category="channel_crash", component="telegram")]
        resolved = engine.verify_recovery(action, current)
        assert resolved is False

    def test_different_component_not_confused(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(category="channel_crash", component="telegram")
        engine._recent_issues.append(issue)
        action = RecoveryAction(issue_id=issue.issue_id, action_type="restart_channel")

        # Different component still has issue — doesn't affect our check
        current = [HealthIssue(category="channel_crash", component="discord")]
        resolved = engine.verify_recovery(action, current)
        assert resolved is True

    def test_unknown_issue_assumed_resolved(self) -> None:
        engine = _make_engine()
        action = RecoveryAction(issue_id="nonexistent", action_type="restart_channel")
        resolved = engine.verify_recovery(action, [])
        assert resolved is True


# ---------------------------------------------------------------------------
# Channel Restart
# ---------------------------------------------------------------------------


class TestRestartChannel:
    @pytest.mark.asyncio
    async def test_restart_disabled_by_config(self) -> None:
        engine = _make_engine(**{"GWENN_SELF_HEALING_CHANNEL_RESTART": False})
        issue = HealthIssue(
            category="channel_crash",
            component="telegram",
            suggested_action="restart_channel",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None

        actions = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions) == 1
        assert actions[0].success is False
        assert "disabled" in actions[0].detail

    @pytest.mark.asyncio
    async def test_restart_no_agent(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="channel_crash",
            component="telegram",
            suggested_action="restart_channel",
        )
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None
        heartbeat._agent = None

        actions = await engine.heal([issue], heartbeat, MagicMock())
        assert len(actions) == 1
        assert actions[0].success is False


# ---------------------------------------------------------------------------
# Action failure handling
# ---------------------------------------------------------------------------


class TestActionFailure:
    @pytest.mark.asyncio
    async def test_action_exception_handled(self) -> None:
        """If a recovery action itself raises, it should be caught and recorded."""
        engine = _make_engine()
        issue = HealthIssue(
            category="memory_pressure",
            component="system",
            suggested_action="force_consolidation",
        )
        agent = MagicMock()
        agent.consolidate_memories = AsyncMock(side_effect=RuntimeError("OOM"))
        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = None

        actions = await engine.heal([issue], heartbeat, agent)
        assert len(actions) == 1
        assert actions[0].success is False


# ---------------------------------------------------------------------------
# Health Summary
# ---------------------------------------------------------------------------


class TestHealthSummary:
    def test_healthy_when_no_actions(self) -> None:
        engine = _make_engine()
        summary = engine.health_summary
        assert summary["status"] == "healthy"
        assert summary["recent_issues_count"] == 0
        assert summary["actions_this_hour"] == 0

    def test_recovering_when_pending(self) -> None:
        engine = _make_engine()
        engine._recent_actions.append(
            RecoveryAction(action_type="restart_channel", success=None)
        )
        summary = engine.health_summary
        assert summary["status"] == "recovering"

    def test_degraded_when_recent_failure(self) -> None:
        engine = _make_engine()
        engine._recent_actions.append(
            RecoveryAction(action_type="restart_channel", success=False)
        )
        engine._action_timestamps.append(time.monotonic())
        summary = engine.health_summary
        assert summary["status"] == "degraded"


# ---------------------------------------------------------------------------
# Pre-recovery Checkpoint
# ---------------------------------------------------------------------------


class TestPreRecoveryCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_triggered_before_risky_action(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="memory_pressure",
            component="system",
            suggested_action="force_consolidation",
        )

        # Set up heartbeat with checkpoint manager that's overdue
        mgr = MagicMock()
        mgr._interval_beats = 50
        mgr._last_checkpoint_beat = 0
        mgr.create_checkpoint = AsyncMock(return_value=MagicMock())
        mgr.save_checkpoint = AsyncMock()

        agent = MagicMock()
        agent.consolidate_memories = AsyncMock()

        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = mgr
        heartbeat._beat_count = 100
        heartbeat._agent = agent

        await engine.heal([issue], heartbeat, agent)
        mgr.create_checkpoint.assert_called_once()
        mgr.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_skipped_if_recent(self) -> None:
        engine = _make_engine()
        issue = HealthIssue(
            category="memory_pressure",
            component="system",
            suggested_action="force_consolidation",
        )

        mgr = MagicMock()
        mgr._interval_beats = 50
        mgr._last_checkpoint_beat = 45  # Recent checkpoint at beat 45
        mgr.create_checkpoint = AsyncMock()

        agent = MagicMock()
        agent.consolidate_memories = AsyncMock()

        heartbeat = MagicMock()
        heartbeat._checkpoint_manager = mgr
        heartbeat._beat_count = 50  # Only 5 beats since last → skip
        heartbeat._agent = agent

        await engine.heal([issue], heartbeat, agent)
        mgr.create_checkpoint.assert_not_called()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestSelfHealingEvents:
    def test_self_heal_event_type(self) -> None:
        event = SelfHealEvent(
            issue_id="abc",
            action_id="def",
            category="channel_crash",
            action_type="restart_channel",
            component="telegram",
        )
        assert event.event_type == "self.heal"

    def test_health_check_event_type(self) -> None:
        event = HealthCheckEvent(
            issues_found=2, actions_taken=1, all_healthy=False,
        )
        assert event.event_type == "health.check"

    def test_health_check_all_healthy(self) -> None:
        event = HealthCheckEvent(
            issues_found=0, actions_taken=0, all_healthy=True,
        )
        assert event.all_healthy is True


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


class TestSelfHealingConfig:
    def test_defaults(self) -> None:
        config = SelfHealingConfig()
        assert config.enabled is True
        assert config.cooldown_seconds == 300.0
        assert config.max_actions_per_hour == 20

    def test_min_values_enforced(self) -> None:
        config = SelfHealingConfig(
            GWENN_SELF_HEALING_COOLDOWN=1.0,
            GWENN_SELF_HEALING_MAX_ACTIONS_HOUR=0,
            GWENN_SELF_HEALING_STUCK_MULTIPLIER=0.5,
            GWENN_SELF_HEALING_MEMORY_THRESHOLD=10.0,
            GWENN_SELF_HEALING_ERROR_RATE_THRESHOLD=0.01,
        )
        assert config.cooldown_seconds >= 10.0
        assert config.max_actions_per_hour >= 1
        assert config.stuck_subagent_timeout_multiplier >= 1.1
        assert config.memory_pressure_threshold >= 50.0
        assert config.error_rate_threshold >= 0.1

    def test_memory_threshold_clamped(self) -> None:
        config = SelfHealingConfig(GWENN_SELF_HEALING_MEMORY_THRESHOLD=999.0)
        assert config.memory_pressure_threshold <= 99.0


# ---------------------------------------------------------------------------
# Provenance Auditing
# ---------------------------------------------------------------------------


class TestProvenanceAudit:
    """Tests for knowledge provenance health checking and deprecation."""

    def _make_engine_with_memory(self):
        """Create engine with wired semantic and episodic memory."""
        from gwenn.memory.episodic import Episode, EpisodicMemory
        from gwenn.memory.semantic import SemanticMemory

        engine = _make_engine()
        em = EpisodicMemory()
        sm = SemanticMemory()
        engine.set_memory_references(sm, em)
        return engine, sm, em

    def test_orphaned_node_detected(self) -> None:
        """A node whose source episodes are missing should be flagged."""
        engine, sm, em = self._make_engine_with_memory()
        # Force the audit interval to trigger on first call
        engine._provenance_audit_interval = 1
        engine._diagnose_count = 0

        # Store a node referencing an episode that doesn't exist
        sm.store_knowledge(
            "test fact", "Some knowledge",
            source_episode_id="nonexistent-ep",
            confidence=0.8,
        )

        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={},
        )
        provenance_issues = [i for i in issues if i.category == "knowledge_provenance"]
        assert len(provenance_issues) == 1
        assert provenance_issues[0].suggested_action == "deprecate_weak_node"
        assert "unsupported" in provenance_issues[0].detail

    def test_supported_node_no_issue(self) -> None:
        """A node whose source episodes exist and match should not be flagged."""
        from gwenn.memory.episodic import Episode

        engine, sm, em = self._make_engine_with_memory()
        engine._provenance_audit_interval = 1
        engine._diagnose_count = 0

        # Create matching episode
        em.encode(Episode(
            episode_id="ep-1",
            content="Python is a popular programming language for scripting",
            importance=0.7,
            tags=["python", "programming"],
        ))
        # Store a node supported by this episode
        sm.store_knowledge(
            "python", "A popular programming language",
            source_episode_id="ep-1",
            confidence=0.8,
        )

        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={},
        )
        provenance_issues = [i for i in issues if i.category == "knowledge_provenance"]
        assert len(provenance_issues) == 0

    def test_audit_skipped_outside_interval(self) -> None:
        """Provenance audit should only run every N diagnose cycles."""
        engine, sm, em = self._make_engine_with_memory()
        engine._provenance_audit_interval = 25
        engine._diagnose_count = 0

        sm.store_knowledge(
            "orphan", "Unknown fact",
            source_episode_id="missing-ep",
            confidence=0.9,
        )

        # First diagnose: count=0, 1%25!=0, so audit skipped
        issues = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={},
        )
        provenance_issues = [i for i in issues if i.category == "knowledge_provenance"]
        assert len(provenance_issues) == 0

    @pytest.mark.asyncio
    async def test_deprecation_reduces_confidence(self) -> None:
        """deprecate_weak_node should halve the node's confidence."""
        engine, sm, em = self._make_engine_with_memory()

        node = sm.store_knowledge(
            "weak fact", "Unsupported claim",
            source_episode_id="gone-ep",
            confidence=0.8,
        )

        issue = HealthIssue(
            category="knowledge_provenance",
            component="semantic_memory",
            suggested_action="deprecate_weak_node",
            target_id=node.node_id,
        )

        actions = await engine.heal([issue], MagicMock(_checkpoint_manager=None), MagicMock())
        assert len(actions) == 1
        assert actions[0].success is True
        assert node.confidence == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_provenance_rate_limiting(self) -> None:
        """Max 5 provenance deprecation actions per hour."""
        engine, sm, em = self._make_engine_with_memory()

        # Fill the provenance rate limit
        now = time.monotonic()
        engine._provenance_actions_timestamps = deque(
            [now] * 5, maxlen=50,
        )

        node = sm.store_knowledge(
            "rate-limited", "Will not be deprecated",
            source_episode_id="x",
            confidence=0.9,
        )

        issue = HealthIssue(
            category="knowledge_provenance",
            component="semantic_memory",
            suggested_action="deprecate_weak_node",
            target_id=node.node_id,
        )

        actions = await engine.heal([issue], MagicMock(_checkpoint_manager=None), MagicMock())
        assert len(actions) == 1
        assert actions[0].success is False
        assert "rate limit" in actions[0].detail
        # Confidence unchanged
        assert node.confidence == pytest.approx(0.9)

    def test_per_node_cooldown(self) -> None:
        """Nodes audited within cooldown window should be skipped."""
        engine, sm, em = self._make_engine_with_memory()
        engine._provenance_audit_interval = 1
        engine._diagnose_count = 0

        sm.store_knowledge(
            "cooled", "Some fact",
            source_episode_id="missing-ep",
            confidence=0.7,
        )

        # First diagnose: detects the issue
        issues1 = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={},
        )
        prov1 = [i for i in issues1 if i.category == "knowledge_provenance"]
        assert len(prov1) == 1

        # Second diagnose: node is within cooldown, should be skipped
        issues2 = engine.diagnose(
            interoceptive=_make_interoceptive(),
            channel_statuses={},
            subagent_statuses={},
        )
        prov2 = [i for i in issues2 if i.category == "knowledge_provenance"]
        assert len(prov2) == 0
