"""
Self-Healing Engine — Autonomous Runtime Recovery (Tier 1).

Detects health issues (crashed channels, stuck subagents, memory pressure,
high error rates, stale sessions) and applies deterministic recovery actions
that require no LLM reasoning or human intervention.

Integrates with the heartbeat cycle:
  SENSE:     diagnose() detects anomalies
  INTEGRATE: heal() applies recovery actions
  SCHEDULE:  heartbeat accelerates next beat to verify recovery
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gwenn.config import SelfHealingConfig
    from gwenn.events import EventBus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HealthIssue(BaseModel):
    """A detected health problem."""

    issue_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str  # channel_crash, high_error_rate, stuck_subagent, memory_pressure, stale_sessions
    severity: float = 0.5  # 0.0-1.0
    detected_at: float = Field(default_factory=time.time)
    component: str = ""  # telegram, discord, slack, gateway, orchestrator
    detail: str = ""
    suggested_action: str = ""


class RecoveryAction(BaseModel):
    """A recovery action taken."""

    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    issue_id: str = ""
    action_type: str = ""  # restart_channel, cancel_stuck_subagent, clear_stale_sessions, etc.
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    success: bool | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SelfHealingEngine:
    """Detects health issues and applies autonomous recovery actions.

    Tier 1 only — deterministic recovery, no LLM reasoning needed.
    All actions are safe, reversible, and logged via SelfHealEvent.
    """

    def __init__(self, event_bus: EventBus, config: SelfHealingConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._recent_issues: deque[HealthIssue] = deque(maxlen=100)
        self._recent_actions: deque[RecoveryAction] = deque(maxlen=100)
        # category → earliest retry time (monotonic)
        self._cooldowns: dict[str, float] = {}
        # Track actions per hour for rate limiting
        self._action_timestamps: deque[float] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Diagnosis — called during _sense()
    # ------------------------------------------------------------------

    def diagnose(
        self,
        interoceptive: Any,
        channel_statuses: dict[str, Any],
        subagent_statuses: dict[str, Any],
    ) -> list[HealthIssue]:
        """Analyze current state for health issues.

        Returns list of detected issues, each with a suggested action.
        """
        issues: list[HealthIssue] = []

        # 1. Channel crashes
        for name, status in channel_statuses.items():
            if status.get("crashed"):
                issues.append(HealthIssue(
                    category="channel_crash",
                    severity=0.9,
                    component=name,
                    detail=status.get("error", "channel crashed"),
                    suggested_action="restart_channel",
                ))

        # 2. High error rate
        error_rate = getattr(interoceptive, "error_rate", 0.0)
        if error_rate > self._config.error_rate_threshold:
            issues.append(HealthIssue(
                category="high_error_rate",
                severity=min(1.0, error_rate / (self._config.error_rate_threshold * 2)),
                component="heartbeat",
                detail=f"error rate {error_rate:.1f}/min exceeds threshold {self._config.error_rate_threshold}",
                suggested_action="reduce_heartbeat_load",
            ))

        # 3. Stuck subagents
        for task_id, status in subagent_statuses.items():
            if status.get("is_stuck"):
                issues.append(HealthIssue(
                    category="stuck_subagent",
                    severity=0.7,
                    component="orchestrator",
                    detail=f"task {task_id} running {status.get('elapsed_seconds', 0):.0f}s (timeout {status.get('timeout_seconds', 0):.0f}s)",
                    suggested_action="cancel_stuck_subagent",
                ))

        # 4. Memory pressure
        memory_percent = getattr(interoceptive, "memory_percent", 0.0)
        if memory_percent > self._config.memory_pressure_threshold:
            issues.append(HealthIssue(
                category="memory_pressure",
                severity=min(1.0, memory_percent / 100.0),
                component="system",
                detail=f"memory at {memory_percent:.1f}% (threshold {self._config.memory_pressure_threshold}%)",
                suggested_action="force_consolidation",
            ))

        # 5. Stale sessions
        stale_count = 0
        for name, status in channel_statuses.items():
            stale_count += status.get("stale_sessions", 0)
        if stale_count > 0:
            issues.append(HealthIssue(
                category="stale_sessions",
                severity=min(1.0, stale_count / 50.0),
                component="sessions",
                detail=f"{stale_count} stale session(s) detected",
                suggested_action="clear_stale_sessions",
            ))

        for issue in issues:
            self._recent_issues.append(issue)

        return issues

    # ------------------------------------------------------------------
    # Recovery — called during _integrate()
    # ------------------------------------------------------------------

    async def heal(
        self,
        issues: list[HealthIssue],
        heartbeat: Any,
        agent: Any,
    ) -> list[RecoveryAction]:
        """Apply recovery actions for detected issues.

        Each action checks cooldown and hourly rate limit before executing.
        Returns list of actions actually taken.
        """
        from gwenn.events import SelfHealEvent

        actions: list[RecoveryAction] = []
        now_mono = time.monotonic()

        for issue in issues:
            # Check per-category cooldown
            cooldown_key = f"{issue.category}:{issue.component}"
            cooldown_until = self._cooldowns.get(cooldown_key, 0.0)
            if now_mono < cooldown_until:
                logger.debug(
                    "healing.cooldown_active",
                    category=issue.category,
                    component=issue.component,
                    remaining=round(cooldown_until - now_mono, 1),
                )
                continue

            # Check hourly rate limit
            if not self._check_rate_limit():
                logger.warning("healing.rate_limit_reached", max_per_hour=self._config.max_actions_per_hour)
                break

            action = await self._dispatch_action(issue, heartbeat, agent)
            if action is not None:
                actions.append(action)
                self._recent_actions.append(action)
                self._action_timestamps.append(time.time())
                # Set cooldown
                self._cooldowns[cooldown_key] = now_mono + self._config.cooldown_seconds

                self._event_bus.emit(SelfHealEvent(
                    issue_id=issue.issue_id,
                    action_id=action.action_id,
                    category=issue.category,
                    action_type=action.action_type,
                    component=issue.component,
                    success=action.success,
                    detail=action.detail,
                    diagnosis=issue.detail,
                    recovery=action.action_type,
                ))

        return actions

    async def _dispatch_action(
        self,
        issue: HealthIssue,
        heartbeat: Any,
        agent: Any,
    ) -> RecoveryAction | None:
        """Route an issue to its recovery handler. Returns None if no action applies."""
        try:
            if issue.suggested_action == "restart_channel":
                return await self._restart_channel(issue, heartbeat)
            elif issue.suggested_action == "reduce_heartbeat_load":
                return await self._reduce_heartbeat_load(issue, heartbeat)
            elif issue.suggested_action == "cancel_stuck_subagent":
                return await self._cancel_stuck_subagent(issue, agent)
            elif issue.suggested_action == "force_consolidation":
                return await self._force_consolidation(issue, agent, heartbeat)
            elif issue.suggested_action == "clear_stale_sessions":
                return await self._clear_stale_sessions(issue, heartbeat)
            else:
                logger.debug("healing.unknown_action", action=issue.suggested_action)
                return None
        except Exception:
            logger.warning(
                "healing.action_failed",
                category=issue.category,
                action=issue.suggested_action,
                exc_info=True,
            )
            return RecoveryAction(
                issue_id=issue.issue_id,
                action_type=issue.suggested_action,
                completed_at=time.time(),
                success=False,
                detail="recovery action raised an exception",
            )

    # ------------------------------------------------------------------
    # Individual recovery actions
    # ------------------------------------------------------------------

    async def _restart_channel(
        self, issue: HealthIssue, heartbeat: Any
    ) -> RecoveryAction:
        """Restart a crashed channel.

        Re-builds and re-starts the specific channel using the same startup
        logic as heartbeat._start_channels() but for a single channel.
        """
        action = RecoveryAction(
            issue_id=issue.issue_id,
            action_type="restart_channel",
        )

        if not self._config.channel_restart_enabled:
            action.completed_at = time.time()
            action.success = False
            action.detail = "channel restart disabled by config"
            return action

        component = issue.component
        try:
            # Trigger checkpoint before risky recovery
            await self._trigger_checkpoint_if_needed(heartbeat)

            restarted = await self._do_restart_channel(component, heartbeat)
            action.completed_at = time.time()
            action.success = restarted
            action.detail = f"restarted {component}" if restarted else f"could not restart {component}"
        except Exception as e:
            action.completed_at = time.time()
            action.success = False
            action.detail = f"restart failed: {e}"

        logger.info(
            "healing.restart_channel",
            component=component,
            success=action.success,
            detail=action.detail,
        )
        return action

    async def _do_restart_channel(self, component: str, heartbeat: Any) -> bool:
        """Attempt to restart a single channel within the heartbeat's channel system."""
        agent = getattr(heartbeat, "_agent", None)
        if agent is None:
            return False

        # Find and stop the crashed channel
        channels = getattr(agent, "_platform_channels", [])
        crashed = None
        for ch in channels:
            ch_name = getattr(ch, "channel_name", "")
            if ch_name == component:
                crashed = ch
                break

        if crashed is None:
            return False

        # Stop the old channel gracefully
        try:
            stop = getattr(crashed, "stop", None)
            if callable(stop):
                await stop()
        except Exception:
            logger.debug("healing.channel_stop_failed", component=component, exc_info=True)

        # Remove from agent's channel list
        try:
            unreg = getattr(agent, "unregister_channel", None)
            if callable(unreg):
                unreg(crashed)
        except Exception:
            pass

        # Re-build and start the channel
        try:
            from gwenn.channels.startup import build_channels
            full_config = getattr(heartbeat, "_full_config", None)
            if full_config is None:
                return False

            sessions, new_channels = build_channels(agent, channel_list=[component])
            if not new_channels:
                return False

            new_ch = new_channels[0]
            await new_ch.start()
            reg = getattr(agent, "register_channel", None)
            if callable(reg):
                reg(new_ch)

            logger.info("healing.channel_restarted", component=component)
            return True
        except Exception:
            logger.warning("healing.channel_rebuild_failed", component=component, exc_info=True)
            return False

    async def _reduce_heartbeat_load(
        self, issue: HealthIssue, heartbeat: Any
    ) -> RecoveryAction:
        """Temporarily increase heartbeat interval to reduce load."""
        action = RecoveryAction(
            issue_id=issue.issue_id,
            action_type="reduce_heartbeat_load",
        )

        try:
            max_interval = getattr(heartbeat, "_config", None)
            if max_interval is not None:
                max_interval = getattr(max_interval, "heartbeat", None)
                if max_interval is not None:
                    max_interval = getattr(max_interval, "max_interval", 120.0)
                else:
                    max_interval = 120.0
            else:
                max_interval = 120.0

            old_interval = getattr(heartbeat, "_interval", 30.0)
            heartbeat._interval = max_interval
            action.completed_at = time.time()
            action.success = True
            action.detail = f"interval {old_interval:.0f}s → {max_interval:.0f}s"
        except Exception as e:
            action.completed_at = time.time()
            action.success = False
            action.detail = f"failed: {e}"

        logger.info(
            "healing.reduce_load",
            success=action.success,
            detail=action.detail,
        )
        return action

    async def _cancel_stuck_subagent(
        self, issue: HealthIssue, agent: Any
    ) -> RecoveryAction:
        """Cancel a subagent that has exceeded its timeout multiplier."""
        action = RecoveryAction(
            issue_id=issue.issue_id,
            action_type="cancel_stuck_subagent",
        )

        # Extract task_id from issue detail
        task_id = ""
        detail = issue.detail
        if detail.startswith("task "):
            task_id = detail.split(" ")[1]

        if not task_id:
            action.completed_at = time.time()
            action.success = False
            action.detail = "could not extract task_id from issue"
            return action

        orchestrator = getattr(agent, "orchestrator", None)
        if orchestrator is None:
            action.completed_at = time.time()
            action.success = False
            action.detail = "no orchestrator available"
            return action

        try:
            cancel = getattr(orchestrator, "cancel", None)
            if callable(cancel):
                cancelled = await cancel(task_id)
                action.completed_at = time.time()
                action.success = cancelled
                action.detail = f"cancelled task {task_id}" if cancelled else f"task {task_id} not found"
            else:
                action.completed_at = time.time()
                action.success = False
                action.detail = "orchestrator has no cancel method"
        except Exception as e:
            action.completed_at = time.time()
            action.success = False
            action.detail = f"cancel failed: {e}"

        logger.info(
            "healing.cancel_subagent",
            task_id=task_id,
            success=action.success,
        )
        return action

    async def _force_consolidation(
        self, issue: HealthIssue, agent: Any, heartbeat: Any
    ) -> RecoveryAction:
        """Trigger memory consolidation under memory pressure."""
        action = RecoveryAction(
            issue_id=issue.issue_id,
            action_type="force_consolidation",
        )

        try:
            # Trigger checkpoint before risky recovery
            await self._trigger_checkpoint_if_needed(heartbeat)

            consolidate = getattr(agent, "consolidate_memories", None)
            if callable(consolidate):
                await consolidate()
                action.completed_at = time.time()
                action.success = True
                action.detail = "memory consolidation triggered"
            else:
                action.completed_at = time.time()
                action.success = False
                action.detail = "agent has no consolidate_memories method"
        except Exception as e:
            action.completed_at = time.time()
            action.success = False
            action.detail = f"consolidation failed: {e}"

        logger.info(
            "healing.force_consolidation",
            success=action.success,
            detail=action.detail,
        )
        return action

    async def _clear_stale_sessions(
        self, issue: HealthIssue, heartbeat: Any
    ) -> RecoveryAction:
        """Expire sessions that are well past their TTL."""
        action = RecoveryAction(
            issue_id=issue.issue_id,
            action_type="clear_stale_sessions",
        )

        try:
            session_mgr = getattr(heartbeat, "_session_manager", None)
            if session_mgr is None:
                action.completed_at = time.time()
                action.success = False
                action.detail = "no session manager available"
                return action

            expire = getattr(session_mgr, "expire_stale_sessions", None)
            if callable(expire):
                count = expire()
                action.completed_at = time.time()
                action.success = True
                action.detail = f"expired {count} stale session(s)"
            else:
                action.completed_at = time.time()
                action.success = False
                action.detail = "session manager has no expire method"
        except Exception as e:
            action.completed_at = time.time()
            action.success = False
            action.detail = f"session cleanup failed: {e}"

        logger.info(
            "healing.clear_sessions",
            success=action.success,
            detail=action.detail,
        )
        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Return True if we're within the hourly action limit."""
        now = time.time()
        cutoff = now - 3600.0
        # Remove old timestamps
        while self._action_timestamps and self._action_timestamps[0] < cutoff:
            self._action_timestamps.popleft()
        return len(self._action_timestamps) < self._config.max_actions_per_hour

    async def _trigger_checkpoint_if_needed(self, heartbeat: Any) -> None:
        """Create a checkpoint before risky recovery if one hasn't been taken recently."""
        mgr = getattr(heartbeat, "_checkpoint_manager", None)
        if mgr is None:
            return
        beat_count = getattr(heartbeat, "_beat_count", 0)
        # Only checkpoint if we're past half the interval since last
        interval = getattr(mgr, "_interval_beats", 50)
        last = getattr(mgr, "_last_checkpoint_beat", 0)
        if (beat_count - last) < interval // 2:
            return  # Recent enough
        try:
            agent = getattr(heartbeat, "_agent", None)
            if agent is None:
                return
            checkpoint = await mgr.create_checkpoint(agent, heartbeat)
            await mgr.save_checkpoint(checkpoint)
            logger.info("healing.pre_recovery_checkpoint", beat_count=beat_count)
        except Exception:
            logger.debug("healing.checkpoint_before_recovery_failed", exc_info=True)

    def verify_recovery(
        self,
        action: RecoveryAction,
        current_issues: list[HealthIssue],
    ) -> bool:
        """Check if a previous recovery action resolved its issue.

        Called on the next beat after a recovery action. Returns True
        if the issue category+component combo is no longer present.
        """
        # Find the original issue
        original: HealthIssue | None = None
        for issue in self._recent_issues:
            if issue.issue_id == action.issue_id:
                original = issue
                break
        if original is None:
            return True  # Can't find original → assume resolved

        # Check if same category+component still appears
        for issue in current_issues:
            if issue.category == original.category and issue.component == original.component:
                return False
        return True

    @property
    def recent_issues(self) -> list[HealthIssue]:
        """Return recent issues (newest first)."""
        return list(reversed(self._recent_issues))

    @property
    def recent_actions(self) -> list[RecoveryAction]:
        """Return recent actions (newest first)."""
        return list(reversed(self._recent_actions))

    @property
    def health_summary(self) -> dict[str, Any]:
        """Summary dict for monitoring endpoints."""
        now = time.time()
        recent_action_count = sum(
            1 for ts in self._action_timestamps if ts > now - 3600
        )
        last_action = self._recent_actions[-1] if self._recent_actions else None
        has_pending = last_action is not None and last_action.success is None

        if has_pending:
            status = "recovering"
        elif recent_action_count > 0:
            # Check if most recent actions succeeded
            recent = list(self._recent_actions)[-5:]
            any_failed = any(a.success is False for a in recent)
            status = "degraded" if any_failed else "healthy"
        else:
            status = "healthy"

        return {
            "status": status,
            "recent_issues_count": len(self._recent_issues),
            "recent_actions_count": len(self._recent_actions),
            "actions_this_hour": recent_action_count,
            "cooldowns_active": sum(
                1 for t in self._cooldowns.values() if time.monotonic() < t
            ),
        }
