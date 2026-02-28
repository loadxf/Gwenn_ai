"""Tests for heartbeat core-mode lifecycle (Phase 1f).

Covers: run(), _start_gateway(), _wake_up(), _wire_gateway(),
_start_channels(), _live(), _sleep(), _request_shutdown(),
_is_nonfatal_channel_error(), _redact_channel_error(),
_on_channel_task_done(), and backward compatibility with legacy mode.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gwenn.config import GwennConfig, HeartbeatConfig
from gwenn.heartbeat import Heartbeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_config(**overrides) -> GwennConfig:
    """Build a GwennConfig with sensible test defaults."""
    cfg = MagicMock(spec=GwennConfig)
    cfg.heartbeat = HeartbeatConfig()
    cfg.daemon = SimpleNamespace(
        gateway_enabled=True,
        gateway_host="127.0.0.1",
        gateway_port=0,  # OS-assigned port
        sessions_dir="/tmp/test_sessions",
        session_max_count=100,
        session_max_messages=1000,
        session_include_preview=False,
        auth_token="test-token",
        heartbeat_core=True,
        legacy_socket_enabled=False,
        mcp_server_enabled=False,
        a2a_enabled=False,
    )
    cfg.channel = SimpleNamespace(
        get_channel_list=lambda: [],
    )
    # Apply overrides
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_agent():
    """Build a mock agent for lifecycle tests."""
    agent = MagicMock()
    agent.respond_lock = asyncio.Lock()
    agent.initialize = AsyncMock()
    agent.start = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.affect_state = SimpleNamespace(
        current_emotion=SimpleNamespace(value="curious"),
        dimensions=SimpleNamespace(arousal=0.3, valence=0.2),
    )
    agent.heartbeat = None
    return agent


# ===========================================================================
# Constructor — dual-mode support
# ===========================================================================


class TestConstructorModes:
    def test_core_mode_from_gwenn_config(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        assert hb._is_core is True
        assert hb._full_config is cfg
        assert hb._agent is None
        assert hb._config is cfg.heartbeat

    def test_legacy_mode_from_heartbeat_config(self):
        agent = _make_mock_agent()
        hb = Heartbeat(HeartbeatConfig(), agent)
        assert hb._is_core is False
        assert hb._full_config is None
        assert hb._agent is agent

    def test_core_mode_subsystems_initially_none(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        assert hb._event_bus is None
        assert hb._gateway is None
        assert hb._router is None
        assert hb._session_store is None
        assert hb._session_manager is None
        assert hb._channel_task is None

    def test_shutdown_event_created(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        assert isinstance(hb._shutdown_event, asyncio.Event)
        assert not hb._shutdown_event.is_set()


# ===========================================================================
# run() — top-level lifecycle
# ===========================================================================


class TestRun:
    @pytest.mark.asyncio
    async def test_run_raises_in_legacy_mode(self):
        agent = _make_mock_agent()
        hb = Heartbeat(HeartbeatConfig(), agent)
        with pytest.raises(RuntimeError, match="core mode"):
            await hb.run()

    @pytest.mark.asyncio
    async def test_run_lifecycle_order(self):
        """Verify run() calls lifecycle stages in the correct order."""
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False
        cfg.channel.get_channel_list = lambda: []

        hb = Heartbeat(cfg)
        call_order = []

        async def mock_start_gateway():
            call_order.append("start_gateway")

        async def mock_wake_up():
            call_order.append("wake_up")
            hb._agent = _make_mock_agent()

        async def mock_start_channels():
            call_order.append("start_channels")

        async def mock_live():
            call_order.append("live")

        async def mock_sleep():
            call_order.append("sleep")

        hb._start_gateway = mock_start_gateway
        hb._wake_up = mock_wake_up
        hb._start_channels = mock_start_channels
        hb._live = mock_live
        hb._sleep = mock_sleep

        await hb.run()

        assert call_order == [
            "start_gateway",
            "wake_up",
            "start_channels",
            "live",
            "sleep",
        ]

    @pytest.mark.asyncio
    async def test_run_starts_and_stops_event_bus(self):
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        hb = Heartbeat(cfg)
        hb._start_gateway = AsyncMock()
        hb._wake_up = AsyncMock(side_effect=lambda: setattr(hb, "_agent", _make_mock_agent()))
        hb._start_channels = AsyncMock()
        hb._live = AsyncMock()
        hb._sleep = AsyncMock()

        await hb.run()

        assert hb._event_bus is not None
        # Event bus should have been stopped in the finally block
        assert not hb._event_bus.is_running

    @pytest.mark.asyncio
    async def test_run_stops_event_bus_on_error(self):
        """Event bus is stopped even when _wake_up raises."""
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        hb = Heartbeat(cfg)
        hb._start_gateway = AsyncMock()
        hb._wake_up = AsyncMock(side_effect=RuntimeError("init failed"))
        hb._start_channels = AsyncMock()
        hb._sleep = AsyncMock()

        with pytest.raises(RuntimeError, match="init failed"):
            await hb.run()

        assert hb._event_bus is not None
        assert not hb._event_bus.is_running

    @pytest.mark.asyncio
    async def test_run_stops_event_bus_even_if_sleep_raises(self):
        """H-2: event bus must stop even when _sleep() raises."""
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        hb = Heartbeat(cfg)
        hb._start_gateway = AsyncMock()
        hb._wake_up = AsyncMock(side_effect=lambda: setattr(hb, "_agent", _make_mock_agent()))
        hb._start_channels = AsyncMock()
        hb._live = AsyncMock()
        hb._sleep = AsyncMock(side_effect=RuntimeError("sleep crash"))

        with pytest.raises(RuntimeError, match="sleep crash"):
            await hb.run()

        assert hb._event_bus is not None
        assert not hb._event_bus.is_running


# ===========================================================================
# _start_gateway() / _wire_gateway()
# ===========================================================================


class TestGateway:
    @pytest.mark.asyncio
    async def test_gateway_disabled_skips(self):
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False
        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()
        await hb._start_gateway()
        assert hb._gateway is None

    @pytest.mark.asyncio
    async def test_start_gateway_creates_session_store(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()
        with patch("gwenn.memory.session_store.SessionStore") as MockStore:
            MockStore.return_value = MagicMock()
            await hb._start_gateway()
            MockStore.assert_called_once()
            assert hb._session_store is not None

    @pytest.mark.asyncio
    async def test_wire_gateway_creates_router_and_server(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()
        hb._agent = _make_mock_agent()
        hb._session_store = MagicMock()

        mock_gateway = MagicMock()
        mock_gateway.start = AsyncMock()

        with (
            patch("gwenn.rpc.RequestRouter") as MockRouter,
            patch("gwenn.gateway.GatewayServer", return_value=mock_gateway),
        ):
            MockRouter.return_value = MagicMock()
            await hb._wire_gateway()

            MockRouter.assert_called_once()
            assert hb._router is not None
            mock_gateway.start.assert_awaited_once()
            assert hb._gateway is mock_gateway

    @pytest.mark.asyncio
    async def test_wire_gateway_skips_when_disabled(self):
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False
        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()
        hb._agent = _make_mock_agent()
        await hb._wire_gateway()
        assert hb._gateway is None


# ===========================================================================
# _wake_up()
# ===========================================================================


class TestWakeUp:
    @pytest.mark.asyncio
    async def test_creates_and_initializes_agent(self):
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        mock_agent = _make_mock_agent()

        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()

        with patch("gwenn.agent.SentientAgent", return_value=mock_agent):
            await hb._wake_up()

        assert hb._agent is mock_agent
        mock_agent.initialize.assert_awaited_once()
        mock_agent.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wake_up_wires_heartbeat_on_agent(self):
        """H-3: agent.heartbeat must point back to the heartbeat instance."""
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        mock_agent = _make_mock_agent()

        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()

        with patch("gwenn.agent.SentientAgent", return_value=mock_agent):
            await hb._wake_up()

        assert mock_agent.heartbeat is hb

    @pytest.mark.asyncio
    async def test_propagates_cognitive_engine_init_error(self):
        cfg = _make_full_config()
        cfg.daemon.gateway_enabled = False

        hb = Heartbeat(cfg)
        hb._event_bus = MagicMock()

        with patch("gwenn.agent.SentientAgent") as MockAgent:
            from gwenn.api.claude import CognitiveEngineInitError

            MockAgent.side_effect = CognitiveEngineInitError("bad key")
            with pytest.raises(CognitiveEngineInitError):
                await hb._wake_up()


# ===========================================================================
# _start_channels()
# ===========================================================================


class TestStartChannels:
    @pytest.mark.asyncio
    async def test_no_channels_configured_skips(self):
        cfg = _make_full_config()
        cfg.channel.get_channel_list = lambda: []
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        await hb._start_channels()
        assert hb._channel_task is None

    @pytest.mark.asyncio
    async def test_slack_channel_creates_task(self):
        """M-5: non-telegram/discord channels should also start."""
        cfg = _make_full_config()
        cfg.channel.get_channel_list = lambda: ["slack"]
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        async def _fake_run(channel_list):
            pass

        hb._run_platform_channels = _fake_run
        await hb._start_channels()
        assert hb._channel_task is not None
        await hb._channel_task

    @pytest.mark.asyncio
    async def test_channels_creates_task(self):
        cfg = _make_full_config()
        cfg.channel.get_channel_list = lambda: ["telegram"]
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        async def _fake_run(channel_list):
            pass

        hb._run_platform_channels = _fake_run

        await hb._start_channels()
        assert hb._channel_task is not None
        # Wait for the task to finish
        await hb._channel_task

    @pytest.mark.asyncio
    async def test_run_platform_channels_calls_startup(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        with (
            patch("gwenn.channels.startup.build_channels") as mock_build,
            patch("gwenn.channels.startup.run_channels_until_shutdown") as mock_run,
        ):
            mock_build.return_value = (MagicMock(), [MagicMock()])
            mock_run.return_value = None
            await hb._run_platform_channels(["telegram"])
            mock_build.assert_called_once()
            mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_platform_channels_empty_channels_returns(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        with patch("gwenn.channels.startup.build_channels") as mock_build:
            mock_build.return_value = (MagicMock(), [])
            await hb._run_platform_channels(["telegram"])
            # No exception, early return

    @pytest.mark.asyncio
    async def test_run_platform_channels_nonfatal_error_logged(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        with (
            patch("gwenn.channels.startup.build_channels") as mock_build,
            patch("gwenn.channels.startup.run_channels_until_shutdown") as mock_run,
        ):
            mock_build.return_value = (MagicMock(), [MagicMock()])
            mock_run.side_effect = ImportError("No module named 'telegram'")
            # Should not raise — nonfatal
            await hb._run_platform_channels(["telegram"])

    @pytest.mark.asyncio
    async def test_run_platform_channels_fatal_error_propagates(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        with (
            patch("gwenn.channels.startup.build_channels") as mock_build,
            patch("gwenn.channels.startup.run_channels_until_shutdown") as mock_run,
        ):
            mock_build.return_value = (MagicMock(), [MagicMock()])
            mock_run.side_effect = RuntimeError("fatal")
            with pytest.raises(RuntimeError, match="fatal"):
                await hb._run_platform_channels(["telegram"])


# ===========================================================================
# _on_channel_task_done()
# ===========================================================================


class TestOnChannelTaskDone:
    def test_cancelled_task_ignored(self):
        hb = Heartbeat(_make_full_config())
        task = MagicMock()
        task.cancelled.return_value = True
        # Should not raise
        hb._on_channel_task_done(task)

    def test_no_exception_ignored(self):
        hb = Heartbeat(_make_full_config())
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        hb._on_channel_task_done(task)

    def test_nonfatal_error_logged_no_shutdown(self):
        hb = Heartbeat(_make_full_config())
        hb._request_shutdown = MagicMock()
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = ImportError("No module named 'telegram'")
        hb._on_channel_task_done(task)
        hb._request_shutdown.assert_not_called()

    def test_fatal_error_triggers_shutdown(self):
        hb = Heartbeat(_make_full_config())
        hb._request_shutdown = MagicMock()
        task = MagicMock()
        task.cancelled.return_value = False
        exc = RuntimeError("fatal crash")
        exc.__traceback__ = None
        task.exception.return_value = exc
        hb._on_channel_task_done(task)
        hb._request_shutdown.assert_called_once_with("heartbeat_channels_task_failed")


# ===========================================================================
# _live() — core-mode cognitive loop
# ===========================================================================


class TestLive:
    @pytest.mark.asyncio
    async def test_live_stops_on_shutdown_event(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        hb._beat = AsyncMock()
        hb._config = HeartbeatConfig(interval=0.01)
        hb._interval = 0.01

        # Trigger shutdown after first beat
        original_beat = hb._beat

        async def beat_then_shutdown():
            await original_beat()
            hb._shutdown_event.set()

        hb._beat = beat_then_shutdown
        await hb._live()
        assert hb._running is False or hb._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_live_emits_heartbeat_beat_event(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        hb._config = HeartbeatConfig(interval=0.01)
        hb._interval = 0.01

        from gwenn.events import EventBus

        bus = EventBus()
        await bus.start()
        hb._event_bus = bus

        emitted = []
        bus.subscribe("heartbeat.beat", lambda e: emitted.append(e))

        beat_count = 0

        async def _mock_beat():
            nonlocal beat_count
            beat_count += 1
            hb._beat_count = beat_count
            if beat_count >= 2:
                hb._shutdown_event.set()

        hb._beat = _mock_beat

        await hb._live()
        # Give dispatcher time to process
        await asyncio.sleep(0.05)

        await bus.stop()
        assert len(emitted) >= 1
        assert emitted[0].beat_count == 1

    @pytest.mark.asyncio
    async def test_live_circuit_breaker(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        hb._config = HeartbeatConfig(
            interval=0.01,
            circuit_max_consecutive=2,
            circuit_base_seconds=0.01,
            circuit_max_seconds=0.05,
        )
        hb._interval = 0.01

        fail_count = 0

        async def _failing_beat():
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 3:
                raise RuntimeError("beat error")
            hb._shutdown_event.set()

        hb._beat = _failing_beat
        await hb._live()
        assert fail_count >= 3


# ===========================================================================
# _sleep() — graceful shutdown
# ===========================================================================


class TestSleep:
    @pytest.mark.asyncio
    async def test_stops_agent(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        await hb._sleep()
        hb._agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_gateway(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        hb._gateway = MagicMock()
        hb._gateway.stop = AsyncMock()
        await hb._sleep()
        hb._gateway.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_joins_channel_task(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        async def noop():
            pass

        hb._channel_task = asyncio.create_task(noop())
        await asyncio.sleep(0)  # let task complete
        await hb._sleep()
        assert hb._channel_task is None

    @pytest.mark.asyncio
    async def test_sets_shutdown_event(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()
        await hb._sleep()
        assert hb._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_no_agent_safe(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        await hb._sleep()  # no agent, no crash

    @pytest.mark.asyncio
    async def test_agent_shutdown_failure_still_stops_gateway(self):
        """H-1: gateway must stop even if agent.shutdown() raises."""
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        agent = _make_mock_agent()
        agent.shutdown = AsyncMock(side_effect=RuntimeError("shutdown crash"))
        hb._agent = agent
        hb._gateway = MagicMock()
        hb._gateway.stop = AsyncMock()

        await hb._sleep()
        hb._gateway.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sleep_ordering(self):
        """M-6: shutdown order must be channels → agent → gateway."""
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        order = []

        agent = _make_mock_agent()
        agent.shutdown = AsyncMock(side_effect=lambda: order.append("agent"))
        hb._agent = agent

        gateway = MagicMock()
        gateway.stop = AsyncMock(side_effect=lambda: order.append("gateway"))
        hb._gateway = gateway

        async def finish():
            order.append("channels")

        hb._channel_task = asyncio.create_task(finish())
        await asyncio.sleep(0)

        await hb._sleep()
        assert order == ["channels", "agent", "gateway"]

    @pytest.mark.asyncio
    async def test_channel_task_exception_swallowed(self):
        cfg = _make_full_config()
        hb = Heartbeat(cfg)
        hb._agent = _make_mock_agent()

        async def failing():
            raise RuntimeError("channel crash")

        hb._channel_task = asyncio.create_task(failing())
        await asyncio.sleep(0)  # let task fail
        # Should not raise
        await hb._sleep()
        assert hb._channel_task is None


# ===========================================================================
# _request_shutdown()
# ===========================================================================


class TestRequestShutdown:
    def test_sets_shutdown_event(self):
        hb = Heartbeat(_make_full_config())
        hb._request_shutdown("test_reason")
        assert hb._shutdown_event.is_set()

    def test_calls_emergency_stop_on_agent(self):
        hb = Heartbeat(_make_full_config())
        agent = _make_mock_agent()
        agent.safety = SimpleNamespace(emergency_stop=MagicMock())
        hb._agent = agent
        hb._request_shutdown("test")
        agent.safety.emergency_stop.assert_called_once_with("test")

    def test_no_agent_safe(self):
        hb = Heartbeat(_make_full_config())
        hb._request_shutdown("test")  # no crash

    def test_emergency_stop_exception_swallowed(self):
        hb = Heartbeat(_make_full_config())
        agent = _make_mock_agent()
        agent.safety = SimpleNamespace(
            emergency_stop=MagicMock(side_effect=RuntimeError("boom"))
        )
        hb._agent = agent
        hb._request_shutdown("test")  # no crash
        assert hb._shutdown_event.is_set()


# ===========================================================================
# Channel error classification
# ===========================================================================


class TestChannelErrorClassification:
    def test_import_error_telegram_nonfatal(self):
        exc = ImportError("No module named 'telegram'")
        assert Heartbeat._is_nonfatal_channel_error(exc) is True

    def test_import_error_discord_nonfatal(self):
        exc = ImportError("No module named 'discord'")
        assert Heartbeat._is_nonfatal_channel_error(exc) is True

    def test_import_error_unrelated_fatal(self):
        exc = ImportError("No module named 'numpy'")
        assert Heartbeat._is_nonfatal_channel_error(exc) is False

    def test_runtime_error_fatal(self):
        assert Heartbeat._is_nonfatal_channel_error(RuntimeError("crash")) is False

    def test_redact_telegram_token(self):
        msg = "Error: bad token 123456789:ABCDEFGHIJKLMNOPQRSTUVwxyz123"
        result = Heartbeat._redact_channel_error(msg)
        assert "123456789" not in result
        assert "[REDACTED_TELEGRAM_TOKEN]" in result

    def test_redact_empty_string(self):
        assert Heartbeat._redact_channel_error("") == ""

    def test_redact_no_token(self):
        msg = "normal error message"
        assert Heartbeat._redact_channel_error(msg) == msg


# ===========================================================================
# _sense() — gateway_connections in snapshot
# ===========================================================================


class TestSenseGatewayConnections:
    def test_sense_includes_gateway_connections_when_gateway_active(self):
        """Core-mode _sense() adds gateway_connections to snapshot."""
        # Create a full agent mock for _sense()
        agent = SimpleNamespace(
            sensory=SimpleNamespace(
                ground_temporal=lambda event_description="": None,
                ground_environmental=lambda key, value, description: None,
            ),
            affect_state=SimpleNamespace(
                to_dict=lambda: {"valence": 0.0, "arousal": 0.3},
                current_emotion=SimpleNamespace(value="neutral"),
                dimensions=SimpleNamespace(arousal=0.3, valence=0.0),
            ),
            working_memory=SimpleNamespace(load_factor=0.2),
            goal_system=SimpleNamespace(get_goals_summary=lambda: ""),
            resilience=SimpleNamespace(status={}),
        )

        hb = Heartbeat(HeartbeatConfig(), agent)
        hb._gateway = SimpleNamespace(active_connection_count=3)
        hb._beat_count = 1

        state = hb._sense()
        assert state["gateway_connections"] == 3

    def test_sense_zero_connections_when_no_gateway(self):
        agent = SimpleNamespace(
            sensory=SimpleNamespace(
                ground_temporal=lambda event_description="": None,
                ground_environmental=lambda key, value, description: None,
            ),
            affect_state=SimpleNamespace(
                to_dict=lambda: {"valence": 0.0, "arousal": 0.3},
                current_emotion=SimpleNamespace(value="neutral"),
                dimensions=SimpleNamespace(arousal=0.3, valence=0.0),
            ),
            working_memory=SimpleNamespace(load_factor=0.2),
            goal_system=SimpleNamespace(get_goals_summary=lambda: ""),
            resilience=SimpleNamespace(status={}),
        )

        hb = Heartbeat(HeartbeatConfig(), agent)
        hb._beat_count = 1
        state = hb._sense()
        assert state["gateway_connections"] == 0


# ===========================================================================
# status property — core mode additions
# ===========================================================================


class TestStatusCoreMode:
    def test_status_includes_is_core(self):
        hb = Heartbeat(_make_full_config())
        assert hb.status["is_core"] is True

    def test_status_legacy_mode(self):
        agent = _make_mock_agent()
        hb = Heartbeat(HeartbeatConfig(), agent)
        assert hb.status["is_core"] is False

    def test_status_includes_gateway_connections(self):
        hb = Heartbeat(_make_full_config())
        hb._gateway = SimpleNamespace(active_connection_count=5)
        assert hb.status["gateway_connections"] == 5

    def test_status_no_gateway(self):
        hb = Heartbeat(_make_full_config())
        assert "gateway_connections" not in hb.status


# ===========================================================================
# Legacy mode — backward compatibility
# ===========================================================================


class TestLegacyMode:
    @pytest.mark.asyncio
    async def test_start_stop_still_works(self):
        """Legacy mode: agent creates heartbeat, calls start()/stop()."""
        agent = SimpleNamespace(
            inner_life=SimpleNamespace(
                autonomous_thought=AsyncMock(return_value="thought"),
                select_mode=MagicMock(return_value=None),
            ),
            heartbeat=None,
        )
        hb = Heartbeat(HeartbeatConfig(interval=0.01), agent)
        assert hb._is_core is False

        # Start creates a task
        await hb.start()
        assert hb._running is True
        assert hb._task is not None

        # Stop cancels the task
        await hb.stop()
        assert hb._running is False

    def test_notify_user_activity(self):
        agent = _make_mock_agent()
        hb = Heartbeat(HeartbeatConfig(), agent)
        before = hb._last_user_activity_mono
        time.sleep(0.01)
        hb.notify_user_activity()
        assert hb._last_user_activity_mono > before
