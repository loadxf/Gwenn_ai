"""Tests for gwenn/interoception.py — System Self-Awareness."""

from __future__ import annotations

import time
from unittest.mock import mock_open, patch

import pytest

from gwenn.interoception import InteroceptiveMonitor, InteroceptiveState


# ---------------------------------------------------------------------------
# InteroceptiveState property calculations
# ---------------------------------------------------------------------------


class TestInteroceptiveState:
    def test_defaults_all_zero(self) -> None:
        state = InteroceptiveState()
        assert state.fatigue == 0.0
        assert state.flow == 1.0  # low latency + low errors = max flow
        assert state.overwhelm == 0.0

    def test_fatigue_high_cpu_and_memory(self) -> None:
        state = InteroceptiveState(cpu_percent=100.0, memory_percent=100.0)
        assert state.fatigue == pytest.approx(0.6, abs=0.01)

    def test_fatigue_high_latency(self) -> None:
        state = InteroceptiveState(response_latency_ms=10000.0)
        assert state.fatigue == pytest.approx(0.4, abs=0.01)

    def test_fatigue_clamped_at_one(self) -> None:
        state = InteroceptiveState(
            cpu_percent=100.0, memory_percent=100.0, response_latency_ms=20000.0,
        )
        assert state.fatigue <= 1.0

    def test_flow_low_latency_low_errors(self) -> None:
        state = InteroceptiveState(response_latency_ms=0.0, error_rate=0.0)
        assert state.flow == 1.0

    def test_flow_high_latency(self) -> None:
        state = InteroceptiveState(response_latency_ms=5000.0)
        assert state.flow == pytest.approx(0.5, abs=0.01)

    def test_flow_high_errors(self) -> None:
        state = InteroceptiveState(error_rate=10.0)
        assert state.flow == pytest.approx(0.5, abs=0.01)

    def test_flow_clamped_at_one(self) -> None:
        state = InteroceptiveState()
        assert state.flow <= 1.0

    def test_overwhelm_high_connections(self) -> None:
        state = InteroceptiveState(active_connections=20)
        assert state.overwhelm == pytest.approx(0.3, abs=0.01)

    def test_overwhelm_high_error_rate(self) -> None:
        state = InteroceptiveState(error_rate=10.0)
        assert state.overwhelm == pytest.approx(0.4, abs=0.01)

    def test_overwhelm_high_memory(self) -> None:
        state = InteroceptiveState(memory_percent=100.0)
        assert state.overwhelm == pytest.approx(0.3, abs=0.01)

    def test_overwhelm_clamped_at_one(self) -> None:
        state = InteroceptiveState(
            active_connections=100, error_rate=50.0, memory_percent=100.0,
        )
        assert state.overwhelm <= 1.0

    def test_beat_drift_stored(self) -> None:
        state = InteroceptiveState(beat_drift_ms=42.5)
        assert state.beat_drift_ms == 42.5


# ---------------------------------------------------------------------------
# InteroceptiveMonitor
# ---------------------------------------------------------------------------


class TestInteroceptiveMonitor:
    def test_snapshot_default(self) -> None:
        mon = InteroceptiveMonitor()
        snap = mon.snapshot()
        assert isinstance(snap, InteroceptiveState)
        assert snap.response_latency_ms == 0.0
        assert snap.error_rate == 0.0
        assert snap.active_connections == 0

    def test_snapshot_with_connections(self) -> None:
        mon = InteroceptiveMonitor()
        snap = mon.snapshot(active_connections=5)
        assert snap.active_connections == 5

    def test_record_response_latency(self) -> None:
        mon = InteroceptiveMonitor()
        mon.record_response_latency(100.0)
        mon.record_response_latency(200.0)
        snap = mon.snapshot()
        assert snap.response_latency_ms == pytest.approx(150.0)

    def test_latency_samples_bounded(self) -> None:
        mon = InteroceptiveMonitor()
        for i in range(100):
            mon.record_response_latency(float(i))
        # maxlen=50 so only last 50 samples retained
        snap = mon.snapshot()
        expected = sum(range(50, 100)) / 50
        assert snap.response_latency_ms == pytest.approx(expected)

    def test_record_error(self) -> None:
        mon = InteroceptiveMonitor()
        mon.record_error()
        mon.record_error()
        mon.record_error()
        snap = mon.snapshot()
        assert snap.error_rate == 3.0  # 3 errors in last minute

    def test_error_rate_only_recent(self) -> None:
        mon = InteroceptiveMonitor()
        # Simulate old errors by inserting timestamps in the past
        old_time = time.monotonic() - 120.0
        mon._error_timestamps.append(old_time)
        snap = mon.snapshot()
        assert snap.error_rate == 0.0  # Old error expired

    def test_record_beat(self) -> None:
        mon = InteroceptiveMonitor()
        mon.record_beat(30.0)
        assert mon._last_beat_time is not None
        assert mon._target_interval == 30.0

    def test_beat_drift_zero_before_first_beat(self) -> None:
        mon = InteroceptiveMonitor()
        snap = mon.snapshot()
        assert snap.beat_drift_ms == 0.0

    def test_beat_drift_zero_after_first_beat(self) -> None:
        """First beat has no previous beat to compare to, so drift stays 0."""
        mon = InteroceptiveMonitor()
        mon.record_beat(30.0)
        snap = mon.snapshot()
        assert snap.beat_drift_ms == 0.0

    def test_beat_drift_measures_between_beats(self) -> None:
        """Drift measures |actual_interval - target| between consecutive beats."""
        mon = InteroceptiveMonitor()
        # Simulate first beat
        mon._last_beat_time = time.monotonic() - 35.0  # 35s ago
        # Second beat with target of 30s → drift = |35 - 30| * 1000 = 5000ms
        mon.record_beat(30.0)
        snap = mon.snapshot()
        assert snap.beat_drift_ms == pytest.approx(5000.0, abs=100.0)

    def test_read_cpu_proc_available(self) -> None:
        """Test CPU reading from /proc/loadavg."""
        data = "0.50 0.40 0.30 1/500 12345"
        with patch("builtins.open", mock_open(read_data=data)):
            with patch("os.cpu_count", return_value=2):
                cpu = InteroceptiveMonitor._read_cpu()
        assert cpu == pytest.approx(25.0)  # 0.50 / 2 * 100

    def test_read_cpu_proc_not_available(self) -> None:
        """Falls back to os.getloadavg() when /proc/loadavg not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("os.getloadavg", return_value=(1.0, 0.8, 0.5)):
                with patch("os.cpu_count", return_value=4):
                    cpu = InteroceptiveMonitor._read_cpu()
        assert cpu == pytest.approx(25.0)  # 1.0 / 4 * 100

    def test_read_cpu_both_fail(self) -> None:
        """Returns 0.0 when both sources fail."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("os.getloadavg", side_effect=OSError):
                cpu = InteroceptiveMonitor._read_cpu()
        assert cpu == 0.0

    def test_read_memory_proc_available(self) -> None:
        """Test memory reading from /proc/meminfo."""
        meminfo = (
            "MemTotal:       16000000 kB\n"
            "MemFree:         1000000 kB\n"
            "MemAvailable:    8000000 kB\n"
            "Buffers:          500000 kB\n"
            "Cached:          3000000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=meminfo)):
            mem = InteroceptiveMonitor._read_memory()
        assert mem == pytest.approx(50.0)  # (16M - 8M) / 16M * 100

    def test_read_memory_proc_not_available(self) -> None:
        """Returns 0.0 when /proc/meminfo not available."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            mem = InteroceptiveMonitor._read_memory()
        assert mem == 0.0

    def test_cpu_clamped_at_100(self) -> None:
        """CPU percent should not exceed 100."""
        data = "10.0 5.0 3.0 1/500 12345"
        with patch("builtins.open", mock_open(read_data=data)):
            with patch("os.cpu_count", return_value=1):
                cpu = InteroceptiveMonitor._read_cpu()
        assert cpu == 100.0
