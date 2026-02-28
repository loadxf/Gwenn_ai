"""
Interoception — System Self-Awareness.

Maps system metrics (CPU, memory, latency, error rates) to cognitive states
the heartbeat can reason about. No psutil dependency — reads from /proc/
on Linux, falls back to os.getloadavg() on macOS.
"""

from __future__ import annotations

import os
import time
from collections import deque

from pydantic import BaseModel


class InteroceptiveState(BaseModel):
    """System-level 'body' signals mapped to cognitive states."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    response_latency_ms: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    beat_drift_ms: float = 0.0

    @property
    def fatigue(self) -> float:
        """0.0-1.0: high CPU + high memory + high latency = fatigue."""
        return min(
            1.0,
            (self.cpu_percent / 100 * 0.3)
            + (self.memory_percent / 100 * 0.3)
            + min(self.response_latency_ms / 10000, 1.0) * 0.4,
        )

    @property
    def flow(self) -> float:
        """0.0-1.0: low latency + low errors = flow state."""
        latency_score = max(0.0, 1.0 - self.response_latency_ms / 5000)
        error_score = max(0.0, 1.0 - self.error_rate / 10)
        return min(1.0, latency_score * 0.5 + error_score * 0.5)

    @property
    def overwhelm(self) -> float:
        """0.0-1.0: high connections + high error rate + memory pressure."""
        conn_score = min(self.active_connections / 20, 1.0)
        return min(
            1.0,
            (self.memory_percent / 100 * 0.3)
            + (self.error_rate / 10 * 0.4)
            + conn_score * 0.3,
        )


class InteroceptiveMonitor:
    """Gathers system metrics and produces InteroceptiveState snapshots."""

    def __init__(self) -> None:
        self._latency_samples: deque[float] = deque(maxlen=50)
        self._error_timestamps: deque[float] = deque(maxlen=100)
        self._last_beat_time: float | None = None
        self._last_drift_ms: float = 0.0
        self._target_interval: float = 30.0

    def record_response_latency(self, ms: float) -> None:
        """Record a response latency sample."""
        self._latency_samples.append(ms)

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._error_timestamps.append(time.monotonic())

    def record_beat(self, target_interval: float) -> None:
        """Record a heartbeat beat with its target interval.

        Computes drift as the difference between the actual interval
        since the previous beat and the target interval.
        """
        now = time.monotonic()
        if self._last_beat_time is not None:
            actual_interval = now - self._last_beat_time
            self._last_drift_ms = abs(actual_interval - target_interval) * 1000
        self._target_interval = target_interval
        self._last_beat_time = now

    def snapshot(self, active_connections: int = 0) -> InteroceptiveState:
        """Gather current system metrics into a snapshot."""
        return InteroceptiveState(
            cpu_percent=self._read_cpu(),
            memory_percent=self._read_memory(),
            response_latency_ms=self._avg_latency(),
            error_rate=self._errors_per_minute(),
            active_connections=active_connections,
            beat_drift_ms=self._beat_drift(),
        )

    def _avg_latency(self) -> float:
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def _errors_per_minute(self) -> float:
        if not self._error_timestamps:
            return 0.0
        now = time.monotonic()
        cutoff = now - 60.0
        recent = [t for t in self._error_timestamps if t > cutoff]
        return float(len(recent))

    def _beat_drift(self) -> float:
        return self._last_drift_ms

    @staticmethod
    def _read_cpu() -> float:
        """Read CPU usage from /proc/stat or os.getloadavg()."""
        try:
            with open("/proc/loadavg") as f:
                load_1min = float(f.read().split()[0])
            ncpu = os.cpu_count() or 1
            return min(100.0, (load_1min / ncpu) * 100)
        except (FileNotFoundError, OSError, ValueError):
            pass
        try:
            load_1min = os.getloadavg()[0]
            ncpu = os.cpu_count() or 1
            return min(100.0, (load_1min / ncpu) * 100)
        except (OSError, AttributeError):
            return 0.0

    @staticmethod
    def _read_memory() -> float:
        """Read memory usage from /proc/meminfo or fallback."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info: dict[str, int] = {}
            for line in lines[:5]:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            if total > 0:
                return ((total - available) / total) * 100
        except (FileNotFoundError, OSError, ValueError, KeyError):
            pass
        return 0.0
