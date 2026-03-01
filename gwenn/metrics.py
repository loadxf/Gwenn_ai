"""
Lightweight metrics collection for Gwenn observability.

Tracks key operational counters and gauges without requiring external
dependencies like Prometheus or StatsD.  Metrics are stored in-process
and can be exported as a JSON snapshot.

Usage:
    from gwenn.metrics import metrics

    metrics.inc("api_calls_total")
    metrics.observe("api_latency_seconds", 1.23)
    snapshot = metrics.snapshot()
"""

from __future__ import annotations

import threading
import time
from typing import Any


class _Histogram:
    """Minimal histogram: tracks count, sum, min, max."""

    __slots__ = ("count", "total", "min_val", "max_val")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_val = float("inf")
        self.max_val = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    def snapshot(self) -> dict[str, Any]:
        avg = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "sum": round(self.total, 4),
            "avg": round(avg, 4),
            "min": round(self.min_val, 4) if self.count else 0.0,
            "max": round(self.max_val, 4) if self.count else 0.0,
        }


class MetricsRegistry:
    """Thread-safe in-process metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._start_time = time.monotonic()

    # -- Counters --

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # -- Gauges --

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    # -- Histograms (latency tracking) --

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram()
            self._histograms[name].observe(value)

    # -- Export --

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: v.snapshot() for k, v in self._histograms.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.monotonic()


# Singleton — import and use from anywhere.
metrics = MetricsRegistry()
