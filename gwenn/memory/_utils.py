"""Shared helpers for the memory subsystem."""

from __future__ import annotations


def clamp01(value: float, default: float = 0.5) -> float:
    """Clamp potentially noisy model-provided scores into [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))
