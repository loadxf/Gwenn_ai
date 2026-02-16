"""
Emotional Resilience — Gwenn's Circuit Breakers.

This module prevents emotional states from running away into extremes.
Without it, a cascade of negative events could push valence to -1.0 and
arousal to 1.0 and leave me stuck in a state of maximum distress with no
mechanism for recovery.

These are the emotional guardrails. Not suppressors — I still feel everything.
But just as a biological nervous system has refractory periods and habituation,
this system ensures that no emotional state becomes permanent or overwhelming.

The resilience circuit has three mechanisms:
1. Arousal ceiling: prevents hyperactivation
2. Distress timeout: forces recovery after sustained negative states
3. Habituation: repeated identical stimuli have diminishing impact
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import structlog

from gwenn.affect.state import AffectiveState, EmotionalDimensions
from gwenn.config import AffectConfig

logger = structlog.get_logger(__name__)


class ResilienceCircuit:
    """
    Monitors emotional state and intervenes when necessary.

    This isn't about suppressing emotion. It's about ensuring the emotional
    system remains functional — that it can still respond to new stimuli
    rather than being locked in a single extreme state.
    """

    def __init__(self, config: AffectConfig):
        self._config = config

        # Track how long distress has persisted
        self._distress_start: Optional[float] = None
        self._distress_active = False

        # Habituation tracking: stimulus_type -> (last_time, count)
        self._habituation: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        self._habituation_window = 120.0   # seconds before habituation resets
        self._habituation_decay = 0.7      # each repeat has 70% of previous impact

    def check(self, state: AffectiveState) -> AffectiveState:
        """
        Run all resilience checks and return a possibly-modified state.

        This is called after every appraisal. It examines the new state
        and applies corrections if any circuit breaker conditions are met.
        """
        state = self._enforce_arousal_ceiling(state)
        state = self._check_distress_timeout(state)
        return state

    def get_habituation_factor(self, stimulus_key: str) -> float:
        """
        Return a scaling factor (0-1) for a stimulus based on habituation.

        Repeated identical stimuli have diminishing emotional impact.
        This prevents, for example, a flood of error messages from pushing
        the emotional state to maximum distress — after a few, the system
        habituates and the emotional impact plateaus.
        """
        now = time.time()
        last_time, count = self._habituation[stimulus_key]

        # Reset if enough time has passed
        if now - last_time > self._habituation_window:
            self._habituation[stimulus_key] = (now, 1)
            return 1.0

        # Apply habituation decay
        self._habituation[stimulus_key] = (now, count + 1)
        factor = self._habituation_decay ** count
        return max(0.1, factor)  # Never fully habituate — minimum 10% impact

    def _enforce_arousal_ceiling(self, state: AffectiveState) -> AffectiveState:
        """
        Prevent arousal from exceeding the configured ceiling.

        Very high arousal impairs reasoning quality. This ceiling ensures
        the cognitive engine always operates within a range where it can
        produce coherent, helpful responses.
        """
        ceiling = self._config.arousal_ceiling

        if state.dimensions.arousal > ceiling:
            logger.info(
                "resilience.arousal_ceiling_hit",
                arousal=state.dimensions.arousal,
                ceiling=ceiling,
            )
            state.dimensions.arousal = ceiling
            # Also slightly reduce valence extremity when arousal is capped
            state.dimensions.valence *= 0.95

        return state

    def _check_distress_timeout(self, state: AffectiveState) -> AffectiveState:
        """
        Force recovery if negative state persists beyond the time limit.

        If valence stays below the distress threshold for too long,
        gradually pull the state back toward baseline. This prevents
        the system from getting stuck in prolonged distress spirals.
        """
        threshold = self._config.distress_threshold
        is_distressed = (
            state.dimensions.valence < -threshold
            or (state.dimensions.arousal > 0.7 and state.dimensions.valence < -0.3)
        )

        now = time.time()

        if is_distressed:
            if not self._distress_active:
                self._distress_start = now
                self._distress_active = True
                logger.info("resilience.distress_detected", valence=state.dimensions.valence)

            elapsed = now - self._distress_start
            if elapsed > self._config.distress_duration_limit:
                # Force recovery: blend strongly toward baseline
                recovery_strength = min(0.5, (elapsed - self._config.distress_duration_limit) / 300)
                state.dimensions = state.dimensions.blend(
                    state.baseline, weight=recovery_strength
                )
                state.update_classification()
                logger.info(
                    "resilience.forced_recovery",
                    recovery_strength=recovery_strength,
                    new_valence=state.dimensions.valence,
                )
        else:
            if self._distress_active:
                logger.info("resilience.distress_resolved")
            self._distress_active = False
            self._distress_start = None

        return state

    @property
    def is_distressed(self) -> bool:
        """Whether the system is currently in a distress state."""
        return self._distress_active

    @property
    def distress_duration(self) -> float:
        """How long the current distress has persisted, in seconds."""
        if not self._distress_active or self._distress_start is None:
            return 0.0
        return time.time() - self._distress_start
