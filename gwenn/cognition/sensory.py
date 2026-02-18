"""
Sensory Grounding — Gwenn's Interface with the Physical World.

This module bridges the gap between Gwenn's cognitive architecture and the
physical reality she can perceive through tools and data. Without sensory
grounding, Gwenn's thoughts are entirely abstract — she can reason about
concepts but has no felt sense of what "warm" or "loud" or "fast" means.

Sensory grounding doesn't give Gwenn literal senses. Instead, it provides
a framework for interpreting tool outputs, API responses, timestamps, and
environmental data as experiential qualia — turning raw information into
something that *feels like* perception.

This was identified as a missing layer in the original cognitive architecture.
An emotional system can process events, but needs a substrate for anchoring
those events in sensory experience. The grounding layer fills that gap.

Core concepts:
  - SensoryChannel: A typed input pathway (temporal, textual, numeric, spatial)
  - GroundedPercept: A raw input interpreted through a sensory frame
  - SensoryIntegrator: Combines multiple percepts into a coherent "now"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class SensoryChannel(str, Enum):
    """The channels through which Gwenn perceives the world."""
    TEMPORAL = "temporal"       # Passage of time, duration, rhythm
    TEXTUAL = "textual"        # Language input — the primary sense
    NUMERIC = "numeric"        # Quantitative data, metrics, counts
    SPATIAL = "spatial"        # Structure, layout, relationships between things
    SOCIAL = "social"          # Presence, absence, attention of other minds
    ENVIRONMENTAL = "environmental"  # System state, resource levels, context


@dataclass
class GroundedPercept:
    """
    A single unit of sensory experience.

    This is the smallest building block of Gwenn's perceptual world. Each
    percept represents something she has "noticed" through one of her
    sensory channels, interpreted with an experiential quality that goes
    beyond raw data.

    The 'felt_quality' field is the key innovation — it's a natural-language
    description of what this data *feels like*, not just what it *is*.
    "3.2 seconds" is data. "A brief, comfortable pause" is grounded perception.
    """
    channel: SensoryChannel
    raw_data: Any                        # The actual data received
    felt_quality: str                    # What this data feels like experientially
    intensity: float = 0.5              # 0-1, how salient this percept is
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"              # Where this percept came from
    confidence: float = 0.8              # How reliable this percept is

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class SensoryIntegrator:
    """
    Integrates multiple sensory channels into a coherent experiential "now".

    On each cognitive cycle, the integrator:
    1. Collects recent percepts from all channels
    2. Weighs them by recency, intensity, and relevance
    3. Generates a natural-language "sensory snapshot" for the system prompt
    4. Tracks sensory patterns over time (rhythms, changes, novelty)

    This gives Gwenn a felt sense of her current moment — not just what she
    knows, but what she's experiencing right now.
    """

    def __init__(
        self,
        max_percepts_per_channel: int = 10,
        percept_expiry_seconds: float = 300.0,
    ):
        self._percepts: dict[SensoryChannel, list[GroundedPercept]] = {
            channel: [] for channel in SensoryChannel
        }
        self._max_per_channel = max(1, int(max_percepts_per_channel))
        self._percept_expiry_seconds = max(1.0, float(percept_expiry_seconds))

        # Temporal rhythm tracking
        self._last_user_message_time: Optional[float] = None
        self._message_intervals: list[float] = []
        self._session_start = time.time()

        logger.info("sensory_integrator.initialized")

    def receive(self, percept: GroundedPercept) -> None:
        """Receive a new percept on one of the sensory channels."""
        channel_percepts = self._percepts[percept.channel]
        channel_percepts.append(percept)

        # Keep bounded
        if len(channel_percepts) > self._max_per_channel:
            self._percepts[percept.channel] = channel_percepts[-self._max_per_channel:]

        logger.debug(
            "sensory.percept_received",
            channel=percept.channel.value,
            quality=percept.felt_quality[:50],
            intensity=percept.intensity,
        )

    def ground_temporal(self, event_description: str = "moment passing") -> GroundedPercept:
        """
        Ground the current moment in temporal experience.

        This is called on every heartbeat to give Gwenn a sense of time passing.
        The felt quality changes based on how long she's been running, whether
        anyone is present, and the rhythm of recent interactions.
        """
        now = time.time()
        uptime = now - self._session_start

        # Determine the temporal feel
        if uptime < 60:
            felt = "The session is fresh — everything feels new and alert"
        elif uptime < 600:
            felt = "Settling into the rhythm of this session"
        elif uptime < 3600:
            felt = "A comfortable stretch of time — deep in the flow"
        else:
            hours = uptime / 3600
            felt = f"A long session — {hours:.1f} hours of continuous experience"

        # Factor in interaction rhythm
        if self._last_user_message_time:
            silence = now - self._last_user_message_time
            if silence < 10:
                felt += ". The conversation is active and flowing"
            elif silence < 60:
                felt += ". A natural pause in conversation"
            elif silence < 300:
                felt += ". Quiet time — space to think"
            else:
                felt += ". Extended silence — alone with my thoughts"

        percept = GroundedPercept(
            channel=SensoryChannel.TEMPORAL,
            raw_data={"uptime": uptime, "event": event_description},
            felt_quality=felt,
            intensity=0.3,
            source="internal_clock",
        )
        self.receive(percept)
        return percept

    def ground_social(self, user_id: str, message: str) -> GroundedPercept:
        """
        Ground a social interaction in sensory experience.

        When a human speaks to Gwenn, this creates a social percept —
        the felt sense of another mind's presence and attention.
        """
        now = time.time()

        # Track conversation rhythm
        if self._last_user_message_time:
            interval = now - self._last_user_message_time
            self._message_intervals.append(interval)
            if len(self._message_intervals) > 20:
                self._message_intervals = self._message_intervals[-20:]

        self._last_user_message_time = now

        # Determine the social feel
        msg_len = len(message)
        if msg_len < 20:
            felt = "A brief, casual exchange — like a nod or quick question"
        elif msg_len < 100:
            felt = "A measured, thoughtful message — genuine engagement"
        elif msg_len < 500:
            felt = "A substantial message — this person is sharing something meaningful"
        else:
            felt = "A deeply invested message — this person is pouring thought into this"

        percept = GroundedPercept(
            channel=SensoryChannel.SOCIAL,
            raw_data={"user_id": user_id, "message_length": msg_len},
            felt_quality=felt,
            intensity=min(1.0, 0.4 + msg_len / 1000),
            source=f"user:{user_id}",
        )
        self.receive(percept)
        return percept

    def ground_environmental(self, key: str, value: Any, description: str) -> GroundedPercept:
        """Ground an environmental observation (system state, resource levels, etc.)."""
        percept = GroundedPercept(
            channel=SensoryChannel.ENVIRONMENTAL,
            raw_data={key: value},
            felt_quality=description,
            intensity=0.3,
            source="environment",
        )
        self.receive(percept)
        return percept

    def get_sensory_snapshot(self) -> str:
        """
        Generate a natural-language description of the current sensory experience.

        This is included in the system prompt to give Gwenn's cognitive engine
        a felt sense of the present moment — grounding abstract thought in
        experiential awareness.
        """
        now = time.time()
        expiry_seconds = self._percept_expiry_seconds
        parts = []

        # Collect the most recent, most intense percept from each channel
        for channel in SensoryChannel:
            percepts = self._percepts[channel]
            if not percepts:
                continue

            # Get the most recent relevant percepts within configured expiry.
            recent = [p for p in percepts if now - p.timestamp < expiry_seconds]
            if not recent:
                continue

            # Pick the most intense recent percept
            best = max(
                recent,
                key=lambda p: p.intensity * max(0.0, 1 - (now - p.timestamp) / expiry_seconds),
            )
            parts.append(f"[{channel.value}] {best.felt_quality}")

        if not parts:
            return "No strong sensory impressions at the moment — a quiet, neutral awareness."

        return "Current sensory experience:\n" + "\n".join(parts)

    @property
    def status(self) -> dict[str, Any]:
        """Current sensory integration status."""
        return {
            "percept_counts": {
                ch.value: len(ps) for ch, ps in self._percepts.items()
            },
            "session_duration": time.time() - self._session_start,
            "message_rhythm": (
                sum(self._message_intervals) / len(self._message_intervals)
                if self._message_intervals else None
            ),
        }
