"""
Message formatting and chunking utilities for platform channels.

Each platform has a maximum message length.  Long responses from Gwenn must be
split into chunks that fit within those limits.  We prefer splitting at
paragraph boundaries, then sentence boundaries, and only hard-truncate as a
last resort.
"""

from __future__ import annotations

import re
from typing import Any

TELEGRAM_MAX_LEN: int = 4096
DISCORD_MAX_LEN: int = 2000


def split_message(text: str, max_len: int) -> list[str]:
    """
    Split *text* into a list of strings, each at most *max_len* characters.

    Splitting priority:
      1. Paragraph boundaries (double-newline)
      2. Single-newline boundaries
      3. Sentence-end punctuation followed by a space
      4. Word boundaries (spaces)
      5. Hard truncation (character boundary)

    Empty input returns an empty list.
    """
    if not text:
        return []

    # Preserve formatting for meaningful content while still treating
    # whitespace-only messages as empty.
    if not text.strip():
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    _recursive_split(text, max_len, chunks)
    return chunks


def _recursive_split(text: str, max_len: int, out: list[str]) -> None:
    """Recursively split text, appending results to *out*."""
    if len(text) <= max_len:
        if text:
            out.append(text)
        return

    # Try paragraph break
    split_pos = _find_split(text, max_len, r"\n\n")
    if split_pos is not None:
        head = text[:split_pos]
        if head:
            out.append(head)
        _recursive_split(text[split_pos:], max_len, out)
        return

    # Try single newline
    split_pos = _find_split(text, max_len, r"\n")
    if split_pos is not None:
        head = text[:split_pos]
        if head:
            out.append(head)
        _recursive_split(text[split_pos:], max_len, out)
        return

    # Try sentence boundary (. ! ?)
    split_pos = _find_split(text, max_len, r"(?<=[.!?])\s+")
    if split_pos is not None:
        head = text[:split_pos]
        if head:
            out.append(head)
        _recursive_split(text[split_pos:], max_len, out)
        return

    # Try word boundary
    split_pos = _find_split(text, max_len, r"\s+")
    if split_pos is not None:
        head = text[:split_pos]
        if head:
            out.append(head)
        _recursive_split(text[split_pos:], max_len, out)
        return

    # Hard truncation
    out.append(text[:max_len])
    _recursive_split(text[max_len:], max_len, out)


def _find_split(text: str, max_len: int, pattern: str) -> int | None:
    """
    Find the last occurrence of *pattern* at or before *max_len*.

    Returns the end-position of the match (i.e. start of the remainder), or
    None if no suitable split point was found.
    """
    best: int | None = None
    for m in re.finditer(pattern, text):
        end = m.end()
        if end > max_len:
            break
        best = end
    return best


def format_for_telegram(text: str) -> list[str]:
    """Return *text* split into chunks suitable for the Telegram Bot API."""
    return split_message(text, TELEGRAM_MAX_LEN)


def format_for_discord(text: str) -> list[str]:
    """Return *text* split into chunks suitable for the Discord API."""
    return split_message(text, DISCORD_MAX_LEN)


def format_uptime(seconds: float) -> str:
    """Format uptime in a plain, human-friendly way."""
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def describe_mood(emotion: str, valence: float, arousal: float) -> str:
    """Convert model emotion dimensions into plain-language mood text."""
    if valence >= 0.25:
        tone = "positive"
    elif valence <= -0.25:
        tone = "low"
    else:
        tone = "steady"

    if arousal >= 0.7:
        energy = "high energy"
    elif arousal >= 0.4:
        energy = "moderate energy"
    else:
        energy = "calm energy"

    return f"{emotion} ({tone}, {energy})"


def describe_focus_load(load: float) -> str:
    """Describe working-memory load in plain language."""
    if load < 0.3:
        level = "light"
    elif load < 0.7:
        level = "moderate"
    else:
        level = "heavy"
    return f"{level} ({load:.1%} of active memory in use)"


def describe_stress_guardrail(resilience: dict[str, Any]) -> str:
    """Describe resilience circuit state in plain language."""
    if resilience.get("breaker_active"):
        duration = format_uptime(float(resilience.get("distress_duration", 0.0)))
        return f"ACTIVE (recovering from sustained stress for {duration})"
    return "normal"


def render_status_text(status: dict[str, Any], markdown_heading: bool = False) -> str:
    """Render a user-friendly status summary for channel commands."""
    title = (
        f"**{status['name']}** — Status"
        if markdown_heading
        else f"{status['name']} — Status"
    )
    mood = describe_mood(status["emotion"], status["valence"], status["arousal"])
    focus = describe_focus_load(status["working_memory_load"])
    stress = describe_stress_guardrail(status["resilience"])
    uptime = format_uptime(status["uptime_seconds"])
    return (
        f"{title}\n"
        f"Mood: {mood}\n"
        f"Focus right now: {focus}\n"
        f"Conversations handled: {status['total_interactions']}\n"
        f"Awake for: {uptime}\n"
        f"Stress guardrail: {stress}"
    )


def render_heartbeat_text(hb: dict[str, Any], markdown_heading: bool = False) -> str:
    """Render heartbeat status text for channel commands."""
    title = "**Heartbeat Status**" if markdown_heading else "Heartbeat Status"
    return (
        f"{title}\n"
        f"Running: {hb['running']}\n"
        f"Beat count: {hb['beat_count']}\n"
        f"Current interval: {hb['current_interval']}s\n"
        f"Beats since consolidation: {hb['beats_since_consolidation']}"
    )
