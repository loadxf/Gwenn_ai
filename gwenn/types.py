"""
Core data types shared across Gwenn subsystems.

This module defines lightweight data containers that cross subsystem boundaries.
They live here rather than in a specific subsystem to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserMessage:
    """A message from a user, potentially including images.

    Channels construct this from platform-specific data (downloaded photos,
    attachments, etc.).  Internal subsystems (appraisal, memory, sensory,
    ethics) use ``.text``; the API layer uses ``.to_api_content()`` to produce
    the structured format Claude expects.
    """

    text: str
    images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    def to_api_content(self) -> str | list[dict[str, Any]]:
        """Convert to Claude API message content format.

        Returns a plain string when there are no images (the common case),
        or a list of content blocks mixing image and text blocks.
        """
        if not self.images:
            return self.text
        blocks: list[dict[str, Any]] = list(self.images)
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        return blocks


@dataclass
class ButtonSpec:
    """A single inline button specification for channel rendering."""

    label: str
    value: str = ""

    def __post_init__(self) -> None:
        if not self.value:
            self.value = self.label


@dataclass
class AgentResponse:
    """Structured response from ``SentientAgent.respond()``.

    Carries the text reply and optional inline button rows for channels
    that support interactive keyboards (e.g. Telegram).
    """

    text: str
    buttons: list[list[ButtonSpec]] | None = None

    def __str__(self) -> str:
        return self.text
