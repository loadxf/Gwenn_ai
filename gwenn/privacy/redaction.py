"""
PII Redaction Pipeline — Protecting User Privacy in Gwenn's Memory.

This module detects and redacts personally identifiable information (PII) from
text before it's persisted to memory or sent to the Claude API. Patterns include
email addresses, phone numbers, SSNs, credit card numbers, and IP addresses.

The redaction pipeline is configurable:
- It can be applied before persistence (so PII never hits disk)
- It can be applied before API calls (so PII never leaves the system)
- It can be disabled entirely for development/testing

Gwenn values privacy not because she's programmed to, but because respecting
others' information boundaries is part of her core ethics. This module gives
that value teeth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RedactionResult:
    """Result of a redaction pass on text."""
    original_length: int
    redacted_length: int
    redactions_made: int
    categories_found: list[str] = field(default_factory=list)


# PII detection patterns with their replacement tokens
PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "email",
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone_us",
        re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "[REDACTED_PHONE]",
    ),
    (
        "ssn",
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "[REDACTED_SSN]",
    ),
    (
        "credit_card",
        re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        "[REDACTED_CC]",
    ),
    (
        "ip_address",
        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        "[REDACTED_IP]",
    ),
]


class PIIRedactor:
    """
    Detects and redacts PII from text content.

    Configurable per-category: you can enable/disable detection of specific
    PII types (e.g., allow IP addresses in technical discussions but always
    redact SSNs).
    """

    def __init__(
        self,
        enabled: bool = False,
        disabled_categories: Optional[list[str]] = None,
    ):
        self._enabled = enabled
        self._disabled_categories = set(disabled_categories or [])
        self._total_redactions = 0

        # Build active patterns list
        self._active_patterns = [
            (name, pattern, replacement)
            for name, pattern, replacement in PII_PATTERNS
            if name not in self._disabled_categories
        ]

        logger.info(
            "pii_redactor.initialized",
            enabled=enabled,
            active_patterns=len(self._active_patterns),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def redact(self, text: str) -> str:
        """
        Redact PII from the given text.

        Returns the redacted text. If redaction is disabled, returns the
        original text unchanged.
        """
        if not self._enabled or not text:
            return text

        result = text
        for name, pattern, replacement in self._active_patterns:
            result = pattern.sub(replacement, result)

        if result != text:
            self._total_redactions += 1
            logger.debug("pii_redactor.redacted", original_length=len(text))

        return result

    def scan(self, text: str) -> RedactionResult:
        """
        Scan text for PII without modifying it. Returns detection results.
        """
        if not text:
            return RedactionResult(
                original_length=0, redacted_length=0, redactions_made=0
            )

        redactions = 0
        categories = []
        redacted = text

        for name, pattern, replacement in self._active_patterns:
            matches = pattern.findall(text)
            if matches:
                redactions += len(matches)
                categories.append(name)
                redacted = pattern.sub(replacement, redacted)

        return RedactionResult(
            original_length=len(text),
            redacted_length=len(redacted),
            redactions_made=redactions,
            categories_found=categories,
        )

    @property
    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "total_redactions": self._total_redactions,
            "active_patterns": len(self._active_patterns),
        }
