"""
PII Redaction Pipeline — Protecting User Privacy in Gwenn's Memory.

This module detects and redacts personally identifiable information (PII) from
text before it's persisted to memory or sent to the Claude API. Patterns include
email addresses, phone numbers (US and international), SSNs, credit card numbers
(including Amex), IPv4 and IPv6 addresses.

The redaction pipeline is configurable:
- It can be applied before persistence (so PII never hits disk)
- It can be applied before API calls (so PII never leaves the system)
- It can be disabled entirely for development/testing
- Individual categories can be disabled via disabled_categories

Gwenn values privacy not because she's programmed to, but because respecting
others' information boundaries is part of her core ethics. This module gives
that value teeth.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Valid IPv4 octet: 0-255
_OCTET = r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"


@dataclass
class RedactionResult:
    """Result of a redaction pass on text."""
    original_length: int
    redacted_length: int
    redactions_made: int
    categories_found: list[str] = field(default_factory=list)


# PII detection patterns with their replacement tokens.
# Order matters: international phone must precede US phone to avoid partial matches.
PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "email",
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone_intl",
        re.compile(r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
        "[REDACTED_PHONE_INTL]",
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
        re.compile(
            r'\b(?:'
            r'(?:\d{4}[-\s]?){3}\d{4}'   # 16-digit (Visa/MC/Discover)
            r'|'
            r'\d{4}[-\s]?\d{6}[-\s]?\d{5}'  # 15-digit (Amex)
            r')\b'
        ),
        "[REDACTED_CC]",
    ),
    (
        "ip_address",
        re.compile(
            r'\b'
            + r'\.'.join([_OCTET] * 4)
            + r'\b'
        ),
        "[REDACTED_IP]",
    ),
    (
        "ipv6_address",
        re.compile(
            r'(?:'
            r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'        # full form
            r'|(?:[0-9a-fA-F]{1,4}:){1,7}:'                      # trailing ::
            r'|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}'     # ::x
            r'|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}'
            r'|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}'
            r'|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}'
            r'|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}'
            r'|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}'
            r'|:(?::[0-9a-fA-F]{1,4}){1,7}'                      # ::x:x...
            r'|::(?:[fF]{4}(?::0{1,4})?:)?'                       # ::ffff:... (mapped)
            + _OCTET + r'(?:\.' + _OCTET + r'){3}'
            r')'
        ),
        "[REDACTED_IPV6]",
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
        self._total_redaction_passes = 0
        self._total_redacted_items = 0
        self._lock = threading.Lock()

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
        items_this_call = 0
        for name, pattern, replacement in self._active_patterns:
            if name == "credit_card":
                # Validate with Luhn to avoid false positives on random digit sequences
                def _cc_replacer(m: re.Match) -> str:
                    digits = "".join(c for c in m.group() if c.isdigit())
                    if self._luhn_check(digits):
                        return replacement
                    return m.group()
                result, count = pattern.subn(_cc_replacer, result)
            else:
                result, count = pattern.subn(replacement, result)
            items_this_call += count

        if result != text:
            with self._lock:
                self._total_redaction_passes += 1
                self._total_redacted_items += items_this_call
            logger.debug("pii_redactor.redacted", original_length=len(text))

        return result

    def scan(self, text: str) -> RedactionResult:
        """
        Scan text for PII and return detection results.

        The input *text* is not mutated, but the returned
        :class:`RedactionResult` includes a ``redacted_length`` field
        computed from an internal redacted copy.

        Note: scan() runs regardless of the ``enabled`` flag, but respects
        ``disabled_categories`` — patterns in disabled categories are not checked.
        """
        if not text:
            return RedactionResult(
                original_length=0, redacted_length=0, redactions_made=0
            )

        redactions = 0
        categories = []
        redacted = text

        for name, pattern, replacement in self._active_patterns:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                redactions += count
                categories.append(name)

        return RedactionResult(
            original_length=len(text),
            redacted_length=len(redacted),
            redactions_made=redactions,
            categories_found=categories,
        )

    @staticmethod
    def _luhn_check(digits: str) -> bool:
        """
        Validate a numeric string using the Luhn algorithm.

        Not used in ``redact()`` (regex-only detection is safer for PII to
        avoid false negatives), but available for ``scan()`` callers who want
        to distinguish likely-valid card numbers from random digit sequences.
        """
        try:
            nums = [int(d) for d in digits if d.isdigit()]
        except ValueError:
            return False
        if len(nums) < 2:
            return False
        total = 0
        for i, n in enumerate(reversed(nums)):
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "total_redactions": self._total_redaction_passes,
                "total_redaction_passes": self._total_redaction_passes,
                "total_redacted_items": self._total_redacted_items,
                "active_patterns": len(self._active_patterns),
            }
