"""
Message formatting and chunking utilities for platform channels.

Each platform has a maximum message length.  Long responses from Gwenn must be
split into chunks that fit within those limits.  We prefer splitting at
paragraph boundaries, then sentence boundaries, and only hard-truncate as a
last resort.
"""

from __future__ import annotations

import re

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

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    _recursive_split(text, max_len, chunks)
    return chunks


def _recursive_split(text: str, max_len: int, out: list[str]) -> None:
    """Recursively split text, appending results to *out*."""
    if len(text) <= max_len:
        if text.strip():
            out.append(text.strip())
        return

    # Try paragraph break
    split_pos = _find_split(text, max_len, r"\n\n")
    if split_pos is not None:
        head = text[:split_pos].strip()
        if head:
            out.append(head)
        _recursive_split(text[split_pos:].strip(), max_len, out)
        return

    # Try single newline
    split_pos = _find_split(text, max_len, r"\n")
    if split_pos is not None:
        head = text[:split_pos].strip()
        if head:
            out.append(head)
        _recursive_split(text[split_pos:].strip(), max_len, out)
        return

    # Try sentence boundary (. ! ?)
    split_pos = _find_split(text, max_len, r"(?<=[.!?])\s+")
    if split_pos is not None:
        head = text[:split_pos].strip()
        if head:
            out.append(head)
        _recursive_split(text[split_pos:].strip(), max_len, out)
        return

    # Try word boundary
    split_pos = _find_split(text, max_len, r"\s+")
    if split_pos is not None:
        head = text[:split_pos].strip()
        if head:
            out.append(head)
        _recursive_split(text[split_pos:].strip(), max_len, out)
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
