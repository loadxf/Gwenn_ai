"""
Message formatting and chunking utilities for platform channels.

Each platform has a maximum message length.  Long responses from Gwenn must be
split into chunks that fit within those limits.  We prefer splitting at
paragraph boundaries, then sentence boundaries, and only hard-truncate as a
last resort.
"""

from __future__ import annotations

import html as _html_mod
import re
from typing import Any

TELEGRAM_MAX_LEN: int = 4096
DISCORD_MAX_LEN: int = 2000

# Markdown is split *before* HTML conversion. HTML tags, entity escaping
# (& → &amp;), and <pre>/<code> wrappers expand the text.  We split at a
# reduced limit so converted chunks stay within TELEGRAM_MAX_LEN.
_TELEGRAM_SPLIT_LEN: int = 3800

# Telegram parse mode used with every bot.send_message / reply_text call that
# delivers Gwenn's AI-generated responses. HTML is more robust than MarkdownV2
# because it doesn't require escaping every punctuation character and handles
# code blocks, bold and italic without ambiguity with bullet-list asterisks.
TELEGRAM_PARSE_MODE: str = "HTML"

# ── Compiled regexes for markdown → Telegram HTML conversion ─────────────────

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# Underscore italic only — single-star italic is ambiguous with bullet lists.
_ITALIC_RE = re.compile(r"(?<![_\w])_([^_\n]+)_(?![_\w])")
_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_HEADER_RE = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)
# Blockquote: lines starting with '> ' (consecutive lines grouped).
_BLOCKQUOTE_RE = re.compile(r"(?:^> .+(?:\n|$))+", re.MULTILINE)

# Regex matching HTML entities (&amp; &lt; &gt; &quot; &#…;).
_HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[a-zA-Z]+);")

# Tag-stripping regex for plain-text fallback when HTML parse fails.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert Claude markdown output to Telegram-compatible HTML.

    Processing order:
      1. Extract fenced code blocks → <pre language="…">…</pre>  (HTML-escaped)
      2. Extract inline code        → <code>…</code>             (HTML-escaped)
      3. HTML-escape all remaining plain text
      4. Convert **bold** → <b>…</b>
      5. Convert ~~strikethrough~~ → <s>…</s>
      6. Convert _italic_ → <i>…</i>   (underscore form only)
      7. Convert ## headings → <b>…</b>
      8. Convert > blockquotes → <blockquote>…</blockquote>
      9. Restore extracted code blocks

    Sentinels (\\x02N\\x03) protect code from HTML-escaping and markdown
    regex passes. They survive html.escape() because \\x02/\\x03 are not
    HTML-special characters.
    """
    sentinel_map: dict[str, str] = {}
    counter = [0]

    def _sentinel() -> str:
        key = f"\x02{counter[0]}\x03"
        counter[0] += 1
        return key

    def _fence_repl(m: re.Match) -> str:
        lang = m.group(1).strip()
        code = _html_mod.escape(m.group(2))
        # Use <pre language="X"> instead of <pre><code class="language-X">
        # to avoid spaces inside tags — the second-pass whitespace split in
        # format_for_telegram could otherwise break a <code class=...> tag.
        # Telegram accepts language= on <pre> but doesn't highlight either way.
        block = f'<pre language="{lang}">{code}</pre>' if lang else f"<pre>{code}</pre>"
        key = _sentinel()
        sentinel_map[key] = block
        return key

    def _inline_repl(m: re.Match) -> str:
        code = _html_mod.escape(m.group(1))
        span = f"<code>{code}</code>"
        key = _sentinel()
        sentinel_map[key] = span
        return key

    def _blockquote_sentinel_repl(m: re.Match) -> str:
        lines = m.group(0).splitlines()
        stripped = "\n".join(
            line.removeprefix("> ").removeprefix(">") for line in lines
        )
        escaped = _html_mod.escape(stripped)
        block = f"<blockquote>{escaped}</blockquote>"
        key = _sentinel()
        sentinel_map[key] = block
        return key

    # Phase 1 — protect code and blockquotes from further processing.
    text = _FENCE_RE.sub(_fence_repl, text)
    text = _INLINE_CODE_RE.sub(_inline_repl, text)
    # Blockquotes must be extracted before HTML-escaping since '>' is a special char.
    text = _BLOCKQUOTE_RE.sub(_blockquote_sentinel_repl, text)

    # Phase 2 — HTML-escape all remaining plain text.
    text = _html_mod.escape(text)

    # Phase 3 — convert markdown formatting to HTML tags.
    # Bold before italic so **bold _and italic_** nests correctly.
    text = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _STRIKETHROUGH_RE.sub(lambda m: f"<s>{m.group(1)}</s>", text)
    text = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _HEADER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)

    # Phase 4 — restore protected blocks.
    for key, block in sentinel_map.items():
        text = text.replace(key, block)

    return text


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from *text* for plain-text fallback delivery."""
    return _HTML_TAG_RE.sub("", text)


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
    _iterative_split(text, max_len, chunks)
    return chunks


def _iterative_split(text: str, max_len: int, out: list[str]) -> None:
    """Iteratively split text, appending results to *out*.

    Uses a loop instead of recursion to avoid hitting Python's recursion
    limit for very long inputs with no natural break points.
    """
    _PATTERNS = [
        r"\n\n",  # Paragraph break
        r"\n",  # Single newline
        r"(?<=[.!?])\s+",  # Sentence boundary
        r"\s+",  # Word boundary
    ]
    while text:
        if len(text) <= max_len:
            out.append(text)
            return

        split_pos = None
        for pattern in _PATTERNS:
            pos = _find_split(text, max_len, pattern)
            # Reject zero-length heads — they'd lose the delimiter and stall.
            if pos is not None and pos > 0:
                split_pos = pos
                break

        if split_pos is not None:
            out.append(text[:split_pos])
            text = text[split_pos:]
        else:
            # Hard truncation — no suitable break point found
            out.append(text[:max_len])
            text = text[max_len:]


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


def _find_safe_html_split(html_text: str, max_len: int) -> int | None:
    """Find a split position in *html_text* that does not break inside an HTML
    tag or an HTML entity (``&amp;``, ``&lt;``, ``&#123;``, etc.).

    Scans backwards from *max_len* looking for whitespace that is outside
    any ``<…>`` tag and not inside an ``&…;`` entity.  Returns the position
    after the whitespace (start of remainder), or None if no safe split
    point exists.
    """
    # Pre-compute entity spans so we can quickly check if a position falls
    # inside one.  Entities are short (2-8 chars) so the set is small.
    entity_ranges: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _HTML_ENTITY_RE.finditer(html_text)
    ]

    def _inside_entity(p: int) -> bool:
        for start, end in entity_ranges:
            if start < p < end:
                return True
            if start >= p:
                break
        return False

    pos = max_len
    while pos > 0:
        if html_text[pos - 1] in (" ", "\n", "\t"):
            # Check we're not inside an HTML tag.
            last_lt = html_text.rfind("<", 0, pos)
            last_gt = html_text.rfind(">", 0, pos)
            if last_lt <= last_gt and not _inside_entity(pos):
                return pos
        pos -= 1
    return None


def format_for_telegram(text: str) -> list[str]:
    """
    Return *text* split into Telegram-sized chunks, each converted to HTML.

    Two-pass approach:
      1. Split raw markdown at ``_TELEGRAM_SPLIT_LEN`` so most chunks convert
         to HTML within ``TELEGRAM_MAX_LEN``.
      2. Convert each chunk to HTML.
      3. If HTML expansion (entity escaping, tags) still pushes a chunk over
         the limit, re-split using a tag-aware splitter that ensures split
         points never fall inside an ``<…>`` tag.
    """
    md_chunks = split_message(text, _TELEGRAM_SPLIT_LEN)
    result: list[str] = []
    for md_chunk in md_chunks:
        html = markdown_to_telegram_html(md_chunk)
        if len(html) <= TELEGRAM_MAX_LEN:
            result.append(html)
        else:
            _split_html_safe(html, TELEGRAM_MAX_LEN, result)
    return result


def _split_html_safe(html: str, max_len: int, out: list[str]) -> None:
    """Split HTML text into chunks without breaking inside tags."""
    while html:
        if len(html) <= max_len:
            out.append(html)
            return
        pos = _find_safe_html_split(html, max_len)
        if pos is not None and pos > 0:
            out.append(html[:pos])
            html = html[pos:]
        else:
            # No safe split found — fall back to hard truncation.
            out.append(html[:max_len])
            html = html[max_len:]


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
    title = f"**{status['name']}** — Status" if markdown_heading else f"{status['name']} — Status"
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
    circuit_open = bool(hb.get("circuit_open"))
    recovery = float(hb.get("circuit_recovery_in", 0.0))
    circuit_text = "open" if circuit_open else "closed"
    if circuit_open:
        circuit_text = f"open (recovering in {recovery:.1f}s)"
    return (
        f"{title}\n"
        f"Running: {hb['running']}\n"
        f"Beat count: {hb['beat_count']}\n"
        f"Current interval: {hb['current_interval']}s\n"
        f"Beats since consolidation: {hb['beats_since_consolidation']}\n"
        f"Circuit breaker: {circuit_text}"
    )
