"""
Unit tests for gwenn.harness.context — The Context Manager.

Tests cover token estimation, compaction triggers, image-block stripping,
conversation compaction (success and failure paths), and the stats property.

Every test is deterministic: no network access or API calls are needed.
A lightweight mock engine is used for compaction tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gwenn.config import ContextConfig
from gwenn.harness.context import ContextManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def context_config() -> ContextConfig:
    """A ContextConfig with small, predictable values for easy math.

    Note: fields with aliases must be set via the alias (env-var name)
    because ContextConfig does not set ``populate_by_name=True``.
    ``chars_per_token`` has no alias so it uses the field name directly.
    """
    return ContextConfig(
        GWENN_CONTEXT_LIMIT=1000,
        GWENN_COMPACTION_TRIGGER=0.8,
        chars_per_token=4.0,
    )


@pytest.fixture()
def ctx(context_config: ContextConfig) -> ContextManager:
    """A fresh ContextManager wired to the test config."""
    return ContextManager(context_config)


# ---------------------------------------------------------------------------
# Mock engine for compaction tests
# ---------------------------------------------------------------------------

@dataclass
class MockCompactResponse:
    """Stand-in for the API response returned by CognitiveEngine.compact."""
    text: str = "Summary of the conversation so far."


class MockEngine:
    """Minimal mock of CognitiveEngine with async compact() and extract_text()."""

    def __init__(self, *, should_fail: bool = False):
        self._should_fail = should_fail
        self.compact_calls: list[tuple[str, list[dict], str]] = []

    async def compact(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        compaction_prompt: str,
    ) -> MockCompactResponse:
        self.compact_calls.append((system_prompt, messages, compaction_prompt))
        if self._should_fail:
            raise RuntimeError("Engine compaction failed")
        return MockCompactResponse()

    @staticmethod
    def extract_text(response: MockCompactResponse) -> str:
        return response.text


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self, ctx: ContextManager) -> None:
        assert ctx.estimate_tokens("") == 0

    def test_short_string(self, ctx: ContextManager) -> None:
        # 20 chars / 4 chars_per_token = 5
        assert ctx.estimate_tokens("a" * 20) == 5

    def test_rounds_down(self, ctx: ContextManager) -> None:
        # 7 chars / 4 = 1.75 -> int(1.75) = 1
        assert ctx.estimate_tokens("abcdefg") == 1


# ---------------------------------------------------------------------------
# estimate_message_tokens
# ---------------------------------------------------------------------------

class TestEstimateMessageTokens:
    def test_string_content(self, ctx: ContextManager) -> None:
        messages = [{"role": "user", "content": "a" * 40}]
        # 40/4 = 10 tokens + 10 overhead = 20
        assert ctx.estimate_message_tokens(messages) == 20

    def test_list_content_text_blocks(self, ctx: ContextManager) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a" * 40},
                ],
            }
        ]
        # 40/4 = 10 tokens + 10 overhead = 20
        assert ctx.estimate_message_tokens(messages) == 20

    def test_image_block_adds_fixed_tokens(self, ctx: ContextManager) -> None:
        """Line 75: image blocks contribute _IMAGE_TOKEN_ESTIMATE (1600) tokens."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                ],
            }
        ]
        # 1600 (image) + 10 (overhead) = 1610
        assert ctx.estimate_message_tokens(messages) == 1610

    def test_image_plus_text_blocks(self, ctx: ContextManager) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                    {"type": "text", "text": "a" * 20},
                ],
            }
        ]
        # 1600 (image) + 5 (text 20/4) + 10 (overhead) = 1615
        assert ctx.estimate_message_tokens(messages) == 1615

    def test_non_dict_block_in_list(self, ctx: ContextManager) -> None:
        """When a content block is a plain string inside a list."""
        messages = [
            {
                "role": "user",
                "content": ["hello world"],  # 11 chars -> 2 tokens
            }
        ]
        # int(11/4)=2 + 10 overhead = 12
        assert ctx.estimate_message_tokens(messages) == 12

    def test_block_with_content_key(self, ctx: ContextManager) -> None:
        """Dict block that uses 'content' key instead of 'text'."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "a" * 40},
                ],
            }
        ]
        # 40/4 = 10 + 10 overhead = 20
        assert ctx.estimate_message_tokens(messages) == 20

    def test_multiple_messages(self, ctx: ContextManager) -> None:
        messages = [
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 40},
        ]
        # (10 + 10) + (10 + 10) = 40
        assert ctx.estimate_message_tokens(messages) == 40

    def test_empty_messages(self, ctx: ContextManager) -> None:
        assert ctx.estimate_message_tokens([]) == 0

    def test_missing_content_key(self, ctx: ContextManager) -> None:
        """Message with no 'content' key should just add overhead."""
        messages = [{"role": "user"}]
        # empty string -> 0 tokens + 10 overhead = 10
        assert ctx.estimate_message_tokens(messages) == 10


# ---------------------------------------------------------------------------
# needs_compaction
# ---------------------------------------------------------------------------

class TestNeedsCompaction:
    def test_below_threshold_returns_false(self, ctx: ContextManager) -> None:
        # threshold = 1000 * 0.8 = 800
        # system: 40/4=10, message: 40/4=10 + 10 overhead = 20, total = 30
        assert ctx.needs_compaction("a" * 40, [{"role": "user", "content": "b" * 40}]) is False

    def test_above_threshold_returns_true(self, ctx: ContextManager) -> None:
        """Lines 101-108: the True path with logging."""
        # threshold = 800
        # system_prompt: 3200 chars -> 800 tokens
        # messages: 1 message of 40 chars -> 10 + 10 = 20 tokens
        # total = 820 > 800
        system_prompt = "x" * 3200
        messages = [{"role": "user", "content": "y" * 40}]
        assert ctx.needs_compaction(system_prompt, messages) is True

    def test_exactly_at_threshold(self, ctx: ContextManager) -> None:
        """At exactly the threshold, should not trigger (only > triggers)."""
        # threshold = 800, we need total == 800 exactly
        # system: 3120 chars / 4 = 780 tokens
        # message: 40 chars / 4 = 10 tokens + 10 overhead = 20
        # total = 780 + 20 = 800  (not > 800)
        system_prompt = "x" * 3120
        messages = [{"role": "user", "content": "y" * 40}]
        assert ctx.needs_compaction(system_prompt, messages) is False


# ---------------------------------------------------------------------------
# _strip_image_blocks
# ---------------------------------------------------------------------------

class TestStripImageBlocks:
    def test_message_with_only_images(self) -> None:
        """Lines 118-142: all image blocks replaced with placeholder, content becomes string."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                ],
            }
        ]
        result = ContextManager._strip_image_blocks(messages)
        assert len(result) == 1
        # One image -> "[1 image shared]" placeholder is the only block
        # len(new_blocks)==1 and type=="text" so content collapses to string
        assert result[0]["content"] == "[1 image shared]"

    def test_message_with_image_and_text(self) -> None:
        """Image block stripped, text retained, placeholder prepended."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                    {"type": "text", "text": "Look at this photo"},
                ],
            }
        ]
        result = ContextManager._strip_image_blocks(messages)
        assert len(result) == 1
        content = result[0]["content"]
        # 2 blocks: placeholder + text -> stays as list
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "[1 image shared]"}
        assert content[1] == {"type": "text", "text": "Look at this photo"}

    def test_multiple_images(self) -> None:
        """Multiple images produce plural placeholder."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "a"}},
                    {"type": "image", "source": {"type": "base64", "data": "b"}},
                    {"type": "image", "source": {"type": "base64", "data": "c"}},
                ],
            }
        ]
        result = ContextManager._strip_image_blocks(messages)
        # 3 images, no other blocks -> placeholder is only block -> collapses to string
        assert result[0]["content"] == "[3 images shared]"

    def test_no_images_in_list_content(self) -> None:
        """List content with no images passes through unmodified (as list)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        result = ContextManager._strip_image_blocks(messages)
        assert len(result) == 1
        # No images, so no placeholder inserted. Two blocks remain -> stays list.
        assert result[0]["content"] == [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]

    def test_string_content_passes_through(self) -> None:
        """Messages with plain string content are unchanged."""
        messages = [{"role": "user", "content": "Just text"}]
        result = ContextManager._strip_image_blocks(messages)
        assert result == [{"role": "user", "content": "Just text"}]

    def test_single_text_block_collapses_to_string(self) -> None:
        """A single text block in a list collapses to its string value."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Only text"}],
            }
        ]
        result = ContextManager._strip_image_blocks(messages)
        # One text block, no images -> single block with type "text" -> collapse to string
        assert result[0]["content"] == "Only text"

    def test_empty_content_list(self) -> None:
        """An empty content list results in empty string content."""
        messages = [{"role": "user", "content": []}]
        result = ContextManager._strip_image_blocks(messages)
        # new_blocks is empty and no images -> not new_blocks is True -> content = ""
        assert result[0]["content"] == ""

    def test_does_not_mutate_original(self) -> None:
        """The original messages list should not be mutated."""
        original_block = {"type": "image", "source": {"type": "base64", "data": "abc"}}
        messages = [{"role": "user", "content": [original_block]}]
        ContextManager._strip_image_blocks(messages)
        # Original still has image block
        assert messages[0]["content"][0]["type"] == "image"

    def test_multiple_messages_mixed(self) -> None:
        """Multiple messages: one string, one with images, one text-only list."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                    {"type": "text", "text": "caption"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "response"}],
            },
        ]
        result = ContextManager._strip_image_blocks(messages)
        assert len(result) == 3
        assert result[0]["content"] == "Hi"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["text"] == "[1 image shared]"
        assert result[2]["content"] == "response"  # single text block collapsed


# ---------------------------------------------------------------------------
# compact — success path
# ---------------------------------------------------------------------------

class TestCompactSuccess:
    @pytest.mark.asyncio
    async def test_too_few_messages_returns_unchanged(self, ctx: ContextManager) -> None:
        """Line 165: <= 4 messages returns the original list."""
        engine = MockEngine()
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = await ctx.compact(engine, "system prompt", messages)
        assert result is messages  # identity — same object returned
        assert len(engine.compact_calls) == 0
        # compaction_count still incremented
        assert ctx.stats["compaction_count"] == 1

    @pytest.mark.asyncio
    async def test_compact_summarizes_old_messages(self, ctx: ContextManager) -> None:
        """Lines 161-225: successful compaction with engine mock."""
        engine = MockEngine()
        messages = [
            {"role": "user", "content": "old msg 1"},
            {"role": "assistant", "content": "old msg 2"},
            {"role": "user", "content": "old msg 3"},
            {"role": "assistant", "content": "old msg 4"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]
        result = await ctx.compact(engine, "system prompt", messages)

        # Engine was called once with old messages (first 4)
        assert len(engine.compact_calls) == 1
        sys_prompt, old_msgs, _ = engine.compact_calls[0]
        assert sys_prompt == "system prompt"
        assert len(old_msgs) == 4

        # Result: 2 summary messages + 4 recent = 6
        assert len(result) == 6
        assert result[0]["role"] == "user"
        assert "CONVERSATION SUMMARY" in result[0]["content"]
        assert "4 messages compacted" in result[0]["content"]
        assert "Summary of the conversation so far." in result[0]["content"]
        assert result[1]["role"] == "assistant"
        assert "reviewed the conversation summary" in result[1]["content"]

        # Recent messages preserved
        assert result[2]["content"] == "recent 1"
        assert result[3]["content"] == "recent 2"
        assert result[4]["content"] == "recent 3"
        assert result[5]["content"] == "recent 4"

    @pytest.mark.asyncio
    async def test_compact_strips_images_before_summarizing(self, ctx: ContextManager) -> None:
        """Image blocks in old messages are stripped before being sent to engine."""
        engine = MockEngine()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "big"}},
                    {"type": "text", "text": "here is a photo"},
                ],
            },
            {"role": "assistant", "content": "Nice photo!"},
            {"role": "user", "content": "old msg 3"},
            {"role": "assistant", "content": "old msg 4"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]
        result = await ctx.compact(engine, "system", messages)

        # The old messages sent to engine should have images stripped
        _, old_msgs, _ = engine.compact_calls[0]
        first_old = old_msgs[0]
        # Image was stripped; content should be list with placeholder + text
        assert isinstance(first_old["content"], list)
        assert first_old["content"][0]["text"] == "[1 image shared]"

    @pytest.mark.asyncio
    async def test_compact_increments_compaction_count(self, ctx: ContextManager) -> None:
        """Each call to compact increments _compaction_count."""
        engine = MockEngine()
        msgs_short = [{"role": "user", "content": f"m{i}"} for i in range(3)]
        msgs_long = [{"role": "user", "content": f"m{i}"} for i in range(6)]

        await ctx.compact(engine, "sp", msgs_short)
        assert ctx.stats["compaction_count"] == 1

        await ctx.compact(engine, "sp", msgs_long)
        assert ctx.stats["compaction_count"] == 2


# ---------------------------------------------------------------------------
# compact — failure path
# ---------------------------------------------------------------------------

class TestCompactFailure:
    @pytest.mark.asyncio
    async def test_engine_failure_produces_fallback_summary(self, ctx: ContextManager) -> None:
        """Lines 190-196: when engine raises, a fallback summary is used."""
        engine = MockEngine(should_fail=True)
        messages = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(8)
        ]
        result = await ctx.compact(engine, "system", messages)

        # Should still produce compacted output (not raise)
        assert len(result) == 6  # 2 summary + 4 recent

        summary_content = result[0]["content"]
        assert "CONVERSATION SUMMARY" in summary_content
        assert "Earlier conversation was summarized due to context limits" in summary_content
        assert "Original messages: 4" in summary_content

        # Recent messages preserved
        assert result[2]["content"] == "msg 4"
        assert result[5]["content"] == "msg 7"


# ---------------------------------------------------------------------------
# stats property
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_expected_keys(self, ctx: ContextManager, context_config: ContextConfig) -> None:
        """Line 229: stats property returns compaction count and config values."""
        stats = ctx.stats
        assert stats == {
            "compaction_count": 0,
            "context_limit": context_config.context_limit,
            "compaction_trigger": context_config.compaction_trigger,
        }

    @pytest.mark.asyncio
    async def test_stats_after_compaction(self, ctx: ContextManager) -> None:
        """Stats reflect compaction count after compaction runs."""
        engine = MockEngine()
        messages = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        await ctx.compact(engine, "sp", messages)
        assert ctx.stats["compaction_count"] == 1
