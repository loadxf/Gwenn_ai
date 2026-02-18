"""
Unit tests for DiscordChannel using discord.py mocks.

These tests exercise routing logic and session management without any real
Discord API calls.  The discord.Client and related objects are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gwenn.channels.discord_channel import DiscordChannel
from gwenn.channels.session import SessionManager


def make_config(allowed_guild_ids=None, sync_guild_id=None):
    cfg = MagicMock()
    cfg.bot_token = "test_token"
    cfg.allowed_guild_ids = allowed_guild_ids or []
    cfg.sync_guild_id = sync_guild_id
    return cfg


def make_channel(allowed_guild_ids=None, sync_guild_id=None):
    """Create a DiscordChannel with mocked agent and sessions."""
    agent = AsyncMock()
    agent.respond = AsyncMock(return_value="Hello from Gwenn")
    agent.apply_startup_onboarding = MagicMock()
    agent.status = {
        "name": "Gwenn",
        "emotion": "calm",
        "valence": 0.6,
        "arousal": 0.2,
        "working_memory_load": 0.3,
        "total_interactions": 5,
        "uptime_seconds": 60.0,
        "resilience": {"breaker_active": False},
    }
    hb = MagicMock()
    hb.status = {
        "running": True,
        "beat_count": 3,
        "current_interval": 30,
        "beats_since_consolidation": 1,
    }
    agent.heartbeat = hb
    identity = MagicMock()
    identity.should_run_startup_onboarding.return_value = False
    agent.identity = identity
    sessions = SessionManager()
    config = make_config(allowed_guild_ids=allowed_guild_ids, sync_guild_id=sync_guild_id)
    ch = DiscordChannel(agent, sessions, config)
    return ch, agent, sessions


def make_dm_message(user_id="12345", content="hello", bot_user=None):
    """Create a mock DM message."""
    import discord as _discord

    msg = MagicMock()
    msg.author = MagicMock()
    msg.author.id = int(user_id)
    msg.content = content
    msg.mentions = []
    # DM channel
    msg.channel = MagicMock(spec=_discord.DMChannel)
    msg.guild = None
    msg.reply = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock().__aenter__)
    # Make typing() work as an async context manager
    typing_cm = AsyncMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    msg.channel.typing = MagicMock(return_value=typing_cm)
    return msg


def make_guild_message(user_id="12345", content="hello", bot_user=None, guild_id=111):
    """Create a mock guild message."""
    msg = MagicMock()
    msg.author = MagicMock()
    msg.author.id = int(user_id)
    msg.content = content
    msg.guild = MagicMock()
    msg.guild.id = guild_id
    # Text channel (not DM)
    msg.channel = MagicMock()
    msg.channel.__class__ = MagicMock  # not DMChannel

    if bot_user:
        msg.mentions = [bot_user]
    else:
        msg.mentions = []

    msg.reply = AsyncMock()
    typing_cm = AsyncMock()
    typing_cm.__aenter__ = AsyncMock(return_value=None)
    typing_cm.__aexit__ = AsyncMock(return_value=False)
    msg.channel.typing = MagicMock(return_value=typing_cm)
    return msg


# ============================================================================
# Guild allowlist tests
# ============================================================================


class TestGuildAllowlist:
    def test_no_allowlist_permits_all_guilds(self):
        ch, _, _ = make_channel(allowed_guild_ids=[])
        assert ch._is_allowed_guild(999) is True

    def test_allowlist_permits_listed_guild(self):
        ch, _, _ = make_channel(allowed_guild_ids=["111", "222"])
        assert ch._is_allowed_guild(111) is True

    def test_allowlist_blocks_unlisted_guild(self):
        ch, _, _ = make_channel(allowed_guild_ids=["111", "222"])
        assert ch._is_allowed_guild(333) is False


# ============================================================================
# make_user_id
# ============================================================================


class TestMakeUserId:
    def test_channel_name_prefix(self):
        ch, _, _ = make_channel()
        assert ch.make_user_id("99") == "discord_99"


# ============================================================================
# _on_message routing tests
# ============================================================================


class TestOnMessage:
    @pytest.fixture(autouse=True)
    def import_discord(self):
        """Skip tests if discord.py is not installed."""
        pytest.importorskip("discord")

    @pytest.mark.asyncio
    async def test_message_requires_setup_when_onboarding_pending(self):
        ch, agent, _ = make_channel()
        agent.identity.should_run_startup_onboarding.return_value = True
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_dm_message(user_id="42", content="hi gwenn")
        msg.author = MagicMock()
        msg.author.id = 42
        ch._client.user.__eq__ = lambda self, other: False

        await ch._on_message(msg)

        agent.respond.assert_not_called()
        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "/setup" in reply_text

    @pytest.mark.asyncio
    async def test_dm_always_responded(self):
        ch, agent, _ = make_channel()
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_dm_message(user_id="42", content="hi gwenn")
        msg.author = MagicMock()
        msg.author.id = 42
        # Not the bot itself
        ch._client.user.__eq__ = lambda self, other: False

        await ch._on_message(msg)
        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args.kwargs
        assert call_kwargs["user_message"] == "hi gwenn"

    @pytest.mark.asyncio
    async def test_replies_disable_allowed_mentions(self):
        import discord as _discord

        ch, _, _ = make_channel()
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_dm_message(user_id="42", content="hi @everyone")
        msg.author = MagicMock()
        msg.author.id = 42
        ch._client.user.__eq__ = lambda self, other: False

        await ch._on_message(msg)

        assert msg.reply.call_count >= 1
        for _args, kwargs in msg.reply.call_args_list:
            assert isinstance(kwargs.get("allowed_mentions"), _discord.AllowedMentions)
            assert kwargs["allowed_mentions"].everyone is False
            assert kwargs["allowed_mentions"].roles is False
            assert kwargs["allowed_mentions"].users is False

    @pytest.mark.asyncio
    async def test_guild_message_ignored_if_not_mentioned(self):
        ch, agent, _ = make_channel()
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_guild_message(user_id="42", content="random chat", bot_user=None)
        msg.author = MagicMock()
        msg.author.id = 42
        ch._client.user.__eq__ = lambda self, other: False

        await ch._on_message(msg)
        agent.respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_guild_message_responded_when_mentioned(self):
        ch, agent, _ = make_channel()
        bot_user = MagicMock()
        bot_user.id = 999
        ch._client = MagicMock()
        ch._client.user = bot_user

        msg = make_guild_message(user_id="42", content="<@999> hello!", bot_user=bot_user)
        msg.author = MagicMock()
        msg.author.id = 42
        # author != bot
        bot_user.__eq__ = lambda self, other: other is bot_user

        await ch._on_message(msg)
        agent.respond.assert_called_once()

    @pytest.mark.asyncio
    async def test_bot_own_messages_ignored(self):
        ch, agent, _ = make_channel()
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_dm_message(user_id="999", content="I said something")
        # Message author IS the bot
        msg.author = ch._client.user

        await ch._on_message(msg)
        agent.respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_mention_stripped_from_content(self):
        ch, agent, _ = make_channel()
        bot_user = MagicMock()
        bot_user.id = 999
        ch._client = MagicMock()
        ch._client.user = bot_user

        msg = make_guild_message(
            user_id="42",
            content="<@999> what is the weather?",
            bot_user=bot_user,
        )
        msg.author = MagicMock()
        msg.author.id = 42
        bot_user.__eq__ = lambda self, other: other is bot_user

        await ch._on_message(msg)
        call_kwargs = agent.respond.call_args.kwargs
        assert "<@999>" not in call_kwargs["user_message"]
        assert "what is the weather?" in call_kwargs["user_message"]

    @pytest.mark.asyncio
    async def test_error_sends_error_reply(self):
        ch, agent, _ = make_channel()
        agent.respond = AsyncMock(side_effect=RuntimeError("boom"))
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        msg = make_dm_message(user_id="77", content="trigger error")
        msg.author = MagicMock()
        msg.author.id = 77
        ch._client.user.__eq__ = lambda self, other: False

        await ch._on_message(msg)
        msg.reply.assert_called_once()
        text = msg.reply.call_args[0][0]
        assert "error" in text.lower()


# ============================================================================
# Guild allowlist enforcement in _on_message
# ============================================================================


class TestGuildAllowlistInMessage:
    @pytest.fixture(autouse=True)
    def import_discord(self):
        pytest.importorskip("discord")

    @pytest.mark.asyncio
    async def test_blocked_guild_not_responded(self):
        ch, agent, _ = make_channel(allowed_guild_ids=["999"])
        bot_user = MagicMock()
        bot_user.id = 1
        ch._client = MagicMock()
        ch._client.user = bot_user

        msg = make_guild_message(user_id="42", content="<@1> hi", bot_user=bot_user, guild_id=555)
        msg.author = MagicMock()
        msg.author.id = 42
        bot_user.__eq__ = lambda self, other: other is bot_user

        await ch._on_message(msg)
        agent.respond.assert_not_called()


class _FakeCommandTree:
    def __init__(self) -> None:
        self.commands = {}

    def command(self, name: str, description: str):
        def _decorator(func):
            self.commands[name] = func
            return func

        return _decorator


class TestSlashAllowlist:
    @pytest.mark.asyncio
    async def test_status_slash_command_blocks_disallowed_guild(self):
        ch, _, _ = make_channel(allowed_guild_ids=["999"])
        tree = _FakeCommandTree()
        ch._register_slash_commands(tree)

        interaction = MagicMock()
        interaction.guild = MagicMock()
        interaction.guild.id = 123
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await tree.commands["status"](interaction)

        interaction.response.send_message.assert_called_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "not allowed" in args[0]
        assert kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_status_slash_command_allows_approved_guild(self):
        ch, _, _ = make_channel(allowed_guild_ids=["123"])
        tree = _FakeCommandTree()
        ch._register_slash_commands(tree)

        interaction = MagicMock()
        interaction.guild = MagicMock()
        interaction.guild.id = 123
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await tree.commands["status"](interaction)

        interaction.response.send_message.assert_called_once()
        args, kwargs = interaction.response.send_message.call_args
        assert "Gwenn" in args[0]
        assert "Mood:" in args[0]
        assert kwargs["ephemeral"] is True
