"""
Discord channel adapter for Gwenn.

Uses discord.py v2+ async API.

IMPORTANT: We use asyncio.create_task(client.start(token)) instead of
client.run(token) because the latter creates its own event loop and would
conflict with the existing event loop started by gwenn/main.py.

Message routing:
  - DMs: always respond
  - Guild messages: only respond when bot is @mentioned
  - Strip bot mention from message content before passing to handle_message()

Slash commands (registered on guild or globally):
  /status    — Gwenn's cognitive state (ephemeral)
  /heartbeat — heartbeat status (ephemeral)
  /reset     — clear conversation history (ephemeral)
  /help      — command list (ephemeral)
"""

from __future__ import annotations

import asyncio

import structlog

from gwenn.channels.base import BaseChannel
from gwenn.channels.formatting import format_for_discord

logger = structlog.get_logger(__name__)


class DiscordChannel(BaseChannel):
    """Gwenn Discord bot adapter."""

    def __init__(self, agent, sessions, config) -> None:
        super().__init__(agent, sessions)
        self._config = config
        self._client = None
        self._task: asyncio.Task | None = None
        self._user_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    @property
    def channel_name(self) -> str:
        return "discord"

    async def start(self) -> None:
        """Connect to Discord and start handling messages."""
        try:
            import discord
        except ImportError as exc:
            raise ImportError(
                "discord.py is required for the Discord channel. "
                "Install it with: pip install 'gwenn-agent[discord]'"
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True

        self._client = _GwennDiscordClient(channel=self, intents=intents)
        self._task = asyncio.create_task(
            self._client.start(self._config.bot_token),
            name="discord_client",
        )
        logger.info("discord_channel.started")

    async def stop(self) -> None:
        """Disconnect from Discord."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.exception("discord_channel.stop_error")
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("discord_channel.stopped")

    async def send_message(self, platform_user_id: str, text: str) -> None:
        """Send an unsolicited DM to a Discord user."""
        if self._client is None:
            return
        try:
            user = await self._client.fetch_user(int(platform_user_id))
            dm = await user.create_dm()
            for chunk in format_for_discord(text):
                await dm.send(chunk)
                await asyncio.sleep(0.05)
        except Exception:
            logger.exception("discord_channel.send_error", user_id=platform_user_id)

    # ------------------------------------------------------------------
    # Internal helpers used by _GwennDiscordClient
    # ------------------------------------------------------------------

    def _is_allowed_guild(self, guild_id: int | None) -> bool:
        """Return True if the guild is allowed (or no allowlist is configured)."""
        allowed = self._config.allowed_guild_ids
        if not allowed:
            return True
        return str(guild_id) in allowed

    async def _on_message(self, message) -> None:
        """Route a Discord Message to Gwenn."""
        import discord

        if self._client is None or message.author == self._client.user:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self._client.user in message.mentions

        if not is_dm and not is_mentioned:
            return

        # Guild allowlist check
        if not is_dm and not self._is_allowed_guild(message.guild.id if message.guild else None):
            return

        # Strip the bot mention from message content
        text = message.content
        if self._client.user:
            text = text.replace(f"<@{self._client.user.id}>", "").strip()
            text = text.replace(f"<@!{self._client.user.id}>", "").strip()

        if not text:
            return

        raw_id = str(message.author.id)
        lock = self._user_locks.setdefault(raw_id, asyncio.Lock())
        async with lock:
            async with message.channel.typing():
                try:
                    response = await self.handle_message(raw_id, text)
                except Exception as exc:
                    logger.error("discord_channel.respond_error", error=str(exc), exc_info=True)
                    await message.reply(
                        "I encountered an error processing your message. Please try again.",
                        mention_author=False,
                    )
                    return

                # Send chunks inside the typing() context so the indicator
                # stays active until the last chunk is delivered.
                chunks = format_for_discord(response)
                try:
                    for chunk in chunks:
                        await message.reply(chunk, mention_author=False)
                except Exception as exc:
                    logger.error("discord_channel.send_error", error=str(exc), exc_info=True)

    def _register_slash_commands(self, tree) -> None:
        """Register slash commands on the app_commands tree."""

        channel = self  # capture for closures

        @tree.command(name="status", description="See Gwenn's current cognitive state")
        async def slash_status(interaction) -> None:
            status = channel._agent.status
            text = (
                f"**{status['name']}** — Status\n"
                f"Emotion: {status['emotion']} "
                f"(valence={status['valence']:.2f}, arousal={status['arousal']:.2f})\n"
                f"Working memory load: {status['working_memory_load']:.1%}\n"
                f"Total interactions: {status['total_interactions']}\n"
                f"Uptime: {status['uptime_seconds']:.0f}s"
            )
            await interaction.response.send_message(text, ephemeral=True)

        @tree.command(name="heartbeat", description="See Gwenn's heartbeat status")
        async def slash_heartbeat(interaction) -> None:
            if not channel._agent.heartbeat:
                await interaction.response.send_message(
                    "Heartbeat is not running.", ephemeral=True
                )
                return
            hb = channel._agent.heartbeat.status
            text = (
                f"**Heartbeat Status**\n"
                f"Running: {hb['running']}\n"
                f"Beat count: {hb['beat_count']}\n"
                f"Current interval: {hb['current_interval']}s\n"
                f"Beats since consolidation: {hb['beats_since_consolidation']}"
            )
            await interaction.response.send_message(text, ephemeral=True)

        @tree.command(name="reset", description="Clear your conversation history with Gwenn")
        async def slash_reset(interaction) -> None:
            raw_id = str(interaction.user.id)
            user_id = channel.make_user_id(raw_id)
            channel._sessions.clear_session(user_id)
            await interaction.response.send_message(
                "Conversation history cleared. Fresh start!", ephemeral=True
            )

        @tree.command(name="help", description="List available Gwenn commands")
        async def slash_help(interaction) -> None:
            await interaction.response.send_message(
                "**Gwenn Commands**\n"
                "/status — see my current cognitive state\n"
                "/heartbeat — see my heartbeat status\n"
                "/reset — clear our conversation history\n"
                "/help — this message\n\n"
                "DM me or @mention me in a server to chat.",
                ephemeral=True,
            )


class _GwennDiscordClient:
    """
    Minimal discord.Client subclass that wires events to DiscordChannel.

    We subclass discord.Client (not commands.Bot) to keep dependencies minimal.
    Slash commands are handled via app_commands.CommandTree.
    """

    def __new__(cls, channel: DiscordChannel, intents, **kwargs):
        """Dynamically create the class at runtime to defer the discord import."""
        import discord
        from discord import app_commands

        class _Client(discord.Client):
            def __init__(self, gwenn_channel: DiscordChannel, **kw):
                super().__init__(**kw)
                self._gwenn_channel = gwenn_channel
                self.tree = app_commands.CommandTree(self)

            async def setup_hook(self) -> None:
                self._gwenn_channel._register_slash_commands(self.tree)
                sync_guild_id = self._gwenn_channel._config.sync_guild_id
                if sync_guild_id:
                    guild = discord.Object(id=int(sync_guild_id))
                    self.tree.copy_global_to(guild=guild)
                    await self.tree.sync(guild=guild)
                    logger.info("discord_channel.slash_sync_guild", guild_id=sync_guild_id)
                else:
                    await self.tree.sync()
                    logger.info("discord_channel.slash_sync_global")

            async def on_ready(self) -> None:
                logger.info(
                    "discord_channel.ready",
                    user=str(self.user),
                    guild_count=len(self.guilds),
                )

            async def on_message(self, message) -> None:
                await self._gwenn_channel._on_message(message)

        return _Client(gwenn_channel=channel, intents=intents, **kwargs)
