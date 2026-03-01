"""
Discord channel adapter for Gwenn.

Uses discord.py v2+ async API.

IMPORTANT: We use asyncio.create_task(client.start(token)) instead of
client.run(token) because the latter creates its own event loop and would
conflict with the existing event loop started by gwenn/main.py.

Message routing:
  - DMs: optional (controlled by config + user allowlist)
  - Guild messages: only respond when bot is @mentioned
  - Strip bot mention from message content before passing to handle_message()

Slash commands (registered on guild or globally):
  /status    — Gwenn's cognitive state (ephemeral)
  /heartbeat — heartbeat status (ephemeral)
  /setup     — first-run onboarding profile (ephemeral)
  /reset     — clear conversation history (ephemeral)
  /help      — command list (ephemeral)
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING

import structlog

from gwenn.channels.base import BaseChannel
from gwenn.types import UserMessage

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager
    from gwenn.config import DiscordConfig
from gwenn.channels.formatting import (
    format_for_discord,
    render_heartbeat_text,
    render_status_text,
)

logger = structlog.get_logger(__name__)


class DiscordChannel(BaseChannel):
    """Gwenn Discord bot adapter."""

    def __init__(
        self,
        agent: SentientAgent,
        sessions: SessionManager,
        config: DiscordConfig,
    ) -> None:
        super().__init__(
            agent,
            sessions,
            user_lock_cache_size=config.user_lock_cache_size,
        )
        self._config = config
        self._client = None
        self._task: asyncio.Task | None = None
        self._ready_event: asyncio.Event = asyncio.Event()

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

        # Suppress discord.py's "PyNaCl is not installed" warning —
        # Gwenn does not use voice features.
        try:
            discord.VoiceClient.warn_nacl = False
        except AttributeError:
            pass

        intents = discord.Intents.default()
        intents.message_content = True

        self._ready_event.clear()
        self._client = _create_discord_client(channel=self, intents=intents)
        self._task = asyncio.create_task(
            self._client.start(self._config.bot_token),
            name="discord_client",
        )
        # Wait for on_ready or an early task failure (bad token, network error).
        ready_fut = asyncio.ensure_future(self._ready_event.wait())
        done, _ = await asyncio.wait(
            {ready_fut, self._task},
            timeout=30.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        ready_fut.cancel()
        if self._task in done:
            # Task finished before ready — clean up and propagate the error.
            try:
                self._task.result()  # raises if the task failed
            except Exception:
                await self.stop()
                raise
        if not self._ready_event.is_set():
            await self.stop()
            raise TimeoutError("Discord client did not become ready within 30s")
        logger.info("discord_channel.started")

    async def stop(self) -> None:
        """Disconnect from Discord."""
        if self._audio_transcriber is not None:
            try:
                await self._audio_transcriber.close()
            except Exception:
                logger.debug("discord_channel.audio_transcriber_close_error", exc_info=True)
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
        uid = self._validate_platform_id(platform_user_id)
        if uid is None:
            return
        try:
            import discord

            user = await self._client.fetch_user(uid)
            dm = await user.create_dm()
            no_mentions = discord.AllowedMentions.none()
            chunks = format_for_discord(text)
            for i, chunk in enumerate(chunks):
                await dm.send(chunk, allowed_mentions=no_mentions)
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
        except Exception:
            logger.exception("discord_channel.send_error", user_id=platform_user_id)

    async def send_proactive(self, text: str) -> None:
        """Send a proactive message to explicitly configured Discord owner users via DM."""
        owner_ids = self._id_set(self._config.owner_user_ids)
        for uid in owner_ids:
            if uid:
                await self.send_message(uid, text)
                await asyncio.sleep(0.5)

    async def send_to_session(self, session_id: str, text: str) -> bool:
        """Send to the originating Discord context of a session."""
        if not session_id.startswith("discord_") or self._client is None:
            return False
        scope = session_id[len("discord_"):]

        if scope.startswith("thread:") or scope.startswith("chat:"):
            channel_id_str = scope.split(":", 1)[1]
            try:
                channel = self._client.get_channel(int(channel_id_str))
                if channel is None:
                    channel = await self._client.fetch_channel(int(channel_id_str))
                for chunk in format_for_discord(text):
                    await channel.send(chunk)
                return True
            except Exception:
                return False
        elif scope.startswith("user:"):
            await self.send_message(scope[len("user:"):], text)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers used by _GwennDiscordClient
    # ------------------------------------------------------------------

    def _is_allowed_guild(self, guild_id: int | None) -> bool:
        """Return True if the guild is allowed (or no allowlist is configured)."""
        allowed = self._id_set(self._config.allowed_guild_ids)
        if not allowed:
            return True
        return str(guild_id) in allowed

    def _is_allowed_user(self, raw_user_id: str) -> bool:
        allowed_users = self._id_set(self._config.allowed_user_ids)
        return not allowed_users or raw_user_id in allowed_users

    def _is_allowed_dm_user(self, raw_user_id: str) -> bool:
        return self._config.allow_direct_messages and self._is_allowed_user(raw_user_id)

    def _is_owner_user(self, raw_user_id: str) -> bool:
        owner_ids = self._id_set(self._config.owner_user_ids)
        if owner_ids:
            return raw_user_id in owner_ids
        return self._is_allowed_user(raw_user_id)

    def _session_scope_mode(self) -> str:
        return self._normalize_scope_mode(self._config.session_scope_mode, default="per_thread")

    def _session_scope_key(
        self,
        *,
        raw_user_id: str,
        raw_chat_id: str | None,
        raw_thread_id: str | None,
    ) -> str:
        return self.make_session_scope_key(
            raw_user_id=raw_user_id,
            raw_chat_id=raw_chat_id,
            raw_thread_id=raw_thread_id,
            scope_mode=self._session_scope_mode(),
        )

    def _session_id_for_interaction(self, interaction, raw_user_id: str) -> str:
        import discord as _discord

        channel_obj = getattr(interaction, "channel", None)
        raw_chat_id = self._normalize_optional_id(getattr(channel_obj, "id", None))
        raw_thread_id = None
        if (
            channel_obj is not None
            and hasattr(_discord, "Thread")
            and isinstance(channel_obj, _discord.Thread)
        ):
            raw_thread_id = raw_chat_id
        scope_key = self._session_scope_key(
            raw_user_id=raw_user_id,
            raw_chat_id=raw_chat_id,
            raw_thread_id=raw_thread_id,
        )
        return self.make_session_id(scope_key)

    # ------------------------------------------------------------------
    # Image attachment helpers
    # ------------------------------------------------------------------

    _SUPPORTED_IMAGE_MIMES: set[str] = {
        "image/jpeg", "image/png", "image/gif", "image/webp",
    }
    _IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    _SUPPORTED_VIDEO_MIMES: set[str] = {
        "video/mp4", "video/webm", "video/quicktime",
    }
    _VIDEO_EXTENSIONS: set[str] = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    _SUPPORTED_AUDIO_MIMES: set[str] = {
        "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav", "audio/webm",
    }
    _AUDIO_EXTENSIONS: set[str] = {".ogg", ".mp3", ".m4a", ".wav"}
    _MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    _MAX_VIDEO_BYTES: int = 20 * 1024 * 1024  # 20 MB

    async def _extract_image_attachments(self, message) -> list[dict]:
        """Download image attachments from a Discord message.

        Returns a list of Claude API image content blocks, or ``[]`` if
        there are none or all downloads fail.
        """
        blocks: list[dict] = []
        for att in message.attachments:
            # Check content type or fall back to extension.
            ct = (att.content_type or "").split(";")[0].strip().lower()
            ext = ("." + att.filename.rsplit(".", 1)[-1]).lower() if "." in att.filename else ""
            if ct not in self._SUPPORTED_IMAGE_MIMES and ext not in self._IMAGE_EXTENSIONS:
                continue
            if att.size and att.size > self._MAX_IMAGE_BYTES:
                logger.warning(
                    "discord_channel.attachment_too_large",
                    filename=att.filename,
                    size=att.size,
                )
                continue
            try:
                data = await att.read()
                b64 = base64.standard_b64encode(data).decode("ascii")
                media_type = ct if ct in self._SUPPORTED_IMAGE_MIMES else "image/jpeg"
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                })
            except Exception as exc:
                logger.warning(
                    "discord_channel.attachment_download_failed",
                    filename=att.filename,
                    error=str(exc),
                )
        return blocks

    async def _extract_video_attachments(self, message) -> list[tuple[bytes, str]]:
        """Download video attachments from a Discord message.

        Returns a list of ``(video_bytes, filename)`` tuples.
        """
        results: list[tuple[bytes, str]] = []
        for att in message.attachments:
            ct = (att.content_type or "").split(";")[0].strip().lower()
            ext = ("." + att.filename.rsplit(".", 1)[-1]).lower() if "." in att.filename else ""
            if ct not in self._SUPPORTED_VIDEO_MIMES and ext not in self._VIDEO_EXTENSIONS:
                continue
            if att.size and att.size > self._MAX_VIDEO_BYTES:
                logger.warning(
                    "discord_channel.video_too_large",
                    filename=att.filename,
                    size=att.size,
                )
                continue
            try:
                data = await att.read()
                results.append((data, att.filename))
            except Exception as exc:
                logger.warning(
                    "discord_channel.video_download_failed",
                    filename=att.filename,
                    error=str(exc),
                )
        return results

    async def _extract_audio_attachments(self, message) -> list[tuple[bytes, str]]:
        """Download audio attachments from a Discord message.

        Returns a list of ``(audio_bytes, filename)`` tuples.
        """
        results: list[tuple[bytes, str]] = []
        for att in message.attachments:
            ct = (att.content_type or "").split(";")[0].strip().lower()
            ext = ("." + att.filename.rsplit(".", 1)[-1]).lower() if "." in att.filename else ""
            if ct not in self._SUPPORTED_AUDIO_MIMES and ext not in self._AUDIO_EXTENSIONS:
                continue
            try:
                data = await att.read()
                results.append((data, att.filename))
            except Exception as exc:
                logger.warning(
                    "discord_channel.audio_download_failed",
                    filename=att.filename,
                    error=str(exc),
                )
        return results

    # ------------------------------------------------------------------
    # Audio transcriber helper
    # ------------------------------------------------------------------

    _audio_transcriber = None

    def _get_audio_transcriber(self):
        """Lazily create an AudioTranscriber from the agent's Groq config."""
        if self._audio_transcriber is not None:
            return self._audio_transcriber
        groq_config = getattr(self._agent._config, "groq", None)
        if groq_config is None or not groq_config.is_available:
            return None
        try:
            from gwenn.media.audio import AudioTranscriber

            self._audio_transcriber = AudioTranscriber(groq_config)
            return self._audio_transcriber
        except Exception as exc:
            logger.warning("discord_channel.transcriber_init_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, message) -> None:
        """Route a Discord Message to Gwenn."""
        import discord

        if self._client is None or message.author == self._client.user:
            return

        raw_id = str(message.author.id)
        is_dm = isinstance(message.channel, discord.DMChannel)

        if is_dm and not self._is_allowed_dm_user(raw_id):
            no_mentions = discord.AllowedMentions.none()
            if self._config.allow_direct_messages:
                reason = "Sorry, I'm not available to you."
            else:
                reason = "Direct messages are disabled for this Gwenn bot."
            await message.reply(
                reason,
                mention_author=False,
                allowed_mentions=no_mentions,
            )
            return

        if not is_dm and not self._is_allowed_user(raw_id):
            return

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

        # Extract media attachments when media is enabled.
        image_blocks: list[dict] = []
        media_descriptions: list[str] = []
        if getattr(self._config, "enable_media", False) and message.attachments:
            image_blocks = await self._extract_image_attachments(message)

            # Video attachments — extract frames + transcribe.
            video_attachments = await self._extract_video_attachments(message)
            if video_attachments:
                from gwenn.media.video import VideoProcessor

                transcriber = self._get_audio_transcriber()
                for video_bytes, filename in video_attachments:
                    frames = await VideoProcessor.extract_frames(video_bytes)
                    image_blocks.extend(frames)
                    transcript: str | None = None
                    if transcriber:
                        transcript = await transcriber.transcribe(video_bytes, filename)
                    desc_parts = [f"[Video: {filename}."]
                    if transcript:
                        desc_parts.append(f" Audio transcript: '{transcript}'.")
                    if frames:
                        desc_parts.append(f" {len(frames)} frames extracted.]")
                    else:
                        desc_parts.append("]")
                    media_descriptions.append("".join(desc_parts))

            # Audio attachments — transcribe.
            audio_attachments = await self._extract_audio_attachments(message)
            if audio_attachments:
                transcriber = self._get_audio_transcriber()
                for audio_bytes, filename in audio_attachments:
                    transcript = None
                    if transcriber:
                        transcript = await transcriber.transcribe(audio_bytes, filename)
                    if transcript:
                        media_descriptions.append(
                            f"[Audio message transcript ({filename}): '{transcript}']"
                        )
                    else:
                        media_descriptions.append(f"[Audio message: {filename}]")

        if not text and not image_blocks and not media_descriptions:
            # Empty content after mention-stripping may indicate the privileged
            # MESSAGE_CONTENT intent is not enabled in the Discord Developer Portal.
            if not is_dm:
                logger.warning(
                    "discord_channel.empty_content_after_strip",
                    user_id=raw_id,
                    hint="Ensure the MESSAGE_CONTENT privileged intent is enabled "
                    "in the Discord Developer Portal.",
                )
            return

        if not self._check_rate_limit(raw_id):
            return

        if self._agent.identity.should_run_startup_onboarding():
            no_mentions = discord.AllowedMentions.none()
            if not self._is_owner_user(raw_id):
                await message.reply(
                    "Gwenn is in primary-user setup mode. Only the configured owner can run `/setup`.",
                    mention_author=False,
                    allowed_mentions=no_mentions,
                )
                return
            await message.reply(
                "Before we begin, run `/setup` so I can tailor how I help you.\n"
                "You can also run `/setup skip`.",
                mention_author=False,
                allowed_mentions=no_mentions,
            )
            return

        raw_chat_id = self._normalize_optional_id(getattr(message.channel, "id", None))
        raw_thread_id = (
            raw_chat_id
            if hasattr(discord, "Thread") and isinstance(message.channel, discord.Thread)
            else None
        )
        session_scope_key = self._session_scope_key(
            raw_user_id=raw_id,
            raw_chat_id=raw_chat_id,
            raw_thread_id=raw_thread_id,
        )

        lock = self._get_user_lock(raw_id)
        try:
            async with lock:
                # Acknowledge receipt with a reaction so the user knows we saw it.
                try:
                    await message.add_reaction("\U0001f916")  # 🤖
                except Exception as exc:
                    logger.debug("discord_channel.reaction_failed", error=str(exc))

                async with message.channel.typing():
                    no_mentions = discord.AllowedMentions.none()
                    try:
                        # Compose final text from user text + media descriptions.
                        msg_text = text
                        if media_descriptions:
                            desc_block = "\n".join(media_descriptions)
                            msg_text = f"{text}\n{desc_block}" if text else desc_block
                        if not msg_text:
                            if image_blocks:
                                msg_text = "[The user sent an image]"
                            else:  # pragma: no cover – msg_text always set by text/attachment handlers above
                                msg_text = "[The user sent a message]"
                        user_msg = UserMessage(
                            text=msg_text,
                            images=image_blocks,
                        )
                        response = await self.handle_message(
                            raw_id,
                            user_msg,
                            session_scope_key=session_scope_key,
                        )
                    except Exception as exc:
                        logger.error("discord_channel.respond_error", error=str(exc), exc_info=True)
                        await message.reply(
                            "I encountered an error processing your message. Please try again.",
                            mention_author=False,
                            allowed_mentions=no_mentions,
                        )
                        return

                    # Guard against empty responses
                    response_text = str(response)
                    if not response_text or not response_text.strip():
                        await message.reply(
                            "I processed your message but didn't generate a response. "
                            "Could you try rephrasing?",
                            mention_author=False,
                            allowed_mentions=no_mentions,
                        )
                        return

                    # Send chunks inside the typing() context so the indicator
                    # stays active until the last chunk is delivered.
                    # Only the first chunk is a reply (anchors the context);
                    # subsequent chunks are plain channel sends to avoid a
                    # wall of reply arrows pointing at the original message.
                    chunks = format_for_discord(response_text)
                    try:
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                await message.reply(
                                    chunk,
                                    mention_author=False,
                                    allowed_mentions=no_mentions,
                                )
                            else:
                                await message.channel.send(
                                    chunk,
                                    allowed_mentions=no_mentions,
                                )
                            if i < len(chunks) - 1:
                                # Brief pause between chunks to respect Discord rate limits.
                                await asyncio.sleep(0.5)
                    except Exception as exc:
                        logger.error("discord_channel.send_error", error=str(exc), exc_info=True)

                # Clear the "received" reaction now that we've replied.
                try:
                    await message.remove_reaction("\U0001f916", self._client.user)
                except Exception as exc:
                    logger.debug("discord_channel.clear_reaction_failed", error=str(exc))
        finally:
            self._release_user_lock(raw_id)

    def _register_slash_commands(self, tree) -> None:
        """Register slash commands on the app_commands tree."""

        channel = self  # capture for closures

        async def _ensure_allowed_interaction(interaction) -> bool:
            raw_user_id = str(interaction.user.id)
            guild = getattr(interaction, "guild", None)
            if guild is None:
                if channel._is_allowed_dm_user(raw_user_id):
                    return True
                if channel._config.allow_direct_messages:
                    reason = "Sorry, I'm not available to you."
                else:
                    reason = "Direct messages are disabled for this Gwenn bot."
                await interaction.response.send_message(reason, ephemeral=True)
                return False
            if channel._is_allowed_guild(getattr(guild, "id", None)):
                if channel._is_allowed_user(raw_user_id):
                    return True
                await interaction.response.send_message(
                    "Sorry, I'm not available to you.",
                    ephemeral=True,
                )
                return False
            await interaction.response.send_message(
                "This server is not allowed for Gwenn commands.",
                ephemeral=True,
            )
            return False

        @tree.command(name="status", description="See Gwenn's current cognitive state")
        async def slash_status(interaction) -> None:
            if not await _ensure_allowed_interaction(interaction):
                return
            status = channel._agent.status
            text = render_status_text(status, markdown_heading=True)
            await interaction.response.send_message(text, ephemeral=True)

        @tree.command(name="heartbeat", description="See Gwenn's heartbeat status")
        async def slash_heartbeat(interaction) -> None:
            if not await _ensure_allowed_interaction(interaction):
                return
            if not channel._agent.heartbeat:
                await interaction.response.send_message("Heartbeat is not running.", ephemeral=True)
                return
            hb = channel._agent.heartbeat.status
            text = render_heartbeat_text(hb, markdown_heading=True)
            await interaction.response.send_message(text, ephemeral=True)

        @tree.command(name="setup", description="Run first-time profile setup")
        async def slash_setup(
            interaction,
            name: str = "",
            role: str = "",
            interests: str = "",
            communication_style: str = "",
            boundaries: str = "",
            skip: bool = False,
        ) -> None:
            if not await _ensure_allowed_interaction(interaction):
                return
            raw_id = str(interaction.user.id)
            user_id = channel.make_user_id(raw_id)

            if not channel._is_owner_user(raw_id):
                await interaction.response.send_message(
                    "Only the configured owner can run `/setup`.",
                    ephemeral=True,
                )
                return

            if skip:
                channel._agent.identity.mark_onboarding_completed({})
                await interaction.response.send_message(
                    "No worries — we'll get to know each other as we go.", ephemeral=True
                )
                return

            profile = {
                "name": name.strip(),
                "role": role.strip(),
                "interests": interests.strip(),
                "communication_style": communication_style.strip(),
                "boundaries": boundaries.strip(),
            }
            if not any(profile.values()):
                await interaction.response.send_message(
                    "Provide at least one field (name/role/interests/style/boundaries), "
                    "or use `skip=true`.",
                    ephemeral=True,
                )
                return

            channel._agent.apply_startup_onboarding(profile, user_id=user_id)
            await interaction.response.send_message(
                "Thank you for sharing that with me. I'll carry this forward.",
                ephemeral=True,
            )

        @tree.command(name="reset", description="Clear your conversation history with Gwenn")
        async def slash_reset(interaction) -> None:
            if not await _ensure_allowed_interaction(interaction):
                return
            raw_id = str(interaction.user.id)
            session_id = channel._session_id_for_interaction(interaction, raw_id)
            channel._sessions.clear_session(session_id)
            await interaction.response.send_message(
                "Conversation history cleared. Fresh start!", ephemeral=True
            )

        @tree.command(name="help", description="List available Gwenn commands")
        async def slash_help(interaction) -> None:
            if not await _ensure_allowed_interaction(interaction):
                return
            if channel._config.allow_direct_messages:
                chat_hint = "DM me or @mention me in a server to chat."
            else:
                chat_hint = "@mention me in an allowed server to chat."
            await interaction.response.send_message(
                "**Gwenn Commands**\n"
                "/status — see my current cognitive state\n"
                "/heartbeat — see my heartbeat status\n"
                "/setup — first-run profile setup\n"
                "/reset — clear our conversation history\n"
                "/help — this message\n\n"
                f"{chat_hint}",
                ephemeral=True,
            )


def _create_discord_client(channel: DiscordChannel, intents, **kwargs):
    """
    Factory that creates a discord.Client wired to *channel*.

    The discord import is deferred to this function so the module can be
    imported even when discord.py is not installed.
    """
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
            self._gwenn_channel._ready_event.set()
            logger.info(
                "discord_channel.ready",
                user=str(self.user),
                guild_count=len(self.guilds),
            )

        async def on_message(self, message) -> None:
            await self._gwenn_channel._on_message(message)

    return _Client(gwenn_channel=channel, intents=intents, **kwargs)
