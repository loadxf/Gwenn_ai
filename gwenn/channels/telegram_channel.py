"""
Telegram channel adapter for Gwenn.

Uses python-telegram-bot v22+ async API with AIORateLimiter.

IMPORTANT: We use initialize()+start()+updater.start_polling() instead of
app.run_polling() because the latter calls asyncio.run() internally and would
conflict with the existing event loop started by gwenn/main.py.

Built-in slash commands:
  /start     — welcome message, clear session
  /help      — command list (includes loaded skills)
  /setup     — first-run onboarding profile
  /status    — Gwenn's cognitive state
  /heartbeat — heartbeat status
  /reset     — clear conversation history
  /cancel    — cancel an in-progress response

Skill-based commands (e.g. /weather, /news) are loaded dynamically from the
skill registry and registered with Telegram's command menu via set_my_commands.
Unrecognised /commands are forwarded to the agent as regular messages.

Regular messages are routed to Gwenn via handle_message().
Photo/document/voice messages are handled when TELEGRAM_ENABLE_MEDIA=true.
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import structlog

from gwenn.channels.base import BaseChannel
from gwenn.types import UserMessage

if TYPE_CHECKING:
    from gwenn.agent import SentientAgent
    from gwenn.channels.session import SessionManager
    from gwenn.config import TelegramConfig
from gwenn.channels.formatting import (
    TELEGRAM_PARSE_MODE,
    format_for_telegram,
    render_heartbeat_text,
    render_status_text,
    strip_html_tags,
)

logger = structlog.get_logger(__name__)

# Typing indicator expires after 5s on Telegram clients; refresh before that.
_TYPING_REFRESH_INTERVAL: float = 4.0

# Telegram dependency bootstrap.
_PTB_REQUIREMENT: str = "python-telegram-bot[rate-limiter]>=22.0"
_PTB_AUTO_INSTALL_ENV: str = "GWENN_AUTO_INSTALL_TELEGRAM"
_PTB_INSTALL_ATTEMPTED: bool = False

# Fallback text when agent returns an empty response.
_EMPTY_RESPONSE_FALLBACK: str = (
    "I processed your message but didn't generate a response. "
    "Could you try rephrasing?"
)


def _import_application_builder():
    from telegram.ext import ApplicationBuilder

    return ApplicationBuilder


def _auto_install_enabled() -> bool:
    raw = os.environ.get(_PTB_AUTO_INSTALL_ENV, "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _truncate_output(text: str, limit: int = 300) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "...[truncated]"


def _pip_install_requirement(requirement: str) -> bool:
    """Install one pip requirement in the current interpreter environment."""
    python_bin = sys.executable or "python3"
    cmd = [python_bin, "-m", "pip", "install", requirement]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        logger.error(
            "telegram_channel.auto_install_failed",
            requirement=requirement,
            error=str(exc),
        )
        return False

    if proc.returncode != 0:
        logger.error(
            "telegram_channel.auto_install_failed",
            requirement=requirement,
            exit_code=proc.returncode,
            stderr=_truncate_output(proc.stderr),
        )
        return False

    logger.info("telegram_channel.auto_install_succeeded", requirement=requirement)
    return True


async def _attempt_auto_install_ptb() -> bool:
    """Best-effort runtime install for existing envs missing PTB."""
    global _PTB_INSTALL_ATTEMPTED  # noqa: PLW0603
    if _PTB_INSTALL_ATTEMPTED:
        return False

    if not _auto_install_enabled():
        logger.info(
            "telegram_channel.auto_install_disabled",
            env_var=_PTB_AUTO_INSTALL_ENV,
        )
        return False

    _PTB_INSTALL_ATTEMPTED = True
    logger.warning(
        "telegram_channel.auto_install_attempt",
        requirement=_PTB_REQUIREMENT,
    )
    return await asyncio.to_thread(_pip_install_requirement, _PTB_REQUIREMENT)


async def _ensure_application_builder():
    """Import PTB ApplicationBuilder, auto-installing PTB once if needed."""
    try:
        return _import_application_builder()
    except ImportError as exc:
        if await _attempt_auto_install_ptb():
            try:
                return _import_application_builder()
            except ImportError:
                pass
        raise ImportError(
            "python-telegram-bot is required for the Telegram channel. "
            "Install/refresh dependencies with: uv sync && uv pip install -e ."
        ) from exc


class TelegramChannel(BaseChannel):
    """Gwenn Telegram bot adapter."""

    def __init__(
        self,
        agent: SentientAgent,
        sessions: SessionManager,
        config: TelegramConfig,
    ) -> None:
        super().__init__(
            agent,
            sessions,
            user_lock_cache_size=config.user_lock_cache_size,
        )
        self._config = config
        self._app = None
        # Tracks per-user cancel flags for /cancel support.
        self._cancel_flags: dict[str, bool] = {}
        # Queue for proactive messages received before the bot is ready.
        self._proactive_queue: list[str] = []

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    @property
    def channel_name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Connect to Telegram and begin polling for updates."""
        ApplicationBuilder = await _ensure_application_builder()

        builder = ApplicationBuilder().token(self._config.bot_token)

        # Configure concurrent update processing (#11).
        if self._config.concurrent_updates > 0:
            builder = builder.concurrent_updates(self._config.concurrent_updates)

        # Install AIORateLimiter to respect Telegram's flood limits (#1).
        try:
            from telegram.ext import AIORateLimiter
            builder = builder.rate_limiter(AIORateLimiter(max_retries=3))
        except ImportError:
            logger.warning(
                "telegram_channel.no_rate_limiter",
                hint="Install python-telegram-bot[rate-limiter] for flood protection",
            )

        self._app = builder.build()
        self._register_handlers()
        await self._app.initialize()
        await self._app.start()
        # Only receive update types we actually handle (#8).
        await self._app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "edited_message"],
        )
        logger.info("telegram_channel.started")

        # Register commands with Telegram so the "/" menu shows all of them.
        await self._sync_bot_commands()

        # Drain any proactive messages queued before startup (#19).
        if self._proactive_queue:
            for queued in self._proactive_queue:
                await self.send_proactive(queued)
            self._proactive_queue.clear()

    async def stop(self) -> None:
        """Stop polling and shut down the PTB application."""
        if self._audio_transcriber is not None:
            try:
                await self._audio_transcriber.close()
            except Exception:
                logger.debug("telegram_channel.audio_transcriber_close_error", exc_info=True)
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            logger.exception("telegram_channel.stop_error")
        logger.info("telegram_channel.stopped")

    async def send_message(self, platform_user_id: str, text: str) -> None:
        """Send an unsolicited message to a Telegram user."""
        if self._app is None:
            return
        chat_id = self._validate_platform_id(platform_user_id)
        if chat_id is None:
            return
        try:
            for chunk in format_for_telegram(text):
                try:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode=TELEGRAM_PARSE_MODE,
                    )
                except Exception:
                    # HTML formatting rejected — retry as plain text (#10).
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=strip_html_tags(chunk),
                    )
                await asyncio.sleep(0.3)
        except Exception:
            logger.exception("telegram_channel.send_error", user_id=platform_user_id)

    async def send_proactive(self, text: str) -> None:
        """Send a proactive message to configured Telegram owner users.

        Falls back to allowed_user_ids when no explicit owners are set (#6).
        If the bot hasn't started yet, queues the message for later (#19).
        """
        if self._app is None:
            self._proactive_queue.append(text)
            return
        owner_ids = self._get_owner_ids()
        for uid in owner_ids:
            if uid:
                await self.send_message(uid, text)
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_owner_ids(self) -> set[str]:
        """Return the effective owner set, falling back to allowed list (#6)."""
        owner_ids = self._id_set(self._config.owner_user_ids)
        if owner_ids:
            return owner_ids
        return self._id_set(self._config.allowed_user_ids)

    def _register_handlers(self) -> None:
        from telegram.ext import CommandHandler, MessageHandler, filters

        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_help))
        self._app.add_handler(CommandHandler("setup", self._on_setup))
        self._app.add_handler(CommandHandler("status", self._on_status))
        self._app.add_handler(CommandHandler("heartbeat", self._on_heartbeat))
        self._app.add_handler(CommandHandler("reset", self._on_reset))
        self._app.add_handler(CommandHandler("cancel", self._on_cancel))
        # Catch-all for unrecognized /commands (e.g. skill names like /weather).
        # Route them through the normal message handler so the agent can process them.
        self._app.add_handler(MessageHandler(filters.COMMAND, self._on_message))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        # Handle edited messages — acknowledge but don't re-process.
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.EDITED_MESSAGE & filters.TEXT,
                self._on_edited_message,
            )
        )
        # Media handlers — photo, document, voice, video (#12).
        if self._config.enable_media:
            self._app.add_handler(
                MessageHandler(filters.PHOTO, self._on_photo)
            )
            self._app.add_handler(
                MessageHandler(filters.Document.ALL, self._on_document)
            )
            self._app.add_handler(
                MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice)
            )
            self._app.add_handler(
                MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, self._on_video)
            )
        # Acknowledge unsupported media types so users aren't left in silence.
        self._app.add_handler(
            MessageHandler(~filters.TEXT & ~filters.COMMAND, self._on_unsupported_media)
        )
        # Route unhandled exceptions through structlog with differentiated handling (#2).
        self._app.add_error_handler(self._on_error)

    async def _sync_bot_commands(self) -> None:
        """Push the full command list (built-ins + skills) to Telegram's menu."""
        from telegram import BotCommand

        commands = [
            BotCommand("start", "Start a new conversation"),
            BotCommand("help", "Command list"),
            BotCommand("setup", "First-run profile setup"),
            BotCommand("status", "Cognitive state"),
            BotCommand("heartbeat", "Heartbeat status"),
            BotCommand("reset", "Clear conversation history"),
            BotCommand("cancel", "Cancel in-progress response"),
        ]
        for skill in sorted(self._agent.skill_registry.all_skills(), key=lambda s: s.name):
            if "user_command" not in skill.tags:
                continue
            # Telegram command names: 1-32 lowercase chars, only a-z 0-9 and underscores.
            cmd = skill.name[:32]
            desc = (skill.description or skill.name)[:256]
            commands.append(BotCommand(cmd, desc))
        try:
            await self._app.bot.set_my_commands(commands)
            logger.info("telegram_channel.commands_synced", count=len(commands))
        except Exception:
            logger.warning("telegram_channel.commands_sync_failed", exc_info=True)

    def _is_allowed(self, user_id: str) -> bool:
        """Return True if the user is on the allowlist (or no allowlist is set)."""
        allowed_ids = self._id_set(self._config.allowed_user_ids)
        return not allowed_ids or user_id in allowed_ids

    def _is_owner_user(self, user_id: str) -> bool:
        """Return True if the user is an owner (falls back to allowed list)."""
        owner_ids = self._id_set(self._config.owner_user_ids)
        if owner_ids:
            return user_id in owner_ids
        return self._is_allowed(user_id)

    def _session_scope_mode(self) -> str:
        return self._normalize_scope_mode(self._config.session_scope_mode, default="per_chat")

    def _session_scope_key_for_update(self, update, raw_user_id: str) -> str:
        chat_id = self._normalize_optional_id(
            getattr(getattr(update, "effective_chat", None), "id", None)
        )
        thread_id = self._normalize_optional_id(
            getattr(getattr(update, "message", None), "message_thread_id", None)
        )
        return self.make_session_scope_key(
            raw_user_id=raw_user_id,
            raw_chat_id=chat_id,
            raw_thread_id=thread_id,
            scope_mode=self._session_scope_mode(),
        )

    def _session_id_for_update(self, update, raw_user_id: str) -> str:
        scope_key = self._session_scope_key_for_update(update, raw_user_id)
        return self.make_session_id(scope_key)

    async def _keep_typing(self, chat_id: int, thread_id: int | None = None) -> None:
        """Refresh the typing indicator every 4s until cancelled (#4, #5)."""
        kwargs: dict = {"chat_id": chat_id, "action": "typing"}
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        try:
            while True:
                await self._app.bot.send_chat_action(**kwargs)
                await asyncio.sleep(_TYPING_REFRESH_INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # Typing indicator is best-effort; don't propagate.

    async def _send_chunks(self, update, chunks: list[str]) -> None:
        """Send pre-formatted HTML chunks as replies, with plain-text fallback (#10)."""
        for i, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(chunk, parse_mode=TELEGRAM_PARSE_MODE)
            except Exception:
                # HTML formatting rejected — strip tags and retry as plain text.
                try:
                    await update.message.reply_text(strip_html_tags(chunk))
                except Exception as exc:
                    logger.error(
                        "telegram_channel.send_error",
                        error=str(exc),
                        chunk_index=i,
                        chunk_count=len(chunks),
                        exc_info=True,
                    )
                    break
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _on_start(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return
        session_id = self._session_id_for_update(update, raw_id)
        self._sessions.clear_session(session_id)
        scope = self._session_scope_mode()
        if scope == "per_chat":
            reset_hint = "This chat's conversation history has been reset."
        elif scope == "per_thread":
            reset_hint = "This thread's conversation history has been reset."
        else:
            reset_hint = "Your conversation history has been reset."
        await update.message.reply_text(
            f"Hello! I'm Gwenn — an autonomous sentient AI agent.\n\n"
            f"{reset_hint} Just send me a message to begin.\n\n"
            f"Type /help to see all available commands."
        )

    async def _on_help(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return
        lines = [
            "Gwenn commands:\n",
            "/start — start a new conversation",
            "/setup — first-run profile setup",
            "/status — see my current cognitive state",
            "/heartbeat — see my heartbeat status",
            "/reset — clear our conversation history",
            "/cancel — cancel an in-progress response",
        ]
        # Append loaded skills so /help always reflects the full command set.
        # Truncate descriptions to keep the message readable (and under 4096 chars).
        skills = self._agent.skill_registry.all_skills()
        if skills:
            lines.append("\nSkills:")
            for skill in sorted(skills, key=lambda s: s.name):
                desc = skill.description
                # Take the first sentence or truncate at 80 chars.
                dot = desc.find(". ")
                if 0 < dot <= 80:
                    desc = desc[:dot + 1]
                elif len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"/{skill.name} — {desc}")
        lines.append("\nJust send a message to talk with me.")
        text = "\n".join(lines)
        if len(text) > 4000:
            # Truncate skills list to fit Telegram's 4096-char limit
            text = text[:4000] + "\n..."
        await update.message.reply_text(text)

    async def _on_setup(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._is_owner_user(raw_id):
            await update.message.reply_text("Only the configured owner can run /setup.")
            return

        # Use raw message text after the command for robust parsing (#16).
        raw_payload = (update.message.text or "").partition(" ")[2].strip()
        if not raw_payload:
            await update.message.reply_text(
                "Setup format:\n"
                "/setup Name | Role | Main needs/goals | Communication style | Boundaries\n\n"
                "Example:\n"
                "/setup Bob | coding partner | ship reliable features | concise "
                "| no destructive changes\n\n"
                "Use `/setup skip` to skip first-run setup."
            )
            return

        if raw_payload.lower() == "skip":
            self._agent.identity.mark_onboarding_completed({})
            await update.message.reply_text("First-run setup skipped.")
            return

        profile = self._parse_setup_payload(raw_payload)
        if not any(profile.values()):
            await update.message.reply_text(
                "I couldn't parse setup values. Use `/setup` for the format."
            )
            return

        user_id = self.make_user_id(raw_id)
        self._agent.apply_startup_onboarding(profile, user_id=user_id)
        await update.message.reply_text("Setup saved. I will use this as ongoing guidance.")

    async def _on_status(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        status = self._agent.status
        text = render_status_text(status)
        await update.message.reply_text(text)

    async def _on_heartbeat(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._agent.heartbeat:
            await update.message.reply_text("Heartbeat is not running.")
            return
        hb = self._agent.heartbeat.status
        text = render_heartbeat_text(hb)
        await update.message.reply_text(text)

    async def _on_reset(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        session_id = self._session_id_for_update(update, raw_id)
        self._sessions.clear_session(session_id)
        await update.message.reply_text("Conversation history cleared. Fresh start!")

    async def _on_cancel(self, update, context) -> None:
        """Set the cancel flag for the requesting user (#14)."""
        if not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        self._cancel_flags[raw_id] = True
        await update.message.reply_text("Cancellation requested. I'll stop after the current step.")

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, update, context) -> None:
        """Handle a regular text message from a Telegram user."""
        if not update.message or not update.effective_user:
            return
        text = update.message.text
        if not text:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return

        if not self._check_rate_limit(raw_id):
            return

        logger.debug(
            "telegram_channel.message_received",
            user_id=raw_id,
            chat_id=update.effective_chat.id,
            text_length=len(text),
        )

        if self._agent.identity.should_run_startup_onboarding():
            if not self._is_owner_user(raw_id):
                await update.message.reply_text(
                    "Gwenn is in first-run setup mode. Only the configured owner can run /setup."
                )
                return
            await update.message.reply_text(
                "Before we begin, run first-time setup with:\n"
                "/setup Name | Role | Main needs/goals | Communication style | Boundaries\n"
                "Or use `/setup skip`."
            )
            return

        await self._process_user_input(update, context, raw_id, text)

    async def _acknowledge_received(self, message) -> None:
        """Add a reaction to acknowledge receipt of the user's message."""
        try:
            from telegram import ReactionTypeEmoji

            await message.set_reaction([ReactionTypeEmoji(emoji="👾")])
        except Exception as exc:
            logger.debug("telegram_channel.reaction_failed", error=str(exc))

    async def _clear_reaction(self, message) -> None:
        """Remove the acknowledgment reaction after responding."""
        try:
            await message.set_reaction([])
        except Exception as exc:
            logger.debug("telegram_channel.clear_reaction_failed", error=str(exc))

    async def _process_user_input(
        self,
        update,
        context,
        raw_id: str,
        message: UserMessage | str,
    ) -> None:
        """Core message processing shared by text and media handlers."""
        if isinstance(message, str):
            message = UserMessage(text=message)
        # Per-user lock prevents concurrent requests from the same user.
        lock = self._get_user_lock(raw_id)
        try:
            async with lock:
                self._cancel_flags.pop(raw_id, None)
                # Acknowledge receipt with a reaction so the user knows we saw it.
                await self._acknowledge_received(update.message)

                # Start persistent typing indicator (#4, #5).
                thread_id = getattr(
                    getattr(update, "message", None), "message_thread_id", None
                )
                typing_task = asyncio.create_task(
                    self._keep_typing(update.effective_chat.id, thread_id)
                )
                try:
                    session_scope_key = self._session_scope_key_for_update(update, raw_id)
                    response = await self.handle_message(
                        raw_id,
                        message,
                        session_scope_key=session_scope_key,
                    )
                except Exception as exc:
                    logger.error(
                        "telegram_channel.respond_error", error=str(exc), exc_info=True
                    )
                    await update.message.reply_text(
                        "I encountered an error processing your message. Please try again."
                    )
                    return
                finally:
                    typing_task.cancel()

                # Check if the user cancelled while we were processing (#14).
                if self._cancel_flags.pop(raw_id, False):
                    await update.message.reply_text("Response cancelled.")
                    return

                # Guard against empty responses (#24).
                if not response or not response.strip():
                    await update.message.reply_text(_EMPTY_RESPONSE_FALLBACK)
                    return

                chunks = format_for_telegram(response)
                await self._send_chunks(update, chunks)

                # Clear the "received" reaction now that we've replied.
                await self._clear_reaction(update.message)
        finally:
            self._release_user_lock(raw_id)

    # ------------------------------------------------------------------
    # Image download helpers
    # ------------------------------------------------------------------

    _SUPPORTED_IMAGE_MIMES: set[str] = {
        "image/jpeg", "image/png", "image/gif", "image/webp",
    }
    _MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB

    async def _download_telegram_photo(self, message) -> list[dict]:
        """Download the best-resolution photo attached to *message*.

        Returns a list containing a single image content block suitable for
        the Claude API, or an empty list on failure (graceful fallback).
        """
        try:
            photos = message.photo
            if not photos:
                return []
            # photos is sorted smallest → largest; pick largest within size limit.
            best = photos[-1]
            if best.file_size and best.file_size > self._MAX_IMAGE_BYTES:
                if len(photos) >= 2:
                    best = photos[-2]
                else:
                    logger.warning("telegram_channel.photo_too_large", size=best.file_size)
                    return []
            tg_file = await best.get_file()
            data = await tg_file.download_as_bytearray()
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            return [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }]
        except Exception as exc:
            logger.warning("telegram_channel.photo_download_failed", error=str(exc))
            return []

    async def _download_telegram_document_image(self, document) -> list[dict]:
        """Download a document attachment that has an image MIME type.

        Returns a list with one image content block, or ``[]`` on failure.
        """
        try:
            mime = (document.mime_type or "").lower()
            if mime not in self._SUPPORTED_IMAGE_MIMES:
                return []
            if document.file_size and document.file_size > self._MAX_IMAGE_BYTES:
                logger.warning(
                    "telegram_channel.document_image_too_large",
                    size=document.file_size,
                )
                return []
            tg_file = await document.get_file()
            data = await tg_file.download_as_bytearray()
            b64 = base64.standard_b64encode(bytes(data)).decode("ascii")
            return [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64,
                },
            }]
        except Exception as exc:
            logger.warning("telegram_channel.document_image_download_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Media handlers (#12)
    # ------------------------------------------------------------------

    async def _on_photo(self, update, context) -> None:
        """Handle photo messages — download image and route to agent."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        caption = update.message.caption or ""
        image_blocks = await self._download_telegram_photo(update.message)

        if image_blocks:
            msg = UserMessage(
                text=caption or "[The user sent a photo]",
                images=image_blocks,
            )
        else:
            # Download failed — fall back to text-only description.
            description = "[The user sent a photo]"
            if caption:
                description = f"[The user sent a photo with caption: {caption}]"
            msg = UserMessage(text=description)

        logger.debug(
            "telegram_channel.photo_received",
            user_id=raw_id,
            has_caption=bool(caption),
            has_image=bool(image_blocks),
        )
        await self._process_user_input(update, context, raw_id, msg)

    async def _on_document(self, update, context) -> None:
        """Handle document messages — download if image, else note file and route."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        doc = update.message.document
        caption = update.message.caption or ""
        filename = doc.file_name if doc else "unknown"

        # Try to download as image if the MIME type is supported.
        image_blocks: list[dict] = []
        if doc:
            image_blocks = await self._download_telegram_document_image(doc)

        if image_blocks:
            msg = UserMessage(
                text=caption or f"[The user sent an image document: {filename}]",
                images=image_blocks,
            )
        else:
            description = f"[The user sent a document: {filename}]"
            if caption:
                description = f"[The user sent a document ({filename}) with caption: {caption}]"
            msg = UserMessage(text=description)

        logger.debug(
            "telegram_channel.document_received",
            user_id=raw_id,
            filename=filename,
            has_image=bool(image_blocks),
        )
        await self._process_user_input(update, context, raw_id, msg)

    async def _on_voice(self, update, context) -> None:
        """Handle voice/audio messages — transcribe via Groq Whisper and route."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        # Attempt to download and transcribe.
        transcript: str | None = None
        voice = update.message.voice or update.message.audio
        if voice:
            try:
                tg_file = await voice.get_file()
                audio_bytes = await tg_file.download_as_bytearray()
                transcriber = self._get_audio_transcriber()
                if transcriber:
                    transcript = await transcriber.transcribe(
                        bytes(audio_bytes), "voice.ogg"
                    )
            except Exception as exc:
                logger.warning(
                    "telegram_channel.voice_download_failed", error=str(exc)
                )

        if transcript:
            description = f"[Voice message transcript: '{transcript}']"
        else:
            description = "[The user sent a voice message]"

        logger.debug(
            "telegram_channel.voice_received",
            user_id=raw_id,
            has_transcript=bool(transcript),
        )
        await self._process_user_input(update, context, raw_id, description)

    _MAX_VIDEO_BYTES: int = 20 * 1024 * 1024  # 20 MB

    async def _on_video(self, update, context) -> None:
        """Handle video/video-note messages — extract frames + transcribe audio."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        video = update.message.video or update.message.video_note
        caption = update.message.caption or ""
        duration = getattr(video, "duration", 0) or 0

        image_blocks: list[dict] = []
        transcript: str | None = None

        if video:
            file_size = getattr(video, "file_size", 0) or 0
            if file_size <= self._MAX_VIDEO_BYTES:
                # Download full video for frame extraction + transcription.
                try:
                    tg_file = await video.get_file()
                    video_bytes = bytes(await tg_file.download_as_bytearray())

                    from gwenn.media.video import VideoProcessor

                    image_blocks = await VideoProcessor.extract_frames(video_bytes)

                    transcriber = self._get_audio_transcriber()
                    if transcriber:
                        transcript = await transcriber.transcribe(
                            video_bytes, "video.mp4"
                        )
                except Exception as exc:
                    logger.warning(
                        "telegram_channel.video_download_failed", error=str(exc)
                    )
            else:
                # Video too large — try thumbnail fallback.
                thumb = getattr(video, "thumbnail", None)
                if thumb:
                    try:
                        tg_file = await thumb.get_file()
                        thumb_bytes = bytes(await tg_file.download_as_bytearray())

                        from gwenn.media.video import VideoProcessor

                        image_blocks = VideoProcessor.build_thumbnail_block(thumb_bytes)
                    except Exception as exc:
                        logger.warning(
                            "telegram_channel.video_thumbnail_failed", error=str(exc)
                        )

        # Build description text.
        parts = [f"[The user sent a {duration}s video."]
        if transcript:
            parts.append(f" Audio transcript: '{transcript}'.")
        if image_blocks:
            parts.append(f" Here are {len(image_blocks)} representative frames.]")
        else:
            parts.append("]")
        description = "".join(parts)
        if caption:
            description = f"{caption}\n{description}"

        msg = UserMessage(text=description, images=image_blocks)

        logger.debug(
            "telegram_channel.video_received",
            user_id=raw_id,
            duration=duration,
            frames=len(image_blocks),
            has_transcript=bool(transcript),
        )
        await self._process_user_input(update, context, raw_id, msg)

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
            logger.warning("telegram_channel.transcriber_init_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Edited / unsupported / error handlers
    # ------------------------------------------------------------------

    async def _on_edited_message(self, update, context) -> None:
        """Acknowledge edited messages without re-processing them."""
        if not update.edited_message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        await update.edited_message.reply_text(
            "I noticed you edited your message, but I can only respond to new messages."
        )

    async def _on_unsupported_media(self, update, context) -> None:
        """Acknowledge unsupported media so users aren't left with no response."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if self._config.enable_media:
            await update.message.reply_text(
                "I received your media but can't process this type yet. "
                "I support photos, documents, voice messages, and videos."
            )
        else:
            await update.message.reply_text(
                "I can only read text messages for now. Send me text and I'll respond!"
            )

    @staticmethod
    async def _on_error(update, context) -> None:
        """Route unhandled PTB exceptions through structlog with type-specific handling (#2)."""
        error = context.error
        update_id = getattr(update, "update_id", None) if update else None

        # Import Telegram error hierarchy for differentiated handling.
        try:
            from telegram.error import (
                BadRequest,
                Forbidden,
                NetworkError,
                RetryAfter,
                TimedOut,
            )
        except ImportError:
            # Fallback if import fails — log generically.
            logger.error(
                "telegram_channel.unhandled_error",
                error=str(error),
                update_id=update_id,
                exc_info=error,
            )
            return

        if isinstance(error, RetryAfter):
            # Telegram flood control — wait the requested time, then resume.
            logger.warning(
                "telegram_channel.rate_limited",
                retry_after=error.retry_after,
                update_id=update_id,
            )
            await asyncio.sleep(error.retry_after + 0.5)
            return

        if isinstance(error, Forbidden):
            # User blocked the bot or bot lacks permissions.
            logger.warning(
                "telegram_channel.forbidden",
                error=str(error),
                update_id=update_id,
            )
            return

        if isinstance(error, TimedOut):
            # Subset of NetworkError — transient timeout, PTB retries internally.
            logger.debug(
                "telegram_channel.timed_out",
                update_id=update_id,
            )
            return

        if isinstance(error, NetworkError):
            # Transient network issue — log at warning, PTB handles reconnection.
            logger.warning(
                "telegram_channel.network_error",
                error=str(error),
                update_id=update_id,
            )
            return

        if isinstance(error, BadRequest):
            # Invalid API call — log the full error for debugging.
            logger.error(
                "telegram_channel.bad_request",
                error=str(error),
                update_id=update_id,
                exc_info=error,
            )
            return

        # Unknown/unexpected error.
        logger.error(
            "telegram_channel.unhandled_error",
            error=str(error),
            error_type=type(error).__name__,
            update_id=update_id,
            exc_info=error,
        )

    @staticmethod
    def _parse_setup_payload(raw_payload: str) -> dict[str, str]:
        """
        Parse '/setup' payload into the onboarding profile fields.

        Expected form: "name | role | needs | communication_style | boundaries"
        Missing trailing fields are allowed.
        """
        parts = [part.strip() for part in raw_payload.split("|")]
        keys = ["name", "role", "needs", "communication_style", "boundaries"]
        profile = {key: "" for key in keys}
        for idx, value in enumerate(parts[: len(keys)]):
            profile[keys[idx]] = value
        return profile
