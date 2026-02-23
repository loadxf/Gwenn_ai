"""
Telegram channel adapter for Gwenn.

Uses python-telegram-bot v22+ async API with AIORateLimiter.

IMPORTANT: We use initialize()+start()+updater.start_polling() instead of
app.run_polling() because the latter calls asyncio.run() internally and would
conflict with the existing event loop started by gwenn/main.py.

Slash commands:
  /start     — welcome message, clear session
  /help      — command list
  /setup     — first-run onboarding profile
  /status    — Gwenn's cognitive state
  /heartbeat — heartbeat status
  /reset     — clear conversation history
  /cancel    — cancel an in-progress response

Regular messages are routed to Gwenn via handle_message().
Photo/document/voice messages are handled when TELEGRAM_ENABLE_MEDIA=true.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import structlog

from gwenn.channels.base import BaseChannel

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

        # Drain any proactive messages queued before startup (#19).
        if self._proactive_queue:
            for queued in self._proactive_queue:
                await self.send_proactive(queued)
            self._proactive_queue.clear()

    async def stop(self) -> None:
        """Stop polling and shut down the PTB application."""
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
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        # Handle edited messages — acknowledge but don't re-process.
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.EDITED_MESSAGE & filters.TEXT,
                self._on_edited_message,
            )
        )
        # Media handlers — photo, document, voice (#12).
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
        # Acknowledge unsupported media types so users aren't left in silence.
        self._app.add_handler(
            MessageHandler(~filters.TEXT & ~filters.COMMAND, self._on_unsupported_media)
        )
        # Route unhandled exceptions through structlog with differentiated handling (#2).
        self._app.add_error_handler(self._on_error)

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
            f"Commands: /help /setup /status /heartbeat /reset /cancel"
        )

    async def _on_help(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            await update.message.reply_text("Sorry, I'm not available to you.")
            return
        await update.message.reply_text(
            "Gwenn commands:\n"
            "/start — start a new conversation\n"
            "/setup — first-run profile setup\n"
            "/status — see my current cognitive state\n"
            "/heartbeat — see my heartbeat status\n"
            "/reset — clear our conversation history\n"
            "/cancel — cancel an in-progress response\n\n"
            "Just send a message to talk with me."
        )

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

    async def _process_user_input(
        self,
        update,
        context,
        raw_id: str,
        text: str,
        *,
        media_description: str | None = None,
    ) -> None:
        """Core message processing shared by text and media handlers."""
        # Per-user lock prevents concurrent requests from the same user.
        lock = self._get_user_lock(raw_id)
        self._cancel_flags.pop(raw_id, None)
        try:
            async with lock:
                # Start persistent typing indicator (#4, #5).
                thread_id = getattr(
                    getattr(update, "message", None), "message_thread_id", None
                )
                typing_task = asyncio.create_task(
                    self._keep_typing(update.effective_chat.id, thread_id)
                )
                try:
                    if media_description:
                        user_text = f"{media_description}\n\n{text}" if text else media_description
                    else:
                        user_text = text
                    session_scope_key = self._session_scope_key_for_update(update, raw_id)
                    response = await self.handle_message(
                        raw_id,
                        user_text,
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
        finally:
            self._release_user_lock(raw_id)

    # ------------------------------------------------------------------
    # Media handlers (#12)
    # ------------------------------------------------------------------

    async def _on_photo(self, update, context) -> None:
        """Handle photo messages — describe the image and route to agent."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        caption = update.message.caption or ""
        # Inform the agent that a photo was received.
        description = "[The user sent a photo]"
        if caption:
            description = f"[The user sent a photo with caption: {caption}]"

        logger.debug(
            "telegram_channel.photo_received",
            user_id=raw_id,
            has_caption=bool(caption),
        )
        await self._process_user_input(
            update, context, raw_id, caption, media_description=description
        )

    async def _on_document(self, update, context) -> None:
        """Handle document messages — note the file and route to agent."""
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
        description = f"[The user sent a document: {filename}]"
        if caption:
            description = f"[The user sent a document ({filename}) with caption: {caption}]"

        logger.debug(
            "telegram_channel.document_received",
            user_id=raw_id,
            filename=filename,
        )
        await self._process_user_input(
            update, context, raw_id, caption, media_description=description
        )

    async def _on_voice(self, update, context) -> None:
        """Handle voice/audio messages — note receipt and route to agent."""
        if not update.message or not update.effective_user:
            return
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._check_rate_limit(raw_id):
            return

        description = "[The user sent a voice message]"
        logger.debug("telegram_channel.voice_received", user_id=raw_id)
        await self._process_user_input(
            update, context, raw_id, "", media_description=description
        )

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
                "I support photos, documents, and voice messages."
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
