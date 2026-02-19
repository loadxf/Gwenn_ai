"""
Telegram channel adapter for Gwenn.

Uses python-telegram-bot v21+ async API.

IMPORTANT: We use initialize()+start()+updater.start_polling() instead of
app.run_polling() because the latter calls asyncio.run() internally and would
conflict with the existing event loop started by gwenn/main.py.

Slash commands:
  /start   — welcome message, clear session
  /help    — command list
  /setup   — first-run onboarding profile
  /status  — Gwenn's cognitive state
  /heartbeat — heartbeat status
  /reset   — clear conversation history

Regular messages are routed to Gwenn via handle_message().
"""

from __future__ import annotations

import asyncio
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
)

logger = structlog.get_logger(__name__)


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

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    @property
    def channel_name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Connect to Telegram and begin polling for updates."""
        try:
            from telegram.ext import ApplicationBuilder
        except ImportError as exc:
            raise ImportError(
                "python-telegram-bot is required for the Telegram channel. "
                "Install it with: pip install 'gwenn-agent[telegram]'"
            ) from exc

        self._app = ApplicationBuilder().token(self._config.bot_token).build()
        self._register_handlers()
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram_channel.started")

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
                    # HTML formatting rejected — retry as plain text.
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                    )
                await asyncio.sleep(0.3)
        except Exception:
            logger.exception("telegram_channel.send_error", user_id=platform_user_id)

    async def send_proactive(self, text: str) -> None:
        """Send a proactive message to explicitly configured Telegram owner users."""
        owner_ids = self._id_set(self._config.owner_user_ids)
        for uid in owner_ids:
            if uid:
                await self.send_message(uid, text)
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        from telegram.ext import CommandHandler, MessageHandler, filters

        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_help))
        self._app.add_handler(CommandHandler("setup", self._on_setup))
        self._app.add_handler(CommandHandler("status", self._on_status))
        self._app.add_handler(CommandHandler("heartbeat", self._on_heartbeat))
        self._app.add_handler(CommandHandler("reset", self._on_reset))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        # Handle edited messages — acknowledge but don't re-process.
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.EDITED_MESSAGE & filters.TEXT,
                self._on_edited_message,
            )
        )
        # Acknowledge unsupported media types so users aren't left in silence.
        self._app.add_handler(
            MessageHandler(~filters.TEXT & ~filters.COMMAND, self._on_unsupported_media)
        )
        # Route unhandled exceptions through structlog.
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
            f"Commands: /help /setup /status /heartbeat /reset"
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
            "/reset — clear our conversation history\n\n"
            "Just send a message to talk with me."
        )

    async def _on_setup(self, update, context) -> None:
        raw_id = str(update.effective_user.id)
        if not self._is_allowed(raw_id):
            return
        if not self._is_owner_user(raw_id):
            await update.message.reply_text("Only the configured owner can run /setup.")
            return

        raw_payload = " ".join(getattr(context, "args", [])).strip()
        if not raw_payload:
            await update.message.reply_text(
                "Setup format:\n"
                "/setup Name | Role | Main needs/goals | Communication style | Boundaries\n\n"
                "Example:\n"
                "/setup Bob | coding partner | ship reliable features | concise | no destructive changes\n\n"
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

        # Per-user lock prevents concurrent requests from the same user.
        lock = self._get_user_lock(raw_id)
        try:
            async with lock:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id, action="typing"
                )
                session_scope_key = self._session_scope_key_for_update(update, raw_id)
                try:
                    response = await self.handle_message(
                        raw_id,
                        text,
                        session_scope_key=session_scope_key,
                    )
                except Exception as exc:
                    logger.error("telegram_channel.respond_error", error=str(exc), exc_info=True)
                    await update.message.reply_text(
                        "I encountered an error processing your message. Please try again."
                    )
                    return

                chunks = format_for_telegram(response)
                for i, chunk in enumerate(chunks):
                    try:
                        await update.message.reply_text(chunk, parse_mode=TELEGRAM_PARSE_MODE)
                    except Exception:
                        # HTML formatting rejected — retry as plain text.
                        try:
                            await update.message.reply_text(chunk)
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
                        # Brief pause between chunks to avoid Telegram flood limits
                        await asyncio.sleep(0.5)
        finally:
            self._release_user_lock(raw_id)

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
        await update.message.reply_text(
            "I can only read text messages for now. Send me text and I'll respond!"
        )

    @staticmethod
    async def _on_error(update, context) -> None:
        """Route unhandled PTB exceptions through structlog."""
        logger.error(
            "telegram_channel.unhandled_error",
            error=str(context.error),
            update_id=getattr(update, "update_id", None) if update else None,
            exc_info=context.error,
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
