"""
Gwenn Daemon — Persistent background process.

Starts SentientAgent + Heartbeat and keeps them running indefinitely.
Exposes a Unix domain socket so the CLI can connect and interact.

Start with: gwenn daemon
Or via systemd: gwenn-daemon (entry point calls run_daemon())

Protocol: Newline-Delimited JSON (NDJSON) over Unix domain socket.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import signal
import sys
import time
from typing import Any, Optional

import structlog

from gwenn.config import GwennConfig
from gwenn.memory.session_store import SessionStore
from gwenn.privacy.redaction import PIIRedactor

logger = structlog.get_logger(__name__)


class GwennDaemon:
    """
    Persistent background daemon that keeps Gwenn alive between CLI sessions.

    Architecture:
      - SentientAgent with continuous heartbeat
      - One asyncio Unix socket server (chmod 0o600)
      - Per-connection conversation history
      - SessionStore auto-saves on disconnect
    """

    def __init__(self, config: GwennConfig) -> None:
        self._config = config
        self._agent: Optional[Any] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._shutdown_event = asyncio.Event()
        self._connection_count = 0  # total connections ever accepted
        self._active_connections = 0  # connections currently open
        # Resolve paths to absolute so CWD changes don't matter
        self._socket_path = config.daemon.socket_path.resolve()
        self._pid_file = config.daemon.pid_file.resolve()
        self._sessions_dir = config.daemon.sessions_dir.resolve()
        self._auth_token = (config.daemon.auth_token or "").strip() or None
        self._max_connections = max(1, int(config.daemon.max_connections))
        self._connection_timeout = max(1.0, float(config.daemon.connection_timeout))
        self._session_include_preview = bool(config.daemon.session_include_preview)
        self._session_redactor = PIIRedactor(enabled=config.daemon.redact_session_content)
        self._session_store = SessionStore(
            self._sessions_dir,
            max_count=config.daemon.session_max_count,
            max_messages=config.daemon.session_max_messages,
        )
        # SentientAgent mutates shared state during respond(); serialize daemon chat calls.
        self._agent_respond_lock = asyncio.Lock()

    async def run(self) -> None:
        """Full daemon lifecycle: init → serve → shutdown."""
        self._write_pid_file()
        try:
            await self._start_agent()
            await self._start_socket_server()
            logger.info(
                "daemon.running",
                socket=str(self._socket_path),
                pid=os.getpid(),
            )
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _write_pid_file(self) -> None:
        """Write PID file, detecting stale files from crashed processes."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self._pid_file.exists():
            try:
                existing_pid = int(self._pid_file.read_text().strip())
                # Check if process is actually alive
                os.kill(existing_pid, 0)
                # If we get here, a daemon is already running
                logger.error("daemon.already_running", pid=existing_pid)
                sys.exit(1)
            except (ValueError, ProcessLookupError, PermissionError):
                # Stale PID file — clean it up
                logger.info("daemon.stale_pid_file_removed", path=str(self._pid_file))
                try:
                    self._pid_file.unlink()
                except OSError:
                    pass
        self._pid_file.write_text(str(os.getpid()))

    async def _start_agent(self) -> None:
        """Create, initialize, and start the SentientAgent."""
        from gwenn.agent import SentientAgent
        from gwenn.api.claude import CognitiveEngineInitError

        try:
            self._agent = SentientAgent(self._config)
        except CognitiveEngineInitError as e:
            logger.error("daemon.agent_init_failed", error=str(e))
            sys.exit(1)

        await self._agent.initialize()
        await self._agent.start()
        logger.info("daemon.agent_started")

        # Start configured channels (telegram, discord) if requested
        channel_list = self._config.daemon.get_channel_list()
        if "telegram" in channel_list or "discord" in channel_list:
            asyncio.create_task(
                self._run_platform_channels(channel_list),
                name="daemon-channels",
            )

    async def _start_socket_server(self) -> None:
        """Open the Unix domain socket server."""
        # Remove stale socket file
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError as e:
                logger.warning("daemon.stale_socket_remove_failed", error=str(e))

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )
        # Restrict access to owner only
        try:
            os.chmod(str(self._socket_path), 0o600)
        except OSError as e:
            logger.warning("daemon.chmod_failed", error=str(e))

    async def _run_platform_channels(self, channel_list: list[str]) -> None:
        """Start Telegram/Discord channels in the daemon's event loop."""
        from gwenn.channels.session import SessionManager

        tg_config = None
        dc_config = None

        if "telegram" in channel_list:
            try:
                from gwenn.config import TelegramConfig
                tg_config = TelegramConfig()
            except Exception as e:
                logger.error("daemon.telegram_config_failed", error=str(e))

        if "discord" in channel_list:
            try:
                from gwenn.config import DiscordConfig
                dc_config = DiscordConfig()
            except Exception as e:
                logger.error("daemon.discord_config_failed", error=str(e))

        if not tg_config and not dc_config:
            return

        if tg_config and dc_config:
            max_history = min(tg_config.max_history_length, dc_config.max_history_length)
            session_ttl = dc_config.session_ttl_seconds
        elif tg_config:
            max_history = tg_config.max_history_length
            session_ttl = 3600.0
        else:
            max_history = dc_config.max_history_length
            session_ttl = dc_config.session_ttl_seconds

        sessions = SessionManager(max_history_length=max_history, session_ttl_seconds=session_ttl)
        channels = []

        if tg_config:
            from gwenn.channels.telegram_channel import TelegramChannel
            channels.append(TelegramChannel(self._agent, sessions, tg_config))

        if dc_config:
            from gwenn.channels.discord_channel import DiscordChannel
            channels.append(DiscordChannel(self._agent, sessions, dc_config))

        for ch in channels:
            await ch.start()

        await self._shutdown_event.wait()

        for ch in channels:
            await ch.stop()

    # ------------------------------------------------------------------
    # Client connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one CLI client connection from connect to disconnect."""
        if self._active_connections >= self._max_connections:
            logger.warning(
                "daemon.connection_rejected_max_connections",
                active_connections=self._active_connections,
                max_connections=self._max_connections,
            )
            try:
                await self._send(
                    writer,
                    {"type": "error", "req_id": "", "message": "max connections reached"},
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return

        self._connection_count += 1
        self._active_connections += 1
        conn_id = self._connection_count
        logger.info("daemon.client_connected", conn_id=conn_id)

        # Per-connection state
        history: list[dict] = []
        started_at = time.time()

        try:
            await self._dispatch_loop(reader, writer, history)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            logger.info("daemon.client_disconnected", conn_id=conn_id)
        except Exception as e:
            logger.error("daemon.client_error", conn_id=conn_id, error=str(e), exc_info=True)
        finally:
            self._active_connections = max(0, self._active_connections - 1)
            # Always save session on disconnect
            if history:
                self._session_store.save_session(
                    history,
                    started_at,
                    text_filter=self._session_redactor.redact,
                )
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug("daemon.client_close_failed", conn_id=conn_id, error=str(e))
            logger.info("daemon.client_closed", conn_id=conn_id)

    async def _dispatch_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        history: list[dict],
    ) -> None:
        """Read NDJSON messages from client and dispatch responses."""
        last_activity = time.monotonic()
        while not self._shutdown_event.is_set():
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.monotonic() - last_activity >= self._connection_timeout:
                    logger.info(
                        "daemon.connection_timed_out",
                        timeout_seconds=self._connection_timeout,
                    )
                    break
                continue
            if not raw:
                break  # Client disconnected

            last_activity = time.monotonic()

            try:
                msg = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("daemon.bad_json")
                continue

            req_id = msg.get("id", "")
            msg_type = msg.get("type", "")

            response = await self._dispatch(msg_type, msg, history, req_id)
            await self._send(writer, response)

            if msg_type == "stop":
                break

    async def _dispatch(
        self,
        msg_type: str,
        msg: dict,
        history: list[dict],
        req_id: str,
    ) -> dict:
        """Route a request to the appropriate handler and return a response."""
        try:
            if not self._is_authorized(msg):
                return {"type": "error", "req_id": req_id, "message": "unauthorized"}

            if msg_type == "ping":
                return {"type": "pong", "req_id": req_id}

            elif msg_type == "chat":
                text = msg.get("text", "")
                if not text:
                    return {"type": "error", "req_id": req_id, "message": "empty text"}
                lock = getattr(self, "_agent_respond_lock", None)
                if lock is None:
                    lock = asyncio.Lock()
                    self._agent_respond_lock = lock
                async with lock:
                    response_text = await self._agent.respond(
                        text,
                        conversation_history=history,
                    )
                    emotion = "neutral"
                    try:
                        emotion = self._agent.affect_state.current_emotion.value
                    except Exception as e:
                        logger.debug("daemon.emotion_snapshot_failed", error=str(e))
                return {
                    "type": "response",
                    "req_id": req_id,
                    "text": response_text,
                    "emotion": emotion,
                }

            elif msg_type == "status":
                status = self._agent.status if self._agent else {}
                return {
                    "type": "status_response",
                    "req_id": req_id,
                    "status": status,
                    "active_connections": self._active_connections,
                }

            elif msg_type == "heartbeat_status":
                hb = {}
                if self._agent and self._agent.heartbeat:
                    hb = self._agent.heartbeat.status
                return {"type": "heartbeat_response", "req_id": req_id, "status": hb}

            elif msg_type == "list_sessions":
                sessions = self._session_store.list_sessions(
                    limit=10,
                    include_preview=self._session_include_preview,
                )
                return {"type": "sessions_list", "req_id": req_id, "sessions": sessions}

            elif msg_type == "load_session":
                session_id = msg.get("session_id", "")
                if not SessionStore.is_valid_session_id(session_id):
                    return {"type": "error", "req_id": req_id, "message": "invalid session id"}
                try:
                    loaded = self._session_store.load_session(session_id)
                    # Replace this connection's history with the loaded session
                    history.clear()
                    history.extend(loaded)
                    return {"type": "session_loaded", "req_id": req_id, "message_count": len(loaded)}
                except FileNotFoundError:
                    return {"type": "error", "req_id": req_id, "message": "session not found"}

            elif msg_type == "stop":
                logger.info("daemon.stop_requested")
                self._shutdown_event.set()
                return {"type": "ack_stop", "req_id": req_id}

            else:
                return {"type": "error", "req_id": req_id, "message": f"unknown type: {msg_type}"}

        except Exception as e:
            logger.error("daemon.dispatch_error", msg_type=msg_type, error=str(e), exc_info=True)
            return {"type": "error", "req_id": req_id, "message": str(e)}

    def _is_authorized(self, msg: dict) -> bool:
        """Validate optional daemon auth token using constant-time comparison."""
        if not self._auth_token:
            return True

        provided = msg.get("auth_token")
        if not isinstance(provided, str):
            logger.warning("daemon.auth_missing")
            return False

        authorized = hmac.compare_digest(provided, self._auth_token)
        if not authorized:
            logger.warning("daemon.auth_rejected")
        return authorized

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, data: dict) -> None:
        """Serialize and write one NDJSON line to the client."""
        line = json.dumps(data, ensure_ascii=False) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    # ------------------------------------------------------------------
    # Shutdown & cleanup
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """Graceful shutdown: stop server, shut down agent, remove files."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._agent:
            await self._agent.shutdown()

        # Remove socket and PID files
        for path in (self._socket_path, self._pid_file):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("daemon.cleanup_unlink_failed", path=str(path), error=str(e))

        logger.info("daemon.stopped")

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass


def run_daemon() -> None:
    """
    Entry point for `gwenn-daemon` console script and `gwenn daemon` subcommand.

    Loads config, creates GwennDaemon, runs the event loop.
    """
    try:
        config = GwennConfig()
    except Exception as e:
        print(f"[daemon] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    daemon = GwennDaemon(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    daemon._install_signal_handlers(loop)
    try:
        loop.run_until_complete(daemon.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
