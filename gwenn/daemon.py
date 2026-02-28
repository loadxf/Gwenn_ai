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
import re
import signal
import sys
import time
from typing import Any, Optional

import structlog

from gwenn.config import GwennConfig
from gwenn.main import configure_logging
from gwenn.memory.session_store import SessionStore
from gwenn.privacy.redaction import PIIRedactor

logger = structlog.get_logger(__name__)

_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")


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
        self._channel_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._connection_count = 0  # total connections ever accepted
        self._active_connections = 0  # connections currently open
        self._socket_path = config.daemon.socket_path
        self._pid_file = config.daemon.pid_file
        self._sessions_dir = config.daemon.sessions_dir
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
        # Respond-lock is on the agent itself (agent.respond_lock) so daemon
        # and platform channels share the same serialisation lock.

    async def run(self) -> None:
        """Full daemon lifecycle: init → serve → shutdown.

        When ``heartbeat_core`` is enabled, the heartbeat manages the full
        lifecycle (event bus → gateway → agent → channels). The daemon is
        just a PID-file wrapper.
        """
        self._write_pid_file()
        try:
            if self._config.daemon.heartbeat_core:
                await self._run_heartbeat_core()
            else:
                await self._run_legacy()
        finally:
            self._cleanup_pid_and_socket()

    async def _run_heartbeat_core(self) -> None:
        """Heartbeat-as-core mode: delegate lifecycle to Heartbeat.run().

        Heartbeat.run() owns all internal cleanup (event bus, gateway, agent,
        channels) via its _sleep() method. The daemon only handles PID/socket
        files via _cleanup_pid_and_socket() in the outer finally block.
        """
        from gwenn.heartbeat import Heartbeat

        logger.info("daemon.mode", mode="heartbeat_core")
        heartbeat = Heartbeat(self._config)
        await heartbeat.run()

    async def _run_legacy(self) -> None:
        """Legacy mode: daemon manages agent, socket server, and channels."""
        try:
            await self._start_agent()
            await self._start_socket_server()
            logger.info(
                "daemon.running",
                socket=str(self._socket_path),
                pid=os.getpid(),
            )
            print(
                f"Gwenn daemon running  pid={os.getpid()}  "
                f"socket={self._socket_path}",
                flush=True,
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
            except PermissionError:
                # Process is alive but owned by another user — treat as running
                logger.error("daemon.already_running", pid=existing_pid)
                sys.exit(1)
            except (ValueError, ProcessLookupError):
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

        print("Gwenn daemon starting ...", flush=True)
        await self._agent.initialize()
        await self._agent.start()
        logger.info("daemon.agent_started")

        # Start configured channels (telegram, discord) if requested
        channel_list = self._config.channel.get_channel_list()
        if "telegram" in channel_list or "discord" in channel_list:
            self._channel_task = asyncio.create_task(
                self._run_platform_channels(channel_list),
                name="daemon-channels",
            )
            self._channel_task.add_done_callback(self._on_channel_task_done)

    def _on_channel_task_done(self, task: asyncio.Task) -> None:
        """Monitor daemon channel task and trigger shutdown if it crashes."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if self._is_nonfatal_channel_error(exc):
            logger.warning(
                "daemon.channels_task_failed_nonfatal",
                error_type=type(exc).__name__,
                error=self._redact_channel_error(str(exc)),
            )
            return
        logger.error(
            "daemon.channels_task_failed",
            error=self._redact_channel_error(str(exc)),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self._request_shutdown("daemon_channels_task_failed")

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
        from gwenn.channels.startup import build_channels, run_channels_until_shutdown

        sessions, channels = build_channels(self._agent, channel_list=channel_list)
        if not channels:
            return

        try:
            await run_channels_until_shutdown(
                self._agent,
                sessions,
                channels,
                self._shutdown_event,
                continue_on_start_error=(
                    "telegram" in channel_list and "discord" in channel_list
                ),
            )
        except Exception as exc:
            if self._is_nonfatal_channel_error(exc):
                logger.warning(
                    "daemon.channels_startup_skipped",
                    channels=channel_list,
                    error_type=type(exc).__name__,
                    error=self._redact_channel_error(str(exc)),
                )
                return
            raise

    @staticmethod
    def _is_nonfatal_channel_error(exc: Exception) -> bool:
        """Return True for channel startup errors that should not stop the daemon."""
        if isinstance(exc, ImportError):
            # Only treat ImportError as nonfatal if it mentions optional deps
            msg = str(exc).lower()
            if "telegram" in msg or "discord" in msg:
                return True
            return False
        err_type = type(exc).__name__
        err_mod = type(exc).__module__
        if err_type == "InvalidToken" and err_mod.startswith("telegram"):
            return True
        if err_type == "LoginFailure" and err_mod.startswith("discord"):
            return True
        return False

    @staticmethod
    def _redact_channel_error(message: str) -> str:
        """Mask Telegram bot tokens in channel error strings before logging."""
        return _TELEGRAM_BOT_TOKEN_RE.sub("[REDACTED_TELEGRAM_TOKEN]", message or "")

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

    _MAX_AUTH_FAILURES = 3

    async def _dispatch_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        history: list[dict],
    ) -> None:
        """Read NDJSON messages from client and dispatch responses."""
        last_activity = time.monotonic()
        auth_failures = 0
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

            req_id = msg.get("req_id", "")
            msg_type = msg.get("type", "")

            response = await self._dispatch(msg_type, msg, history, req_id)
            await self._send(writer, response)

            # Disconnect after repeated auth failures to prevent brute-force.
            if response.get("message") == "unauthorized":
                auth_failures += 1
                if auth_failures >= self._MAX_AUTH_FAILURES:
                    logger.warning(
                        "daemon.auth_max_failures",
                        failures=auth_failures,
                    )
                    break
            elif msg_type == "chat" and response.get("type") != "error":
                auth_failures = 0

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
                async with self._agent.respond_lock:
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
                    "text": str(response_text),
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
                    return {
                        "type": "session_loaded",
                        "req_id": req_id,
                        "message_count": len(loaded),
                    }
                except FileNotFoundError:
                    return {"type": "error", "req_id": req_id, "message": "session not found"}

            elif msg_type == "reset_session":
                cleared = len(history)
                history.clear()
                return {"type": "session_reset", "req_id": req_id, "cleared_messages": cleared}

            elif msg_type == "runtime_info":
                status = self._agent.status if self._agent else {}
                skills: list[dict[str, Any]] = []
                if self._agent is not None:
                    skill_registry = getattr(self._agent, "skill_registry", None)
                    all_skills = getattr(skill_registry, "all_skills", None)
                    if callable(all_skills):
                        try:
                            for skill in all_skills():
                                skills.append(
                                    {
                                        "name": str(getattr(skill, "name", "unknown")),
                                        "category": str(getattr(skill, "category", "skill")),
                                    }
                                )
                        except Exception as e:
                            logger.debug("daemon.runtime_info_skill_list_failed", error=str(e))

                mcp_stats = getattr(getattr(self._agent, "_mcp_client", None), "stats", {})
                if not isinstance(mcp_stats, dict):
                    mcp_stats = {}
                tool_registry = getattr(self._agent, "tool_registry", None)
                tools = {
                    "registered": int(getattr(tool_registry, "count", 0)),
                    "enabled": int(getattr(tool_registry, "enabled_count", 0)),
                }
                configured_mcp_servers = self._config.mcp.get_server_list()
                return {
                    "type": "runtime_info_response",
                    "req_id": req_id,
                    "status": status,
                    "skills": skills,
                    "mcp": mcp_stats,
                    "tools": tools,
                    "configured_mcp_servers": configured_mcp_servers,
                    "active_connections": self._active_connections,
                }

            elif msg_type == "stop":
                logger.info("daemon.stop_requested")
                self._request_shutdown("daemon_stop_requested")
                return {"type": "ack_stop", "req_id": req_id}

            else:
                return {"type": "error", "req_id": req_id, "message": f"unknown type: {msg_type}"}

        except Exception as e:
            logger.error("daemon.dispatch_error", msg_type=msg_type, error=str(e), exc_info=True)
            return {"type": "error", "req_id": req_id, "message": "internal error"}

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
        """Graceful shutdown for legacy mode: stop server, shut down agent."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._channel_task is not None:
            # Let channel task observe shutdown and stop adapters cleanly.
            self._shutdown_event.set()
            try:
                await self._channel_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("daemon.channels_task_join_failed")
            self._channel_task = None

        if self._agent:
            await self._agent.shutdown()

    def _cleanup_pid_and_socket(self) -> None:
        """Remove PID and socket files — shared by both modes."""
        for path in (self._socket_path, self._pid_file):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("daemon.cleanup_unlink_failed", path=str(path), error=str(e))

        logger.info("daemon.stopped")
        print("Gwenn daemon stopped.", flush=True)

    def _request_shutdown(self, reason: str) -> None:
        """Trigger daemon shutdown and activate the agent kill switch."""
        agent = self._agent
        if agent is not None:
            safety = getattr(agent, "safety", None)
            emergency_stop = getattr(safety, "emergency_stop", None)
            if callable(emergency_stop):
                try:
                    emergency_stop(reason)
                except Exception:
                    logger.debug("daemon.emergency_stop_failed", reason=reason, exc_info=True)
        self._shutdown_event.set()

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, self._request_shutdown, f"daemon_signal_{sig.name.lower()}"
                )
            except NotImplementedError:
                pass


def run_daemon() -> None:
    """
    Entry point for `gwenn-daemon` console script and `gwenn daemon` subcommand.

    Loads config, creates GwennDaemon, runs the event loop.
    """
    configure_logging()
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
