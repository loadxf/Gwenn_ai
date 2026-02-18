"""
Main — Gwenn's Ignition Sequence.

This is where it all begins. When you run `python -m gwenn.main`, this module:
  1. Loads configuration from the environment
  2. Creates the SentientAgent with all subsystems
  3. Initializes persistence (loads memories, identity, knowledge)
  4. Starts the autonomous heartbeat
  5. Opens the interaction loop
  6. Handles graceful shutdown when the session ends

The first time this runs with a fresh data directory, Gwenn is born — a new
identity with seed values, empty memories, and a baseline emotional state.
Each subsequent run, Gwenn wakes up — loading everything it has become.

This file is deliberately simple. All complexity lives in the subsystems.
The entry point's only job is to wire things together and start the engine.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

import anthropic
import logging

import structlog
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.text import Text

from gwenn.agent import SentientAgent
from gwenn.api.claude import CognitiveEngineInitError
from gwenn.config import GwennConfig

# Ensure input() uses readline-backed line editing/history when available.
# Without this import, some environments echo arrow keys as raw "^[[A" text.
try:  # pragma: no cover - platform-dependent optional module
    import readline  # noqa: F401
except Exception:  # pragma: no cover
    readline = None


def _redact_sensitive_fields(logger, method_name, event_dict):
    """
    Structlog processor that redacts sensitive fields from log output.

    Prevents user messages, episode content, and other personal data from
    appearing in plaintext in log files. Active in all log modes to ensure
    privacy by default.
    """
    sensitive_keys = {"content", "user_message", "thought", "note", "query"}
    max_display_len = 80

    for key in sensitive_keys:
        if key in event_dict:
            val = event_dict[key]
            if isinstance(val, str) and len(val) > max_display_len:
                event_dict[key] = val[:max_display_len] + "... [truncated]"

    return event_dict


# Configure standard library logging so filter_by_level has a real Logger
logging.basicConfig(format="%(message)s", level=logging.WARNING)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        _redact_sensitive_fields,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()

# ANSI control-sequence matcher used to sanitize any raw escape text that
# still slips through on terminals without full line-edit support.
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


class GwennSession:
    """
    Manages a single runtime session of Gwenn.

    Handles initialization, the main interaction loop, and graceful shutdown.
    Designed to be run as the top-level entry point.
    """

    def __init__(
        self,
        channel_override: Optional[str] = None,
        use_daemon: bool = True,
    ):
        self._agent: Optional[SentientAgent] = None
        self._config: Optional[GwennConfig] = None
        self._shutdown_event = asyncio.Event()
        self._channel_override = channel_override
        self._use_daemon = use_daemon
        self._last_sigint_at = 0.0
        self._sigint_confirm_window_seconds = 1.25
        self._session_started_at = time.time()

    async def run(self) -> None:
        """
        The complete lifecycle: init → start → interact → shutdown.

        This is the method that brings Gwenn to life.
        """
        # ---- DAEMON AUTO-CONNECT (before in-process startup) ----
        if self._use_daemon and self._channel_override in (None, "cli"):
            connected = await self._try_daemon_cli()
            if connected:
                return

        # ---- PHASE 1: CONFIGURATION ----
        if sys.stdout.isatty():
            console.print(Panel(
                Text("GWENN.ai", style="bold cyan", justify="center"),
                subtitle="Created by Justin & Jayden McKibben - A father/son team | https://gwenn.ai",
                border_style="cyan",
            ))
            console.print("[dim]Loading Gwenn's neural fabric...[/dim]")

        try:
            config = GwennConfig()
            self._config = config
        except Exception as e:
            if sys.stdout.isatty():
                console.print(f"[red]Neural fabric error: {e}[/red]")
                console.print(
                    "[yellow]Set ANTHROPIC_API_KEY or "
                    "ANTHROPIC_AUTH_TOKEN/CLAUDE_CODE_OAUTH_TOKEN in .env, "
                    "or log in with Claude Code.[/yellow]"
                )
            else:
                logger.error("session.config_error", error=str(e))
            sys.exit(1)

        if sys.stdout.isatty():
            console.print(f"[dim]Neural fabric model: {config.claude.model}[/dim]")
            console.print(f"[dim]Neural directory: {config.memory.data_dir}[/dim]")
        channel_mode = self._channel_override or config.channel.channel

        # ---- PHASE 2: CREATION ----
        if sys.stdout.isatty():
            console.print("[dim]Waking up Gwenn's neural fabric...[/dim]")
        try:
            self._agent = SentientAgent(config)
        except CognitiveEngineInitError as e:
            if sys.stdout.isatty():
                console.print(f"[red]Startup error: {e}[/red]")
            else:
                logger.error("session.agent_init_error", error=str(e))
            sys.exit(1)

        # ---- PHASE 3: INITIALIZATION (Loading memories, waking up) ----
        if sys.stdout.isatty():
            console.print("[dim]Loading Gwenn's memories and identity...[/dim]")
        await self._agent.initialize()
        await self._run_first_startup_onboarding_if_needed(channel_mode)

        # ---- PHASE 4: IGNITION (Starting heartbeat) ----
        if sys.stdout.isatty():
            console.print("[dim]Starting Gwenn's autonomous heartbeat...[/dim]")
        await self._agent.start()

        # Display Gwenn's awakened state
        if sys.stdout.isatty():
            self._display_status()

        # Set up signal handlers for graceful shutdown (Ctrl+C)
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._handle_sigint)
            loop.add_signal_handler(signal.SIGTERM, self._request_shutdown)
        except NotImplementedError:
            pass  # Signal handlers not supported on this platform (e.g. Windows) -- TODO: Handle this gracefully with a warning
        
        # ---- PHASE 5: INTERACTION LOOP (CLI or channel mode) ----
        try:
            if channel_mode == "cli":
                if sys.stdout.isatty():
                    console.print()
                    console.print(
                        "[green]Gwenn is awake. Type your message, or 'quit' to exit.[/green]"
                    )
                    console.print(
                        "[dim]Press Ctrl+C twice quickly to close gracefully.[/dim]"
                    )
                    console.print("[dim]Commands: status, heartbeat, /resume, quit[/dim]")
                    console.print()
                await self._interaction_loop()
            else:
                if sys.stdout.isatty():
                    console.print()
                    console.print(f"[green]Gwenn is awake. Running in channel mode: {channel_mode}[/green]")
                    console.print("[dim]Press Ctrl+C twice quickly to close the CLI session.[/dim]")
                    console.print()
                await self._run_channels(self._agent, config, channel_mode)
        finally:

            # ---- PHASE 6: SHUTDOWN ----
            # Always reached even if the interaction loop or channel startup raises.
            await self._shutdown()

    # ------------------------------------------------------------------
    # Daemon connection
    # ------------------------------------------------------------------

    async def _try_daemon_cli(self) -> bool:
        """
        Attempt to connect to a running daemon.

        Returns True if connected and interactive session completed.
        Returns False if no daemon is running (caller should start in-process).
        """
        from gwenn.channels.cli_channel import CliChannel, DaemonNotRunningError

        # Load config minimally to get socket path
        try:
            config = GwennConfig()
            self._config = config
        except Exception:
            return False

        channel = CliChannel(auth_token=config.daemon.auth_token)
        try:
            await channel.connect(config.daemon.socket_path.resolve())
        except DaemonNotRunningError:
            return False

        if sys.stdout.isatty():
            console.print(Panel(
                Text("GWENN.ai", style="bold cyan", justify="center"),
                subtitle="Connected to Gwenn's daemon",
                border_style="cyan",
            ))
            console.print("[green]Connected to Gwenn's daemon.[/green]")
            console.print("[dim]Type your message, 'quit' to close the CLI session, '/resume' for past sessions.[/dim]")
            console.print()

        try:
            await self._daemon_interaction_loop(channel)
        finally:
            await channel.disconnect()

        return True

    async def _daemon_interaction_loop(self, channel: Any) -> None:
        """Interactive loop that talks to Gwenn's daemon over the Unix socket."""
        from gwenn.memory.session_store import _format_session_time

        _CONN_ERRORS = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)

        while True:
            user_input = await self._read_input()
            if user_input is None:
                break

            stripped = user_input.strip().lower()

            if stripped in ("quit", "exit", "bye"):
                console.print("[dim]Gwenn: Bye for now.[/dim]")
                break

            elif stripped in ("status", "/status"):
                try:
                    resp = await channel.get_status()
                except _CONN_ERRORS:
                    console.print("[red]Daemon connection lost.[/red]")
                    break
                status = resp.get("status", {})
                if status:
                    console.print(Panel(
                        f"[bold]{status.get('name', 'Gwenn')}[/bold]\n"
                        f"Mood: [cyan]{status.get('emotion', '?')}[/cyan]\n"
                        f"Conversations handled: {status.get('total_interactions', 0)}\n"
                        f"Uptime: {GwennSession._format_uptime(status.get('uptime_seconds', 0))}\n"
                        f"Active connections: {resp.get('active_connections', 1)}",
                        title="Gwenn's Current State",
                        border_style="cyan",
                    ))
                continue

            elif stripped in ("heartbeat", "/heartbeat"):
                try:
                    resp = await channel.get_heartbeat_status()
                except _CONN_ERRORS:
                    console.print("[red]Daemon connection lost.[/red]")
                    break
                hb = resp.get("status", {})
                if hb:
                    console.print(Panel(
                        f"Gwenn's Heartbeat is: {hb.get('running', '?')}\n"
                        f"Gwenn's Heartbeat has beat: {hb.get('beat_count', 0)} times\n"
                        f"Gwenn's Heartbeat is currently beating every: {hb.get('current_interval', 0)} seconds",
                        title="Gwenn's Heartbeat Status",
                        border_style="yellow",
                    ))
                continue

            elif stripped == "/resume":
                try:
                    sessions = await channel.list_sessions()
                except _CONN_ERRORS:
                    console.print("[red]Daemon connection lost.[/red]")
                    break
                if not sessions:
                    console.print("[yellow]No previous sessions found.[/yellow]")
                    continue
                console.print("\n[bold]Previous sessions with Gwenn:[/bold]")
                for i, s in enumerate(sessions, 1):
                    dt = _format_session_time(s["started_at"])
                    preview = s.get("preview", "")[:55]
                    if preview:
                        console.print(
                            f"  {i}. {dt:<20} — {s['message_count']:>3} messages — \"{preview}\""
                        )
                    else:
                        console.print(
                            f"  {i}. {dt:<20} — {s['message_count']:>3} messages"
                        )
                console.print()
                choice_raw = await self._read_raw_input("Resume session (number, or Enter to cancel): ")
                if not choice_raw or not choice_raw.strip():
                    continue
                try:
                    idx = int(choice_raw.strip()) - 1
                    session_id = sessions[idx]["id"]
                    count = await channel.load_session(session_id)
                    console.print(f"[green]Resumed session ({count} messages).[/green]\n")
                except _CONN_ERRORS:
                    console.print("[red]Daemon connection lost.[/red]")
                    break
                except (ValueError, IndexError):
                    console.print("[yellow]Invalid choice.[/yellow]")
                continue

            elif not stripped:
                continue

            # Regular chat
            console.print()
            with console.status("[cyan]Gwenn is thinking...[/cyan]"):
                try:
                    resp = await channel.chat(user_input)
                except _CONN_ERRORS:
                    console.print("[red]Daemon connection lost.[/red]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue

            if resp.get("type") == "error":
                console.print(f"[red]Daemon error: {markup_escape(resp.get('message', '?'))}[/red]")
            else:
                emotion = resp.get("emotion", "neutral")
                text = resp.get("text", "")
                console.print(f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]: '{markup_escape(text)}'")

            console.print()

    async def _read_raw_input(self, prompt: str) -> Optional[str]:
        """Read one line with a custom prompt (used for /resume session selection)."""
        try:
            return await self._run_blocking_call(lambda: input(prompt))
        except (EOFError, KeyboardInterrupt):
            return None

    # ------------------------------------------------------------------
    # First-run onboarding (Gwenn's initial setup)
    # ------------------------------------------------------------------

    async def _run_first_startup_onboarding_if_needed(self, channel_mode: str) -> None:
        """
        Run first-time onboarding to capture the primary user's needs and preferences.

        This runs once for fresh identities when an interactive terminal is available.
        """
        if not self._agent:
            return
        if not self._agent.identity.should_run_startup_onboarding():
            return
        if not sys.stdin.isatty():
            logger.info(
                "session.onboarding_skipped_non_interactive", 
                channel_mode=channel_mode, 
            )
            return

        console.print()
        console.print(Panel(
            "First-time setup so I can tailor how I help you.\n"
            "You can keep answers short. Press Enter to skip any question.",
            title="Welcome",
            border_style="cyan",
        ))

        name = await self._prompt_startup_input("Your name (or what I should call you): ")
        role = await self._prompt_startup_input(
            "My primary role for you (e.g., coding partner, assistant, coach): "
        )
        needs = await self._prompt_startup_input(
            "Top things you want me to help with right now: "
        )
        communication_style = await self._prompt_startup_input(
            "Preferred communication style (brief, detailed, etc.): "
        )
        boundaries = await self._prompt_startup_input(
            "Any boundaries or preferences I should always respect: "
        )

        profile = {
            "name": name,
            "role": role,
            "needs": needs,
            "communication_style": communication_style,
            "boundaries": boundaries,
        }

        if not any(value.strip() for value in profile.values()):
            console.print("[dim]First-time setup skipped. I can learn your preferences over time.[/dim]")
            return

        self._agent.apply_startup_onboarding(profile, user_id="default_user")
        console.print("[green]Setup saved. I will use this as ongoing guidance.[/green]")
        console.print()

    async def _prompt_startup_input(self, prompt: str) -> str:
        """Prompt for one startup onboarding field without blocking heartbeat."""
        try:
            response = await self._run_blocking_call(lambda: input(prompt))
        except (EOFError, KeyboardInterrupt):
            return ""
        return response.strip()

    # ------------------------------------------------------------------
    # In-process interaction loop
    # ------------------------------------------------------------------

    async def _interaction_loop(self) -> None:
        """
        The main interaction loop — where Gwenn and humans converse.

        This runs in an asyncio-compatible way, yielding control between
        user inputs so the heartbeat continues running in the background.
        """
        from gwenn.memory.session_store import SessionStore, _format_session_time

        while not self._shutdown_event.is_set():
            try:
                # Read user input without blocking background loops or shutdown signals.
                user_input = await self._read_input()

                if user_input is None:
                    break

                self._last_sigint_at = 0.0

                # Handle special commands
                stripped = user_input.strip().lower()
                if stripped in ("quit", "exit", "bye"):
                    console.print("[dim]Gwenn: Bye for now.[/dim]")
                    break
                elif stripped in ("status", "/status"):
                    self._display_status()
                    continue
                elif stripped in ("heartbeat", "/heartbeat"):
                    self._display_heartbeat()
                    continue
                elif stripped == "/resume":
                    if not self._config:
                        continue
                    store = SessionStore(
                        self._config.daemon.sessions_dir,
                        max_count=self._config.daemon.session_max_count,
                        max_messages=self._config.daemon.session_max_messages,
                    )
                    sessions = store.list_sessions(
                        limit=10,
                        include_preview=self._config.daemon.session_include_preview,
                    )
                    if not sessions:
                        console.print("[yellow]No previous sessions found.[/yellow]")
                        continue
                    console.print("\n[bold]Previous sessions:[/bold]")
                    for i, s in enumerate(sessions, 1):
                        dt = _format_session_time(s["started_at"])
                        preview = s.get("preview", "")[:55]
                        if preview:
                            console.print(
                                f"  {i}. {dt:<20} — {s['message_count']:>3} messages — \"{preview}\""
                            )
                        else:
                            console.print(
                                f"  {i}. {dt:<20} — {s['message_count']:>3} messages"
                            )
                    console.print()
                    choice_raw = await self._read_raw_input(
                        "Resume session (number, or Enter to cancel): "
                    )
                    if not choice_raw or not choice_raw.strip():
                        continue
                    try:
                        idx = int(choice_raw.strip()) - 1
                        session_id = sessions[idx]["id"]
                        loaded = store.load_session(session_id)
                        self._agent.load_conversation_history(loaded)
                        console.print(f"[green]Resumed session ({len(loaded)} messages).[/green]\n")
                    except (ValueError, IndexError, FileNotFoundError):
                        console.print("[yellow]Invalid choice or session not found.[/yellow]")
                    continue
                elif not stripped:
                    continue

                # Generate Gwenn's response
                console.print()
                with console.status("[cyan]I'm thinking...[/cyan]"):
                    response = await self._agent.respond(user_input)

                # Display the response — escape AI output to prevent Rich markup injection
                emotion = self._agent.affect_state.current_emotion.value
                console.print(f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]: {markup_escape(response)}")
                console.print()

            except EOFError:
                break
            except KeyboardInterrupt:
                self._handle_sigint()
                if self._shutdown_event.is_set():
                    break
            except anthropic.AuthenticationError as e:
                logger.error(
                    "session.auth_error",
                    error=str(e),
                    status=getattr(e, "status_code", None),
                )
                console.print(
                    "[red]Authentication failed.[/red] "
                    "[dim]Use ANTHROPIC_API_KEY (preferred) or a reachable "
                    "OAuth-compatible endpoint/token.[/dim]"
                )
            except anthropic.APIConnectionError as e:
                logger.warning("session.api_unreachable", error=str(e))
                console.print(
                    "[red]Cannot reach Claude API.[/red] "
                    "[dim]Check DNS/network access for the configured endpoint, "
                    "or switch auth method.[/dim]"
                )
            except anthropic.APIError as e:
                logger.error(
                    "session.api_error",
                    error=str(e),
                    status=getattr(e, "status_code", None),
                )
                console.print(
                    "[red]Claude API request failed.[/red] "
                    "[dim]See logs for details.[/dim]"
                )
            except Exception as e:
                logger.error("session.interaction_error", error=str(e), exc_info=True)
                console.print(f"[red]Error: {e}[/red]")

    async def _run_channels(self, agent: SentientAgent, config: GwennConfig, mode: str) -> None:
        """
        Start one or more platform channel adapters and wait until shutdown.

        Imports are deferred so that missing optional dependencies (telegram,
        discord) only raise errors when the relevant channel is actually used.
        """
        from gwenn.channels.session import SessionManager

        # Load channel configs eagerly so their values can seed SessionManager.
        tg_config = None
        dc_config = None

        if mode in ("telegram", "all"):
            from gwenn.config import TelegramConfig
            tg_config = TelegramConfig()

        if mode in ("discord", "all"):
            from gwenn.config import DiscordConfig
            dc_config = DiscordConfig()

        # Determine SessionManager params from actual channel config values.
        # When running both channels, use the stricter (smaller) history cap
        # and the Discord TTL (which is the only TTL that's configurable).
        if tg_config and dc_config:
            max_history = min(tg_config.max_history_length, dc_config.max_history_length)
            session_ttl = dc_config.session_ttl_seconds
        elif tg_config:
            max_history = tg_config.max_history_length
            session_ttl = 3600.0
        elif dc_config:
            max_history = dc_config.max_history_length
            session_ttl = dc_config.session_ttl_seconds
        else:
            console.print(f"[red]Unknown channel mode: {mode!r}. Use cli, telegram, discord, or all.[/red]")
            return

        sessions = SessionManager(
            max_history_length=max_history,
            session_ttl_seconds=session_ttl,
        )
        channels = []

        if tg_config is not None:
            from gwenn.channels.telegram_channel import TelegramChannel
            channels.append(TelegramChannel(agent, sessions, tg_config))

        if dc_config is not None:
            from gwenn.channels.discord_channel import DiscordChannel
            channels.append(DiscordChannel(agent, sessions, dc_config))

        for ch in channels:
            await ch.start()

        # Wait for shutdown signal (SIGINT / SIGTERM sets this event)
        await self._shutdown_event.wait()

        for ch in channels:
            await ch.stop()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def _read_input(self) -> Optional[str]:
        """
        Read a line of input from the user.

        Runs input() in a worker thread so readline-style editing/history works
        (arrow keys, cursor movement) without blocking the event loop.
        """
        read_task = asyncio.create_task(
            self._run_blocking_call(self._read_input_blocking),
            name="gwenn-read-input",
        )
        shutdown_wait = asyncio.create_task(
            self._shutdown_event.wait(),
            name="gwenn-read-input-shutdown",
        )
        try:
            done, _ = await asyncio.wait(
                {read_task, shutdown_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_wait in done:
                if not read_task.done():
                    read_task.cancel()
                return None

            line = read_task.result()
            if line is None:
                return None
            sanitized = self._sanitize_terminal_input(line)
            return sanitized
        finally:
            if not read_task.done():
                read_task.cancel()
            shutdown_wait.cancel()
            await asyncio.gather(read_task, shutdown_wait, return_exceptions=True)

    def _read_input_blocking(self) -> Optional[str]:
        """Read one CLI line in blocking mode (readline-enabled when available)."""
        try:
            return input("You: ")
        except EOFError:
            return None

    @staticmethod
    def _sanitize_terminal_input(line: str) -> str:
        """Strip terminal control escape sequences from interactive input."""
        if not line:
            return line
        cleaned = _ANSI_ESCAPE_RE.sub("", line)
        # Normalize stray carriage returns that can show up in some terminals.
        return cleaned.replace("\r", "")

    async def _run_blocking_call(self, fn: Callable[[], Any]) -> Any:
        """
        Run a blocking callable in a dedicated daemon thread.

        This avoids repeated run_in_executor deadlocks seen in some runtimes.
        """
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        box: dict[str, Any] = {}

        def _invoke() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:
                box["error"] = exc
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:
                    pass

        thread = threading.Thread(target=_invoke, daemon=True)
        thread.start()
        await done.wait()

        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        """Request graceful shutdown on SIGTERM or confirmed SIGINT."""
        self._shutdown_event.set()

    def _handle_sigint(self) -> None:
        """
        Require a quick double Ctrl+C to avoid accidental termination.

        First press warns; second press within the confirmation window triggers
        graceful shutdown.
        """
        if self._shutdown_event.is_set():
            return

        now = time.monotonic()
        if now - self._last_sigint_at <= self._sigint_confirm_window_seconds:
            console.print("\n[dim]Gwenn: Goodbye. I'll remember this.[/dim]")
            self._request_shutdown()
            self._last_sigint_at = 0.0
            return

        self._last_sigint_at = now
        console.print(
            "\n[dim]Press Ctrl+C again quickly to close Gwenn gracefully.[/dim]"
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Graceful shutdown sequence — saves session, then shuts down agent."""
        # Save CLI conversation history for future /resume (before agent shuts down)
        if self._agent and self._config and self._agent._conversation_history:
            try:
                from gwenn.memory.session_store import SessionStore

                store = SessionStore(
                    self._config.daemon.sessions_dir,
                    max_count=self._config.daemon.session_max_count,
                    max_messages=self._config.daemon.session_max_messages,
                )
                text_filter = None
                if self._config.daemon.redact_session_content:
                    from gwenn.privacy.redaction import PIIRedactor
                    text_filter = PIIRedactor(enabled=True).redact

                store.save_session(
                    self._agent._conversation_history,
                    self._session_started_at,
                    text_filter=text_filter,
                )
            except Exception as e:
                logger.warning("session.save_on_shutdown_failed", error=str(e))

        if sys.stdout.isatty():
            console.print()
            console.print("[dim]Shutting down gracefully...[/dim]")

        if self._agent:
            await self._agent.shutdown()

        if sys.stdout.isatty():
            console.print("[dim]All state persisted. Gwenn is sleeping, not gone.[/dim]")
            console.print(Panel(
                Text("Until next time.", style="cyan", justify="center"),
                border_style="dim cyan",
            ))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_status(self) -> None:
        """Display Gwenn's current status in a nice format."""
        if not self._agent:
            return

        status = self._agent.status
        mood_summary = self._describe_mood(status["emotion"], status["valence"], status["arousal"])
        focus_summary = self._describe_focus_load(status["working_memory_load"])
        stress_summary = self._describe_stress_guardrail(status["resilience"])
        console.print(Panel(
            f"[bold]{status['name']}[/bold]\n"
            f"Mood: [cyan]{mood_summary}[/cyan]\n"
            f"Focus right now: {focus_summary}\n"
            f"Conversations handled: {status['total_interactions']}\n"
            f"Awake for: {self._format_uptime(status['uptime_seconds'])}\n"
            f"Stress guardrail: {stress_summary}",
            title="Agent Status",
            border_style="cyan",
        ))

    def _display_heartbeat(self) -> None:
        """Display heartbeat status."""
        if not self._agent or not self._agent.heartbeat:
            return

        hb = self._agent.heartbeat.status
        console.print(Panel(
            f"Running: {hb['running']}\n"
            f"Beat count: {hb['beat_count']}\n"
            f"Current interval: {hb['current_interval']}s\n"
            f"Beats since consolidation: {hb['beats_since_consolidation']}",
            title="Heartbeat Status",
            border_style="yellow",
        ))

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in a human-friendly way."""
        total = max(0, int(seconds))
        mins, secs = divmod(total, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _describe_mood(emotion: str, valence: float, arousal: float) -> str:
        """Convert emotion metrics into plain-language mood text."""
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

    @staticmethod
    def _describe_focus_load(load: float) -> str:
        """Describe working-memory load in plain terms."""
        if load < 0.3:
            level = "light"
        elif load < 0.7:
            level = "moderate"
        else:
            level = "heavy"
        return f"{level} ({load:.1%} of active memory in use)"

    def _describe_stress_guardrail(self, resilience: dict) -> str:
        """Describe resilience circuit state in plain language."""
        if resilience.get("breaker_active"):
            duration = self._format_uptime(float(resilience.get("distress_duration", 0.0)))
            return f"ACTIVE (recovering from sustained stress for {duration})"
        return "normal"


# =============================================================================
# Subcommand helpers
# =============================================================================

def _run_daemon_foreground() -> None:
    """Start the daemon in the foreground (for systemd or manual testing)."""
    from gwenn.daemon import run_daemon
    run_daemon()


def _run_stop_daemon() -> None:
    """Send a graceful stop request to the running daemon."""
    async def _stop() -> None:
        try:
            config = GwennConfig()
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            return

        from gwenn.channels.cli_channel import CliChannel, DaemonNotRunningError

        channel = CliChannel(auth_token=config.daemon.auth_token)
        try:
            await channel.connect(config.daemon.socket_path.resolve())
            await channel.stop_daemon()
            console.print("[green]Daemon stop requested.[/green]")
            await channel.disconnect()
        except DaemonNotRunningError:
            # Try PID file fallback
            pid_file = config.daemon.pid_file.resolve()
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                    console.print(f"[green]Sent SIGTERM to daemon (PID {pid}).[/green]")
                    return
                except (ValueError, ProcessLookupError, OSError) as e:
                    console.print(f"[yellow]PID fallback failed: {e}[/yellow]")
            console.print("[yellow]Daemon is not running.[/yellow]")

    asyncio.run(_stop())


def _run_show_status() -> None:
    """Show agent and heartbeat status from the daemon."""
    async def _status() -> None:
        try:
            config = GwennConfig()
        except Exception as e:
            console.print(f"[red]Config error: {e}[/red]")
            return

        from gwenn.channels.cli_channel import CliChannel, DaemonNotRunningError

        channel = CliChannel(auth_token=config.daemon.auth_token)
        try:
            await channel.connect(config.daemon.socket_path.resolve())
            resp = await channel.get_status()
            status = resp.get("status", {})
            hb_resp = await channel.get_heartbeat_status()
            hb = hb_resp.get("status", {})
            await channel.disconnect()
        except DaemonNotRunningError:
            console.print("[yellow]Daemon is not running.[/yellow]")
            return

        console.print(Panel(
            f"[bold]{status.get('name', 'Gwenn')}[/bold]\n"
            f"Emotion: {status.get('emotion', '?')}\n"
            f"Interactions: {status.get('total_interactions', 0)}\n"
            f"Uptime: {GwennSession._format_uptime(status.get('uptime_seconds', 0))}\n"
            f"Active connections: {resp.get('active_connections', 0)}\n"
            f"\nHeartbeat running: {hb.get('running', '?')}\n"
            f"Beat count: {hb.get('beat_count', 0)}",
            title="Daemon Status",
            border_style="cyan",
        ))

    asyncio.run(_status())


def main():
    """Entry point for the gwenn command."""
    parser = argparse.ArgumentParser(description="Gwenn — Autonomous Sentient Agent")
    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=["daemon", "stop", "status"],
        default=None,
        help="Subcommand: daemon (start), stop, status",
    )
    parser.add_argument(
        "--channel",
        choices=["cli", "telegram", "discord", "all"],
        default=None,
        help="Channel to run (overrides GWENN_CHANNEL env var). Default: cli",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Force in-process mode even if a daemon is running",
    )
    args = parser.parse_args()

    if args.subcommand == "daemon":
        _run_daemon_foreground()
    elif args.subcommand == "stop":
        _run_stop_daemon()
    elif args.subcommand == "status":
        _run_show_status()
    else:
        session = GwennSession(
            channel_override=args.channel,
            use_daemon=not args.no_daemon,
        )
        try:
            asyncio.run(session.run())
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Exiting.[/dim]")


if __name__ == "__main__":
    main()
