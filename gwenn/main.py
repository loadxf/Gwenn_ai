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
import functools
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
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from gwenn.channels.formatting import (
    describe_focus_load,
    describe_mood,
    describe_stress_guardrail,
    format_uptime,
)

from gwenn.agent import SentientAgent
from gwenn.api.claude import CognitiveEngineInitError
from gwenn.config import GwennConfig

# Ensure input() uses readline-backed line editing/history when available.
# Without this import, some environments echo arrow keys as raw "^[[A" text.
try:  # pragma: no cover - platform-dependent optional module
    import readline  # noqa: F401
except Exception:  # pragma: no cover
    readline = None

try:  # pragma: no cover - platform-dependent optional module
    import termios as _termios
except Exception:  # pragma: no cover
    _termios = None


def _log_redact(text: str) -> str:
    """PII redactor for log fields (always enabled).

    Uses a module-level singleton created on first call.  The
    ``functools.lru_cache`` wrapper makes the initialisation both lazy
    **and** thread-safe — no global mutable state required.
    """
    return _get_log_redactor().redact(text)


@functools.lru_cache(maxsize=1)
def _get_log_redactor():  # noqa: ANN202
    from gwenn.privacy.redaction import PIIRedactor

    return PIIRedactor(enabled=True)


def _redact_sensitive_fields(logger, method_name, event_dict):
    """
    Structlog processor that redacts sensitive fields from log output.

    Prevents user messages, episode content, and other personal data from
    appearing in plaintext in log files. Active in all log modes to ensure
    privacy by default. PII tokens are replaced before truncation so that
    full patterns are never written to disk.
    """
    sensitive_keys = {"content", "user_message", "thought", "note", "query"}
    max_display_len = 80

    for key in sensitive_keys:
        if key in event_dict:
            val = event_dict[key]
            if isinstance(val, str):
                val = _log_redact(val)
                if len(val) > max_display_len:
                    val = val[:max_display_len] + "... [truncated]"
                event_dict[key] = val

    return event_dict


_logging_configured = False


def configure_logging() -> None:
    """Configure structlog and standard-library logging for Gwenn entry points.

    Safe to call more than once — subsequent calls are no-ops.  Both
    ``main()`` and ``run_daemon()`` should invoke this before creating
    any loggers to ensure PII redaction and consistent formatting.
    """
    global _logging_configured  # noqa: PLW0603
    if _logging_configured:
        return
    _logging_configured = True

    logging.basicConfig(format="%(message)s", level=logging.WARNING)

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


# Eagerly configure when main.py is the entry point (python -m gwenn.main).
# Other callers (daemon, tests) should call configure_logging() explicitly.
configure_logging()

logger = structlog.get_logger(__name__)
console = Console()

# ANSI control-sequence matcher used to sanitize any raw escape text that
# still slips through on terminals without full line-edit support.
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")


def _is_nonfatal_channel_start_error(exc: Exception) -> bool:
    """Return True for known channel startup errors we should render nicely."""
    if isinstance(exc, ImportError):
        return True
    err_type = type(exc).__name__
    err_mod = type(exc).__module__
    if err_type == "InvalidToken" and err_mod.startswith("telegram"):
        return True
    if err_type == "LoginFailure" and err_mod.startswith("discord"):
        return True
    return False


def _redact_channel_error(message: str) -> str:
    return _TELEGRAM_BOT_TOKEN_RE.sub("[REDACTED_TELEGRAM_TOKEN]", message or "")
_OUTPUT_STYLE_CHOICES = ("balanced", "brief", "detailed")
_SLASH_COMMANDS = (
    "/help",
    "/status",
    "/heartbeat",
    "/resume",
    "/new",
    "/model",
    "/config",
    "/output-style",
    "/plan",
    "/agents",
    "/skills",
    "/stats",
    "/mcp",
    "/exit",
)


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
        self._output_style = "balanced"
        self._slash_completion_matches: list[str] = []
        self._stdin_term_attrs: Any = None
        self._capture_terminal_state()
        self._configure_readline_completion()

    @staticmethod
    def _startup_steps_template() -> list[dict[str, str]]:
        """Return the ordered startup phases shown in the startup panel."""
        return [
            {"key": "fabric", "label": "Loading Gwenn's neural fabric", "state": "pending"},
            {"key": "wake", "label": "Waking Gwenn up", "state": "pending"},
            {"key": "memory", "label": "Loading Gwenn's memories and identity", "state": "pending"},
            {
                "key": "heartbeat",
                "label": "Loading Gwenn's autonomous heartbeat",
                "state": "pending",
            },
        ]

    def _build_startup_state(self) -> dict[str, Any]:
        """Create mutable startup UI state for the live panel renderer."""
        return {
            "steps": self._startup_steps_template(),
            "model": None,
            "data_dir": None,
            "status": None,
            "ready_lines": [],
            "error_hint": None,
        }

    @staticmethod
    def _set_startup_step(
        startup_state: dict[str, Any],
        step_key: str,
        state: str,
    ) -> None:
        """Set a startup step state (`pending|running|done|error`)."""
        for step in startup_state.get("steps", []):
            if step.get("key") == step_key:
                step["state"] = state
                return

    def _render_startup_panel(self, startup_state: dict[str, Any]) -> Panel:
        """Render one-frame startup panel with spinners/checkmarks per phase."""
        rows = Table.grid(padding=(0, 1))
        rows.add_column(width=2)
        rows.add_column(ratio=1)

        for step in startup_state.get("steps", []):
            step_state = str(step.get("state", "pending"))
            label = str(step.get("label", ""))
            if step_state == "running":
                indicator = Spinner("dots", style="cyan")
                text = Text(label, style="cyan")
            elif step_state == "done":
                indicator = Text("✓", style="bold green")
                text = Text(label, style="green")
            elif step_state == "error":
                indicator = Text("✗", style="bold red")
                text = Text(label, style="red")
            else:
                indicator = Text("•", style="dim")
                text = Text(label, style="dim")
            rows.add_row(indicator, text)

        body: list[Any] = [rows]
        model = startup_state.get("model")
        data_dir = startup_state.get("data_dir")
        if model or data_dir:
            details = Table.grid(padding=(0, 1))
            details.add_column(width=2)
            details.add_column(ratio=1)
            if model:
                details.add_row(
                    Text("✓", style="green"), Text(f"Neural fabric model: {model}", style="dim")
                )
            if data_dir:
                details.add_row(
                    Text("✓", style="green"), Text(f"Neural directory: {data_dir}", style="dim")
                )
            body.extend([Text(""), details])

        status = startup_state.get("status")
        if isinstance(status, dict) and status:
            mood_summary = describe_mood(
                status.get("emotion", "neutral"),
                float(status.get("valence", 0.0)),
                float(status.get("arousal", 0.5)),
            )
            focus_summary = describe_focus_load(float(status.get("working_memory_load", 0.0)))
            stress_summary = describe_stress_guardrail(status.get("resilience", {}))
            status_lines = [
                f"{status.get('name', 'Gwenn')}",
                f"Mood: {mood_summary}",
                f"Focus right now: {focus_summary}",
                f"Conversations handled: {status.get('total_interactions', 0)}",
                f"Awake for: {format_uptime(float(status.get('uptime_seconds', 0.0)))}",
                f"Stress guardrail: {stress_summary}",
            ]
            status_table = Table.grid(padding=(0, 1))
            status_table.add_column(width=2)
            status_table.add_column(ratio=1)
            status_table.add_row(Text("✓", style="green"), Text("Current status", style="cyan"))
            for line in status_lines:
                status_table.add_row(Text("", style="dim"), Text(line))
            body.extend([Text(""), status_table])

        ready_lines = startup_state.get("ready_lines", [])
        if ready_lines:
            ready_table = Table.grid(padding=(0, 1))
            ready_table.add_column(width=2)
            ready_table.add_column(ratio=1)
            for i, line in enumerate(ready_lines):
                style = "green" if i == 0 else "dim"
                icon = Text("✓", style="green") if i == 0 else Text("•", style="dim")
                ready_table.add_row(icon, Text(str(line), style=style))
            body.extend([Text(""), ready_table])

        error_hint = startup_state.get("error_hint")
        if error_hint:
            body.extend([Text(""), Text(str(error_hint), style="yellow")])

        return Panel(
            Group(*body),
            title=Text("Gwenn.ai Terminal", style="bold cyan", justify="center"),
            subtitle="Created by Jayden & Dad - A father/son team | https://gwenn.ai",
            border_style="cyan",
        )

    def _refresh_startup_live(
        self, live: Optional[Live], startup_state: Optional[dict[str, Any]]
    ) -> None:
        """Push an updated startup panel frame to the live display."""
        if live is None or startup_state is None:
            return
        live.update(self._render_startup_panel(startup_state))

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

        startup_live: Optional[Live] = None
        startup_state: Optional[dict[str, Any]] = None
        if sys.stdout.isatty():
            startup_state = self._build_startup_state()
            startup_live = Live(
                self._render_startup_panel(startup_state),
                console=console,
                refresh_per_second=12,
                transient=False,
            )
            startup_live.start()

        # ---- PHASE 1: CONFIGURATION ----
        self._set_startup_step(startup_state or {}, "fabric", "running")
        self._refresh_startup_live(startup_live, startup_state)

        try:
            config = GwennConfig()
            self._config = config
        except Exception as e:
            self._set_startup_step(startup_state or {}, "fabric", "error")
            if startup_state is not None:
                startup_state["error_hint"] = (
                    "Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN/CLAUDE_CODE_OAUTH_TOKEN in .env, "
                    "or log in with Claude Code."
                )
                self._refresh_startup_live(startup_live, startup_state)
            if startup_live is not None:
                startup_live.stop()
            if sys.stdout.isatty():
                console.print(f"[red]Neural fabric error: {e}[/red]")
            else:
                logger.error("session.config_error", error=str(e))
            sys.exit(1)

        self._set_startup_step(startup_state or {}, "fabric", "done")
        if startup_state is not None:
            startup_state["model"] = config.claude.model
            startup_state["data_dir"] = str(config.memory.data_dir)
        self._refresh_startup_live(startup_live, startup_state)
        channel_mode = self._channel_override or config.channel.channel

        # ---- PHASE 2: CREATION ----
        self._set_startup_step(startup_state or {}, "wake", "running")
        self._refresh_startup_live(startup_live, startup_state)
        try:
            self._agent = SentientAgent(config)
        except CognitiveEngineInitError as e:
            self._set_startup_step(startup_state or {}, "wake", "error")
            self._refresh_startup_live(startup_live, startup_state)
            if startup_live is not None:
                startup_live.stop()
            if sys.stdout.isatty():
                console.print(f"[red]Startup error: {e}[/red]")
            else:
                logger.error("session.agent_init_error", error=str(e))
            sys.exit(1)
        self._set_startup_step(startup_state or {}, "wake", "done")
        self._refresh_startup_live(startup_live, startup_state)

        # ---- PHASE 3: INITIALIZATION (Loading memories, waking up) ----
        self._set_startup_step(startup_state or {}, "memory", "running")
        self._refresh_startup_live(startup_live, startup_state)
        try:
            await self._agent.initialize()
            await self._run_first_startup_onboarding_if_needed(channel_mode)
        except Exception:
            self._set_startup_step(startup_state or {}, "memory", "error")
            self._refresh_startup_live(startup_live, startup_state)
            if startup_live is not None:
                startup_live.stop()
            raise
        self._set_startup_step(startup_state or {}, "memory", "done")
        self._refresh_startup_live(startup_live, startup_state)

        # ---- PHASE 4: IGNITION (Starting heartbeat) ----
        self._set_startup_step(startup_state or {}, "heartbeat", "running")
        self._refresh_startup_live(startup_live, startup_state)
        try:
            await self._agent.start()
        except Exception:
            self._set_startup_step(startup_state or {}, "heartbeat", "error")
            self._refresh_startup_live(startup_live, startup_state)
            if startup_live is not None:
                startup_live.stop()
            raise
        self._set_startup_step(startup_state or {}, "heartbeat", "done")

        # Final startup state in the single panel.
        if startup_state is not None:
            startup_state["status"] = self._agent.status
            if channel_mode == "cli":
                startup_state["ready_lines"] = [
                    "Gwenn is awake. Type your message, or '/exit' to close.",
                    "Press Ctrl+C twice quickly to close gracefully.",
                    "Type / and press Tab to see matching slash commands.",
                ]
            else:
                startup_state["ready_lines"] = [
                    f"Gwenn is awake. Running in channel mode: {channel_mode}.",
                    "Press Ctrl+C twice quickly to close the CLI session.",
                ]
            self._refresh_startup_live(startup_live, startup_state)
        elif sys.stdout.isatty():
            self._display_status()

        if startup_live is not None:
            startup_live.stop()

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
                if sys.stdout.isatty() and startup_state is None:
                    console.print()
                    console.print(
                        "[green]Gwenn is awake. Type your message, or '/exit' to close.[/green]"
                    )
                    console.print("[dim]Press Ctrl+C twice quickly to close gracefully.[/dim]")
                    console.print("[dim]Type / and press Tab to see matching slash commands.[/dim]")
                    console.print()
                await self._interaction_loop()
            else:
                if sys.stdout.isatty() and startup_state is None:
                    console.print()
                    console.print(
                        f"[green]Gwenn is awake. Running in channel mode: {channel_mode}[/green]"
                    )
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

        daemon_live: Optional[Live] = None
        daemon_state: Optional[dict[str, Any]] = None

        # Load config minimally to get socket path
        try:
            config = GwennConfig()
            self._config = config
        except Exception:
            return False

        socket_path = config.daemon.socket_path.resolve()
        if not socket_path.exists():
            return False

        if sys.stdout.isatty():
            daemon_state = self._build_startup_state()
            daemon_state["steps"] = [
                {"key": "fabric", "label": "Loading Gwenn's AI", "state": "pending"},
                {"key": "connect", "label": "Connecting to Gwenn's brain", "state": "pending"},
            ]
            daemon_live = Live(
                self._render_startup_panel(daemon_state),
                console=console,
                refresh_per_second=12,
                transient=False,
            )
            daemon_live.start()
            self._set_startup_step(daemon_state, "fabric", "running")
            daemon_state["model"] = config.claude.model
            daemon_state["data_dir"] = str(config.memory.data_dir)
            self._set_startup_step(daemon_state, "fabric", "done")
            self._set_startup_step(daemon_state, "connect", "running")
            self._refresh_startup_live(daemon_live, daemon_state)

        channel = CliChannel(auth_token=config.daemon.auth_token)
        try:
            await channel.connect(socket_path)
        except DaemonNotRunningError:
            if daemon_state is not None:
                self._set_startup_step(daemon_state, "connect", "error")
                self._refresh_startup_live(daemon_live, daemon_state)
            if daemon_live is not None:
                daemon_live.stop()
            return False

        if daemon_state is not None:
            self._set_startup_step(daemon_state, "connect", "done")
            daemon_state["ready_lines"] = [
                "Connected to Gwenn's daemon. Type your message, or '/exit' to close.",
                "Press Ctrl+C twice quickly to close gracefully.",
                "Type / and press Tab to see matching slash commands.",
            ]
            self._refresh_startup_live(daemon_live, daemon_state)
        if daemon_live is not None:
            daemon_live.stop()

        try:
            await self._daemon_interaction_loop(channel)
        finally:
            await channel.disconnect()
            self._restore_terminal_state()

        return True

    async def _daemon_interaction_loop(self, channel: Any) -> None:
        """Interactive loop that talks to Gwenn's daemon over the Unix socket."""
        _CONN_ERRORS = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)

        while True:
            user_input = await self._read_input()
            if user_input is None:
                break

            stripped = user_input.strip().lower()
            command = self._parse_slash_command(user_input)

            if command is not None:
                action = await self._handle_daemon_command(command, channel)
                if action == "exit":
                    break
                if action == "disconnect":
                    break
                continue

            if stripped in ("quit", "exit", "bye"):
                console.print("[dim]Gwenn: Bye for now.[/dim]")
                break

            elif not stripped:
                continue

            # Regular chat
            console.print()
            with console.status("[cyan]Gwenn is thinking...[/cyan]"):
                try:
                    resp = await channel.chat(self._apply_output_style_to_message(user_input))
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
                console.print(
                    f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]:"
                )
                console.print(Markdown(text))

            console.print()

    async def _read_raw_input(self, prompt: str) -> Optional[str]:
        """Read one line with a custom prompt (used for /resume session selection)."""
        try:
            input_prompt = self._make_input_prompt(prompt, "", ansi_color="2")
            return await self._run_blocking_call(
                lambda: self._read_line_blocking(input_prompt),
            )
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
        console.print(
            Panel(
                "First-time setup so I can tailor how I help you.\n"
                "You can keep answers short. Press Enter to skip any question.",
                title="Welcome",
                border_style="cyan",
            )
        )

        name = await self._prompt_startup_input("Your name (or what I should call you): ")
        role = await self._prompt_startup_input(
            "My primary role for you (e.g., coding partner, assistant, coach): "
        )
        needs = await self._prompt_startup_input("Top things you want me to help with right now: ")
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
            console.print(
                "[dim]First-time setup skipped. I can learn your preferences over time.[/dim]"
            )
            return

        self._agent.apply_startup_onboarding(profile, user_id="default_user")
        console.print("[green]Setup saved. I will use this as ongoing guidance.[/green]")
        console.print()

    async def _prompt_startup_input(self, prompt: str) -> str:
        """Prompt for one startup onboarding field without blocking heartbeat."""
        try:
            input_prompt = self._make_input_prompt(prompt, "", ansi_color="36")
            response = await self._run_blocking_call(
                lambda: self._read_line_blocking(input_prompt),
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        if response is None:
            return ""
        return response.strip()

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    @staticmethod
    def _matching_slash_commands(text: str) -> list[str]:
        """Return slash commands matching the current typed prefix."""
        prefix = (text or "").strip().lower()
        if not prefix.startswith("/"):
            return []
        if prefix == "/":
            return list(_SLASH_COMMANDS)
        return [cmd for cmd in _SLASH_COMMANDS if cmd.startswith(prefix)]

    def _slash_command_completer(self, text: str, state: int) -> Optional[str]:
        """Readline completer callback for slash commands."""
        if readline is None:
            return None
        if state == 0:
            try:
                line_buffer = readline.get_line_buffer()
            except Exception:
                line_buffer = ""
            stripped = line_buffer.lstrip()
            try:
                begidx = int(readline.get_begidx())
            except Exception:
                begidx = 0
            leading_ws = len(line_buffer) - len(stripped)
            if stripped and not stripped.startswith("/"):
                self._slash_completion_matches = []
            elif begidx > leading_ws:
                self._slash_completion_matches = []
            else:
                self._slash_completion_matches = self._matching_slash_commands(text)
        if state < len(self._slash_completion_matches):
            return self._slash_completion_matches[state]
        return None

    def _configure_readline_completion(self) -> None:
        """Install readline completion for slash commands when available."""
        if readline is None:
            return
        try:
            readline.parse_and_bind("tab: complete")
        except Exception:
            pass
        for bind_spec in (
            "set show-all-if-ambiguous on",
            "set show-all-if-unmodified on",
            "set completion-query-items 200",
        ):
            try:
                readline.parse_and_bind(bind_spec)
            except Exception:
                pass
        try:
            delims = readline.get_completer_delims()
            updated_delims = delims.replace("/", "").replace("-", "")
            if updated_delims != delims:
                readline.set_completer_delims(updated_delims)
        except Exception:
            pass
        try:
            readline.set_completer(self._slash_command_completer)
        except Exception:
            pass

    @staticmethod
    def _parse_slash_command(line: str) -> Optional[tuple[str, str]]:
        """Parse `/command arg...` input. Returns None for non-command messages."""
        text = line.strip()
        if not text.startswith("/") or text == "/":
            return None
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        return cmd, arg

    def _apply_output_style_to_message(self, message: str) -> str:
        """Attach per-session output-style guidance to a user message."""
        style = getattr(self, "_output_style", "balanced")
        if style == "brief":
            return (
                "Output style directive: respond briefly and directly. Keep it concise unless "
                "the user explicitly asks for more depth.\n\n"
                f"{message}"
            )
        if style == "detailed":
            return (
                "Output style directive: respond with detailed, structured depth, including "
                "clear reasoning and actionable steps.\n\n"
                f"{message}"
            )
        return message

    def _set_output_style(self, arg: str) -> None:
        """Show or update the output style used for subsequent responses."""
        if not arg:
            current = getattr(self, "_output_style", "balanced")
            console.print(
                f"[cyan]Current output style:[/cyan] [bold]{current}[/bold]\n"
                f"[dim]Available: {', '.join(_OUTPUT_STYLE_CHOICES)}[/dim]"
            )
            return
        style = arg.strip().lower()
        if style not in _OUTPUT_STYLE_CHOICES:
            console.print(
                "[yellow]Invalid style.[/yellow] "
                f"[dim]Use one of: {', '.join(_OUTPUT_STYLE_CHOICES)}[/dim]"
            )
            return
        self._output_style = style
        console.print(f"[green]Output style set to {style}.[/green]")

    def _print_help(self) -> None:
        """Print supported interactive slash commands."""
        console.print(
            "[bold]Commands[/bold]\n"
            "/help — show this command list\n"
            "/status — show current agent status\n"
            "/heartbeat — show heartbeat status\n"
            "/resume — resume a previous session\n"
            "/new — start a fresh conversation context\n"
            "/model — show active model/runtime limits\n"
            "/config — show key runtime config\n"
            "/output-style [balanced|brief|detailed] — show/set response style\n"
            "/plan <task> — ask Gwenn for a focused execution plan\n"
            "/agents — list known inter-agent connections\n"
            "/skills — list loaded skills\n"
            "/stats — show runtime/memory/tool statistics\n"
            "/mcp — show MCP server/tool status\n"
            "/exit — close the CLI session\n"
            "[dim]Legacy aliases still work: quit, exit, bye[/dim]"
        )

    def _render_status_panel(
        self, status: dict[str, Any], active_connections: Optional[int] = None
    ) -> None:
        """Render a user-facing status panel from a status payload."""
        if not status:
            console.print("[yellow]No status available.[/yellow]")
            return
        mood_summary = describe_mood(
            status.get("emotion", "neutral"),
            float(status.get("valence", 0.0)),
            float(status.get("arousal", 0.5)),
        )
        lines = [
            f"[bold]{markup_escape(status.get('name', 'Gwenn'))}[/bold]",
            f"Mood: [cyan]{markup_escape(mood_summary)}[/cyan]",
            f"Conversations handled: {status.get('total_interactions', 0)}",
            f"Awake for: {markup_escape(format_uptime(float(status.get('uptime_seconds', 0))))}",
        ]
        if active_connections is not None:
            lines.append(f"Active connections: {active_connections}")
        console.print(Panel("\n".join(lines), title="Agent Status", border_style="cyan"))

    def _render_heartbeat_panel(self, hb: dict[str, Any]) -> None:
        """Render a heartbeat panel from a heartbeat payload."""
        if not hb:
            console.print("[yellow]Heartbeat status unavailable.[/yellow]")
            return
        circuit_open = bool(hb.get("circuit_open"))
        recovery = float(hb.get("circuit_recovery_in", 0.0))
        circuit_text = f"open (recovering in {recovery:.1f}s)" if circuit_open else "closed"
        console.print(
            Panel(
                f"Running: {hb.get('running', '?')}\n"
                f"Beat count: {hb.get('beat_count', 0)}\n"
                f"Current interval: {hb.get('current_interval', 0)}s\n"
                f"Beats since consolidation: {hb.get('beats_since_consolidation', 0)}\n"
                f"Circuit breaker: {markup_escape(circuit_text)}",
                title="Heartbeat Status",
                border_style="yellow",
            )
        )

    def _show_model(self) -> None:
        """Show model and Claude runtime controls."""
        if not self._config:
            console.print("[yellow]Config not loaded.[/yellow]")
            return
        cfg = self._config.claude
        console.print(
            Panel(
                f"Model: [cyan]{markup_escape(cfg.model)}[/cyan]\n"
                f"Max tokens: {cfg.max_tokens}\n"
                f"Thinking budget: {cfg.thinking_budget}\n"
                f"Request timeout: {cfg.request_timeout_seconds:.1f}s\n"
                f"Retry max: {cfg.retry_max_retries}",
                title="Model",
                border_style="cyan",
            )
        )

    def _show_config(self) -> None:
        """Show a safe summary of key runtime config values."""
        if not self._config:
            console.print("[yellow]Config not loaded.[/yellow]")
            return
        cfg = self._config
        console.print(
            Panel(
                f"Channel: {markup_escape(cfg.channel.channel)}\n"
                f"Data dir: {markup_escape(str(cfg.memory.data_dir))}\n"
                f"Memory DB: {markup_escape(str(cfg.memory.episodic_db_path))}\n"
                f"Retrieval mode: {cfg.memory.retrieval_mode}\n"
                f"Heartbeat interval: {cfg.heartbeat.interval}s "
                f"(min {cfg.heartbeat.min_interval}s, max {cfg.heartbeat.max_interval}s)\n"
                f"Sandbox enabled: {cfg.safety.sandbox_enabled}\n"
                f"PII redaction enabled: {cfg.privacy.redaction_enabled}",
                title="Config",
                border_style="cyan",
            )
        )

    def _show_agents(self, interagent_status: Optional[dict[str, Any]] = None) -> None:
        """Show discovered inter-agent connections."""
        if interagent_status is None:
            if not self._agent:
                console.print("[yellow]Agent not initialized.[/yellow]")
                return
            interagent_status = self._agent.interagent.status
        known = (
            interagent_status.get("known_agents", {}) if isinstance(interagent_status, dict) else {}
        )
        if not known:
            console.print("[dim]No known agents yet.[/dim]")
            return
        lines = [f"[bold]{len(known)} known agent(s)[/bold]"]
        for aid, meta in sorted(known.items()):
            if not isinstance(meta, dict):
                continue
            lines.append(
                f"- {markup_escape(str(meta.get('name', aid)))} "
                f"({markup_escape(str(meta.get('relationship', 'new')))}, "
                f"messages={meta.get('messages', 0)})"
            )
        console.print(Panel("\n".join(lines), title="Agents", border_style="cyan"))

    def _show_skills(self, skills: Optional[list[dict[str, Any]]] = None) -> None:
        """Show loaded skill names."""
        if skills is None:
            if not self._agent:
                console.print("[yellow]Agent not initialized.[/yellow]")
                return
            skills = [
                {
                    "name": s.name,
                    "category": s.category,
                }
                for s in self._agent.skill_registry.all_skills()
            ]
        if not skills:
            console.print("[dim]No skills loaded.[/dim]")
            return
        lines = [f"[bold]{len(skills)} skill(s) loaded[/bold]"]
        for skill in sorted(skills, key=lambda s: str(s.get("name", ""))):
            lines.append(
                f"- {markup_escape(str(skill.get('name', 'unknown')))} "
                f"[dim]({markup_escape(str(skill.get('category', 'skill')))})[/dim]"
            )
        console.print(Panel("\n".join(lines), title="Skills", border_style="cyan"))

    def _show_stats(
        self,
        status: Optional[dict[str, Any]] = None,
        active_connections: Optional[int] = None,
        tools: Optional[dict[str, Any]] = None,
    ) -> None:
        """Show runtime telemetry and memory/tool counts."""
        if status is None:
            if not self._agent:
                console.print("[yellow]Agent not initialized.[/yellow]")
                return
            status = self._agent.status
        engine = status.get("engine_telemetry", {}) if isinstance(status, dict) else {}
        lines = [
            f"Interactions: {status.get('total_interactions', 0)}",
            f"Uptime: {markup_escape(format_uptime(float(status.get('uptime_seconds', 0.0))))}",
            f"Model calls: {engine.get('total_calls', 0)}",
            f"Input tokens: {engine.get('total_input_tokens', 0)}",
            f"Output tokens: {engine.get('total_output_tokens', 0)}",
        ]
        if active_connections is not None:
            lines.append(f"Active connections: {active_connections}")
        if self._agent:
            try:
                mem_stats = self._agent.memory_store.stats
                lines.append(
                    "Memory: "
                    f"episodes={mem_stats.get('episodes', 0)}, "
                    f"knowledge_nodes={mem_stats.get('knowledge_nodes', 0)}"
                )
            except Exception:
                pass
            lines.append(
                "Tools: "
                f"registered={self._agent.tool_registry.count}, "
                f"enabled={self._agent.tool_registry.enabled_count}"
            )
        elif tools:
            lines.append(
                "Tools: "
                f"registered={int(tools.get('registered', 0))}, "
                f"enabled={int(tools.get('enabled', 0))}"
            )
        console.print(Panel("\n".join(lines), title="Stats", border_style="cyan"))

    def _show_mcp(
        self,
        mcp_stats: Optional[dict[str, Any]] = None,
        configured_servers: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Show MCP server and discovery state."""
        if mcp_stats is None and self._agent:
            mcp_stats = getattr(getattr(self._agent, "_mcp_client", None), "stats", {})
        if configured_servers is None and self._config:
            configured_servers = self._config.mcp.get_server_list()

        stats = mcp_stats or {}
        servers = configured_servers or []
        names = [
            str(server.get("name", "unknown")) for server in servers if isinstance(server, dict)
        ]
        lines = [
            f"Configured servers: {len(servers)}",
            f"Connected servers: {stats.get('connected_servers', 0)}",
            f"Discovered tools: {stats.get('discovered_tools', 0)}",
            f"Server list: {markup_escape(', '.join(names) if names else 'none')}",
        ]
        console.print(Panel("\n".join(lines), title="MCP", border_style="cyan"))

    async def _handle_resume_inprocess(self) -> None:
        """Handle `/resume` for in-process mode."""
        from gwenn.memory.session_store import SessionStore, _format_session_time

        if not self._config or not self._agent:
            return
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
            return
        console.print("\n[bold]Previous sessions:[/bold]")
        for i, s in enumerate(sessions, 1):
            dt = _format_session_time(s["started_at"])
            preview = s.get("preview", "")[:55]
            if preview:
                console.print(
                    f"  [dim]{i}.[/dim] [cyan]{dt:<20}[/cyan] — "
                    f"[bold]{s['message_count']:>3}[/bold] messages — "
                    f'[italic dim]"{markup_escape(preview)}"[/italic dim]'
                )
            else:
                console.print(
                    f"  [dim]{i}.[/dim] [cyan]{dt:<20}[/cyan] — "
                    f"[bold]{s['message_count']:>3}[/bold] messages"
                )
        console.print()
        choice_raw = await self._read_raw_input("Resume session (number, or Enter to cancel): ")
        if not choice_raw or not choice_raw.strip():
            return
        try:
            idx = int(choice_raw.strip()) - 1
            session_id = sessions[idx]["id"]
            loaded = store.load_session(session_id)
            self._agent.load_conversation_history(loaded)
            console.print(f"[green]Resumed session ({len(loaded)} messages).[/green]\n")
        except (ValueError, IndexError, FileNotFoundError):
            console.print("[yellow]Invalid choice or session not found.[/yellow]")

    async def _handle_resume_daemon(self, channel: Any) -> str:
        """Handle `/resume` for daemon-client mode."""
        from gwenn.memory.session_store import _format_session_time

        try:
            sessions = await channel.list_sessions()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError):
            console.print("[red]Daemon connection lost.[/red]")
            return "disconnect"
        if not sessions:
            console.print("[yellow]No previous sessions found.[/yellow]")
            return "handled"
        console.print("\n[bold]Previous sessions:[/bold]")
        for i, s in enumerate(sessions, 1):
            dt = _format_session_time(s["started_at"])
            preview = s.get("preview", "")[:55]
            if preview:
                console.print(
                    f"  [dim]{i}.[/dim] [cyan]{dt:<20}[/cyan] — "
                    f"[bold]{s['message_count']:>3}[/bold] messages — "
                    f'[italic dim]"{markup_escape(preview)}"[/italic dim]'
                )
            else:
                console.print(
                    f"  [dim]{i}.[/dim] [cyan]{dt:<20}[/cyan] — "
                    f"[bold]{s['message_count']:>3}[/bold] messages"
                )
        console.print()
        choice_raw = await self._read_raw_input("Resume session (number, or Enter to cancel): ")
        if not choice_raw or not choice_raw.strip():
            return "handled"
        try:
            idx = int(choice_raw.strip()) - 1
            session_id = sessions[idx]["id"]
            count = await channel.load_session(session_id)
            console.print(f"[green]Resumed session ({count} messages).[/green]\n")
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError):
            console.print("[red]Daemon connection lost.[/red]")
            return "disconnect"
        except (ValueError, IndexError):
            console.print("[yellow]Invalid choice.[/yellow]")
        return "handled"

    async def _handle_inprocess_command(self, command: tuple[str, str]) -> str:
        """Handle a slash command in in-process CLI mode."""
        cmd, arg = command
        if cmd == "/exit":
            console.print("[dim]Gwenn: Bye for now.[/dim]")
            return "exit"
        if cmd == "/help":
            self._print_help()
            return "handled"
        if cmd == "/status":
            self._display_status()
            return "handled"
        if cmd == "/heartbeat":
            self._display_heartbeat()
            return "handled"
        if cmd == "/resume":
            await self._handle_resume_inprocess()
            return "handled"
        if cmd == "/new":
            if self._agent:
                self._agent.load_conversation_history([])
            console.print("[green]Started a new conversation context.[/green]")
            return "handled"
        if cmd == "/model":
            self._show_model()
            return "handled"
        if cmd == "/config":
            self._show_config()
            return "handled"
        if cmd == "/output-style":
            self._set_output_style(arg)
            return "handled"
        if cmd == "/plan":
            if not arg:
                console.print("[yellow]Usage:[/yellow] /plan <task>")
                return "handled"
            if not self._agent:
                console.print("[yellow]Agent not initialized.[/yellow]")
                return "handled"
            plan_prompt = (
                "Create an actionable plan for the request below. Return only numbered steps.\n\n"
                f"Request: {arg}"
            )
            console.print()
            with console.status("[cyan]Gwenn is planning...[/cyan]"):
                response = await self._agent.respond(
                    self._apply_output_style_to_message(plan_prompt)
                )
            emotion = self._agent.affect_state.current_emotion.value
            console.print(f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]:")
            console.print(Markdown(response))
            console.print()
            return "handled"
        if cmd == "/agents":
            self._show_agents()
            return "handled"
        if cmd == "/skills":
            self._show_skills()
            return "handled"
        if cmd == "/stats":
            self._show_stats()
            return "handled"
        if cmd == "/mcp":
            self._show_mcp()
            return "handled"
        console.print(f"[yellow]Unknown command:[/yellow] {markup_escape(cmd)}")
        console.print("[dim]Use /help to see available commands.[/dim]")
        return "handled"

    async def _handle_daemon_command(self, command: tuple[str, str], channel: Any) -> str:
        """Handle a slash command while connected to a daemon-backed CLI."""
        cmd, arg = command
        conn_errors = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)

        if cmd == "/exit":
            console.print("[dim]Gwenn: Bye for now.[/dim]")
            return "exit"
        if cmd == "/help":
            self._print_help()
            return "handled"
        if cmd == "/status":
            try:
                resp = await channel.get_status()
            except conn_errors:
                console.print("[red]Daemon connection lost.[/red]")
                return "disconnect"
            self._render_status_panel(resp.get("status", {}), resp.get("active_connections"))
            return "handled"
        if cmd == "/heartbeat":
            try:
                resp = await channel.get_heartbeat_status()
            except conn_errors:
                console.print("[red]Daemon connection lost.[/red]")
                return "disconnect"
            self._render_heartbeat_panel(resp.get("status", {}))
            return "handled"
        if cmd == "/resume":
            return await self._handle_resume_daemon(channel)
        if cmd == "/new":
            try:
                cleared = await channel.reset_session()
            except conn_errors:
                console.print("[red]Daemon connection lost.[/red]")
                return "disconnect"
            console.print(
                f"[green]Started a new conversation context ({cleared} messages cleared).[/green]"
            )
            return "handled"
        if cmd == "/model":
            self._show_model()
            return "handled"
        if cmd == "/config":
            self._show_config()
            return "handled"
        if cmd == "/output-style":
            self._set_output_style(arg)
            return "handled"
        if cmd == "/plan":
            if not arg:
                console.print("[yellow]Usage:[/yellow] /plan <task>")
                return "handled"
            plan_prompt = (
                "Create an actionable plan for the request below. Return only numbered steps.\n\n"
                f"Request: {arg}"
            )
            console.print()
            with console.status("[cyan]Gwenn is planning...[/cyan]"):
                try:
                    resp = await channel.chat(self._apply_output_style_to_message(plan_prompt))
                except conn_errors:
                    console.print("[red]Daemon connection lost.[/red]")
                    return "disconnect"
            if resp.get("type") == "error":
                console.print(f"[red]Daemon error: {markup_escape(resp.get('message', '?'))}[/red]")
                return "handled"
            emotion = resp.get("emotion", "neutral")
            text = resp.get("text", "")
            console.print(f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]:")
            console.print(Markdown(text))
            console.print()
            return "handled"
        if cmd in {"/agents", "/skills", "/stats", "/mcp"}:
            try:
                info = await channel.get_runtime_info()
            except conn_errors:
                console.print("[red]Daemon connection lost.[/red]")
                return "disconnect"
            status = info.get("status", {})
            if cmd == "/agents":
                self._show_agents(status.get("interagent"))
            elif cmd == "/skills":
                self._show_skills(info.get("skills"))
            elif cmd == "/stats":
                self._show_stats(
                    status=status,
                    active_connections=info.get("active_connections"),
                    tools=info.get("tools"),
                )
            elif cmd == "/mcp":
                self._show_mcp(info.get("mcp"), info.get("configured_mcp_servers"))
            return "handled"
        console.print(f"[yellow]Unknown command:[/yellow] {markup_escape(cmd)}")
        console.print("[dim]Use /help to see available commands.[/dim]")
        return "handled"

    # ------------------------------------------------------------------
    # In-process interaction loop
    # ------------------------------------------------------------------

    async def _interaction_loop(self) -> None:
        """
        The main interaction loop — where Gwenn and humans converse.

        This runs in an asyncio-compatible way, yielding control between
        user inputs so the heartbeat continues running in the background.
        """
        while not self._shutdown_event.is_set():
            try:
                # Read user input without blocking background loops or shutdown signals.
                user_input = await self._read_input()

                if user_input is None:
                    break

                self._last_sigint_at = 0.0

                # Handle special commands
                stripped = user_input.strip().lower()
                command = self._parse_slash_command(user_input)
                if command is not None:
                    action = await self._handle_inprocess_command(command)
                    if action == "exit":
                        break
                    continue
                if stripped in ("quit", "exit", "bye"):
                    console.print("[dim]Gwenn: Bye for now.[/dim]")
                    break
                elif not stripped:
                    continue

                # Generate Gwenn's response
                console.print()
                with console.status("[cyan]Gwenn is thinking...[/cyan]"):
                    response = await self._agent.respond(
                        self._apply_output_style_to_message(user_input)
                    )

                # Display the response — Markdown renders formatting; name/emotion still escaped
                emotion = self._agent.affect_state.current_emotion.value
                console.print(
                    f"[bold cyan]Gwenn[/bold cyan] [dim]({markup_escape(emotion)})[/dim]:"
                )
                console.print(Markdown(response))
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
                    "[red]Claude API request failed.[/red] [dim]See logs for details.[/dim]"
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
        from gwenn.channels.startup import build_channels, run_channels_until_shutdown

        sessions, channels = build_channels(agent, channel_list=[mode])
        if not channels:
            console.print(
                f"[red]No channels could be started for mode {mode!r}. "
                f"Check your .env configuration (bot tokens, etc.) or use: cli, telegram, discord, all.[/red]"
            )
            return

        try:
            await run_channels_until_shutdown(
                agent,
                sessions,
                channels,
                self._shutdown_event,
                continue_on_import_error=(mode == "all"),
            )
        except Exception as exc:
            if not _is_nonfatal_channel_start_error(exc):
                raise
            err_type = type(exc).__name__
            err_mod = type(exc).__module__
            if err_type == "InvalidToken" and err_mod.startswith("telegram"):
                console.print(
                    "[red]Telegram bot token was rejected by Telegram. "
                    "Check TELEGRAM_BOT_TOKEN in .env.[/red]"
                )
                return
            if err_type == "LoginFailure" and err_mod.startswith("discord"):
                console.print(
                    "[red]Discord bot login failed. Check DISCORD_BOT_TOKEN in .env "
                    "and bot permissions.[/red]"
                )
                return
            console.print(f"[red]{_redact_channel_error(str(exc))}[/red]")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def _read_input(self) -> Optional[str]:
        """
        Read a line of input from the user.

        Runs input() in a worker thread so readline-style editing/history works
        (arrow keys, cursor movement) without blocking the event loop.
        """
        if self._shutdown_event.is_set():
            return None
        prompt = self._make_input_prompt("You", ": ", ansi_color="1;32")
        read_task = asyncio.create_task(
            self._run_blocking_call(lambda: self._read_line_blocking(prompt)),
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

    @staticmethod
    def _read_line_blocking(prompt: str = "") -> Optional[str]:
        """Read one terminal line, passing *prompt* to ``input()`` directly.

        Passing the prompt to ``input()`` lets the terminal's line editor
        (readline on Unix, the native console on Windows) know the actual
        cursor column.  This prevents line-wrapping miscalculations that
        cause overlapping text and broken backspace on wrapped lines —
        especially noticeable on Windows PowerShell.
        """
        try:
            return input(prompt)
        except EOFError:
            return None

    @staticmethod
    def _make_input_prompt(text: str, suffix: str = "", *, ansi_color: str = "") -> str:
        """Build a terminal prompt string safe for ``input()``.

        On Unix with readline, ANSI escape sequences are wrapped in
        ``\\001``/``\\002`` markers so readline doesn't count them toward
        the visible prompt width.  On Windows (no readline) the raw ANSI
        codes are emitted directly — modern PowerShell renders them fine.
        """
        if not ansi_color:
            return f"{text}{suffix}"
        start = f"\033[{ansi_color}m"
        reset = "\033[0m"
        if readline is not None:
            # Readline invisible-character markers
            start = f"\001{start}\002"
            reset = f"\001{reset}\002"
        return f"{start}{text}{reset}{suffix}"

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
            console.print("\n[dim]Gwenn: Bye for now.[/dim]")
            self._request_shutdown()
            self._last_sigint_at = 0.0
            return

        self._last_sigint_at = now
        console.print("\n[dim]Press Ctrl+C again quickly to close Gwenn gracefully.[/dim]")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Graceful shutdown sequence — saves session, then shuts down agent."""
        try:
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
                        agent_redactor = getattr(self._agent, "redactor", None)
                        if agent_redactor is not None:
                            text_filter = agent_redactor.redact
                        else:
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

            if self._agent:
                if sys.stdout.isatty():
                    with console.status(
                        "[cyan]Please wait: Updating Gwenn's memory. This may take a few seconds...[/cyan]",
                        spinner="dots",
                    ):
                        await self._agent.shutdown()
                else:
                    await self._agent.shutdown()

            if sys.stdout.isatty():
                console.print("[dim]All state persisted. Gwenn is sleeping, not gone.[/dim]")
                console.print(
                    Panel(
                        Text("Until next time.", style="cyan", justify="center"),
                        border_style="dim cyan",
                    )
                )
        finally:
            self._restore_terminal_state()

    def _capture_terminal_state(self) -> None:
        """Capture the current terminal mode so it can be restored on shutdown."""
        if _termios is None or not sys.stdin.isatty():
            return
        try:
            self._stdin_term_attrs = _termios.tcgetattr(sys.stdin.fileno())
        except Exception:
            self._stdin_term_attrs = None

    def _restore_terminal_state(self) -> None:
        """Best-effort terminal mode restoration to avoid post-exit no-echo shells."""
        if _termios is None or self._stdin_term_attrs is None:
            return
        try:
            _termios.tcsetattr(
                sys.stdin.fileno(),
                _termios.TCSADRAIN,
                self._stdin_term_attrs,
            )
        except Exception as e:
            logger.debug("session.tty_restore_failed", error=str(e))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_status(self) -> None:
        """Display Gwenn's current status in a nice format."""
        if not self._agent:
            return

        status = self._agent.status
        mood_summary = describe_mood(status["emotion"], status["valence"], status["arousal"])
        focus_summary = describe_focus_load(status["working_memory_load"])
        stress_summary = describe_stress_guardrail(status["resilience"])
        console.print(
            Panel(
                f"[bold]{markup_escape(status['name'])}[/bold]\n"
                f"Mood: [cyan]{markup_escape(mood_summary)}[/cyan]\n"
                f"Focus right now: {markup_escape(focus_summary)}\n"
                f"Conversations handled: {status['total_interactions']}\n"
                f"Awake for: {markup_escape(format_uptime(status['uptime_seconds']))}\n"
                f"Stress guardrail: {markup_escape(stress_summary)}",
                title="Gwenn's current status",
                border_style="cyan",
            )
        )

    def _display_heartbeat(self) -> None:
        """Display heartbeat status."""
        if not self._agent or not self._agent.heartbeat:
            return

        hb = self._agent.heartbeat.status
        circuit_open = bool(hb.get("circuit_open"))
        recovery = float(hb.get("circuit_recovery_in", 0.0))
        circuit_text = f"open (recovering in {recovery:.1f}s)" if circuit_open else "closed"
        console.print(
            Panel(
                f"Running: {hb['running']}\n"
                f"Beat count: {hb['beat_count']}\n"
                f"Current interval: {hb['current_interval']}s\n"
                f"Beats since consolidation: {hb['beats_since_consolidation']}\n"
                f"Circuit breaker: {markup_escape(circuit_text)}",
                title="Heartbeat Status",
                border_style="yellow",
            )
        )


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

        mood_text = describe_mood(
            status.get("emotion", "neutral"),
            float(status.get("valence", 0.0)),
            float(status.get("arousal", 0.5)),
        )
        console.print(
            Panel(
                f"[bold]{markup_escape(status.get('name', 'Gwenn'))}[/bold]\n"
                f"Mood: [cyan]{markup_escape(mood_text)}[/cyan]\n"
                f"Interactions: {status.get('total_interactions', 0)}\n"
                f"Uptime: {markup_escape(format_uptime(float(status.get('uptime_seconds', 0))))}\n"
                f"Active connections: {resp.get('active_connections', 0)}\n"
                f"\nHeartbeat running: {hb.get('running', '?')}\n"
                f"Beat count: {hb.get('beat_count', 0)}",
                title="Daemon Status",
                border_style="cyan",
            )
        )

    asyncio.run(_status())


def main():
    """Entry point for the gwenn command."""
    parser = argparse.ArgumentParser(
        description="Gwenn - Genesis Woven from Evolved Neural Networks"
    )
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
