"""CLI formatters — color helpers, state indicators, table formatting."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


def get_console(no_color: bool = False) -> Console:
    """Get a Rich Console, optionally with color disabled."""
    return Console(no_color=no_color)


def status_indicator(status: str) -> Text:
    """Map a status string to a colored indicator."""
    mapping = {
        "connected": Text("> ", style="green"),
        "active": Text("> ", style="green"),
        "running": Text("> ", style="green"),
        "idle": Text("- ", style="dim"),
        "down": Text("x ", style="red"),
        "error": Text("! ", style="yellow"),
        "disconnected": Text("x ", style="red"),
    }
    return mapping.get(status, Text("? ", style="dim"))


def format_duration(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def build_table(title: str, columns: list[str], rows: list[list[Any]]) -> Table:
    """Build a Rich table with standard styling."""
    table = Table(title=title, show_header=True, header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    return table
