"""CLI application — Click-based command hierarchy for Gwenn.

The main CLI group and global flags. Subcommand modules register
themselves by importing and adding to the group.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import click


def async_cmd(func):
    """Decorator to run an async Click command via asyncio.run()."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@click.group(invoke_without_command=True)
@click.option("--json", "json_output", is_flag=True, help="Structured JSON output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--verbose", "-v", is_flag=True, help="Extended details")
@click.option("--no-color", is_flag=True, help="Disable ANSI colors")
@click.pass_context
def cli(ctx: click.Context, json_output: bool, quiet: bool, verbose: bool, no_color: bool) -> None:
    """Gwenn - Genesis Woven from Evolved Neural Networks."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["no_color"] = no_color

    if ctx.invoked_subcommand is None:
        # Default: launch interactive REPL
        from gwenn.cli.repl import run_repl

        run_repl(ctx.obj)


# ---------------------------------------------------------------------------
# Register subcommand modules
# ---------------------------------------------------------------------------

def _register_subcommands() -> None:
    """Import and register all subcommand groups/commands."""
    from gwenn.cli.monitoring import status_cmd, feed_cmd
    from gwenn.cli.dashboard import dashboard_cmd
    from gwenn.cli.agents import agents_group
    from gwenn.cli.channels import channels_group
    from gwenn.cli.service import install_cmd, uninstall_cmd, restart_cmd
    from gwenn.cli.config_cmd import config_group
    from gwenn.cli.doctor import doctor_cmd
    from gwenn.cli.daemon_cmd import daemon_cmd, stop_cmd

    cli.add_command(daemon_cmd)
    cli.add_command(stop_cmd)
    cli.add_command(status_cmd)
    cli.add_command(feed_cmd)
    cli.add_command(dashboard_cmd)
    cli.add_command(agents_group)
    cli.add_command(channels_group)
    cli.add_command(install_cmd)
    cli.add_command(uninstall_cmd)
    cli.add_command(restart_cmd)
    cli.add_command(config_group)
    cli.add_command(doctor_cmd)


_register_subcommands()
