"""Subagent management commands — list, inspect, spawn, stop."""

from __future__ import annotations

import json as json_mod

import click

from gwenn.cli.app import async_cmd


@click.group("agents")
def agents_group() -> None:
    """Manage subagents and swarms."""
    pass


@agents_group.command("list")
@click.option("--all", "show_all", is_flag=True, help="Include completed")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
@async_cmd
async def agents_list(ctx: click.Context, show_all: bool, json_output: bool) -> None:
    """List active subagents."""
    from gwenn.cli.connection import DaemonConnection, DaemonNotRunning

    from gwenn.config import DaemonConfig

    try:
        config = DaemonConfig()
    except Exception:
        config = None

    host = getattr(config, "gateway_host", "127.0.0.1") if config else "127.0.0.1"
    port = getattr(config, "gateway_port", 18900) if config else 18900

    conn = DaemonConnection(host=str(host), port=int(port))
    try:
        await conn.connect()
    except DaemonNotRunning:
        click.echo("Gwenn daemon is not running.")
        return

    try:
        result = await conn.rpc("runtime_info")
        if json_output:
            click.echo(json_mod.dumps(result, indent=2))
        else:
            click.echo("Subagent management requires orchestration module.")
            click.echo("Use 'gwenn status --verbose' for current agent info.")
    finally:
        await conn.disconnect()
