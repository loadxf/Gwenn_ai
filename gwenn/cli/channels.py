"""Channel management commands — list, restart, test."""

from __future__ import annotations

import json as json_mod

import click

from gwenn.cli.app import async_cmd


@click.group("channels")
def channels_group() -> None:
    """Manage communication channels."""
    pass


@channels_group.command("list")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
@async_cmd
async def channels_list(ctx: click.Context, json_output: bool) -> None:
    """List channels and connection status."""
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
        result = await conn.rpc("status")
        if json_output:
            click.echo(json_mod.dumps(result, indent=2))
        else:
            click.echo("Channel status available via: gwenn status --verbose")
    finally:
        await conn.disconnect()
