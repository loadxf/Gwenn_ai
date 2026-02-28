"""Configuration commands — get, set, unset, init, validate, list."""

from __future__ import annotations

import math

import click

from gwenn.config_file import (
    delete_value,
    find_config,
    generate_template,
    get_value,
    load_config,
    set_value,
    write_config,
)


@click.group("config", invoke_without_command=True)
@click.pass_context
def config_group(ctx: click.Context) -> None:
    """Manage Gwenn configuration (gwenn.toml)."""
    if ctx.invoked_subcommand is None:
        # Default: list all config
        _list_config()


@config_group.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value by dotted key."""
    path = find_config()
    if not path:
        raise click.ClickException("No gwenn.toml found")
    data = load_config(path)
    try:
        value = get_value(data, key)
        click.echo(repr(value))
    except KeyError:
        raise click.ClickException(f"Key not found: {key}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value by dotted key."""
    from pathlib import Path

    path = find_config()
    if path is None:
        path = Path("gwenn.toml")

    data = load_config(path) if path.is_file() else {}

    # Type coercion
    parsed = _parse_value(value)

    set_value(data, key, parsed)
    write_config(path, data)
    click.echo(f"Set {key} = {parsed!r} in {path}")


@config_group.command("unset")
@click.argument("key")
def config_unset(key: str) -> None:
    """Remove a configuration override."""
    path = find_config()
    if not path:
        raise click.ClickException("No gwenn.toml found")
    data = load_config(path)
    try:
        delete_value(data, key)
        write_config(path, data)
        click.echo(f"Removed {key} from {path}")
    except KeyError:
        raise click.ClickException(f"Key not found: {key}")


@config_group.command("init")
def config_init() -> None:
    """Generate a gwenn.toml template in the current directory."""
    from pathlib import Path

    target = Path("gwenn.toml")
    if target.exists():
        raise click.ClickException(f"{target} already exists")
    target.write_text(generate_template())
    click.echo(f"Created {target}")


@config_group.command("validate")
def config_validate() -> None:
    """Validate the current configuration."""
    try:
        from gwenn.config import GwennConfig

        config = GwennConfig()
        click.echo("Configuration is valid.")
    except ValueError as e:
        err = str(e)
        if "api_key" in err.lower() or "anthropic" in err.lower():
            click.echo(f"Warning: {err}")
            click.echo("Configuration structure is valid (API key not set).")
        else:
            raise click.ClickException(f"Invalid configuration: {e}")
    except Exception as e:
        raise click.ClickException(f"Invalid configuration: {e}")


def _list_config() -> None:
    """List all configuration from gwenn.toml."""
    path = find_config()
    data = load_config(path) if path else {}
    click.echo(f"TOML file: {path or '(none)'}")
    if data:
        for section, values in sorted(data.items()):
            if isinstance(values, dict):
                for k, v in sorted(values.items()):
                    click.echo(f"  {section}.{k} = {v!r}")
            else:
                click.echo(f"  {section} = {values!r}")
    else:
        click.echo("  (no TOML overrides)")


def _parse_value(raw: str):
    """Parse a string value into the appropriate Python type."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        val = int(raw)
        return val
    except ValueError:
        pass
    try:
        val = float(raw)
        if not math.isfinite(val):
            raise click.ClickException(f"Invalid float value: {raw}")
        return val
    except ValueError:
        pass
    return raw
