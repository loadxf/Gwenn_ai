"""TOML configuration file utilities.

Reading: uses tomllib (stdlib, Python >=3.11)
Writing: uses tomli-w (only write dependency needed)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_config() -> Path | None:
    """Search for gwenn.toml in standard locations.

    Search order:
    1. Current working directory
    2. ~/.config/gwenn/gwenn.toml
    3. Project root (where the gwenn package lives)

    Returns None if not found.
    """
    # 1. Current working directory
    cwd = Path.cwd() / "gwenn.toml"
    if cwd.is_file():
        return cwd

    # 2. XDG config directory
    xdg = Path.home() / ".config" / "gwenn" / "gwenn.toml"
    if xdg.is_file():
        return xdg

    # 3. Project root
    project = _PROJECT_ROOT / "gwenn.toml"
    if project.is_file():
        return project

    return None


def load_config(path: Path) -> dict:
    """Load and parse a gwenn.toml file."""
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def write_config(path: Path, data: dict) -> None:
    """Atomic write with tempfile + rename.

    Creates parent directories if needed.
    Uses tempfile in same directory for atomic rename.
    """
    import tomli_w

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=".gwenn_config_"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            tomli_w.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _validate_dotted_key(dotted_key: str) -> list[str]:
    """Validate and split a dotted key. Raises ValueError on empty segments."""
    if not dotted_key or not dotted_key.strip():
        raise ValueError("Key must not be empty")
    keys = dotted_key.split(".")
    if any(not k for k in keys):
        raise ValueError(f"Key contains empty segments: {dotted_key!r}")
    return keys


def get_value(data: dict, dotted_key: str) -> Any:
    """Get a nested value by dotted key (e.g., 'heartbeat.interval')."""
    keys = _validate_dotted_key(dotted_key)
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(dotted_key)
        current = current[key]
    return current


def set_value(data: dict, dotted_key: str, value: Any) -> dict:
    """Set a nested value by dotted key. Returns modified dict."""
    keys = _validate_dotted_key(dotted_key)
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return data


def delete_value(data: dict, dotted_key: str) -> dict:
    """Delete a nested value by dotted key. Returns modified dict.

    Raises KeyError if the key does not exist.
    """
    keys = _validate_dotted_key(dotted_key)
    current = data
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(dotted_key)
        current = current[key]
    if not isinstance(current, dict) or keys[-1] not in current:
        raise KeyError(dotted_key)
    del current[keys[-1]]
    return data


def generate_template() -> str:
    """Generate an annotated gwenn.toml template with all sections commented out."""
    return '''\
# Gwenn Configuration
# This file is loaded after .env. Environment variables override these values.
# See .env.example for the full list of available settings.

[claude]
# model = "claude-sonnet-4-5-20250929"
# max_tokens = 8192
# thinking_budget = 16000

[memory]
# data_dir = "./gwenn_data"
# retrieval_mode = "keyword"
# working_memory_slots = 7

[heartbeat]
# interval = 30.0
# min_interval = 5.0
# max_interval = 120.0
# proactive_messages = false

[affect]
# arousal_ceiling = 0.92

[context]
# context_limit = 180000

[safety]
# max_tool_iterations = 150
# sandbox_enabled = true

[mcp]
# servers = "[]"

[sensory]
# max_percepts_per_channel = 10

[ethics]
# assessment_history_size = 100

[interagent]
# self_id = "gwenn"

[goals]
# need_decay_rate_multiplier = 1.0

[inner_life]
# variety_pressure_seconds = 300.0

[metacognition]
# max_calibration_records = 1000

[theory_of_mind]
# belief_staleness_days = 30.0

[groq]
# whisper_model = "whisper-large-v3-turbo"

[orchestration]
# enabled = true
# max_concurrent_subagents = 5

[privacy]
# redaction_enabled = false

[telegram]
# bot_token = ""
# allowed_user_ids = []

[discord]
# bot_token = ""

[slack]
# bot_token = ""
# app_token = ""

[skills]
# skills_dir = "./gwenn_skills"

[channels]
# cli_enabled = true
# telegram_enabled = false
# discord_enabled = false
# slack_enabled = false

[daemon]
# gateway_enabled = true
# gateway_host = "127.0.0.1"
# gateway_port = 18900
'''
