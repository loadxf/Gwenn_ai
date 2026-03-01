# Phase 3: Structured TOML Configuration

**Risk: LOW** — Adds configuration layer without changing runtime behavior.

**Prerequisites:** Phase 2 complete (service management installed).

---

## Goal

Add `gwenn.toml` using pydantic-settings native TOML source. Keep `.env` as override.

---

## Technical Background

### pydantic-settings TOML Support

- `TomlConfigSettingsSource` was introduced in **pydantic-settings v2.2.0** (February 2023)
- The original plan stated v2.6.0+ — this was incorrect, but bumping to >=2.6.0 is still reasonable for overall improvements
- Current stable: **v2.13.1** (February 2026)
- On Python >=3.11 (our minimum), TOML reading uses stdlib `tomllib` — no extra install needed
- **Critical**: Setting `toml_file` in `SettingsConfigDict` alone does NOT activate TOML loading. You MUST override `settings_customise_sources()` and explicitly include `TomlConfigSettingsSource` in the returned tuple.

### Config Class Count

The codebase has **22 config classes** (not 26 as originally stated):
- 21 `BaseSettings` subclasses (ClaudeConfig, MemoryConfig, HeartbeatConfig, AffectConfig, ContextConfig, SafetyConfig, MCPConfig, SensoryConfig, EthicsConfig, InterAgentConfig, GoalConfig, InnerLifeConfig, MetacognitionConfig, TheoryOfMindConfig, GroqConfig, OrchestrationConfig, PrivacyConfig, TelegramConfig, DiscordConfig, SkillsConfig, ChannelConfig, DaemonConfig)
- 1 plain `GwennConfig` master class that composes the 21 above

**Important**: `GwennConfig` is NOT a `BaseSettings` subclass — it's a plain Python class whose `__init__` instantiates all 21 config classes. It cannot inherit from `GwennSettingsBase`.

---

## Loading Priority (highest wins)

1. Environment variables (OS-level)
2. `.env` file
3. `gwenn.toml`
4. Field defaults

This maps to the `settings_customise_sources()` return tuple:

```python
@classmethod
def settings_customise_sources(
    cls,
    settings_cls: type[BaseSettings],
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
) -> tuple[PydanticBaseSettingsSource, ...]:
    return (
        init_settings,           # Highest: explicit constructor args
        env_settings,            # OS environment variables
        dotenv_settings,         # .env file
        TomlConfigSettingsSource(settings_cls),  # gwenn.toml
        file_secret_settings,    # Lowest (secrets directory)
    )
```

---

## New Files

### `gwenn/config_file.py` — TOML Utilities

```python
"""TOML configuration file utilities.

Reading: uses tomllib (stdlib, Python >=3.11)
Writing: uses tomli-w (only write dependency needed)
"""

def find_config() -> Path | None:
    """Search for gwenn.toml in standard locations.

    Search order:
    1. Current working directory
    2. ~/.config/gwenn/gwenn.toml
    3. Project root (detected by .git or pyproject.toml)

    Returns None if not found.
    """

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
    # Write to .tmp file, then os.rename() for atomicity

def get_value(data: dict, dotted_key: str) -> Any:
    """Get a nested value by dotted key (e.g., 'heartbeat.interval')."""

def set_value(data: dict, dotted_key: str, value: Any) -> dict:
    """Set a nested value by dotted key. Returns modified dict."""

def generate_template(config: GwennConfig) -> str:
    """Generate annotated gwenn.toml from current config with comments."""
```

### `gwenn/templates/gwenn.toml.example` — Annotated Example Config

```toml
# Gwenn Configuration
# This file is loaded after .env. Environment variables override these values.
# See .env.example for the full list of available settings.

[claude]
# model = "claude-sonnet-4-6"
# max_tokens = 16384
# thinking_budget = 10000

[memory]
# data_dir = "./gwenn_data"
# retrieval_mode = "hybrid"

[heartbeat]
# interval = 30.0
# min_interval = 5.0
# max_interval = 120.0
# proactive_messages = false

[affect]
# arousal_ceiling = 0.85

[context]
# context_limit = 180000
# working_memory_slots = 7

[safety]
# max_tool_iterations = 25
# sandbox_enabled = true

[daemon]
# gateway_enabled = true
# gateway_host = "127.0.0.1"
# gateway_port = 18900

[telegram]
# bot_token = ""
# allowed_user_ids = []

[discord]
# bot_token = ""

[slack]
# enabled = false
# bot_token = ""
# app_token = ""

[orchestration]
# enabled = true
# max_concurrent_subagents = 5
```

### `tests/test_config_file.py`

- Test `find_config()` with various directory layouts (mock filesystem)
- Test `load_config()` with valid/invalid TOML
- Test `write_config()` atomic write (verify no partial writes on error)
- Test `get_value()` / `set_value()` with dotted keys
- Test `generate_template()` output is valid TOML
- Test full priority chain: env var > .env > gwenn.toml > default

---

## Modified Files

### `gwenn/config.py`

**Step 1:** Create `GwennSettingsBase(BaseSettings)` with TOML source:

```python
from pydantic_settings import BaseSettings, TomlConfigSettingsSource

class GwennSettingsBase(BaseSettings):
    """Base for all Gwenn config classes. Adds TOML file support."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        toml_file="gwenn.toml",  # Searched by pydantic-settings
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
```

**Step 2:** Change the 21 leaf config classes: `BaseSettings` → `GwennSettingsBase`

Each class also adds a `toml_table_header` to map to its TOML section:
```python
class HeartbeatConfig(GwennSettingsBase):
    model_config = SettingsConfigDict(
        toml_file="gwenn.toml",
        toml_table_header=("heartbeat",),
        # ... existing env_file, extra, populate_by_name ...
    )
```

**Step 3:** `GwennConfig` (the master class) remains a plain class — it is NOT changed to `GwennSettingsBase`. It continues to compose all 21 leaf configs.

**TOML section mapping:**

| Config Class | TOML Section | Example Key |
|-------------|-------------|-------------|
| `ClaudeConfig` | `[claude]` | `claude.model` |
| `MemoryConfig` | `[memory]` | `memory.data_dir` |
| `HeartbeatConfig` | `[heartbeat]` | `heartbeat.interval` |
| `AffectConfig` | `[affect]` | `affect.arousal_ceiling` |
| `ContextConfig` | `[context]` | `context.context_limit` |
| `SafetyConfig` | `[safety]` | `safety.max_tool_iterations` |
| `MCPConfig` | `[mcp]` | `mcp.servers` |
| `SensoryConfig` | `[sensory]` | `sensory.max_percepts_per_channel` |
| `EthicsConfig` | `[ethics]` | `ethics.history_size` |
| `InterAgentConfig` | `[interagent]` | `interagent.agent_id` |
| `GoalConfig` | `[goals]` | `goals.max_active` |
| `InnerLifeConfig` | `[inner_life]` | `inner_life.variety_bonus` |
| `MetacognitionConfig` | `[metacognition]` | `metacognition.max_active_concerns` |
| `TheoryOfMindConfig` | `[theory_of_mind]` | `theory_of_mind.max_models` |
| `GroqConfig` | `[groq]` | `groq.api_key` |
| `OrchestrationConfig` | `[orchestration]` | `orchestration.enabled` |
| `PrivacyConfig` | `[privacy]` | `privacy.redaction_enabled` |
| `TelegramConfig` | `[telegram]` | `telegram.bot_token` |
| `DiscordConfig` | `[discord]` | `discord.bot_token` |
| `SkillsConfig` | `[skills]` | `skills.dir` |
| `ChannelConfig` | `[channels]` | `channels.enabled` |
| `DaemonConfig` | `[daemon]` | `daemon.gateway_port` |

### `gwenn/main.py` — Add `config` Subcommand

Add to argparse choices:
```python
choices=["daemon", "stop", "status", "install", "uninstall", "restart", "config"]
```

Sub-operations via second positional arg:
```
gwenn config                 # Show current config (resolved from all sources)
gwenn config get <key>       # Get specific value (dotted key like "heartbeat.interval")
gwenn config set <key> <val> # Set value in gwenn.toml
gwenn config unset <key>     # Remove override from gwenn.toml
gwenn config init            # Generate gwenn.toml template from current values
gwenn config validate        # Validate all config sources
gwenn config list            # List all config keys with their sources
```

### `pyproject.toml`

- Bump `pydantic-settings` from `>=2.1.0` to `>=2.6.0`
- Add `"tomli-w>=1.0.0"` to dependencies

---

## Implementation Sub-Steps

```
3a. Create gwenn/config_file.py (find, load, write, get, set, generate_template)
3b. Add GwennSettingsBase to config.py, change 21 leaf classes to use it
3c. Add config subcommand to main.py
3d. Create gwenn/templates/gwenn.toml.example
3e. Bump pydantic-settings, add tomli-w in pyproject.toml
3f. Write tests
```

All in a single commit.

---

## Verification

- `gwenn config init` → creates `gwenn.toml` in CWD with annotated defaults
- `gwenn config get heartbeat.interval` → shows `30.0`
- `gwenn config set heartbeat.interval 15.0` → writes to gwenn.toml
- `GWENN_HEARTBEAT_INTERVAL=10 gwenn config get heartbeat.interval` → shows `10` (env var wins)
- `gwenn config validate` → reports all sources and any validation errors
- `gwenn config list` → shows all keys with source indicators (env/toml/default)
- Existing `.env`-based configuration continues to work unchanged
