"""Tests for Phase 1c config additions: gateway fields, SlackConfig, ChannelConfig.slack_enabled."""

from __future__ import annotations

import pytest

from gwenn.config import ChannelConfig, DaemonConfig, GwennConfig, SlackConfig


# ---------------------------------------------------------------------------
# DaemonConfig — gateway fields
# ---------------------------------------------------------------------------


class TestDaemonConfigGateway:
    """Tests for the new gateway-related fields on DaemonConfig."""

    def test_gateway_defaults(self) -> None:
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_enabled is True
        assert cfg.gateway_host == "127.0.0.1"
        assert cfg.gateway_port == 18900
        assert cfg.legacy_socket_enabled is True
        assert cfg.mcp_server_enabled is False
        assert cfg.a2a_enabled is False
        assert cfg.heartbeat_core is True

    def test_gateway_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_PORT", "9999")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_port == 9999

    def test_gateway_port_clamped_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_PORT", "0")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_port == 1

    def test_gateway_port_clamped_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_PORT", "70000")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_port == 65535

    def test_gateway_host_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_HOST", "0.0.0.0")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_host == "0.0.0.0"

    def test_gateway_host_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_HOST", "  127.0.0.1  ")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_host == "127.0.0.1"

    def test_gateway_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_GATEWAY_ENABLED", "false")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.gateway_enabled is False

    def test_legacy_socket_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_LEGACY_SOCKET_ENABLED", "false")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.legacy_socket_enabled is False

    def test_mcp_server_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_MCP_SERVER_ENABLED", "true")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.mcp_server_enabled is True

    def test_a2a_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_A2A_ENABLED", "true")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.a2a_enabled is True

    def test_heartbeat_core_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_HEARTBEAT_CORE", "false")
        cfg = DaemonConfig(_env_file=None)
        assert cfg.heartbeat_core is False


# ---------------------------------------------------------------------------
# SlackConfig
# ---------------------------------------------------------------------------


class TestSlackConfig:
    """Tests for the new SlackConfig class."""

    def test_slack_defaults(self) -> None:
        cfg = SlackConfig(_env_file=None)
        assert cfg.bot_token is None
        assert cfg.app_token is None

    def test_slack_tokens_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "xoxb-test-token")
        monkeypatch.setenv("GWENN_SLACK_APP_TOKEN", "xapp-test-token")
        cfg = SlackConfig(_env_file=None)
        assert cfg.bot_token == "xoxb-test-token"
        assert cfg.app_token == "xapp-test-token"

    def test_slack_partial_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "xoxb-only-bot")
        cfg = SlackConfig(_env_file=None)
        assert cfg.bot_token == "xoxb-only-bot"
        assert cfg.app_token is None

    def test_slack_tokens_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "  xoxb-test  ")
        monkeypatch.setenv("GWENN_SLACK_APP_TOKEN", "  xapp-test  ")
        cfg = SlackConfig(_env_file=None)
        assert cfg.bot_token == "xoxb-test"
        assert cfg.app_token == "xapp-test"

    def test_slack_empty_token_becomes_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "   ")
        cfg = SlackConfig(_env_file=None)
        assert cfg.bot_token is None

    def test_is_available_both_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("GWENN_SLACK_APP_TOKEN", "xapp-test")
        cfg = SlackConfig(_env_file=None)
        assert cfg.is_available is True

    def test_is_available_missing_app_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "xoxb-test")
        cfg = SlackConfig(_env_file=None)
        assert cfg.is_available is False

    def test_is_available_no_tokens(self) -> None:
        cfg = SlackConfig(_env_file=None)
        assert cfg.is_available is False


# ---------------------------------------------------------------------------
# ChannelConfig — slack_enabled
# ---------------------------------------------------------------------------


class TestChannelConfigSlack:
    """Tests for the slack_enabled flag on ChannelConfig."""

    def test_slack_disabled_by_default(self) -> None:
        cfg = ChannelConfig(_env_file=None)
        assert cfg.slack_enabled is False
        assert "slack" not in cfg.get_channel_list()

    def test_slack_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_ENABLED", "true")
        cfg = ChannelConfig(_env_file=None)
        assert cfg.slack_enabled is True
        assert "slack" in cfg.get_channel_list()

    def test_all_channels_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLI_ENABLED", "true")
        monkeypatch.setenv("TELEGRAM_ENABLED", "true")
        monkeypatch.setenv("DISCORD_ENABLED", "true")
        monkeypatch.setenv("SLACK_ENABLED", "true")
        cfg = ChannelConfig(_env_file=None)
        channels = cfg.get_channel_list()
        assert channels == ["cli", "telegram", "discord", "slack"]

    def test_channel_list_order_without_slack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLI_ENABLED", "true")
        monkeypatch.setenv("TELEGRAM_ENABLED", "true")
        monkeypatch.setenv("DISCORD_ENABLED", "false")
        monkeypatch.setenv("SLACK_ENABLED", "false")
        cfg = ChannelConfig(_env_file=None)
        assert cfg.get_channel_list() == ["cli", "telegram"]


# ---------------------------------------------------------------------------
# GwennConfig — slack attribute
# ---------------------------------------------------------------------------


class TestGwennConfigSlack:
    """Tests that GwennConfig composes SlackConfig."""

    def test_gwenn_config_has_slack(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GWENN_DATA_DIR", str(tmp_path / "data"))
        cfg = GwennConfig()
        assert hasattr(cfg, "slack")
        assert isinstance(cfg.slack, SlackConfig)
        assert cfg.slack.is_available is False

    def test_gwenn_config_slack_with_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GWENN_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("GWENN_SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("GWENN_SLACK_APP_TOKEN", "xapp-test")
        cfg = GwennConfig()
        assert cfg.slack.bot_token == "xoxb-test"
        assert cfg.slack.is_available is True
