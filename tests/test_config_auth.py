from __future__ import annotations

from gwenn.config import ClaudeConfig


def test_claude_config_accepts_claude_code_oauth_token_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test-token")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = ClaudeConfig(_env_file=None)

    assert cfg.api_key is None
    assert cfg.auth_token == "sk-ant-oat01-test-token"
