"""Tests for GwennConfig path derivation and resolution."""

from __future__ import annotations

from pathlib import Path

from gwenn.config import GwennConfig


def test_gwenn_config_default_paths_are_absolute(monkeypatch, tmp_path):
    """All path fields should be resolved to absolute paths."""
    data_dir = tmp_path / "gwenn_data"
    monkeypatch.setenv("GWENN_DATA_DIR", str(data_dir))
    # Prevent real API key lookup
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    cfg = GwennConfig()

    assert cfg.memory.data_dir.is_absolute()
    assert cfg.memory.episodic_db_path.is_absolute()
    assert cfg.memory.semantic_db_path.is_absolute()
    assert cfg.daemon.socket_path.is_absolute()
    assert cfg.daemon.pid_file.is_absolute()
    assert cfg.daemon.sessions_dir.is_absolute()
    assert cfg.skills_dir.is_absolute()


def test_gwenn_config_daemon_paths_derived_from_data_dir(monkeypatch, tmp_path):
    """Daemon paths should follow data_dir when not explicitly overridden."""
    data_dir = tmp_path / "custom_data"
    monkeypatch.setenv("GWENN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Clear any daemon overrides
    monkeypatch.delenv("GWENN_DAEMON_SOCKET", raising=False)
    monkeypatch.delenv("GWENN_DAEMON_PID", raising=False)
    monkeypatch.delenv("GWENN_DAEMON_SESSIONS_DIR", raising=False)

    cfg = GwennConfig()

    assert cfg.daemon.socket_path == (data_dir / "gwenn.sock").resolve()
    assert cfg.daemon.pid_file == (data_dir / "gwenn.pid").resolve()
    assert cfg.daemon.sessions_dir == (data_dir / "sessions").resolve()


def test_gwenn_config_data_dir_created(monkeypatch, tmp_path):
    """GwennConfig.__init__ should create the data directory."""
    data_dir = tmp_path / "new_data"
    monkeypatch.setenv("GWENN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert not data_dir.exists()
    GwennConfig()
    assert data_dir.exists()


def test_gwenn_config_skills_dir_env_override(monkeypatch, tmp_path):
    """GWENN_SKILLS_DIR env var should override the default skills path."""
    skills = tmp_path / "my_skills"
    monkeypatch.setenv("GWENN_SKILLS_DIR", str(skills))
    monkeypatch.setenv("GWENN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    cfg = GwennConfig()

    assert cfg.skills_dir == skills.resolve()


def test_gwenn_config_memory_sub_paths_derived(monkeypatch, tmp_path):
    """Episodic/semantic DB paths should derive from data_dir by default."""
    data_dir = tmp_path / "derived"
    monkeypatch.setenv("GWENN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Clear overrides
    monkeypatch.delenv("GWENN_EPISODIC_DB", raising=False)
    monkeypatch.delenv("GWENN_SEMANTIC_DB", raising=False)

    cfg = GwennConfig()

    assert cfg.memory.episodic_db_path == (data_dir / "episodic.db").resolve()
    assert cfg.memory.semantic_db_path == (data_dir / "semantic_vectors").resolve()
