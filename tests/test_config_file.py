"""Tests for gwenn/config_file.py — TOML configuration utilities."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------

class TestFindConfig:
    """Tests for find_config() search order."""

    def test_finds_in_cwd(self, tmp_path, monkeypatch):
        from gwenn.config_file import find_config

        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 15.0\n")
        monkeypatch.chdir(tmp_path)
        result = find_config()
        assert result is not None
        assert result == toml

    def test_finds_in_xdg_config(self, tmp_path, monkeypatch):
        from gwenn.config_file import find_config

        xdg_dir = tmp_path / ".config" / "gwenn"
        xdg_dir.mkdir(parents=True)
        toml = xdg_dir / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 15.0\n")
        # Ensure CWD doesn't have gwenn.toml
        (tmp_path / "somewhere_else").mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path / "somewhere_else")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        result = find_config()
        assert result is not None
        assert result == toml

    def test_finds_in_project_root(self, tmp_path, monkeypatch):
        from gwenn.config_file import find_config
        import gwenn.config_file as cf

        # Use tmp_path as project root to avoid touching real filesystem
        project_root = tmp_path / "project"
        project_root.mkdir()
        toml_path = project_root / "gwenn.toml"
        toml_path.write_text("[heartbeat]\ninterval = 15.0\n")

        # Change CWD to somewhere without gwenn.toml
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
        monkeypatch.setattr(cf, "_PROJECT_ROOT", project_root)

        result = find_config()
        assert result is not None
        assert result == toml_path

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        from gwenn.config_file import find_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        # Patch _PROJECT_ROOT to a dir without gwenn.toml
        import gwenn.config_file as cf
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        result = find_config()
        assert result is None

    def test_cwd_takes_priority_over_xdg(self, tmp_path, monkeypatch):
        from gwenn.config_file import find_config

        # Create both CWD and XDG configs
        (tmp_path / "gwenn.toml").write_text("[claude]\nmodel = 'cwd'\n")
        xdg_dir = tmp_path / ".config" / "gwenn"
        xdg_dir.mkdir(parents=True)
        (xdg_dir / "gwenn.toml").write_text("[claude]\nmodel = 'xdg'\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        result = find_config()
        assert result == tmp_path / "gwenn.toml"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_valid_toml(self, tmp_path):
        from gwenn.config_file import load_config

        toml = tmp_path / "gwenn.toml"
        toml.write_text(textwrap.dedent("""\
            [heartbeat]
            interval = 15.0
            proactive_messages = true

            [claude]
            model = "claude-opus-4-6"
        """))
        data = load_config(toml)
        assert data["heartbeat"]["interval"] == 15.0
        assert data["heartbeat"]["proactive_messages"] is True
        assert data["claude"]["model"] == "claude-opus-4-6"

    def test_load_empty_toml(self, tmp_path):
        from gwenn.config_file import load_config

        toml = tmp_path / "gwenn.toml"
        toml.write_text("")
        data = load_config(toml)
        assert data == {}

    def test_load_invalid_toml_raises(self, tmp_path):
        from gwenn.config_file import load_config

        toml = tmp_path / "gwenn.toml"
        toml.write_text("this is not valid [toml = = =")
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            load_config(toml)

    def test_load_nonexistent_raises(self, tmp_path):
        from gwenn.config_file import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")


# ---------------------------------------------------------------------------
# write_config
# ---------------------------------------------------------------------------

class TestWriteConfig:
    """Tests for write_config() atomic write."""

    def test_write_creates_file(self, tmp_path):
        from gwenn.config_file import load_config, write_config

        dest = tmp_path / "gwenn.toml"
        write_config(dest, {"heartbeat": {"interval": 15.0}})
        assert dest.exists()
        data = load_config(dest)
        assert data["heartbeat"]["interval"] == 15.0

    def test_write_overwrites_existing(self, tmp_path):
        from gwenn.config_file import load_config, write_config

        dest = tmp_path / "gwenn.toml"
        write_config(dest, {"heartbeat": {"interval": 15.0}})
        write_config(dest, {"heartbeat": {"interval": 30.0}})
        data = load_config(dest)
        assert data["heartbeat"]["interval"] == 30.0

    def test_write_creates_parent_directories(self, tmp_path):
        from gwenn.config_file import write_config

        dest = tmp_path / "a" / "b" / "gwenn.toml"
        write_config(dest, {"claude": {"model": "test"}})
        assert dest.exists()

    def test_write_atomic_no_partial_on_error(self, tmp_path):
        """If tomli_w.dump raises, the original file should be untouched."""
        from gwenn.config_file import write_config

        dest = tmp_path / "gwenn.toml"
        write_config(dest, {"original": True})

        # Write with a value that can't be serialized
        with pytest.raises(Exception):
            write_config(dest, {"bad": object()})

        # Original file should be untouched
        import tomllib
        with open(dest, "rb") as f:
            data = tomllib.load(f)
        assert data == {"original": True}

    def test_write_cleans_up_temp_file_on_error(self, tmp_path):
        from gwenn.config_file import write_config

        dest = tmp_path / "gwenn.toml"
        try:
            write_config(dest, {"bad": object()})
        except Exception:
            pass
        # No .tmp files should remain
        tmp_files = list(tmp_path.glob(".gwenn_config_*"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# get_value / set_value / delete_value
# ---------------------------------------------------------------------------

class TestGetValue:
    """Tests for get_value()."""

    def test_top_level_key(self):
        from gwenn.config_file import get_value

        assert get_value({"foo": 42}, "foo") == 42

    def test_nested_key(self):
        from gwenn.config_file import get_value

        data = {"heartbeat": {"interval": 15.0}}
        assert get_value(data, "heartbeat.interval") == 15.0

    def test_deeply_nested_key(self):
        from gwenn.config_file import get_value

        data = {"a": {"b": {"c": "deep"}}}
        assert get_value(data, "a.b.c") == "deep"

    def test_missing_key_raises(self):
        from gwenn.config_file import get_value

        with pytest.raises(KeyError):
            get_value({"heartbeat": {"interval": 15.0}}, "heartbeat.missing")

    def test_missing_intermediate_key_raises(self):
        from gwenn.config_file import get_value

        with pytest.raises(KeyError):
            get_value({"heartbeat": {"interval": 15.0}}, "missing.interval")

    def test_non_dict_intermediate_raises(self):
        from gwenn.config_file import get_value

        with pytest.raises(KeyError):
            get_value({"heartbeat": 42}, "heartbeat.interval")

    def test_empty_key_raises(self):
        from gwenn.config_file import get_value

        with pytest.raises(ValueError):
            get_value({"foo": 1}, "")

    def test_dot_only_key_raises(self):
        from gwenn.config_file import get_value

        with pytest.raises(ValueError):
            get_value({"foo": 1}, "foo..bar")


class TestSetValue:
    """Tests for set_value()."""

    def test_set_top_level(self):
        from gwenn.config_file import set_value

        data = {}
        result = set_value(data, "foo", 42)
        assert result["foo"] == 42

    def test_set_nested(self):
        from gwenn.config_file import set_value

        data = {"heartbeat": {}}
        result = set_value(data, "heartbeat.interval", 15.0)
        assert result["heartbeat"]["interval"] == 15.0

    def test_set_creates_intermediate_dicts(self):
        from gwenn.config_file import set_value

        data = {}
        result = set_value(data, "a.b.c", "deep")
        assert result["a"]["b"]["c"] == "deep"

    def test_set_overwrites_existing(self):
        from gwenn.config_file import set_value

        data = {"heartbeat": {"interval": 30.0}}
        result = set_value(data, "heartbeat.interval", 15.0)
        assert result["heartbeat"]["interval"] == 15.0

    def test_set_replaces_non_dict_intermediate(self):
        from gwenn.config_file import set_value

        data = {"heartbeat": 42}
        result = set_value(data, "heartbeat.interval", 15.0)
        assert result["heartbeat"]["interval"] == 15.0

    def test_set_returns_same_dict(self):
        from gwenn.config_file import set_value

        data = {"x": 1}
        result = set_value(data, "y", 2)
        assert result is data


class TestDeleteValue:
    """Tests for delete_value()."""

    def test_delete_top_level(self):
        from gwenn.config_file import delete_value

        data = {"foo": 42, "bar": 99}
        result = delete_value(data, "foo")
        assert "foo" not in result
        assert result["bar"] == 99

    def test_delete_nested(self):
        from gwenn.config_file import delete_value

        data = {"heartbeat": {"interval": 15.0, "min_interval": 5.0}}
        result = delete_value(data, "heartbeat.interval")
        assert "interval" not in result["heartbeat"]
        assert result["heartbeat"]["min_interval"] == 5.0

    def test_delete_missing_raises(self):
        from gwenn.config_file import delete_value

        with pytest.raises(KeyError):
            delete_value({"heartbeat": {}}, "heartbeat.interval")

    def test_delete_missing_intermediate_raises(self):
        from gwenn.config_file import delete_value

        with pytest.raises(KeyError):
            delete_value({}, "heartbeat.interval")

    def test_delete_returns_same_dict(self):
        from gwenn.config_file import delete_value

        data = {"foo": 42}
        result = delete_value(data, "foo")
        assert result is data


# ---------------------------------------------------------------------------
# generate_template
# ---------------------------------------------------------------------------

class TestGenerateTemplate:
    """Tests for generate_template()."""

    def test_output_is_valid_commented_toml(self):
        from gwenn.config_file import generate_template

        template = generate_template()
        # Should contain expected sections
        assert "[heartbeat]" in template
        assert "[claude]" in template
        assert "[daemon]" in template
        assert "[memory]" in template
        assert "[slack]" in template

    def test_all_values_commented_out(self):
        from gwenn.config_file import generate_template

        template = generate_template()
        for line in template.splitlines():
            stripped = line.strip()
            # Non-empty, non-section lines should be comments
            if stripped and not stripped.startswith("[") and not stripped.startswith("#"):
                pytest.fail(f"Uncommented config line: {line!r}")

    def test_template_is_parseable_when_uncommented(self):
        """Uncommented template should be valid TOML."""
        import tomllib
        from gwenn.config_file import generate_template

        template = generate_template()
        # Uncomment all lines
        lines = []
        for line in template.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and "=" in stripped:
                lines.append(stripped[2:])
            elif stripped.startswith("#") and not stripped.startswith("# "):
                # Comment-only lines (header comments)
                lines.append(stripped)
            else:
                lines.append(line)
        uncommented = "\n".join(lines)
        # Should parse without error
        data = tomllib.loads(uncommented)
        assert "heartbeat" in data
        assert "claude" in data


# ---------------------------------------------------------------------------
# GwennSettingsBase TOML integration
# ---------------------------------------------------------------------------

class TestGwennSettingsBase:
    """Tests for the GwennSettingsBase TOML source integration."""

    def _patch_toml_and_env(self, cfg, toml_path, monkeypatch=None):
        """Patch toml_file on the child class and neutralize .env file."""
        originals = {}
        # Patch toml_file on the specific child class's merged model_config
        for cls_name in ("HeartbeatConfig", "AffectConfig"):
            cls = getattr(cfg, cls_name, None)
            if cls:
                originals[f"{cls_name}_toml"] = cls.model_config.get("toml_file")
                cls.model_config["toml_file"] = str(toml_path)
                # Neutralize .env file so it doesn't override TOML values
                originals[f"{cls_name}_env"] = cls.model_config.get("env_file")
                cls.model_config["env_file"] = str(toml_path.parent / ".env.nonexistent")
        return originals

    def _restore(self, cfg, originals):
        for key, val in originals.items():
            cls_name, field = key.rsplit("_", 1)
            cls = getattr(cfg, cls_name, None)
            if cls and val is not None:
                config_key = "toml_file" if field == "toml" else "env_file"
                cls.model_config[config_key] = val

    def test_toml_values_loaded(self, tmp_path, monkeypatch):
        """Values from gwenn.toml should be picked up by config classes."""
        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 99.0\n")
        import gwenn.config as cfg
        originals = self._patch_toml_and_env(cfg, toml)
        # Clear env var so TOML is the source
        monkeypatch.delenv("GWENN_HEARTBEAT_INTERVAL", raising=False)
        try:
            hb = cfg.HeartbeatConfig()
            assert hb.interval == 99.0
        finally:
            self._restore(cfg, originals)

    def test_env_var_overrides_toml(self, tmp_path, monkeypatch):
        """Environment variables should take priority over TOML values."""
        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 99.0\n")
        import gwenn.config as cfg
        originals = self._patch_toml_and_env(cfg, toml)
        monkeypatch.setenv("GWENN_HEARTBEAT_INTERVAL", "7.0")
        try:
            hb = cfg.HeartbeatConfig()
            assert hb.interval == 7.0
        finally:
            self._restore(cfg, originals)

    def test_missing_toml_file_is_fine(self, tmp_path, monkeypatch):
        """Config should work when no gwenn.toml exists."""
        import gwenn.config as cfg
        originals = self._patch_toml_and_env(cfg, tmp_path / "nonexistent.toml")
        monkeypatch.delenv("GWENN_HEARTBEAT_INTERVAL", raising=False)
        try:
            hb = cfg.HeartbeatConfig()
            assert hb.interval == 30.0  # default value
        finally:
            self._restore(cfg, originals)

    def test_toml_only_affects_correct_section(self, tmp_path, monkeypatch):
        """Values from [heartbeat] should not leak into [affect]."""
        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 99.0\n[affect]\narousal_ceiling = 0.5\n")
        import gwenn.config as cfg
        originals = self._patch_toml_and_env(cfg, toml)
        monkeypatch.delenv("GWENN_HEARTBEAT_INTERVAL", raising=False)
        monkeypatch.delenv("GWENN_AROUSAL_CEILING", raising=False)
        try:
            hb = cfg.HeartbeatConfig()
            af = cfg.AffectConfig()
            assert hb.interval == 99.0
            assert af.arousal_ceiling == 0.5
        finally:
            self._restore(cfg, originals)

    def test_toml_works_for_class_without_explicit_populate_by_name(self, tmp_path, monkeypatch):
        """ContextConfig (no explicit populate_by_name) should still read TOML via base class."""
        toml = tmp_path / "gwenn.toml"
        toml.write_text("[context]\ncontext_limit = 999999\n")
        import gwenn.config as cfg
        orig_toml = cfg.ContextConfig.model_config.get("toml_file")
        orig_env = cfg.ContextConfig.model_config.get("env_file")
        cfg.ContextConfig.model_config["toml_file"] = str(toml)
        cfg.ContextConfig.model_config["env_file"] = str(toml.parent / ".nonexistent")
        monkeypatch.delenv("GWENN_CONTEXT_LIMIT", raising=False)
        try:
            ctx = cfg.ContextConfig()
            assert ctx.context_limit == 999999
        finally:
            if orig_toml is not None:
                cfg.ContextConfig.model_config["toml_file"] = orig_toml
            if orig_env is not None:
                cfg.ContextConfig.model_config["env_file"] = orig_env

    def test_toml_section_mapping_covers_all_classes(self):
        """Every leaf config class should have a TOML section mapping."""
        from gwenn.config import _TOML_SECTIONS, GwennSettingsBase, GwennConfig
        import gwenn.config as cfg

        # Find all GwennSettingsBase subclasses in config module
        leaf_classes = []
        for name in dir(cfg):
            obj = getattr(cfg, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, GwennSettingsBase)
                and obj is not GwennSettingsBase
            ):
                leaf_classes.append(name)

        for cls_name in leaf_classes:
            assert cls_name in _TOML_SECTIONS, f"{cls_name} missing from _TOML_SECTIONS"

    def test_toml_section_names_are_unique(self):
        """Each TOML section should map to a unique config class."""
        from gwenn.config import _TOML_SECTIONS

        sections = [v for v in _TOML_SECTIONS.values()]
        assert len(sections) == len(set(sections)), "Duplicate TOML section names"

    def test_all_leaf_classes_inherit_from_base(self):
        """All 23 leaf config classes should inherit from GwennSettingsBase."""
        from gwenn.config import GwennSettingsBase, _TOML_SECTIONS
        import gwenn.config as cfg

        for cls_name in _TOML_SECTIONS:
            cls = getattr(cfg, cls_name)
            assert issubclass(cls, GwennSettingsBase), (
                f"{cls_name} does not inherit from GwennSettingsBase"
            )


# ---------------------------------------------------------------------------
# Config subcommand (main.py)
# ---------------------------------------------------------------------------

class TestConfigSubcommand:
    """Tests for the 'gwenn config' subcommand in main.py."""

    def test_config_init(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        _run_config(["init"])
        assert (tmp_path / "gwenn.toml").exists()
        content = (tmp_path / "gwenn.toml").read_text()
        assert "[heartbeat]" in content

    def test_config_init_existing_file(self, tmp_path, monkeypatch, capsys):
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("[existing]\n")
        _run_config(["init"])
        # Should not overwrite
        content = (tmp_path / "gwenn.toml").read_text()
        assert "[existing]" in content

    def test_config_get(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 42.0\n")
        monkeypatch.chdir(tmp_path)
        # Patch find_config to return our temp file
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        _run_config(["get", "heartbeat.interval"])

    def test_config_get_missing_key(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        toml = tmp_path / "gwenn.toml"
        toml.write_text("[heartbeat]\ninterval = 42.0\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        # Should not raise, just print message
        _run_config(["get", "heartbeat.missing"])

    def test_config_get_no_toml(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _run_config(["get", "heartbeat.interval"])

    def test_config_set(self, tmp_path, monkeypatch):
        from gwenn.config_file import load_config
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("")
        _run_config(["set", "heartbeat.interval", "42.0"])
        data = load_config(tmp_path / "gwenn.toml")
        assert data["heartbeat"]["interval"] == 42.0

    def test_config_set_bool(self, tmp_path, monkeypatch):
        from gwenn.config_file import load_config
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("")
        _run_config(["set", "heartbeat.proactive_messages", "true"])
        data = load_config(tmp_path / "gwenn.toml")
        assert data["heartbeat"]["proactive_messages"] is True

    def test_config_set_int(self, tmp_path, monkeypatch):
        from gwenn.config_file import load_config
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("")
        _run_config(["set", "context.context_limit", "200000"])
        data = load_config(tmp_path / "gwenn.toml")
        assert data["context"]["context_limit"] == 200000

    def test_config_set_string(self, tmp_path, monkeypatch):
        from gwenn.config_file import load_config
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("")
        _run_config(["set", "claude.model", "claude-opus-4-6"])
        data = load_config(tmp_path / "gwenn.toml")
        assert data["claude"]["model"] == "claude-opus-4-6"

    def test_config_unset(self, tmp_path, monkeypatch):
        from gwenn.config_file import load_config
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "gwenn.toml").write_text("[heartbeat]\ninterval = 42.0\nmin_interval = 5.0\n")
        _run_config(["unset", "heartbeat.interval"])
        data = load_config(tmp_path / "gwenn.toml")
        assert "interval" not in data["heartbeat"]
        assert data["heartbeat"]["min_interval"] == 5.0

    def test_config_unset_missing_key(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "gwenn.toml").write_text("[heartbeat]\ninterval = 42.0\n")
        # Should not raise
        _run_config(["unset", "heartbeat.missing"])

    def test_config_validate_no_toml(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _run_config(["validate"])

    def test_config_validate_with_toml(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "gwenn.toml").write_text("[heartbeat]\ninterval = 42.0\n")
        _run_config(["validate"])

    def test_config_list_empty(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _run_config([])

    def test_config_list_with_toml(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "gwenn.toml").write_text("[heartbeat]\ninterval = 42.0\n")
        _run_config(["list"])

    def test_config_get_missing_arg(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        _run_config(["get"])

    def test_config_set_missing_args(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        _run_config(["set"])
        _run_config(["set", "key_only"])

    def test_config_unset_missing_arg(self, tmp_path, monkeypatch):
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        _run_config(["unset"])

    def test_config_set_no_existing_file(self, tmp_path, monkeypatch):
        """set should create gwenn.toml at project root if CWD doesn't have one."""
        from gwenn.config_file import load_config
        from gwenn.main import _run_config
        import gwenn.config_file as cf

        monkeypatch.chdir(tmp_path)
        # Patch _PROJECT_ROOT in config_file module used by _run_config
        monkeypatch.setattr(cf, "_PROJECT_ROOT", tmp_path)
        _run_config(["set", "heartbeat.interval", "42.0"])
        data = load_config(tmp_path / "gwenn.toml")
        assert data["heartbeat"]["interval"] == 42.0


# ---------------------------------------------------------------------------
# main() routing
# ---------------------------------------------------------------------------

class TestMainConfigRouting:
    """Tests for config subcommand routing via click CLI in main()."""

    def test_config_subcommand_routes_via_click(self, monkeypatch, tmp_path):
        """Verify main() routes 'config get' to click config command."""
        from gwenn.main import main

        # Create a gwenn.toml with a value to get
        toml_file = tmp_path / "gwenn.toml"
        toml_file.write_text("[heartbeat]\ninterval = 42\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["gwenn", "config", "get", "heartbeat.interval"])
        monkeypatch.setattr("gwenn.main._logging_configured", False)
        main()  # Should not raise — click config get handled

    def test_config_default_lists_via_click(self, monkeypatch, tmp_path):
        """'gwenn config' with no subcommand invokes the default list."""
        from gwenn.main import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["gwenn", "config"])
        monkeypatch.setattr("gwenn.main._logging_configured", False)
        main()  # Should not raise

    def test_non_config_subcommand_rejects_extra_args(self, monkeypatch):
        """Non-config subcommands should reject extra positional args."""
        from gwenn.main import main

        monkeypatch.setattr("sys.argv", ["gwenn", "status", "extra_arg"])
        monkeypatch.setattr("gwenn.main._logging_configured", False)
        with pytest.raises((SystemExit, Exception)):
            main()

    def test_config_set_nan_rejected(self, tmp_path, monkeypatch):
        """config set should reject nan values."""
        from gwenn.main import _run_config

        monkeypatch.chdir(tmp_path)
        (tmp_path / "gwenn.toml").write_text("")
        _run_config(["set", "heartbeat.interval", "nan"])
        # File should still be empty (nan rejected)
        from gwenn.config_file import load_config
        data = load_config(tmp_path / "gwenn.toml")
        assert data == {}
