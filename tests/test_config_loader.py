"""Tests for ``codeflow.config.loader.load_config``.

Covers the migrations that drop / relocate retired blocks from old
configs, plus the default-config fallback path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeflow.config.loader import load_config


def _write(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    """No file → default Config — loader must not raise."""
    cfg = load_config(tmp_path / "does_not_exist.json")

    assert cfg.agents.defaults.max_tool_iterations == 40


def test_legacy_everos_block_silently_dropped(tmp_path: Path) -> None:
    """Old configs may still carry ``agents.defaults.everos``. The
    migration strips it so model_validate doesn't reject the file."""
    p = tmp_path / "config.json"
    _write(
        p,
        {
            "agents": {
                "defaults": {
                    "everos": {"enabled": True, "enableSkill": True},
                },
            },
        },
    )
    cfg = load_config(p)
    assert not hasattr(cfg.agents.defaults, "everos")


def test_legacy_everos_skill_light_is_dropped(
    tmp_path: Path,
) -> None:
    """The retired remembered-Skill block is ignored during migration."""
    p = tmp_path / "config.json"
    _write(
        p,
        {
            "agents": {
                "defaults": {
                    "everosSkillLight": {"enabled": True},
                },
            },
        },
    )
    cfg = load_config(p)
    assert not hasattr(cfg.agents.defaults, "everosSkillLight")
    assert not hasattr(cfg.agents.defaults, "everos_skill_light")


def test_legacy_everos_skill_light_block_is_not_relocated() -> None:
    from codeflow.config.loader import _migrate_config

    out = _migrate_config(
        {
            "agents": {
                "defaults": {
                    "everosSkillLight": {
                        "enabled": True,
                        "minMessages": 4,
                        "minToolCalls": 2,
                        "maxSkillsTopK": 5,
                    },
                },
            },
        },
        pop_extension_keys=False,
    )
    assert "skillForge" not in out


def test_legacy_everos_skill_light_snake_case_block_is_not_relocated() -> None:
    from codeflow.config.loader import _migrate_config

    out = _migrate_config(
        {
            "agents": {
                "defaults": {
                    "everos_skill_light": {
                        "min_messages": 4,
                        "min_tool_calls": 2,
                        "enabled": False,
                    },
                },
            },
        },
        pop_extension_keys=False,
    )
    assert "skillForge" not in out


def test_corrupted_json_falls_back_to_defaults(tmp_path: Path) -> None:
    """A mid-write race can leave the file half-flushed; tolerate it."""
    p = tmp_path / "config.json"
    p.write_text("{this is not json", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.agents.defaults.max_tool_iterations == 40


def test_schema_validation_error_raises(tmp_path: Path) -> None:
    """A user / programmer config error must NOT silently fall back to
    defaults — that masks misconfig as "feature X did nothing"."""
    p = tmp_path / "config.json"

    _write(
        p,
        {
            "agents": {"defaults": {"max_tool_iterations": "not-an-int"}},
        },
    )
    with pytest.raises(ValueError, match="schema validation"):
        load_config(p)


@pytest.mark.parametrize(
    "name",
    ["telegram", "slack", "discord", "whatsapp", "matrix", "mochat", "dingtalk", "email", "weixin"],
)
def test_removed_channel_config_is_rejected_with_migration_diagnostic(tmp_path: Path, name: str) -> None:
    p = tmp_path / "config.json"
    _write(p, {"channels": {name: {"enabled": False}}})

    with pytest.raises(ValueError, match=rf"channels\.{name}.*no longer supported.*remove"):
        load_config(p)


def test_removed_media_generation_config_is_rejected_with_diagnostic(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    _write(p, {"tools": {"media": {"image": {"model": "legacy-image-model"}}}})

    with pytest.raises(ValueError, match=r"tools\.media.*no longer supported.*remove"):
        load_config(p)


@pytest.mark.parametrize("key", ["deepResearch", "deep_research"])
def test_removed_deep_research_config_is_rejected_with_diagnostic(tmp_path: Path, key: str) -> None:
    p = tmp_path / "config.json"
    _write(p, {"tools": {key: {"apiKey": "legacy-key"}}})

    with pytest.raises(ValueError, match=r"tools\.deepResearch.*no longer supported.*remove"):
        load_config(p)


def test_removed_sentinel_config_is_rejected_with_diagnostic(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    _write(p, {"sentinel": {"enabled": False}})

    with pytest.raises(ValueError, match=r"sentinel.*no longer supported.*remove"):
        load_config(p)


def test_removed_gateway_heartbeat_config_is_rejected_with_diagnostic(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    _write(p, {"gateway": {"heartbeat": {"enabled": True}}})

    with pytest.raises(ValueError, match=r"gateway\.heartbeat.*no longer supported.*remove"):
        load_config(p)


@pytest.mark.parametrize(
    "router",
    [
        {"hub": {"endpoint": "https://skills.example.test"}},
        {"weights": {"local": 1.0, "hub": 0.85}},
    ],
)
def test_removed_skill_hub_config_is_rejected_with_diagnostic(
    tmp_path: Path,
    router: dict,
) -> None:
    p = tmp_path / "config.json"
    _write(p, {"skillForge": {"router": router}})

    with pytest.raises(ValueError, match=r"skillForge\.router\.(?:hub|weights\.hub).*no longer supported.*remove"):
        load_config(p)


def test_read_raw_or_raise_absent_returns_empty(tmp_path: Path) -> None:
    from codeflow.config.loader import read_raw_or_raise

    assert read_raw_or_raise(tmp_path / "nope.json") == {}


def test_read_raw_or_raise_valid(tmp_path: Path) -> None:
    from codeflow.config.loader import read_raw_or_raise

    p = tmp_path / "c.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert read_raw_or_raise(p) == {"a": 1}


def test_read_raw_or_raise_malformed_raises(tmp_path: Path) -> None:
    from codeflow.config.loader import ConfigReadError, read_raw_or_raise

    p = tmp_path / "bad.json"
    p.write_text("{  // comment\n}", encoding="utf-8")
    with pytest.raises(ConfigReadError):
        read_raw_or_raise(p)


def test_load_config_malformed_warns_loudly_and_uses_defaults(tmp_path: Path, capsys) -> None:
    from codeflow.config.loader import load_config
    from codeflow.config.schema import Config

    p = tmp_path / "bad.json"
    p.write_text("{  // comment\n}", encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert "IGNORING" in capsys.readouterr().err


def test_read_raw_or_raise_empty_file_is_empty_dict(tmp_path: Path) -> None:
    from codeflow.config.loader import read_raw_or_raise

    p = tmp_path / "empty.json"
    p.write_text("   \n", encoding="utf-8")
    assert read_raw_or_raise(p) == {}


def test_read_raw_or_raise_json_null_is_empty_dict(tmp_path: Path) -> None:
    from codeflow.config.loader import read_raw_or_raise

    p = tmp_path / "null.json"
    p.write_text("null", encoding="utf-8")
    assert read_raw_or_raise(p) == {}


def test_config_read_error_is_not_runtimeerror() -> None:

    from codeflow.config.loader import ConfigReadError

    assert not issubclass(ConfigReadError, RuntimeError)
    assert issubclass(ConfigReadError, Exception)
