"""Unit tests for ``codeflow.config.update`` — the misc-ops write path.

Companion to ``test_config_update_providers.py`` /
``test_config_update_channels.py``. Covers the small focused helpers that
patch one or two fields without re-serializing the entire Pydantic model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeflow.config.update import (
    init_extension_block_defaults,
    reset_cron_config,
    set_default_model,
    set_memory_backend,
    set_sandbox_backend,
    update_cron_config,
)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_set_default_model_writes_into_empty_config(cfg_path: Path) -> None:
    prev = set_default_model("openrouter/anthropic/claude-sonnet-4-5", config_path=cfg_path)
    assert prev is None
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "openrouter/anthropic/claude-sonnet-4-5"


def test_set_default_model_returns_previous_value(cfg_path: Path) -> None:
    cfg_path.write_text(json.dumps({"agents": {"defaults": {"model": "openai/gpt-4o"}}}))
    prev = set_default_model("anthropic/claude-sonnet-4-5", config_path=cfg_path)
    assert prev == "openai/gpt-4o"
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "anthropic/claude-sonnet-4-5"


def test_set_default_model_preserves_sibling_fields(cfg_path: Path) -> None:
    cfg_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "model": "old-model",
                        "maxTokens": 4096,
                        "temperature": 0.5,
                    }
                },
                "providers": {"openai": {"apiKey": "sk-keep-me"}},
            }
        )
    )
    set_default_model("new-model", config_path=cfg_path)
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "new-model"
    assert data["agents"]["defaults"]["maxTokens"] == 4096
    assert data["agents"]["defaults"]["temperature"] == 0.5
    assert data["providers"]["openai"]["apiKey"] == "sk-keep-me"


def test_set_default_model_creates_nested_structure_when_missing(cfg_path: Path) -> None:
    cfg_path.write_text(json.dumps({"providers": {}}))
    set_default_model("some-model", config_path=cfg_path)
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "some-model"
    assert data["providers"] == {}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_update_cron_config_writes_into_empty_config(cfg_path: Path) -> None:
    prev = update_cron_config("forward_channels", ["telegram"], config_path=cfg_path)
    assert prev is None
    data = _read(cfg_path)
    assert data["cron"]["forwardChannels"] == ["telegram"]


def test_update_cron_config_returns_previous_value(cfg_path: Path) -> None:
    update_cron_config("forward_channels", ["telegram"], config_path=cfg_path)
    prev = update_cron_config("forward_channels", ["feishu"], config_path=cfg_path)
    assert prev == ["telegram"]
    data = _read(cfg_path)
    assert data["cron"]["forwardChannels"] == ["feishu"]


def test_update_cron_config_default_timezone(cfg_path: Path) -> None:
    update_cron_config("default_timezone", "America/Vancouver", config_path=cfg_path)
    data = _read(cfg_path)
    assert data["cron"]["defaultTimezone"] == "America/Vancouver"


def test_update_cron_config_unknown_key_raises(cfg_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown cron config key"):
        update_cron_config("nonexistent_key", "x", config_path=cfg_path)


def test_reset_cron_config_removes_section(cfg_path: Path) -> None:
    update_cron_config("forward_channels", ["telegram"], config_path=cfg_path)
    update_cron_config("default_timezone", "UTC", config_path=cfg_path)
    reset_cron_config(config_path=cfg_path)
    data = _read(cfg_path)
    assert "cron" not in data


def test_update_cron_preserves_sibling_sections(cfg_path: Path) -> None:
    cfg_path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"model": "openai/gpt-4o"}},
                "providers": {"openai": {"apiKey": "sk-keep-me"}},
            }
        )
    )
    update_cron_config("forward_channels", ["telegram"], config_path=cfg_path)
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "openai/gpt-4o"
    assert data["providers"]["openai"]["apiKey"] == "sk-keep-me"
    assert data["cron"]["forwardChannels"] == ["telegram"]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_set_sandbox_backend_writes_and_returns_prev(cfg_path: Path) -> None:

    assert set_sandbox_backend("boxlite", config_path=cfg_path) is None
    assert _read(cfg_path)["tools"]["sandbox"]["backend"] == "boxlite"
    prev = set_sandbox_backend("none", config_path=cfg_path)
    assert prev == "boxlite"
    assert _read(cfg_path)["tools"]["sandbox"]["backend"] == "none"


def test_set_sandbox_backend_preserves_siblings(cfg_path: Path) -> None:
    cfg_path.write_text(json.dumps({"providers": {"openai": {"apiKey": "sk-keep"}}}))
    set_sandbox_backend("boxlite", config_path=cfg_path)
    data = _read(cfg_path)
    assert data["providers"]["openai"]["apiKey"] == "sk-keep"
    assert data["tools"]["sandbox"]["backend"] == "boxlite"


def test_set_sandbox_backend_survives_reload(cfg_path: Path) -> None:

    from codeflow.config.loader import load_config

    set_sandbox_backend("boxlite", config_path=cfg_path)
    cfg = load_config(cfg_path)
    assert cfg.tools.sandbox.backend == "boxlite"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_set_memory_backend_example_then_none(cfg_path: Path) -> None:
    assert set_memory_backend("example", config_path=cfg_path) is None
    assert _read(cfg_path)["memory"]["backend"] == "example"
    prev = set_memory_backend(None, config_path=cfg_path)
    assert prev == "example"
    assert _read(cfg_path)["memory"]["backend"] is None


def test_set_memory_backend_preserves_siblings(cfg_path: Path) -> None:
    cfg_path.write_text(json.dumps({"agents": {"defaults": {"model": "openai/gpt-4o"}}}))
    set_memory_backend("example", config_path=cfg_path)
    data = _read(cfg_path)
    assert data["agents"]["defaults"]["model"] == "openai/gpt-4o"
    assert data["memory"]["backend"] == "example"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_init_extension_defaults_seeds_safe_subset(cfg_path: Path) -> None:
    init_extension_block_defaults(config_path=cfg_path)
    data = _read(cfg_path)

    assert data["memory"] == {
        "backend": "myna",
        "userId": "default",
        "memoryTopK": 5,
    }
    assert data["plugins"]["disabled"] == []
    assert data["plugins"]["config"] == {}
    assert data["skillForge"]["enabled"] is True
    assert data["skillForge"]["router"] == {"enabled": True}
    assert "hub" not in data["skillForge"]["router"]


def test_init_extension_defaults_omits_internal_infra_fields(cfg_path: Path) -> None:

    init_extension_block_defaults(config_path=cfg_path)
    sf = _read(cfg_path)["skillForge"]
    for leaked in (
        "embeddingUrl",
        "embeddingApiKey",
        "rerankerUrl",
        "rerankerApiKey",
        "massLibraryDb",
        "embedding_url",
        "embedding_api_key",
    ):
        assert leaked not in sf


def test_init_extension_defaults_is_idempotent_and_non_clobbering(cfg_path: Path) -> None:

    cfg_path.write_text(json.dumps({"memory": {"backend": None, "memoryTopK": 20}}))
    init_extension_block_defaults(config_path=cfg_path)
    first = _read(cfg_path)
    assert first["memory"]["backend"] is None
    assert first["memory"]["memoryTopK"] == 20
    assert first["memory"]["userId"] == "default"

    init_extension_block_defaults(config_path=cfg_path)
    assert _read(cfg_path) == first


def test_init_extension_defaults_round_trips_through_loader(cfg_path: Path) -> None:
    from codeflow.config.codeflow import load_codeflow_config

    init_extension_block_defaults(config_path=cfg_path)
    rc = load_codeflow_config(cfg_path)
    assert rc.memory.memory_top_k == 5
    assert rc.skill_forge.router.top_k == 5

    assert rc.skill_forge.embedding_url == "http://localhost:1357"
    assert rc.skill_forge.embedding_api_key is None
    assert rc.skill_forge.mass_library_db is None


def test_malformed_config_refuses_write_and_preserves_file(cfg_path: Path) -> None:

    from codeflow.config.loader import ConfigReadError
    from codeflow.config.update import set_default_model, set_language

    original = '{\n  "providers": {"openai": {"apiKey": "sk-o"}},\n  // comment => invalid JSON\n}\n'
    cfg_path.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigReadError):
        set_language("zh", config_path=cfg_path)
    assert cfg_path.read_text(encoding="utf-8") == original
    with pytest.raises(ConfigReadError):
        set_default_model("openrouter/x", config_path=cfg_path)
    assert cfg_path.read_text(encoding="utf-8") == original
