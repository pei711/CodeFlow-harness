"""Tests for ``load_codeflow_config`` reading CodeFlow extension blocks
(``skill_forge`` / ``context`` / ``token_wise``) from the
same JSON file as the base Config.

Pre-fix the loader silently ignored the extension keys — every install
got default ``SkillForgeConfig`` regardless of what the user wrote.
These tests pin the post-fix behavior so it doesn't regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeflow.config import codeflow as ec_module


def _write_config(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def stub_config_path(monkeypatch, tmp_path: Path):
    """Redirect ``get_config_path()`` (used by both base loader and the
    new extension-block reader) to a tmp file we control per test."""
    p = tmp_path / "config.json"

    def _stub() -> Path:
        return p

    monkeypatch.setattr("codeflow.config.loader.get_config_path", _stub)
    monkeypatch.setattr("codeflow.config.codeflow.get_config_path", _stub)
    return p


def test_missing_config_falls_through_to_defaults(stub_config_path) -> None:

    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.enabled is True
    assert cfg.skill_forge.top_k == 5


def test_skill_forge_block_loaded_from_snake_case(stub_config_path: Path) -> None:
    _write_config(
        stub_config_path,
        {
            "skill_forge": {
                "enabled": True,
                "top_k": 3,
                "reranker_enabled": False,
            },
        },
    )
    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.enabled is True
    assert cfg.skill_forge.top_k == 3
    assert cfg.skill_forge.reranker_enabled is False


def test_skill_forge_block_loaded_from_camel_case(stub_config_path: Path) -> None:
    """Match the format ``codeflow onboard`` writes (camelCase via
    ``model_dump(by_alias=True)``)."""
    _write_config(
        stub_config_path,
        {
            "skillForge": {
                "enabled": True,
                "topK": 7,
                "rerankerEnabled": False,
            },
        },
    )
    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.enabled is True
    assert cfg.skill_forge.top_k == 7
    assert cfg.skill_forge.reranker_enabled is False


def test_explicit_null_falls_back_to_defaults(stub_config_path: Path) -> None:
    """A user editing config and leaving ``"skill_forge": null`` must not
    crash the loader — treat as 'use defaults'."""
    _write_config(stub_config_path, {"skill_forge": None})
    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.enabled is True


def test_only_specified_block_overrides(stub_config_path: Path) -> None:
    """Setting just ``skill_forge`` leaves other extension defaults intact."""
    _write_config(
        stub_config_path,
        {
            "skill_forge": {"enabled": True},
        },
    )
    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.enabled is True
    assert cfg.call_efficiency.mode == "observe"
    assert cfg.token_wise is cfg.call_efficiency


def test_legacy_token_wise_config_migrates_without_enabling_request_rewrites(
    stub_config_path: Path,
) -> None:
    _write_config(
        stub_config_path,
        {
            "tokenWise": {
                "enabled": True,
                "cacheOptimization": True,
                "maxCacheBreakpoints": 3,
            },
        },
    )

    cfg = ec_module.load_codeflow_config()

    assert cfg.call_efficiency.mode == "observe"
    assert cfg.call_efficiency.max_cache_breakpoints == 3
    assert cfg.token_wise is cfg.call_efficiency


def test_legacy_token_wise_disable_maps_to_call_efficiency_off(
    stub_config_path: Path,
) -> None:
    _write_config(stub_config_path, {"token_wise": {"enabled": False}})

    cfg = ec_module.load_codeflow_config()

    assert cfg.call_efficiency.effective_mode == "off"


def test_explicit_call_efficiency_mode_wins_over_legacy_block(
    stub_config_path: Path,
) -> None:
    _write_config(
        stub_config_path,
        {
            "tokenWise": {"enabled": False},
            "callEfficiency": {"mode": "optimize", "maxCacheBreakpoints": 2},
        },
    )

    cfg = ec_module.load_codeflow_config()

    assert cfg.call_efficiency.effective_mode == "optimize"
    assert cfg.call_efficiency.max_cache_breakpoints == 2


def test_mass_library_db_path_round_trips(stub_config_path: Path) -> None:
    """The ignored compatibility field still round-trips while deprecated."""
    _write_config(
        stub_config_path,
        {
            "skill_forge": {
                "enabled": True,
                "massLibraryDb": "/tmp/some/path/skills.db",
            },
        },
    )
    cfg = ec_module.load_codeflow_config()
    assert cfg.skill_forge.mass_library_db == "/tmp/some/path/skills.db"


def test_invalid_json_falls_through(stub_config_path: Path) -> None:
    stub_config_path.write_text("{ this is not valid json", encoding="utf-8")
    cfg = ec_module.load_codeflow_config()

    assert cfg.skill_forge.enabled is True


def test_retired_everos_skill_fields_are_ignored(stub_config_path: Path) -> None:
    _write_config(
        stub_config_path,
        {
            "agents": {
                "defaults": {
                    "everos": {"enabled": False},
                    "everosSkillLight": {"enabled": True},
                },
            },
            "skillForge": {
                "everos": {"enabled": True},
                "router": {"weights": {"local": 1.0, "everos": 0.9}},
            },
        },
    )

    cfg = ec_module.load_codeflow_config()
    assert not hasattr(cfg.skill_forge, "everos")
    assert not hasattr(cfg.skill_forge.router, "weights")


def test_retired_memory_agent_id_is_dropped_without_rewriting_backend(
    stub_config_path: Path,
) -> None:
    _write_config(
        stub_config_path,
        {
            "memory": {
                "backend": "everos",
                "userId": "default",
                "agentId": "default",
                "memoryTopK": 5,
            },
        },
    )

    cfg = ec_module.load_codeflow_config()

    assert cfg.memory.backend == "everos"
    assert not hasattr(cfg.memory, "agent_id")


def test_extension_keys_with_unknown_field_rejected(stub_config_path: Path) -> None:
    """Pydantic should reject unknown fields under skill_forge to catch
    typos in user config — better a loud error than silent default.
    ``_Base`` is configured ``extra='forbid'`` so the loader raises a
    ``ValidationError`` instead of silently dropping the typo."""
    _write_config(
        stub_config_path,
        {
            "skill_forge": {
                "enabled": True,
                "totally_made_up_field": "oops",
            },
        },
    )
    with pytest.raises(ValidationError, match="totally_made_up_field"):
        ec_module.load_codeflow_config()
