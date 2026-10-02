"""``codeflow plugins`` reports the installed Myna Plugin without starting it."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from codeflow.plugin import (
    Contributes,
    DiscoveredPlugin,
    MemoryBackendContribution,
    PluginManifest,
    Source,
)


@pytest.fixture(autouse=True)
def _installed_myna(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("myna.integrations.codeflow.backend")
    module.make_backend = lambda context: context
    monkeypatch.setitem(sys.modules, module.__name__, module)
    discovered = DiscoveredPlugin(
        manifest=PluginManifest(
            id="myna-memory",
            version="0.1.1rc3",
            codeflow=">=0.1,<0.2",
            enabled_by_default=True,
            contributes=Contributes(
                memory_backends=[
                    MemoryBackendContribution(
                        name="myna",
                        factory="myna.integrations.codeflow:make_backend",
                    )
                ]
            ),
        ),
        source=Source.ENTRY_POINTS,
        location=None,
    )
    monkeypatch.setattr("codeflow.plugin.discover.PluginDiscovery.discover", lambda _self: [discovered])


def _make_runner_args(tmp_path: Path, config: dict[str, Any]) -> list[str]:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return ["plugins", "-c", str(config_path)]


def _invoke(args: list[str], tmp_path: Path):
    from codeflow.cli.commands import app

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HOME": str(fake_home), "COLUMNS": "200"}
    return CliRunner().invoke(app, args, env=env)


def test_lists_installed_myna_plugin(tmp_path: Path) -> None:
    result = _invoke(_make_runner_args(tmp_path, {}), tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "myna-memory" in result.stdout
    assert "0.1.1rc3" in result.stdout
    assert "entry_points" in result.stdout


def test_shows_active_myna_backend(tmp_path: Path) -> None:
    result = _invoke(
        _make_runner_args(
            tmp_path,
            {"memory": {"backend": "myna", "userId": "alice"}},
        ),
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    assert "Active memory backend" in result.stdout
    assert "myna" in result.stdout
    assert "from plugin: myna-memory" in result.stdout
    assert "User id:" in result.stdout
    assert "alice" in result.stdout
    assert "Agent id:" not in result.stdout


def test_null_backend_reports_memory_disabled(tmp_path: Path) -> None:
    result = _invoke(
        _make_runner_args(tmp_path, {"memory": {"backend": None}}),
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    assert "Active memory backend" in result.stdout
    assert "none" in result.stdout
    assert "Memory is disabled" in result.stdout
    assert "legacy" not in result.stdout


def test_unknown_backend_is_fail_closed(tmp_path: Path) -> None:
    result = _invoke(
        _make_runner_args(tmp_path, {"memory": {"backend": "nonexistent"}}),
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    assert "nonexistent" in result.stdout
    assert "not available" in result.stdout
    assert "fail closed" in result.stdout
    assert "fall back" not in result.stdout


def test_disabled_myna_plugin_status(tmp_path: Path) -> None:
    result = _invoke(
        _make_runner_args(
            tmp_path,
            {"plugins": {"disabled": ["myna-memory"]}},
        ),
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    assert "myna-memory" in result.stdout
    assert "disabled" in result.stdout
    assert "not available" in result.stdout


def test_verbose_shows_myna_factory(tmp_path: Path) -> None:
    args = _make_runner_args(tmp_path, {})
    args.append("--verbose")

    result = _invoke(args, tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "myna.integrations.codeflow:make_backend" in result.stdout
