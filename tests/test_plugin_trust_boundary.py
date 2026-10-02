"""Security contracts for Plugin discovery and factory import timing."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeflow.cli._plugin_stack import build_plugin_registry, plugin_discovery_sources
from codeflow.config.codeflow import CodeFlowConfig, MemoryConfig, PluginsConfig
from codeflow.plugin import PluginDiscovery, ServiceLocator, assemble_plugin_registry


@pytest.fixture(autouse=True)
def _restore_import_state():
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    yield
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        if name.startswith("plugin_trust_fixture"):
            sys.modules.pop(name, None)


def _write_tool_plugin(root: Path, *, marker: Path) -> Path:
    plugin_dir = root / "plugin-trust-fixture"
    package = plugin_dir / "plugin_trust_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factories.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            Path({str(marker)!r}).write_text("imported", encoding="utf-8")

            def make_tool(ctx):
                return {{"workspace": str(ctx.services.workspace)}}
            """
        ),
        encoding="utf-8",
    )
    (plugin_dir / "codeflow-plugin.toml").write_text(
        textwrap.dedent(
            """
            [plugin]
            id = "plugin-trust-fixture"
            version = "0.1.0"
            enabled_by_default = true

            [[plugin.contributes.tools]]
            name = "plugin_trust_fixture_tool"
            factory = "plugin_trust_fixture.factories:make_tool"
            """
        ),
        encoding="utf-8",
    )
    return plugin_dir


def _memory_off_config() -> CodeFlowConfig:
    return CodeFlowConfig(
        memory=MemoryConfig(backend=None),
        plugins=PluginsConfig(),
    )


def test_shared_cli_tui_gateway_registry_ignores_checkout_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    marker = tmp_path / "host-imported"
    _write_tool_plugin(checkout / ".codeflow" / "plugins", marker=marker)
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))

    sources = plugin_discovery_sources()
    registry = build_plugin_registry(_memory_off_config())

    assert sources["project_dir"] is None
    assert "plugin-trust-fixture" not in registry.activated_ids()
    assert not marker.exists()


def test_codeflow_plugins_does_not_import_checkout_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    marker = tmp_path / "plugins-command-imported"
    _write_tool_plugin(checkout / ".codeflow" / "plugins", marker=marker)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"memory": {"backend": None}}), encoding="utf-8")
    monkeypatch.chdir(checkout)

    from codeflow.cli.commands import app

    result = CliRunner().invoke(
        app,
        ["plugins", "--config", str(config_path)],
        env={**os.environ, "CODEFLOW_HOME": str(tmp_path / "codeflow-home"), "COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.stdout
    assert "plugin-trust-fixture" not in result.stdout
    assert not marker.exists()


def test_codeflow_plugins_inspects_user_manifest_without_importing_factory(tmp_path: Path) -> None:
    codeflow_home = tmp_path / "codeflow-home"
    marker = tmp_path / "user-command-imported"
    plugin_dir = _write_tool_plugin(codeflow_home / "plugins", marker=marker)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"memory": {"backend": None}}), encoding="utf-8")

    from codeflow.cli.commands import app

    result = CliRunner().invoke(
        app,
        ["plugins", "--config", str(config_path)],
        env={**os.environ, "CODEFLOW_HOME": str(codeflow_home), "COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.stdout
    assert "plugin-trust-fixture" in result.stdout
    assert str(plugin_dir) not in sys.path
    assert not marker.exists()


def test_manifest_admission_defers_user_plugin_import_until_build(tmp_path: Path) -> None:
    user_plugins = tmp_path / "user-plugins"
    marker = tmp_path / "user-imported"
    plugin_dir = _write_tool_plugin(user_plugins, marker=marker)

    registry = assemble_plugin_registry(
        user_dir=user_plugins,
        entry_points_group=None,
    )

    assert registry.activated_ids() == ["plugin-trust-fixture"]
    assert registry.tool_names() == ["plugin_trust_fixture_tool"]
    assert str(plugin_dir) not in sys.path
    assert not marker.exists()

    tool = registry.build_tool(
        "plugin_trust_fixture_tool",
        config={},
        services=ServiceLocator(workspace=tmp_path),
    )

    assert tool == {"workspace": str(tmp_path)}
    assert str(plugin_dir) in sys.path
    assert marker.read_text(encoding="utf-8") == "imported"


def test_explicit_project_discovery_remains_available(tmp_path: Path) -> None:
    project_plugins = tmp_path / "project-plugins"
    marker = tmp_path / "project-imported"
    plugin_dir = _write_tool_plugin(project_plugins, marker=marker)

    discovered = PluginDiscovery(
        project_dir=project_plugins,
        entry_points_group=None,
    ).discover()
    registry = assemble_plugin_registry(
        project_dir=project_plugins,
        entry_points_group=None,
    )

    assert [plugin.manifest.id for plugin in discovered] == ["plugin-trust-fixture"]
    assert str(plugin_dir) not in sys.path
    assert not marker.exists()

    registry.build_tool(
        "plugin_trust_fixture_tool",
        config={},
        services=ServiceLocator(workspace=tmp_path),
    )

    assert marker.read_text(encoding="utf-8") == "imported"
