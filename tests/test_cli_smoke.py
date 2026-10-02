"""Smoke tests covering every CLI command surface.

This file's job is to catch *crash-class* regressions (``NameError`` /
``AttributeError`` / ``ImportError``) that slip through the more focused
per-command tests. It walks every top-level command + every subcommand
group's ``--help`` and asserts:

1. exit code 0
2. no ``Traceback`` printed
3. ``r.exception`` is None (typer.testing.CliRunner captures crashes here)

It also catches the historical regression where ``agent_commands.py`` was
missing the ``sync_workspace_templates`` import after the CLI modularize
refactor — that bug would have been caught here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from codeflow.cli.commands import app
from codeflow.config.loader import set_config_path

runner = CliRunner()


def _registered_command_names() -> set[str]:
    """All top-level command / subcommand-group names on the root app."""
    return set(typer.main.get_command(app).commands.keys())


def _public_command_names() -> set[str]:
    """Top-level commands shown in root help and shell completion."""
    root = typer.main.get_command(app)
    return {name for name, command in root.commands.items() if not command.hidden}


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    set_config_path(cfg)
    yield cfg
    set_config_path(None)  # type: ignore[arg-type]


TOP_LEVEL_COMMANDS = [
    "onboard",
    "gateway",
    "run",
    "status",
    "doctor",
    "evolve",
    "channels",
    "cron",
    "provider",
    "skills",
    "sessions",
    "plugins",
    "tracing",
]


@pytest.mark.parametrize("command", TOP_LEVEL_COMMANDS)
def test_top_level_command_help_does_not_crash(command: str) -> None:
    """Every top-level command's ``--help`` exits 0 with no leaked crash."""
    r = runner.invoke(app, [command, "--help"])
    assert r.exit_code == 0, f"{command} --help exited {r.exit_code}: {r.stdout}"
    assert r.exception is None, f"{command} --help raised an unexpected exception: {r.exception!r}"


def test_root_help_does_not_crash() -> None:
    """``codeflow --help`` should list every command without crashing."""
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert r.exception is None
    for cmd in TOP_LEVEL_COMMANDS:
        assert cmd in r.stdout, f"missing command in root --help: {cmd}"


CHANNEL_SUBCOMMANDS = [
    "status",
    "login",
    "enable",
    "disable",
    "set",
    "get",
    "reset",
    "show",
    "list",
]


@pytest.mark.parametrize("subcmd", CHANNEL_SUBCOMMANDS)
def test_channels_subcommand_help_does_not_crash(subcmd: str) -> None:
    """Every ``channels`` subcommand's ``--help`` exits cleanly."""
    r = runner.invoke(app, ["channels", subcmd, "--help"])
    assert r.exit_code == 0, f"channels {subcmd} --help exited {r.exit_code}"
    assert r.exception is None


SKILL_SUBCOMMANDS = ["list", "get"]


@pytest.mark.parametrize("subcmd", SKILL_SUBCOMMANDS)
def test_skills_subcommand_help_does_not_crash(subcmd: str) -> None:
    """Every ``skills`` subcommand's ``--help`` exits cleanly."""
    r = runner.invoke(app, ["skills", subcmd, "--help"])
    assert r.exit_code == 0, f"skills {subcmd} --help exited {r.exit_code}"
    assert r.exception is None


def test_status_command_body_does_not_crash(tmp_config: Path) -> None:
    """``codeflow status`` reads config + prints rows without crashing."""
    r = runner.invoke(app, ["status"])
    assert r.exception is None, f"status crashed: {r.exception!r}"
    assert r.exit_code == 0


def test_channels_list_body_does_not_crash(tmp_config: Path) -> None:
    """``codeflow channels list`` enumerates channels without crashing."""
    r = runner.invoke(app, ["channels", "list"])
    assert r.exception is None
    assert r.exit_code == 0


def test_cron_list_body_does_not_crash(tmp_config: Path) -> None:
    """``codeflow cron list`` reads cron jobs without crashing."""
    r = runner.invoke(app, ["cron", "list"])
    assert r.exception is None
    assert r.exit_code == 0


REGISTERED_COMMAND_NAMES = {
    "channels",
    "cron",
    "doctor",
    "evolve",
    "gateway",
    "onboard",
    "plugins",
    "provider",
    "sandbox",
    "sessions",
    "skills",
    "status",
    "tracing",
    "run",
}


def test_version_flag_matches_installed_metadata() -> None:
    """``codeflow --version`` reports the installed distribution version, not a
    hand-written literal, so it can never drift from ``pyproject.toml``."""
    from importlib.metadata import version as pkg_version

    r = runner.invoke(app, ["--version"])
    assert r.exception is None
    assert r.exit_code == 0
    assert f"CodeFlow v{pkg_version('codeflow-harness')}" in r.stdout


def test_cli_import_does_not_pull_litellm() -> None:
    """The CLI entry module must not eagerly import litellm (it dominates cold
    start). Checked in a subprocess: ``sys.modules`` is process-global, so a full
    ``pytest tests/`` run pollutes it via sibling tests that import litellm and an
    in-process assertion would false-fail.
    """
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "import codeflow.cli.commands, sys; assert 'litellm' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_no_logs_subcommand_registered() -> None:
    """There is no ``codeflow logs`` command; adding one must break this test."""
    assert "logs" not in _registered_command_names()


def test_no_sentinel_subcommand_registered() -> None:
    assert "sentinel" not in _registered_command_names()


def test_public_command_set_is_pinned() -> None:
    assert _public_command_names() == set(TOP_LEVEL_COMMANDS)


@pytest.mark.parametrize("removed", ["agent", "skill", "tui", "upgrade"])
def test_removed_root_commands_are_unavailable(removed: str) -> None:
    assert removed not in _registered_command_names()


def test_sandbox_is_hidden_but_still_available_as_advanced_help() -> None:
    assert "sandbox" in _registered_command_names()
    assert "sandbox" not in _public_command_names()
    result = runner.invoke(app, ["sandbox", "--help"])
    assert result.exit_code == 0, result.stdout


def test_registered_command_set_is_pinned() -> None:
    """Pin the exact command surface so any add/remove trips a test."""
    assert _registered_command_names() == REGISTERED_COMMAND_NAMES
