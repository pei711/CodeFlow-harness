"""CLI tests for ``codeflow doctor``.

Static checks are validated against on-disk config produced by the
``tmp_config`` / ``healthy_config`` fixtures. The probe boundary
(:func:`codeflow.cli.doctor_commands.send_probe`) is monkeypatched
so tests never touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeflow.cli import doctor_commands
from codeflow.cli.commands import app
from codeflow.config.loader import save_config, set_config_path
from codeflow.config.schema import Config

runner = CliRunner()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Point the loader at a tmp config file; tests opt-in via save_config."""
    cfg = tmp_path / "config.json"
    set_config_path(cfg)
    yield cfg
    set_config_path(None)  # type: ignore[arg-type]


@pytest.fixture
def healthy_config(tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Persist a config that routes cleanly to a real provider name."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tui_bundle = tmp_path / "entry.js"
    tui_bundle.write_text("", encoding="utf-8")
    monkeypatch.setattr("codeflow.cli.tui_commands.resolve_dist_entry", lambda: tui_bundle)
    from codeflow.cli._plugin_stack import MemoryBackendStatus

    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.inspect_memory_backend",
        lambda _config: MemoryBackendStatus(
            backend="myna",
            state="available",
            plugin_id="myna-memory",
            plugin_version="0.1.1rc3",
        ),
    )

    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-sonnet-4-5"
    cfg.agents.defaults.workspace = str(workspace)
    cfg.providers.anthropic.api_key = "sk-fake"
    save_config(cfg)
    return tmp_config


def test_doctor_help_lists_all_flags() -> None:
    """``--help`` exposes the full flag surface."""
    r = runner.invoke(app, ["doctor", "--help"])
    assert r.exit_code == 0, r.stdout
    for flag in ("--probe", "--json", "--timeout"):
        assert flag in r.stdout, f"missing flag in help: {flag}"


def test_doctor_default_on_missing_config_exit1(tmp_config: Path) -> None:
    """No config file → exit 1 with a hint to run ``onboard``."""
    assert not tmp_config.exists()
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 1, r.stdout
    assert "not configured" in r.stdout
    assert "codeflow onboard" in r.stdout


def test_doctor_default_healthy_exit0(healthy_config: Path) -> None:
    """Resolved routing + no probe → exit 0, no network call made."""
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.stdout

    assert "anthropic" in r.stdout.lower()
    assert "Configuration looks healthy" in r.stdout or "All checks passed" in r.stdout
    assert "myna-memory 0.1.1rc3" in r.stdout


def test_doctor_memory_plugin_error_fails_closed(
    healthy_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli._plugin_stack import MemoryBackendStatus

    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.inspect_memory_backend",
        lambda _config: MemoryBackendStatus(
            backend="myna",
            state="error",
            error="install the myna-memory distribution and run 'myna init'",
        ),
    )

    r = runner.invoke(app, ["doctor", "--json"])

    assert r.exit_code == 1, r.stdout
    data = json.loads(r.stdout)
    assert data["memory"]["state"] == "error"
    assert "myna init" in data["memory"]["error"]


def test_doctor_checks_current_project_workspace_and_state(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    product_home = tmp_path / "codeflow-home"
    tui_bundle = tmp_path / "entry.js"
    tui_bundle.write_text("", encoding="utf-8")
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    monkeypatch.chdir(project)
    monkeypatch.setattr("codeflow.cli.tui_commands.resolve_dist_entry", lambda: tui_bundle)
    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-sonnet-4-5"
    cfg.providers.anthropic.api_key = "sk-fake"
    save_config(cfg)
    from codeflow.config.update import set_memory_backend

    set_memory_backend(None)

    r = runner.invoke(app, ["doctor", "--json"])

    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["paths"]["workspace_path"] == str(project)
    assert Path(data["paths"]["state_path"]).parent == product_home / "projects"
    assert data["paths"]["state_writable"] is True


def test_doctor_unresolved_routing_exit1(tmp_config: Path) -> None:
    """Model that no configured provider can serve → exit 1."""
    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-sonnet-4-5"

    save_config(cfg)
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 1, r.stdout
    assert "unresolved" in r.stdout.lower() or "could not be routed" in r.stdout


def test_doctor_missing_tui_bundle_exit1(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("codeflow.cli.tui_commands.resolve_dist_entry", lambda: None)

    r = runner.invoke(app, ["doctor", "--json"])

    assert r.exit_code == 1, r.stdout
    data = json.loads(r.stdout)
    assert data["features"]["tui_bundle_available"] is False


def test_doctor_missing_enabled_channel_extra_exit1(
    healthy_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-sonnet-4-5"
    cfg.agents.defaults.workspace = str(healthy_config.parent / "workspace")
    cfg.providers.anthropic.api_key = "sk-fake"
    cfg.channels.feishu.enabled = True
    save_config(cfg)
    monkeypatch.setattr(
        doctor_commands,
        "_channel_extra_available",
        lambda channel: channel != "feishu",
    )

    r = runner.invoke(app, ["doctor", "--json"])

    assert r.exit_code == 1, r.stdout
    data = json.loads(r.stdout)
    assert data["features"]["missing_channel_extras"] == ["feishu"]


@pytest.mark.parametrize(
    "channel, maturity",
    [("qq", "beta"), ("wecom", "beta"), ("feishu", "live-gated")],
)
def test_doctor_labels_enabled_channel_maturity(healthy_config: Path, channel: str, maturity: str) -> None:
    """The Features channel line states each adapter's declared evidence level."""
    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-sonnet-4-5"
    cfg.agents.defaults.workspace = str(healthy_config.parent / "workspace")
    cfg.providers.anthropic.api_key = "sk-fake"
    setattr(getattr(cfg.channels, channel), "enabled", True)
    save_config(cfg)

    r = runner.invoke(app, ["doctor"])

    assert f"{channel} [{maturity}]" in r.stdout


def test_doctor_unwritable_workspace_exit1(
    healthy_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor_commands, "_workspace_is_writable", lambda _path: False)

    r = runner.invoke(app, ["doctor", "--json"])

    assert r.exit_code == 1, r.stdout
    data = json.loads(r.stdout)
    assert data["paths"]["workspace_writable"] is False


def test_doctor_shows_gateway_running(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A held instance lock surfaces as ``running (pid …)`` in the Gateway section."""
    from codeflow.cli import _gateway_lock

    monkeypatch.setattr(
        _gateway_lock,
        "read_status",
        lambda now: _gateway_lock.LockInfo(pid=999, started_at=1_700_000_000.0, config_path=""),
    )
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.stdout
    assert "Gateway" in r.stdout
    assert "running" in r.stdout
    assert "999" in r.stdout


def test_doctor_shows_gateway_not_running(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codeflow.cli import _gateway_lock

    monkeypatch.setattr(_gateway_lock, "read_status", lambda now: None)
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.stdout
    assert "not running" in r.stdout


def test_doctor_probe_success(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--probe`` invokes send_probe → exit 0, response shown in output."""
    monkeypatch.setattr(
        doctor_commands,
        "send_probe",
        lambda **_: ("Hello!", 42, 1.5),
    )
    r = runner.invoke(app, ["doctor", "--probe"])
    assert r.exit_code == 0, r.stdout
    assert "Hello!" in r.stdout
    assert "42 tokens" in r.stdout


def test_doctor_probe_failure_exit2(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Static checks pass but probe raises → exit 2."""

    def _boom(**_):
        raise RuntimeError("auth failed")

    monkeypatch.setattr(doctor_commands, "send_probe", _boom)
    r = runner.invoke(app, ["doctor", "--probe"])
    assert r.exit_code == 2, r.stdout
    assert "auth failed" in r.stdout


def test_doctor_timeout_flag_passed_through(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--timeout 3`` reaches ``send_probe`` as ``timeout_s=3``."""
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return ("ok", 1, 0.1)

    monkeypatch.setattr(doctor_commands, "send_probe", _capture)
    r = runner.invoke(app, ["doctor", "--probe", "--timeout", "3"])
    assert r.exit_code == 0, r.stdout
    assert captured.get("timeout_s") == 3


def test_doctor_json_default_structure(healthy_config: Path) -> None:
    """``--json`` emits a parseable doc with the documented top-level keys."""
    r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["version"] == 1
    for key in ("paths", "routing", "features", "gateway"):
        assert key in data, f"missing top-level key: {key}"
    assert "running" in data["gateway"]

    assert data["probe"] is None


def test_doctor_json_with_probe_structure(healthy_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--json --probe`` populates the probe key with the result fields."""
    monkeypatch.setattr(
        doctor_commands,
        "send_probe",
        lambda **_: ("hi", 10, 0.2),
    )
    r = runner.invoke(app, ["doctor", "--json", "--probe"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert isinstance(data["probe"], dict)
    assert data["probe"]["ok"] is True
    assert data["probe"]["text"] == "hi"
    assert data["probe"]["tokens"] == 10
