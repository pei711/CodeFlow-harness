"""CLI contract tests for CodeFlow's retained Channels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeflow.cli.commands import app
from codeflow.config.loader import set_config_path

runner = CliRunner()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    set_config_path(path)
    yield path
    set_config_path(None)  # type: ignore[arg-type]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_enable_get_disable_round_trip_redacts_secret(tmp_config: Path) -> None:
    result = runner.invoke(app, ["channels", "enable", "qq", "--app-id", "10001", "--secret", "private-value"])
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(app, ["channels", "get", "qq"])
    assert result.exit_code == 0, result.stdout
    assert "****set****" in result.stdout
    assert "private-value" not in result.stdout

    result = runner.invoke(app, ["channels", "disable", "qq"])
    assert result.exit_code == 0, result.stdout
    section = _read(tmp_config)["channels"]["qq"]
    assert section["enabled"] is False
    assert section["secret"] == "private-value"


def test_enable_feishu_accepts_kebab_flags(tmp_config: Path) -> None:
    result = runner.invoke(
        app,
        ["channels", "enable", "feishu", "--app-id", "X", "--app-secret", "Y"],
    )
    assert result.exit_code == 0, result.stdout
    section = _read(tmp_config)["channels"]["feishu"]
    assert section["appId"] == "X"
    assert section["appSecret"] == "Y"


def test_get_can_show_secrets_explicitly(tmp_config: Path) -> None:
    runner.invoke(app, ["channels", "enable", "qq", "--secret", "plain-value"])
    result = runner.invoke(app, ["channels", "get", "qq", "--show-secrets"])
    assert result.exit_code == 0
    assert "plain-value" in result.stdout


def test_set_supports_equals_and_bool_negative_forms(tmp_config: Path) -> None:
    runner.invoke(app, ["channels", "enable", "feishu", "--app-secret", "first"])
    result = runner.invoke(app, ["channels", "set", "feishu", "--app-secret=second"])
    assert result.exit_code == 0, result.stdout
    assert _read(tmp_config)["channels"]["feishu"]["appSecret"] == "second"

    result = runner.invoke(app, ["channels", "set", "feishu", "--no-enabled"])
    assert result.exit_code == 0, result.stdout
    assert _read(tmp_config)["channels"]["feishu"]["enabled"] is False


def test_unknown_field_reports_name(tmp_config: Path) -> None:
    result = runner.invoke(app, ["channels", "set", "qq", "--secrt", "value"])
    assert result.exit_code != 0
    output = result.stdout + (result.output or "")
    assert "secrt" in output


def test_invalid_literal_fails_validation(tmp_config: Path) -> None:
    result = runner.invoke(app, ["channels", "set", "feishu", "--group-policy", "invalid"])
    assert result.exit_code != 0
    output = result.stdout + (result.output or "")
    assert "validation" in output.lower()


def test_show_lists_retained_schema_and_required_column(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["channels", "show", "feishu"])
    assert result.exit_code == 0, result.stdout
    assert "--app-id" in result.stdout
    assert "--app-secret" in result.stdout
    assert "--group-policy" in result.stdout
    assert "Required?" in result.stdout


def test_enable_and_set_without_fields_show_schema_without_writing(tmp_config: Path) -> None:
    for verb in ("enable", "set"):
        result = runner.invoke(app, ["channels", verb, "qq"])
        assert result.exit_code == 0, result.stdout
        assert "--secret" in result.stdout
        assert "Tip:" in result.stdout
    assert not tmp_config.exists()


def test_channels_list_exposes_only_retained_channels(tmp_config: Path) -> None:
    result = runner.invoke(app, ["channels", "list"])
    assert result.exit_code == 0, result.stdout
    for name in ("feishu", "qq", "wecom"):
        assert name in result.stdout
    for name in ("telegram", "slack", "discord", "whatsapp", "matrix", "mochat", "dingtalk", "email", "weixin"):
        assert name not in result.stdout


def test_channels_list_reflects_enabled_state(tmp_config: Path) -> None:
    runner.invoke(app, ["channels", "enable", "qq", "--secret", "value"])
    result = runner.invoke(app, ["channels", "list"])
    lines = result.stdout.splitlines()
    qq_row = next(line for line in lines if "qq" in line)
    feishu_row = next(line for line in lines if "feishu" in line)
    assert "✓" in qq_row
    assert "✓" not in feishu_row


@pytest.mark.parametrize(
    "command, name, maturity",
    [
        ("list", "qq", "beta"),
        ("list", "wecom", "beta"),
        ("list", "feishu", "live-gated"),
        ("status", "QQ", "beta"),
        ("status", "WeCom", "beta"),
        ("status", "Feishu", "live-gated"),
    ],
)
def test_channels_list_and_status_label_maturity(tmp_config: Path, command: str, name: str, maturity: str) -> None:
    """Maturity comes from ChannelSpec, so no CLI surface can claim more
    evidence than the adapter declares."""
    result = runner.invoke(app, ["channels", command])
    assert result.exit_code == 0, result.stdout
    row = next(line for line in result.stdout.splitlines() if name in line)
    assert maturity in row


@pytest.mark.parametrize("command", ["list", "status"])
def test_channels_list_and_status_explain_maturity(tmp_config: Path, command: str) -> None:
    result = runner.invoke(app, ["channels", command])
    assert "deterministic contract and security evidence only" in result.stdout


def test_channels_group_help_and_bare_command_are_available() -> None:
    result = runner.invoke(app, ["channels", "--help"])
    assert result.exit_code == 0, result.stdout
    assert "channels list" in result.stdout

    result = runner.invoke(app, ["channels"])
    assert result.exit_code == 2, result.stdout
    assert "list" in result.stdout


def test_reset_confirmation_abort_and_accept(tmp_config: Path) -> None:
    runner.invoke(app, ["channels", "enable", "qq", "--secret", "value"])
    result = runner.invoke(app, ["channels", "reset", "qq"], input="N\n")
    assert result.exit_code == 0
    assert _read(tmp_config)["channels"]["qq"]["secret"] == "value"

    result = runner.invoke(app, ["channels", "reset", "qq"], input="y\n")
    assert result.exit_code == 0
    assert _read(tmp_config)["channels"]["qq"]["secret"] == ""


def test_all_retained_channels_default_allow_from_to_wildcard() -> None:
    from codeflow.config.schema import ChannelsConfig

    config = ChannelsConfig()
    assert config.feishu.allow_from == ["*"]
    assert config.qq.allow_from == ["*"]
    assert config.wecom.allow_from == ["*"]


def test_login_unknown_channel_lists_only_retained_channels(tmp_config: Path) -> None:
    result = runner.invoke(app, ["channels", "login", "unknown"])
    assert result.exit_code == 1
    assert "Available: feishu, qq, wecom" in result.stdout


@pytest.mark.parametrize("name", ["feishu", "qq", "wecom"])
def test_retained_channels_need_no_interactive_login(tmp_config: Path, name: str) -> None:
    result = runner.invoke(app, ["channels", "login", name])
    assert result.exit_code == 0, result.stdout
    assert "needs no interactive login" in result.stdout
    assert f"channels set {name}" in result.stdout
