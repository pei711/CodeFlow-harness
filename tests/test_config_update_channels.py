"""Tests for the retained Channel configuration write path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from codeflow.config.update_channels import (
    channel_field_specs,
    disable_channel,
    enable_channel,
    get_channel_config,
    reset_channel,
    set_channel_fields,
)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_channel_names() -> list[str]:
    from codeflow.config.schema import ChannelsConfig

    return [
        name
        for name, field in ChannelsConfig.model_fields.items()
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
    ]


ALL_CHANNELS = _all_channel_names()


def test_schema_exposes_only_retained_channels() -> None:
    assert ALL_CHANNELS == ["feishu", "qq", "wecom"]


def test_enable_channel_writes_camel_case_and_enabled(cfg_path: Path) -> None:
    enable_channel(
        "feishu",
        {"app_id": "cli_xxx", "app_secret": "sec_yyy", "encrypt_key": "ekey"},
        config_path=cfg_path,
    )

    section = _read(cfg_path)["channels"]["feishu"]
    assert section["enabled"] is True
    assert section["appId"] == "cli_xxx"
    assert section["appSecret"] == "sec_yyy"
    assert section["encryptKey"] == "ekey"


def test_unknown_channel_raises(cfg_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown channel 'foobar'"):
        enable_channel("foobar", {}, config_path=cfg_path)


def test_unknown_field_lists_available_fields(cfg_path: Path) -> None:
    with pytest.raises(KeyError) as exc_info:
        set_channel_fields("qq", {"secrt": "value"}, config_path=cfg_path)
    message = str(exc_info.value)
    assert "secrt" in message
    assert "secret" in message
    assert "Available fields" in message


def test_invalid_literal_is_atomic(cfg_path: Path) -> None:
    enable_channel("feishu", {"app_secret": "original"}, config_path=cfg_path)
    before = _read(cfg_path)

    with pytest.raises(ValidationError):
        set_channel_fields("feishu", {"group_policy": "invalid"}, config_path=cfg_path)

    assert _read(cfg_path) == before


def test_set_returns_previous_value(cfg_path: Path) -> None:
    enable_channel("qq", {"secret": "first"}, config_path=cfg_path)
    previous = set_channel_fields("qq", {"secret": "second"}, config_path=cfg_path)
    assert previous == {"secret": "first"}
    assert _read(cfg_path)["channels"]["qq"]["secret"] == "second"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_bool_coercion(cfg_path: Path, raw: str, expected: bool) -> None:
    set_channel_fields("qq", {"enabled": raw}, config_path=cfg_path)
    assert _read(cfg_path)["channels"]["qq"]["enabled"] is expected


def test_list_coercion_accepts_csv_and_json(cfg_path: Path) -> None:
    set_channel_fields("qq", {"allow_from": "alice,bob"}, config_path=cfg_path)
    assert _read(cfg_path)["channels"]["qq"]["allowFrom"] == ["alice", "bob"]

    set_channel_fields("qq", {"allow_from": '["U1","U2"]'}, config_path=cfg_path)
    assert _read(cfg_path)["channels"]["qq"]["allowFrom"] == ["U1", "U2"]


def test_get_redacts_secrets_and_preserves_public_fields(cfg_path: Path) -> None:
    enable_channel(
        "feishu",
        {"app_id": "cli_x", "app_secret": "p1", "encrypt_key": "p2"},
        config_path=cfg_path,
    )
    redacted = get_channel_config("feishu", config_path=cfg_path)
    assert redacted["app_secret"] == "****set****"
    assert redacted["encrypt_key"] == "****set****"
    assert redacted["app_id"] == "cli_x"

    plain = get_channel_config("feishu", redact_secrets=False, config_path=cfg_path)
    assert plain["app_secret"] == "p1"
    assert plain["encrypt_key"] == "p2"


def test_get_marks_unset_secret_empty(cfg_path: Path) -> None:
    enable_channel("feishu", {"app_id": "X"}, config_path=cfg_path)
    cfg = get_channel_config("feishu", config_path=cfg_path)
    assert cfg["app_secret"] == "(empty)"
    assert cfg["app_id"] == "X"


def test_disable_preserves_credentials_and_reset_clears_them(cfg_path: Path) -> None:
    enable_channel("qq", {"app_id": "10001", "secret": "keep"}, config_path=cfg_path)
    disable_channel("qq", config_path=cfg_path)
    disabled = _read(cfg_path)["channels"]["qq"]
    assert disabled["enabled"] is False
    assert disabled["appId"] == "10001"
    assert disabled["secret"] == "keep"

    reset_channel("qq", config_path=cfg_path)
    reset = _read(cfg_path)["channels"]["qq"]
    assert reset["enabled"] is False
    assert reset["appId"] == ""
    assert reset["secret"] == ""


def test_writing_one_channel_preserves_another(cfg_path: Path) -> None:
    enable_channel("qq", {"app_id": "10001", "secret": "secret"}, config_path=cfg_path)
    qq_before = _read(cfg_path)["channels"]["qq"]

    set_channel_fields("feishu", {"app_id": "cli_value"}, config_path=cfg_path)
    disable_channel("wecom", config_path=cfg_path)
    reset_channel("feishu", config_path=cfg_path)

    assert _read(cfg_path)["channels"]["qq"] == qq_before


@pytest.mark.parametrize(
    "name",
    ["telegram", "slack", "discord", "whatsapp", "matrix", "mochat", "dingtalk", "email", "weixin"],
)
def test_removed_channel_config_rejects_all_writes_without_mutation(cfg_path: Path, name: str) -> None:
    legacy = {"channels": {name: {"enabled": False}}}
    cfg_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"channels\.{name}.*no longer supported.*remove"):
        enable_channel("feishu", {"app_id": "X", "app_secret": "Y"}, config_path=cfg_path)

    assert _read(cfg_path) == legacy


def test_field_specs_cover_retained_credentials_and_literal_choices() -> None:
    feishu = channel_field_specs("feishu")
    qq = channel_field_specs("qq")
    wecom = channel_field_specs("wecom")

    assert feishu["app_id"]["required"] is True
    assert feishu["app_id"]["is_secret"] is False
    assert feishu["app_secret"]["required"] is True
    assert feishu["app_secret"]["is_secret"] is True
    assert feishu["encrypt_key"]["required"] is False
    assert qq["secret"]["required"] is True
    assert qq["secret"]["is_secret"] is True
    assert wecom["bot_id"]["required"] is True
    assert wecom["secret"]["is_secret"] is True
    assert feishu["group_policy"]["type"] == "Literal"
    assert "open" in feishu["group_policy"]["description"]
    assert "mention" in feishu["group_policy"]["description"]


@pytest.mark.parametrize("name", ALL_CHANNELS)
def test_retained_channel_crud_round_trip(name: str, cfg_path: Path) -> None:
    specs = channel_field_specs(name)
    assert specs["enabled"]["type"] == "bool"
    assert specs["enabled"]["default"] is False

    enable_channel(name, config_path=cfg_path)
    assert _read(cfg_path)["channels"][name]["enabled"] is True
    assert set(get_channel_config(name, config_path=cfg_path)) == set(specs)

    disable_channel(name, config_path=cfg_path)
    assert _read(cfg_path)["channels"][name]["enabled"] is False

    reset_channel(name, config_path=cfg_path)
    assert _read(cfg_path)["channels"][name]["enabled"] is False


def test_malformed_config_refuses_write_and_preserves_file(cfg_path: Path) -> None:
    from codeflow.config.loader import ConfigReadError

    original = '{\n  "channels": {"qq": {"enabled": true}},\n  // invalid JSON\n}\n'
    cfg_path.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigReadError):
        enable_channel("qq", {"secret": "x"}, config_path=cfg_path)
    assert cfg_path.read_text(encoding="utf-8") == original
    with pytest.raises(ConfigReadError):
        set_channel_fields("qq", {"secret": "y"}, config_path=cfg_path)
    assert cfg_path.read_text(encoding="utf-8") == original
