"""Tests for ``config.get`` / ``config.set`` RPC handlers (specs §3.6).

v0.1 hot-changeable whitelist (per specs §3.6):
    - ``agent.thinking_budget``
    - ``agent.temperature``
    - ``tui.theme``
    - ``tui.show_token_usage``

Writes to non-whitelisted keys → -32010 ``config_field_readonly``.
Writes that fail Pydantic-style validation → -32011 ``config_validation_error``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codeflow.tui_rpc.errors import (
    ConfigFieldReadonlyError,
    ConfigValidationError,
    ModelNotAvailableError,
    ModelSwitchInTurnError,
)
from codeflow.tui_rpc.methods.config import (
    CONFIG_WRITABLE_KEYS,
    config_get,
    config_set,
)


@pytest.fixture
def fake_home(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


async def test_config_get_no_keys_returns_all_writable(fake_home: Path) -> None:
    result = await config_get({})
    assert "config" in result
    cfg = result["config"]

    assert set(cfg.keys()) == set(CONFIG_WRITABLE_KEYS)


async def test_config_get_specific_keys_returns_subset(fake_home: Path) -> None:
    result = await config_get({"keys": ["tui.theme", "agent.temperature"]})
    assert set(result["config"].keys()) == {"tui.theme", "agent.temperature"}


async def test_config_get_unknown_keys_silently_omitted(fake_home: Path) -> None:
    result = await config_get({"keys": ["nope.invalid", "tui.theme"]})

    assert "nope.invalid" not in result["config"]
    assert "tui.theme" in result["config"]


async def test_config_get_reads_persisted_values(fake_home: Path) -> None:
    (fake_home / ".codeflow").mkdir()
    (fake_home / ".codeflow" / "config.json").write_text(json.dumps({"tui": {"theme": "solarized-dark"}}))
    result = await config_get({"keys": ["tui.theme"]})
    assert result["config"]["tui.theme"] == "solarized-dark"


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


async def test_config_set_whitelisted_returns_applied(fake_home: Path) -> None:
    result = await config_set({"key": "tui.theme", "value": "dark"})
    assert result["applied"] is True
    assert "previous" in result


async def test_config_set_non_whitelisted_raises_readonly(fake_home: Path) -> None:
    with pytest.raises(ConfigFieldReadonlyError):
        await config_set({"key": "secret.api_key", "value": "x"})


async def test_config_set_invalid_theme_raises_validation(fake_home: Path) -> None:
    with pytest.raises(ConfigValidationError):
        await config_set({"key": "tui.theme", "value": "@@@nope@@@"})


async def test_config_set_invalid_temperature_raises_validation(fake_home: Path) -> None:

    with pytest.raises(ConfigValidationError):
        await config_set({"key": "agent.temperature", "value": "hot"})

    with pytest.raises(ConfigValidationError):
        await config_set({"key": "agent.temperature", "value": 99})


async def test_config_set_persists_to_config_json(fake_home: Path) -> None:
    await config_set({"key": "tui.theme", "value": "dracula"})
    cfg_path = fake_home / ".codeflow" / "config.json"
    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text())
    assert payload["tui"]["theme"] == "dracula"


async def test_config_set_previous_value_returned(fake_home: Path) -> None:

    res1 = await config_set({"key": "tui.show_token_usage", "value": True})
    assert res1["applied"] is True
    assert res1["previous"] is None

    res2 = await config_set({"key": "tui.show_token_usage", "value": False})
    assert res2["applied"] is True
    assert res2["previous"] is True


async def test_config_set_creates_config_when_missing(fake_home: Path) -> None:
    """When ~/.codeflow/config.json doesn't exist yet, set must create it."""
    assert not (fake_home / ".codeflow" / "config.json").exists()
    await config_set({"key": "tui.theme", "value": "ok"})
    assert (fake_home / ".codeflow" / "config.json").exists()


async def test_config_set_missing_key_param_raises_validation(fake_home: Path) -> None:
    with pytest.raises(ConfigValidationError):
        await config_set({"value": "x"})
    with pytest.raises(ConfigValidationError):
        await config_set({"key": "tui.theme"})


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


async def test_config_set_model_reassigns_loop_and_persists(fake_home: Path, monkeypatch) -> None:
    import codeflow.tui_rpc.methods.config as config_mod

    loop = SimpleNamespace(provider="old-prov", model="old-model")
    new_provider = SimpleNamespace(name="new-prov")

    monkeypatch.setattr(config_mod, "has_active_turns", lambda: False)
    monkeypatch.setattr(config_mod, "make_provider", lambda _cfg: new_provider)
    monkeypatch.setattr(
        config_mod,
        "load_runtime_config",
        lambda *a, **k: SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(model="", provider="auto"))),
    )

    result = await config_set(
        {
            "key": "model",
            "value": "anthropic/claude-opus-4-8",
            "provider": "anthropic",
            "session_id": "tui:default",
        },
        agent_loop_factory=lambda: loop,
    )

    assert result["applied"] is True
    assert result["value"] == "anthropic/claude-opus-4-8"
    assert loop.model == "anthropic/claude-opus-4-8"
    assert loop.provider is new_provider

    cfg = json.loads((fake_home / ".codeflow" / "config.json").read_text())
    assert cfg["agents"]["defaults"]["model"] == "anthropic/claude-opus-4-8"
    assert cfg["agents"]["defaults"]["provider"] == "anthropic"


async def test_config_set_model_uses_atomic_runtime_replacement(fake_home: Path, monkeypatch) -> None:
    import codeflow.tui_rpc.methods.config as config_mod

    calls: list[tuple[object, str | None]] = []
    loop = SimpleNamespace(
        provider="old-prov",
        model="old-model",
        replace_provider=lambda provider, *, model=None: calls.append((provider, model)),
    )
    new_provider = SimpleNamespace(name="new-prov")
    monkeypatch.setattr(config_mod, "has_active_turns", lambda: False)
    monkeypatch.setattr(config_mod, "make_provider", lambda _cfg: new_provider)
    monkeypatch.setattr(
        config_mod,
        "load_runtime_config",
        lambda *a, **k: SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(model="", provider="auto"))),
    )

    await config_set(
        {
            "key": "model",
            "value": "deepseek/deepseek-v4-flash",
            "provider": "deepseek",
            "session_id": "tui:default",
        },
        agent_loop_factory=lambda: loop,
    )

    assert calls == [(new_provider, "deepseek/deepseek-v4-flash")]
    assert loop.provider == "old-prov"
    assert loop.model == "old-model"


async def test_config_set_model_bare_derives_provider(fake_home: Path) -> None:

    result = await config_set(
        {"key": "model", "value": "anthropic/claude-opus-4-8"},
        agent_loop_factory=lambda: None,
    )
    assert result["applied"] is True
    cfg = json.loads((fake_home / ".codeflow" / "config.json").read_text())
    assert cfg["agents"]["defaults"]["model"] == "anthropic/claude-opus-4-8"
    assert cfg["agents"]["defaults"]["provider"] == "anthropic"


async def test_config_set_model_rejected_while_any_session_has_active_turn(fake_home: Path, monkeypatch) -> None:
    import codeflow.tui_rpc.methods.turn as turn_mod

    monkeypatch.setitem(turn_mod._active_turns, "tui:other-session", object())

    with pytest.raises(ModelSwitchInTurnError):
        await config_set(
            {
                "key": "model",
                "value": "anthropic/claude-opus-4-8",
                "session_id": "tui:requesting-session",
            },
            agent_loop_factory=lambda: SimpleNamespace(provider=None, model="x"),
        )


async def test_config_set_model_unconstructable_preserves_previous(fake_home: Path, monkeypatch) -> None:
    import codeflow.tui_rpc.methods.config as config_mod

    (fake_home / ".codeflow").mkdir()
    (fake_home / ".codeflow" / "config.json").write_text(
        json.dumps({"agents": {"defaults": {"model": "anthropic/claude-sonnet-4-5"}}})
    )

    def _boom(_cfg):
        raise RuntimeError("no api key")

    monkeypatch.setattr(config_mod, "has_active_turns", lambda: False)
    monkeypatch.setattr(config_mod, "make_provider", _boom)
    monkeypatch.setattr(
        config_mod,
        "load_runtime_config",
        lambda *a, **k: SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(model="", provider="auto"))),
    )

    loop = SimpleNamespace(provider="keep-prov", model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ModelNotAvailableError):
        await config_set(
            {
                "key": "model",
                "value": "broken/model",
                "session_id": "tui:default",
            },
            agent_loop_factory=lambda: loop,
        )

    assert loop.model == "anthropic/claude-sonnet-4-5"
    assert loop.provider == "keep-prov"
    cfg = json.loads((fake_home / ".codeflow" / "config.json").read_text())
    assert cfg["agents"]["defaults"]["model"] == "anthropic/claude-sonnet-4-5"


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


async def test_config_methods_registered_via_helper(fake_home: Path) -> None:
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.methods.config import register_config_methods

    d = Dispatcher()
    register_config_methods(d)
    resp = await d.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "config.set",
            "params": {"key": "tui.theme", "value": "ok"},
        }
    )
    assert "error" not in resp
    assert resp["result"]["applied"] is True

    resp = await d.dispatch({"jsonrpc": "2.0", "id": 2, "method": "config.get", "params": {"keys": ["tui.theme"]}})
    assert resp["result"]["config"]["tui.theme"] == "ok"

    resp = await d.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "config.set",
            "params": {"key": "secret.api_key", "value": "x"},
        }
    )
    assert resp["error"]["code"] == -32010


async def test_config_set_refuses_malformed_config_and_preserves_file(fake_home: Path) -> None:

    cfg = fake_home / ".codeflow" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    original = '{\n  "tui": {"theme": "dark"},\n  // comment => invalid JSON\n}\n'
    cfg.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        await config_set({"key": "tui.theme", "value": "dracula"})
    assert cfg.read_text(encoding="utf-8") == original


async def test_config_get_refuses_malformed_config(fake_home: Path) -> None:

    cfg = fake_home / ".codeflow" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('{\n  "tui": {"theme": "dark"},\n  // bad\n}\n', encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        await config_get({"keys": ["tui.theme"]})
