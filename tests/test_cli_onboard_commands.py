"""CLI tests for ``codeflow onboard`` - the four-step first-use wizard.

Most tests exercise ``--non-interactive`` so we can drive the wizard
deterministically without a real TTY. Interactive paths are covered by
stubbing the per-step helper functions directly (``_select_provider``,
``_prompt_api_key``, etc.) — that's cheaper and more readable than
patching :mod:`questionary` internals.

Network is mocked at the ops-library boundary
(``codeflow.config.update_providers.test_provider``) and at the first-Turn
boundary (``codeflow.cli.onboard_commands.run_first_turn``).
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from codeflow.cli import onboard_commands
from codeflow.cli.commands import app
from codeflow.config.loader import set_config_path

runner = CliRunner()


def _async_return(value: Any):
    """Build an async method stub that always returns ``value``."""

    async def _login(self, *args, **kwargs):  # noqa: ANN001
        return value

    return _login


def _async_iter(values):
    """Build an async method stub that returns successive ``values`` per call."""

    async def _login(self, *args, **kwargs):  # noqa: ANN001
        return next(values)

    return _login


def _must_not_call(name: str):
    """Build a stub that fails the test if invoked (guards 'never reached').

    Raises ``BaseException`` so a stray call inside a ``try/except Exception``
    (e.g. ``_scancode_login``'s login guard) still surfaces instead of being
    swallowed.
    """

    def _boom(*args, **kwargs):
        raise BaseException(f"{name} should not have been called")  # noqa: TRY002

    return _boom


@pytest.fixture(autouse=True)
def _restore_event_loop():
    """Keep ``asyncio.run`` side effects from leaking across tests.

    ``_scancode_login`` calls ``asyncio.run()``, which closes the loop and
    unsets the thread's current loop. Tests elsewhere that still use the legacy
    ``asyncio.get_event_loop()`` pattern then fail with "no current event loop".
    Hand each test a fresh loop and install another afterward.
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    yield
    asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect config_path + workspace_path under tmp_path; stub template sync.

    ``_bootstrap_empty_config`` uses lazy imports, so we patch the *source*
    modules (``codeflow.config.paths`` / ``codeflow.utils.helpers``) rather
    than the consumer.
    """
    cfg = tmp_path / "config.json"
    workspace = tmp_path / "workspace"
    set_config_path(cfg)
    monkeypatch.setattr(
        "codeflow.config.paths.get_workspace_path",
        lambda: workspace,
    )
    monkeypatch.setattr(
        "codeflow.utils.helpers.sync_workspace_templates",
        lambda _: None,
    )
    yield cfg
    set_config_path(None)  # type: ignore[arg-type]


@pytest.fixture
def stub_verify(monkeypatch: pytest.MonkeyPatch):
    """Default: provider verification succeeds with an empty catalog.

    An empty ``model_ids`` makes ``_pick_model`` fall back to
    ``spec.default_model``, which the non-interactive happy-path tests rely
    on. Tests that need a populated catalog should patch ``test_provider``
    directly with a richer payload.
    """

    def _ok(name: str, *args, **kwargs) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "valid",
            "models_count": 0,
            "model_ids": [],
            "elapsed_ms": 12,
        }

    monkeypatch.setattr("codeflow.config.update_providers.test_provider", _ok)
    return _ok


@pytest.fixture
def stub_step3(monkeypatch: pytest.MonkeyPatch):
    """Default: the first Runtime Turn succeeds. Tests can override."""

    monkeypatch.setattr(onboard_commands, "_prepare_myna", lambda **_: True)
    monkeypatch.setattr(
        onboard_commands,
        "run_first_turn",
        lambda: ("hi there", 24, 0.5),
    )


def test_onboard_help_lists_all_flags() -> None:
    """``codeflow onboard --help`` exposes the full flag surface."""
    r = runner.invoke(app, ["onboard", "--help"])
    assert r.exit_code == 0, r.stdout
    out = r.stdout
    for flag in (
        "--provider",
        "--api-key",
        "--base-url",
        "--model",
        "--channel",
        "--skip-sandbox",
        "--skip-channel",
        "--skip-memory",
        "--non-interactive",
        "--yes",
        "--reset",
    ):
        assert flag in out, f"missing flag in help: {flag}"
    assert "--skip-deep-research" not in out


def test_onboard_non_interactive_minimum_flags(tmp_env: Path, stub_verify, stub_step3) -> None:
    """Minimum non-interactive invocation runs all four steps and writes config."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-fake-test-key",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert "Welcome to the CodeFlow setup wizard" in r.stdout
    assert "Connected" in r.stdout
    assert "Setup complete" in r.stdout

    data = json.loads(tmp_env.read_text())
    assert data["providers"]["openai"]["apiKey"] == "sk-fake-test-key"
    assert data["agents"]["defaults"]["model"] == "openai/gpt-5.5"


def test_onboard_non_interactive_skips_optional_steps(tmp_env: Path, stub_verify, stub_step3) -> None:
    """Non-interactive setup keeps Myna selected and proves the first Turn."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-fake",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert "Keeping run location: host" in r.stdout
    assert "Agent: hi there" in r.stdout
    assert "Setup complete" in r.stdout
    data = json.loads(tmp_env.read_text())
    assert data["memory"]["backend"] == "myna"


def test_onboard_skip_channel_default(tmp_env: Path, stub_verify, stub_step3) -> None:
    """``--skip-channel`` produces the dim skip line in Step 3."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-fake",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0
    assert "Skipped via --skip-channel" in r.stdout


def test_onboard_non_interactive_missing_provider_fails(tmp_env: Path) -> None:
    """Without ``--provider`` non-interactive mode can't proceed."""
    r = runner.invoke(
        app,
        ["onboard", "--non-interactive", "--skip-channel", "--yes"],
    )
    assert r.exit_code != 0
    assert "--provider is required" in r.stderr


def test_onboard_non_interactive_custom_requires_base_url(
    tmp_env: Path,
) -> None:
    """``custom`` provider needs ``--base-url`` when non-interactive."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "custom",
            "--api-key",
            "sk-fake",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code != 0
    assert "--base-url is required" in r.stderr


def test_onboard_oauth_non_interactive_errors(tmp_env: Path) -> None:
    """OAuth providers can't run headless — wizard must surface that."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "github_copilot",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code != 0
    assert "OAuth providers require an interactive browser flow" in r.stdout


def test_onboard_non_tty_no_flag_fails(tmp_env: Path) -> None:
    """Without a TTY and without ``--non-interactive`` we give a clear hint.

    ``CliRunner`` captures stdout into a buffer, so ``isatty()`` already
    returns False here — no extra patching needed to trigger the bail.
    """
    r = runner.invoke(app, ["onboard"])
    assert r.exit_code == 2
    assert "Non-interactive terminal detected" in r.stdout


def test_onboard_existing_config_blocks_without_yes(tmp_env: Path, stub_verify, stub_step3) -> None:
    """Re-running over an existing populated config fails closed."""

    runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-existing",
            "--skip-channel",
            "--yes",
        ],
    )

    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "anthropic",
            "--api-key",
            "sk-newer",
            "--skip-channel",
        ],
    )
    assert r.exit_code == 2
    assert "Existing config detected" in r.stdout

    data = json.loads(tmp_env.read_text())
    assert data["providers"]["openai"]["apiKey"] == "sk-existing"


def test_onboard_reset_flag_forces_redo(tmp_env: Path, stub_verify, stub_step3) -> None:
    """``--reset`` bypasses the existing-config guard."""
    runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-old",
            "--skip-channel",
            "--yes",
        ],
    )
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-new",
            "--skip-channel",
            "--reset",
        ],
    )
    assert r.exit_code == 0, r.stdout
    data = json.loads(tmp_env.read_text())
    assert data["providers"]["openai"]["apiKey"] == "sk-new"


def test_onboard_provider_test_failure_warns_but_continues(
    tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_step3
) -> None:
    """``test_provider`` failure should warn + continue in non-interactive mode."""

    def _fail(name: str, *args, **kwargs) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "invalid_key",
            "models_count": None,
            "elapsed_ms": 5,
            "error": "401 Unauthorized",
        }

    monkeypatch.setattr("codeflow.config.update_providers.test_provider", _fail)

    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-bad",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0
    assert "Auth failed" in r.stdout

    assert "didn't pass a connectivity test" in r.stdout


def test_onboard_test_probe_failure_shows_warning_footer(
    tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_verify
) -> None:
    """When the Step 1 test message raises, the footer must reflect the failure."""

    def _boom() -> tuple[str, int | None, float]:
        raise RuntimeError("AuthenticationError: bogus key")

    monkeypatch.setattr(onboard_commands, "_prepare_myna", lambda **_: True)
    monkeypatch.setattr(onboard_commands, "run_first_turn", _boom)

    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-fake",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0
    assert "Test failed" in r.stdout
    assert "Setup finished" in r.stdout
    assert "Setup complete" not in r.stdout
    assert "didn't pass a connectivity test" in r.stdout


def test_onboard_interactive_uses_stubbed_pickers(
    tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_verify, stub_step3
) -> None:
    """Interactive path: stub the per-step helpers and assert ops-lib is hit."""

    monkeypatch.setattr(onboard_commands, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(onboard_commands, "_pick_language", lambda: None)
    monkeypatch.setattr(onboard_commands, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(onboard_commands, "_prompt_api_key", lambda provider, **kw: "sk-int-test")

    monkeypatch.setattr(
        onboard_commands,
        "_pick_model",
        lambda spec, **_: spec.default_model,
    )

    monkeypatch.setattr(onboard_commands, "_step3_sandbox", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step4_channel", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step2_memory", lambda **_: None)

    r = runner.invoke(app, ["onboard"])
    assert r.exit_code == 0, r.stdout

    data = json.loads(tmp_env.read_text())
    assert data["providers"]["anthropic"]["apiKey"] == "sk-int-test"
    assert data["agents"]["defaults"]["model"] == "anthropic/claude-sonnet-5"


def test_step1_writes_via_ops_lib(tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_verify) -> None:
    """Step 1's write path must go through ``set_provider_fields``."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def _spy(name: str, fields: dict[str, Any], **_) -> dict[str, Any]:
        calls.append((name, dict(fields)))
        return {}

    monkeypatch.setattr("codeflow.config.update_providers.set_provider_fields", _spy)
    monkeypatch.setattr(onboard_commands, "_prepare_myna", lambda **_: True)
    monkeypatch.setattr(onboard_commands, "run_first_turn", lambda: ("hi", 1, 0.1))

    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-spy",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert calls, "set_provider_fields was never called"
    name, fields = calls[0]
    assert name == "openai"
    assert fields == {"api_key": "sk-spy"}


def test_styles_module_loads() -> None:
    """``_styles.py`` import must not crash and must export ``CODEFLOW_STYLE``."""
    from codeflow.cli._styles import CODEFLOW_STYLE  # noqa: F401

    assert CODEFLOW_STYLE is not None


def test_run_first_turn_uses_public_run_command(
    tmp_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class _Completed:
        returncode = 0
        stdout = "Hello from the Runtime\n"
        stderr = ""

    def _run(command: list[str], **kwargs: Any) -> _Completed:
        calls.append((command, kwargs))
        return _Completed()

    monkeypatch.setattr(onboard_commands.subprocess, "run", _run)

    text, tokens, elapsed = onboard_commands.run_first_turn()

    command, kwargs = calls[0]
    assert command[:4] == [onboard_commands.sys.executable, "-m", "codeflow.cli.commands", "run"]
    assert command[4:6] == ["-m", onboard_commands.DEFAULT_PROBE_MESSAGE]
    assert command[-2:] == ["--config", str(tmp_env)]
    assert kwargs["timeout"] == 120
    assert text == "Hello from the Runtime"
    assert tokens is None
    assert elapsed >= 0


def test_step1_model_flag_overrides_picker(tmp_env: Path, stub_verify, stub_step3) -> None:
    """``--model X`` short-circuits the picker, even when a catalog exists."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openrouter",
            "--api-key",
            "sk-or-fake",
            "--model",
            "openrouter/openai/gpt-4o",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    data = json.loads(tmp_env.read_text())
    assert data["agents"]["defaults"]["model"] == "openrouter/openai/gpt-4o"


def test_step1_falls_back_to_spec_default_in_non_interactive(tmp_env: Path, stub_verify, stub_step3) -> None:
    """Without --model + non-interactive → write whatever ProviderSpec says."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "anthropic",
            "--api-key",
            "sk-ant-fake",
            "--skip-channel",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    data = json.loads(tmp_env.read_text())
    assert data["agents"]["defaults"]["model"] == "anthropic/claude-sonnet-5"


def test_step1_picker_uses_catalog_when_available(tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_step3) -> None:
    """When ``/v1/models`` returns a list and we're interactive, the picker
    feeds that list to ``questionary.autocomplete`` and writes the choice."""

    captured_choices: dict[str, list[str]] = {}

    def _ok_with_catalog(name: str, *args, **kwargs) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "valid",
            "models_count": 3,
            "model_ids": ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"],
            "elapsed_ms": 9,
        }

    monkeypatch.setattr("codeflow.config.update_providers.test_provider", _ok_with_catalog)
    monkeypatch.setattr(onboard_commands, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(onboard_commands, "_pick_language", lambda: None)
    monkeypatch.setattr(onboard_commands, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(onboard_commands, "_prompt_api_key", lambda provider, **kw: "sk-ant-test")

    import questionary

    class _FakeQuestion:
        def __init__(self, answer: Any) -> None:
            self._answer = answer

        def ask(self) -> Any:
            return self._answer

    def _fake_autocomplete(message, choices, default=None, **kwargs):
        captured_choices["choices"] = list(choices)
        captured_choices["default"] = default
        return _FakeQuestion("claude-haiku-4-5")

    monkeypatch.setattr(questionary, "autocomplete", _fake_autocomplete)
    monkeypatch.setattr(onboard_commands, "_step3_sandbox", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step4_channel", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step2_memory", lambda **_: None)

    r = runner.invoke(app, ["onboard"])
    assert r.exit_code == 0, r.stdout

    assert captured_choices["choices"] == [
        "anthropic/claude-opus-4-5",
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    ]
    assert captured_choices["default"] == "anthropic/claude-opus-4-5"

    data = json.loads(tmp_env.read_text())
    assert data["agents"]["defaults"]["model"] == "claude-haiku-4-5"


def test_format_model_for_provider_prefix_rules() -> None:
    """Provider's ``litellm_prefix`` is applied unless model_id already has one."""
    from codeflow.providers.registry import find_by_name

    openrouter = find_by_name("openrouter")
    deepseek = find_by_name("deepseek")
    openai = find_by_name("openai")

    assert (
        onboard_commands._format_model_for_provider(openrouter, "anthropic/claude-sonnet-4-5")
        == "openrouter/anthropic/claude-sonnet-4-5"
    )

    assert (
        onboard_commands._format_model_for_provider(openrouter, "openrouter/anthropic/claude-sonnet-4-5")
        == "openrouter/anthropic/claude-sonnet-4-5"
    )

    assert onboard_commands._format_model_for_provider(openai, "gpt-4o-mini") == "gpt-4o-mini"

    assert onboard_commands._format_model_for_provider(deepseek, "deepseek/deepseek-chat") == "deepseek/deepseek-chat"
    assert onboard_commands._format_model_for_provider(deepseek, "deepseek-chat") == "deepseek/deepseek-chat"


def test_model_routes_to_provider_heuristic() -> None:
    """Mirror of ``Config._match_provider``: prefix match wins, else keyword."""
    from codeflow.providers.registry import find_by_name

    openrouter = find_by_name("openrouter")
    anthropic = find_by_name("anthropic")
    openai = find_by_name("openai")

    assert onboard_commands._model_routes_to_provider("openrouter/anthropic/claude-sonnet-4-5", openrouter)

    assert not onboard_commands._model_routes_to_provider("openrouter/anthropic/claude-sonnet-4-5", anthropic)

    assert onboard_commands._model_routes_to_provider("claude-sonnet-4-5", anthropic)
    assert onboard_commands._model_routes_to_provider("gpt-4o-mini", openai)

    assert not onboard_commands._model_routes_to_provider("gemini-2.5-flash", openai)

    assert not onboard_commands._model_routes_to_provider("", anthropic)
    assert not onboard_commands._model_routes_to_provider("claude", None)


def test_registry_default_models_present() -> None:
    """Each curated provider must carry a ``default_model`` in its ``ProviderSpec``."""
    from codeflow.providers.registry import find_by_name

    for name in (
        "openrouter",
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "github_copilot",
        "openai_codex",
    ):
        spec = find_by_name(name)
        assert spec is not None, f"missing provider in registry: {name}"
        assert spec.default_model, f"{name} has empty default_model"


def _seed_provider(provider: str = "openai", key: str = "sk-seed", model: str = "openai/gpt-4o-mini") -> None:
    """Write a minimal populated config via the ops layer."""
    from codeflow.config.update import set_default_model
    from codeflow.config.update_providers import set_provider_fields

    set_provider_fields(provider, {"api_key": key})
    set_default_model(model)


def test_is_config_populated_requires_provider_and_model(tmp_env: Path) -> None:
    """Gate criterion: provider key + default model are BOTH required."""
    from codeflow.config.update import set_default_model
    from codeflow.config.update_providers import set_provider_fields

    assert onboard_commands._is_config_populated() is False
    set_provider_fields("openai", {"api_key": "sk-x"})

    data = json.loads(tmp_env.read_text()) if tmp_env.exists() else {}
    if not data.get("agents", {}).get("defaults", {}).get("model"):
        assert onboard_commands._is_config_populated() is False
    set_default_model("openai/gpt-4o-mini")
    assert onboard_commands._is_config_populated() is True


def test_ensure_configured_short_circuits_when_complete(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate returns True (no wizard) when config is already complete."""
    _seed_provider()
    ran: list[bool] = []
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **_: ran.append(True))
    assert onboard_commands.ensure_configured_or_onboard() is True
    assert ran == []


def test_ensure_configured_runs_wizard_when_missing(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate runs the wizard when the required config is missing."""
    ran: list[bool] = []
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **_: ran.append(True))
    assert onboard_commands.ensure_configured_or_onboard() is False
    assert ran == [True]


def test_run_gate_triggers_when_missing(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``codeflow run`` (interactive, TTY, missing config) enters the wizard."""
    from codeflow.cli import agent_commands

    monkeypatch.setattr(agent_commands, "_stdout_isatty", lambda: True)
    gate_called: list[bool] = []

    def _gate(**_):
        gate_called.append(True)
        raise typer.Exit(0)

    monkeypatch.setattr(onboard_commands, "ensure_configured_or_onboard", _gate)

    r = runner.invoke(app, ["run"])
    assert gate_called == [True]
    assert r.exit_code == 0


def test_run_gate_skips_when_populated(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``codeflow run`` with complete config does not enter the wizard."""
    from codeflow.cli import agent_commands

    _seed_provider()
    monkeypatch.setattr(agent_commands, "_stdout_isatty", lambda: True)
    gate_called: list[bool] = []
    monkeypatch.setattr(
        onboard_commands,
        "ensure_configured_or_onboard",
        lambda **_: gate_called.append(True),
    )

    def _boom(*a, **kw):
        raise typer.Exit(0)

    monkeypatch.setattr("codeflow.cli._helpers.load_runtime_config", _boom)
    runner.invoke(app, ["run"])

    assert gate_called == []


def test_run_gate_skips_oneshot_message(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``codeflow run -m '...'`` (one-shot) must not enter the wizard even on a
    TTY with missing config — scripted use fails loudly later instead."""
    from codeflow.cli import agent_commands

    monkeypatch.setattr(agent_commands, "_stdout_isatty", lambda: True)
    gate_called: list[bool] = []
    monkeypatch.setattr(
        onboard_commands,
        "ensure_configured_or_onboard",
        lambda **_: gate_called.append(True),
    )
    monkeypatch.setattr(
        "codeflow.cli._helpers.load_runtime_config",
        lambda *a, **kw: (_ for _ in ()).throw(typer.Exit(0)),
    )
    runner.invoke(app, ["run", "-m", "hi"])
    assert gate_called == []


def test_run_gate_skips_non_tty(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-TTY (piped) ``codeflow run`` must not enter the wizard."""
    from codeflow.cli import agent_commands

    monkeypatch.setattr(agent_commands, "_stdout_isatty", lambda: False)
    gate_called: list[bool] = []
    monkeypatch.setattr(
        onboard_commands,
        "ensure_configured_or_onboard",
        lambda **_: gate_called.append(True),
    )
    monkeypatch.setattr(
        "codeflow.cli._helpers.load_runtime_config",
        lambda *a, **kw: (_ for _ in ()).throw(typer.Exit(0)),
    )
    runner.invoke(app, ["run"])
    assert gate_called == []


def test_tui_gate_triggers_when_missing(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare ``codeflow`` enters onboarding before launching the native TUI."""
    from codeflow.cli import tui_commands

    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: True)
    gate_called: list[bool] = []

    def _gate(**_):
        gate_called.append(True)
        raise typer.Exit(0)

    monkeypatch.setattr(onboard_commands, "ensure_configured_or_onboard", _gate)
    r = runner.invoke(app, [])
    assert gate_called == [True]
    assert r.exit_code == 0


def test_tui_gate_skips_check_flag(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``codeflow --check`` bypasses onboarding."""
    from codeflow.cli import tui_commands

    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: True)
    gate_called: list[bool] = []
    monkeypatch.setattr(
        onboard_commands,
        "ensure_configured_or_onboard",
        lambda **_: gate_called.append(True),
    )

    monkeypatch.setattr(tui_commands, "find_node", lambda: (None, None))
    runner.invoke(app, ["--check"])
    assert gate_called == []


def test_sandbox_backend_persisted_via_ops(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Picking 'host' writes sandbox.backend=none through the ops layer."""
    import questionary

    class _FQ:
        def __init__(self, a):
            self._a = a

        def ask(self):
            return self._a

    monkeypatch.setattr(questionary, "select", lambda *a, **kw: _FQ("none"))
    onboard_commands._step3_sandbox(skip=False, non_interactive=False)
    data = json.loads(tmp_env.read_text())
    assert data["tools"]["sandbox"]["backend"] == "none"


def test_sandbox_boxlite_probe_failure_falls_back(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Boxlite probe failure → submenu → fall back to host."""
    import questionary

    answers = iter(["boxlite"])

    class _FQ:
        def __init__(self, a):
            self._a = a

        def ask(self):
            return self._a

    monkeypatch.setattr(questionary, "select", lambda *a, **kw: _FQ(next(answers)))
    monkeypatch.setattr(onboard_commands, "_probe_boxlite", lambda: (False, "missing"))

    monkeypatch.setattr(onboard_commands, "_failure_choice", lambda options, *, non_interactive: "host")
    onboard_commands._step3_sandbox(skip=False, non_interactive=False)
    data = json.loads(tmp_env.read_text())
    assert data["tools"]["sandbox"]["backend"] == "none"


def test_sandbox_keep_current_first_option(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-configured sandbox offers a 'keep current' first choice."""
    from codeflow.config.update import set_sandbox_backend

    set_sandbox_backend("boxlite")
    captured: dict[str, list] = {}
    import questionary

    class _FQ:
        def ask(self):
            return "keep"

    def _select(message, choices, **kw):
        captured["choices"] = [getattr(c, "value", c) for c in choices]
        return _FQ()

    monkeypatch.setattr(questionary, "select", _select)
    onboard_commands._step3_sandbox(skip=False, non_interactive=False)
    assert "keep" in captured["choices"]

    assert json.loads(tmp_env.read_text())["tools"]["sandbox"]["backend"] == "boxlite"


def test_memory_skip_sets_backend_null(
    tmp_env: Path,
) -> None:
    """The explicit skip path selects the only supported Memory-off value."""
    onboard_commands._step2_memory(
        skip=True,
        non_interactive=False,
        main_model="openai/gpt-4o-mini",
        warnings=[],
        skip_test=True,
    )
    data = json.loads(tmp_env.read_text())
    assert data["memory"]["backend"] is None
    from codeflow.config.codeflow import load_codeflow_config

    assert load_codeflow_config().memory.backend is None


def test_memory_step_selects_myna_without_codeflow_side_overrides(
    tmp_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Onboarding chooses the installed Adapter without writing its configuration."""
    onboard_commands._set_memory_backend("myna")
    monkeypatch.setattr(onboard_commands, "_prepare_myna", lambda **_: True)
    monkeypatch.setattr(onboard_commands, "run_first_turn", lambda: ("ready", 1, 0.1))
    onboard_commands._step2_memory(
        skip=False,
        non_interactive=False,
        main_model="openrouter/anthropic/claude-sonnet-4-5",
        warnings=[],
    )

    data = json.loads(tmp_env.read_text())
    assert data["memory"]["backend"] == "myna"
    assert "myna" not in data.get("plugins", {}).get("config", {})
    assert "myna-memory" not in data.get("plugins", {}).get("config", {})
    from codeflow.config.codeflow import load_codeflow_config

    assert load_codeflow_config().memory.backend == "myna"


def _install_fake_myna_descriptor(monkeypatch: pytest.MonkeyPatch, descriptor: Any) -> None:
    package = types.ModuleType("myna")
    package.__path__ = []  # type: ignore[attr-defined]
    integrations = types.ModuleType("myna.integrations")
    integrations.__path__ = []  # type: ignore[attr-defined]
    codeflow = types.ModuleType("myna.integrations.codeflow")
    codeflow.descriptor = lambda: descriptor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "myna", package)
    monkeypatch.setitem(sys.modules, "myna.integrations", integrations)
    monkeypatch.setitem(sys.modules, "myna.integrations.codeflow", codeflow)


def test_prepare_myna_applies_consent_bound_repository_setup(
    tmp_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    onboard_commands._bootstrap_empty_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.inspect_memory_backend",
        lambda config: SimpleNamespace(state="available", error=None),
    )
    applied: list[str] = []
    descriptor = SimpleNamespace(
        protocol="myna.codeflow-app-integration.v1",
        preview_setup=lambda workspace: {
            "state": "setup_required",
            "workspace": str(workspace),
            "repo_key": "example/repository",
            "creates": [str(tmp_path / ".git" / "myna.toml"), str(tmp_path / "runtime")],
            "retrieval": {"profile": "fastembed"},
            "consent_token": "consent-token",
        },
        apply_setup=lambda token: applied.append(token) or {"state": "initialized", "repo_key": "example/repository"},
    )
    _install_fake_myna_descriptor(monkeypatch, descriptor)

    assert onboard_commands._prepare_myna(non_interactive=True, yes=True) is True
    assert applied == ["consent-token"]
    assert json.loads(tmp_env.read_text())["memory"]["backend"] == "myna"


def test_prepare_myna_fails_closed_when_plugin_is_missing(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    onboard_commands._bootstrap_empty_config()
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.inspect_memory_backend",
        lambda config: SimpleNamespace(state="error", error="plugin unavailable"),
    )

    with pytest.raises(typer.Exit) as exc_info:
        onboard_commands._prepare_myna(non_interactive=True, yes=True)

    assert exc_info.value.exit_code == 2
    assert json.loads(tmp_env.read_text())["memory"]["backend"] == "myna"


def test_retained_channels_do_not_use_interactive_login() -> None:
    f = onboard_commands._channel_uses_interactive_login
    assert all(f(name) is False for name in ("feishu", "qq", "wecom"))


def test_channel_order_contains_only_retained_channels() -> None:
    assert onboard_commands._ordered_channel_names() == ["feishu", "qq", "wecom"]


@pytest.mark.parametrize("channel", ["qq", "wecom"])
def test_beta_channel_selection_prints_one_honest_note(channel: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Picking a Beta channel states its evidence level without claiming a
    live integration."""
    onboard_commands._print_maturity_note(channel)
    out = " ".join(capsys.readouterr().out.split())
    assert f"{channel} is Beta" in out
    assert "deterministic contract and security checks only" in out
    assert "no live send/receive evidence yet" in out


def test_live_gated_channel_prints_no_beta_note(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A live-gated adapter does not print the Beta evidence warning."""
    onboard_commands._print_maturity_note("feishu")
    assert capsys.readouterr().out.strip() == ""


def test_add_one_channel_announces_maturity_before_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The note lands before any credential prompt, so the user sees it before
    committing secrets."""
    order: list[str] = []
    monkeypatch.setattr(onboard_commands, "_select_channel", lambda: "qq")
    monkeypatch.setattr(
        onboard_commands,
        "_print_maturity_note",
        lambda name: order.append(f"note:{name}"),
    )
    monkeypatch.setattr(onboard_commands, "_prompt_channel_fields", lambda name: order.append(f"prompt:{name}") or {})
    monkeypatch.setattr(onboard_commands, "_enable_channel", lambda name, fields: order.append(f"enable:{name}"))

    onboard_commands._add_one_channel()

    assert order == ["note:qq", "prompt:qq", "enable:qq"]


def test_provider_remove_clears_key(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing a provider clears its api_key (disable, not hard-delete)."""
    from codeflow.config.update_providers import set_provider_fields

    set_provider_fields("openai", {"api_key": "sk-a"})
    set_provider_fields("anthropic", {"api_key": "sk-b"})

    import questionary

    class _FQ:
        def __init__(self, a):
            self._a = a

        def ask(self):
            return self._a

    select_answers = iter(["anthropic", "remove", onboard_commands._BACK])
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: _FQ(next(select_answers)))

    onboard_commands._manage_existing_providers(non_interactive=False)
    data = json.loads(tmp_env.read_text())
    assert not data["providers"]["anthropic"].get("apiKey")
    assert data["providers"]["openai"]["apiKey"] == "sk-a"

    assert onboard_commands._configured_providers() == ["openai"]


def test_provider_picker_back_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider picker surfaces a back sentinel choice."""
    import questionary

    captured: dict[str, list] = {}

    class _FQ:
        def ask(self):
            return onboard_commands._BACK

    def _select(message, choices, **kw):
        captured["values"] = [getattr(c, "value", None) for c in choices]
        return _FQ()

    monkeypatch.setattr(questionary, "select", _select)
    result = onboard_commands._select_provider()
    assert result is onboard_commands._BACK
    assert onboard_commands._BACK in captured["values"]


def test_back_navigation_rewinds_one_screen(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A screen returning _BACK rewinds the state machine by one index."""
    calls: list[str] = []

    def _s1(**_):
        calls.append("s1")
        return None

    def _s2(**_):
        calls.append("s2")

        return onboard_commands._BACK if calls.count("s2") == 1 else None

    def _s3(**_):
        calls.append("s3")
        return None

    monkeypatch.setattr(onboard_commands, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(onboard_commands, "_pick_language", lambda: None)
    monkeypatch.setattr(onboard_commands, "_handle_existing_config", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_bootstrap_empty_config", lambda: None)
    monkeypatch.setattr(onboard_commands, "_step1_provider", _s1)
    monkeypatch.setattr(onboard_commands, "_step3_sandbox", _s3)
    monkeypatch.setattr(onboard_commands, "_step4_channel", _s3)
    monkeypatch.setattr(onboard_commands, "_step2_memory", _s2)

    onboard_commands.run_wizard(non_interactive=False)

    assert calls == ["s1", "s2", "s1", "s2", "s3", "s3"]


def test_first_screen_back_does_not_skip_step1(
    tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_verify, stub_step3
) -> None:
    """BUG-1 regression: Back on the first screen must NOT skip required Step 1.

    Drives the REAL ``_step1_provider``: the picker first returns the back
    sentinel (which used to fall through and skip provider config entirely,
    leaving config unpopulated and re-tripping the gate), then a real provider.
    The wizard must re-display Step 1 and only advance once a provider+model
    are written.
    """
    picks = iter([onboard_commands._BACK, "openai"])
    monkeypatch.setattr(onboard_commands, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(onboard_commands, "_pick_language", lambda: None)
    monkeypatch.setattr(onboard_commands, "_select_provider", lambda: next(picks))
    monkeypatch.setattr(onboard_commands, "_prompt_api_key", lambda provider, **kw: "sk-back-test")
    monkeypatch.setattr(onboard_commands, "_pick_model", lambda spec, **_: spec.default_model)

    monkeypatch.setattr(onboard_commands, "_step3_sandbox", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step4_channel", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step2_memory", lambda **_: None)

    onboard_commands.run_wizard(non_interactive=False)

    data = json.loads(tmp_env.read_text())
    assert data["providers"]["openai"]["apiKey"] == "sk-back-test"
    assert data["agents"]["defaults"]["model"] == "openai/gpt-5.5"
    assert onboard_commands._is_config_populated() is True


def test_switch_provider_returns_to_picker_keeps_steps(
    tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_step3
) -> None:
    """BUG-2 regression: 'Switch provider' on a verify failure re-runs the
    picker instead of exiting the whole wizard."""

    calls = {"n": 0}

    def _verify(name, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "ok": False,
                "status": "invalid_key",
                "models_count": None,
                "model_ids": None,
                "elapsed_ms": 1,
                "error": "401",
            }
        return {"ok": True, "status": "valid", "models_count": 0, "model_ids": [], "elapsed_ms": 1}

    monkeypatch.setattr("codeflow.config.update_providers.test_provider", _verify)
    monkeypatch.setattr(onboard_commands, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(onboard_commands, "_pick_language", lambda: None)

    picks = iter(["anthropic", "openai"])
    monkeypatch.setattr(onboard_commands, "_select_provider", lambda: next(picks))
    monkeypatch.setattr(onboard_commands, "_prompt_api_key", lambda provider, **kw: f"sk-{provider}")
    monkeypatch.setattr(onboard_commands, "_pick_model", lambda spec, **_: spec.default_model)

    monkeypatch.setattr(onboard_commands, "_failure_choice", lambda options, *, non_interactive: "switch")
    monkeypatch.setattr(onboard_commands, "_step3_sandbox", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step4_channel", lambda **_: None)
    monkeypatch.setattr(onboard_commands, "_step2_memory", lambda **_: None)

    onboard_commands.run_wizard(non_interactive=False)
    data = json.loads(tmp_env.read_text())

    assert data["providers"]["openai"]["apiKey"] == "sk-openai"
    assert data["agents"]["defaults"]["model"] == "openai/gpt-5.5"


def test_add_provider_keeps_existing(tmp_env: Path, monkeypatch: pytest.MonkeyPatch, stub_verify, stub_step3) -> None:
    """Adding a second provider in the existing-config entry doesn't drop the first."""
    _seed_provider("openai", "sk-first", "openai/gpt-4o-mini")

    import questionary

    class _FQ:
        def __init__(self, a):
            self._a = a

        def ask(self):
            return self._a

    entry_answers = iter(["add", "done"])
    monkeypatch.setattr(questionary, "select", lambda *a, **kw: _FQ(next(entry_answers)))
    monkeypatch.setattr(onboard_commands, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(onboard_commands, "_prompt_api_key", lambda provider, **kw: "sk-second")
    monkeypatch.setattr(onboard_commands, "_pick_model", lambda spec, **_: spec.default_model)

    onboard_commands._step1_provider(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        non_interactive=False,
        warnings=[],
    )

    data = json.loads(tmp_env.read_text())
    assert data["providers"]["openai"]["apiKey"] == "sk-first"
    assert data["providers"]["anthropic"]["apiKey"] == "sk-second"


def test_skip_memory_disables_backend_effective(tmp_env: Path, stub_verify, stub_step3) -> None:
    """The explicit skip leaves effective memory.backend=None."""
    r = runner.invoke(
        app,
        [
            "onboard",
            "--non-interactive",
            "--provider",
            "openai",
            "--api-key",
            "sk-fake",
            "--skip-channel",
            "--skip-memory",
            "--yes",
        ],
    )
    assert r.exit_code == 0, r.stdout
    from codeflow.config.codeflow import load_codeflow_config

    assert load_codeflow_config().memory.backend is None


def test_fresh_bootstrap_defaults_memory_backend_myna(
    tmp_env: Path, stub_verify, stub_step3, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh config selects the installed Myna contribution."""
    onboard_commands._bootstrap_empty_config()
    from codeflow.config.codeflow import load_codeflow_config

    assert load_codeflow_config().memory.backend == "myna"


def test_fresh_bootstrap_seeds_extension_blocks(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap materializes the memory / plugins / skillForge safe subset so a
    fresh config exposes the knobs without writing optional service endpoints
    or bearer tokens into the user's plaintext config."""
    onboard_commands._bootstrap_empty_config()
    data = json.loads(tmp_env.read_text())

    assert data["memory"]["backend"] == "myna"
    assert data["memory"]["memoryTopK"] == 5
    assert data["plugins"]["config"] == {}
    assert data["skillForge"]["router"] == {"enabled": True}
    assert "hub" not in data["skillForge"]["router"]

    for leaked in ("embeddingApiKey", "rerankerApiKey", "massLibraryDb"):
        assert leaked not in data["skillForge"]


def test_bootstrap_backfills_preexisting_config(tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config that predates the extension blocks gets them backfilled on the
    next onboard — without clobbering values the user already set."""

    tmp_env.write_text(
        json.dumps(
            {
                "providers": {"openai": {"apiKey": "sk-keep"}},
                "agents": {"defaults": {"model": "openai/gpt-4o"}},
                "memory": {"backend": None, "memoryTopK": 20},
            }
        )
    )

    onboard_commands._bootstrap_empty_config()
    data = json.loads(tmp_env.read_text())

    assert data["providers"]["openai"]["apiKey"] == "sk-keep"
    assert data["memory"]["backend"] is None
    assert data["memory"]["memoryTopK"] == 20

    assert data["memory"]["userId"] == "default"
    assert data["plugins"]["config"] == {}
    assert data["skillForge"]["router"] == {"enabled": True}


def test_prompt_channel_fields_gates_skip_on_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional fields get an ``(optional)`` label + skip hint; a required field
    that is not the first prompt (feishu ``app_secret``) must NOT show a skip
    hint. Regression guard for the ``idx>0`` heuristic that told users they
    could skip a required credential.
    """
    import questionary

    monkeypatch.setattr(onboard_commands, "_LANG", "en")
    captured: list[tuple[str, Any]] = []

    class _Prompt:
        def __init__(self, label: str, placeholder: Any = None, **_: Any) -> None:
            self._label = label
            self._placeholder = placeholder

        def ask(self) -> str:
            captured.append((self._label, self._placeholder))
            return "x"

    monkeypatch.setattr(questionary, "text", lambda label, **kw: _Prompt(label, **kw))
    monkeypatch.setattr(questionary, "password", lambda label, **kw: _Prompt(label, **kw))

    onboard_commands._prompt_channel_fields("feishu")

    def _ph_text(placeholder: Any) -> Any:
        return placeholder[0][1] if placeholder else None

    app_id_lbl, app_id_ph = captured[0]
    app_secret_lbl, app_secret_ph = captured[1]
    encrypt_lbl, encrypt_ph = captured[2]

    assert "(optional)" not in app_id_lbl
    assert "(optional)" not in app_secret_lbl
    assert "(optional)" in encrypt_lbl

    assert "back" in _ph_text(app_id_ph)
    assert app_secret_ph is None
    assert "skip" in _ph_text(encrypt_ph)


def test_total_steps_is_four() -> None:
    assert onboard_commands._TOTAL_STEPS == 4


def test_load_raw_config_raises_on_malformed(tmp_env: Path) -> None:

    from codeflow.config.loader import ConfigReadError

    tmp_env.write_text("{  // comment => invalid JSON\n}", encoding="utf-8")
    with pytest.raises(ConfigReadError):
        onboard_commands._load_raw_config()
