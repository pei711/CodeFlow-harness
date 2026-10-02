"""Regression tests for the ``--config`` flag precedence in
``codeflow run`` / ``gateway`` subcommands.

Pre-fix, both subcommands called ``load_codeflow_config()`` BEFORE
``_load_runtime_config(--config)``. Because ``load_codeflow_config()``
reads the module-global ``_current_config_path`` (which
``_load_runtime_config`` sets via ``set_config_path``), the extension
blocks (``skill_forge`` / ``context`` / ``token_wise``)
silently fell back to ``~/.codeflow/config.json`` regardless of the
``--config`` flag.

Real-world impact: every claweval / PinchBench experiment passing a
custom config via ``--config`` got the user's global skill_forge config
instead of the per-experiment one, making A/B comparisons meaningless.

These tests pin the post-fix order so it cannot regress silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _minimal_base_config(skill_forge_block: dict) -> dict:
    """Build a JSON dict valid enough for both ``_load_runtime_config``
    (base Config) and ``load_codeflow_config`` (extension blocks)."""
    return {
        "agents": {
            "defaults": {
                "model": "test-model",
                "provider": "custom",
            }
        },
        "providers": {
            "custom": {
                "apiKey": "test",
                "apiBase": "http://localhost",
            }
        },
        "skill_forge": skill_forge_block,
    }


@pytest.fixture
def isolated_config_state(tmp_path: Path, monkeypatch):
    """Isolate Path.home() AND reset the module-global ``_current_config_path``.

    Without resetting the global, a leaked state from a prior test (or
    the developer's shell) would give false positives — the CLI would
    appear to honor ``--config`` even when the order is wrong.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    import codeflow.config.loader as loader

    monkeypatch.setattr(loader, "_current_config_path", None)

    return fake_home


def test_load_codeflow_config_reads_from_set_config_path(
    isolated_config_state: Path,
    tmp_path: Path,
    monkeypatch,
):
    """Contract: after ``set_config_path(X)``, ``load_codeflow_config()``
    must read from X — not from ``~/.codeflow/config.json``.

    This is the invariant the CLI relies on. If a future refactor
    breaks it (e.g. ``load_codeflow_config`` starts reading
    ``Path.home() / ".codeflow" / "config.json"`` directly), the CLI's
    ``--config`` flag becomes a no-op for extension blocks again.
    """

    global_cfg = isolated_config_state / ".codeflow" / "config.json"
    _write_json(
        global_cfg,
        _minimal_base_config(
            {
                "enabled": True,
                "injection_mode": "summary",
            }
        ),
    )

    custom_cfg = tmp_path / "custom-config.json"
    _write_json(
        custom_cfg,
        _minimal_base_config(
            {
                "enabled": True,
                "injection_mode": "full_body",
                "inject_max": 2,
            }
        ),
    )

    from codeflow.config import loader
    from codeflow.config.codeflow import load_codeflow_config

    cfg_before = load_codeflow_config()
    assert cfg_before.skill_forge.injection_mode == "summary", (
        "Without set_config_path, load_codeflow_config must read the "
        "default ~/.codeflow/config.json (which we stubbed to summary)."
    )

    loader.set_config_path(custom_cfg)
    cfg_after = load_codeflow_config()
    assert cfg_after.skill_forge.injection_mode == "full_body", (
        f"After set_config_path({custom_cfg}), load_codeflow_config "
        f"should read injection_mode from custom (full_body), got "
        f"{cfg_after.skill_forge.injection_mode!r}. "
        f"This means either set_config_path is not honored, or "
        f"load_codeflow_config reads from a hardcoded path."
    )


class _CaptureAndStop(Exception):
    """Raised after we've captured the loaded config, to short-circuit
    the rest of ``run()`` (which would otherwise try to build a real
    AgentLoop, network provider, etc.)."""


@pytest.mark.parametrize("subcommand", ["run", "gateway"])
def test_cli_subcommand_loads_extension_blocks_from_custom_config(
    subcommand: str,
    isolated_config_state: Path,
    tmp_path: Path,
    monkeypatch,
):
    """``codeflow {run,gateway} --config X`` must load skill_forge from
    X, not from ``~/.codeflow/config.json``.

    Before the fix, both subcommands called ``load_codeflow_config()``
    before ``_load_runtime_config(--config)``, so the global config
    won. This test:

      1. Stubs ~/.codeflow/config.json with skill_forge.injection_mode=summary
      2. Writes a custom config with injection_mode=full_body
      3. Patches ``load_codeflow_config`` to capture the loaded value
         then raise (so the rest of the subcommand prologue is skipped).
      4. Invokes ``codeflow {subcommand} --config <custom>``.
      5. Asserts the captured value has injection_mode=full_body.

    If the order regresses, the captured value will be 'summary' and
    the assertion fails with a pointer to the bug.
    """

    global_cfg = isolated_config_state / ".codeflow" / "config.json"
    _write_json(
        global_cfg,
        _minimal_base_config(
            {
                "enabled": True,
                "injection_mode": "summary",
            }
        ),
    )

    custom_cfg = tmp_path / "custom-config.json"
    _write_json(
        custom_cfg,
        _minimal_base_config(
            {
                "enabled": True,
                "injection_mode": "full_body",
                "inject_max": 2,
            }
        ),
    )

    captured: dict = {}
    import codeflow.config.codeflow as ec_module

    real_load = ec_module.load_codeflow_config

    def capture_and_stop():
        captured["ec_config"] = real_load()
        raise _CaptureAndStop()

    monkeypatch.setattr(ec_module, "load_codeflow_config", capture_and_stop)

    from codeflow.cli.commands import app

    runner = CliRunner()
    if subcommand == "run":
        cmd = [
            "run",
            "--config",
            str(custom_cfg),
            "--workspace",
            str(tmp_path / "ws"),
            "--message",
            "test",
            "--no-logs",
        ]
    else:
        cmd = ["gateway", "--config", str(custom_cfg), "--workspace", str(tmp_path / "ws"), "--port", "0"]
    result = runner.invoke(app, cmd, catch_exceptions=True)

    assert "ec_config" in captured, (
        f"load_codeflow_config was never invoked by `codeflow {subcommand}`. "
        f"This means the prologue diverged from the documented pattern "
        f"(or the test stub is wired wrong). "
        f"exit_code={result.exit_code}, stdout={result.stdout!r}"
    )
    sf = captured["ec_config"].skill_forge
    assert sf.injection_mode == "full_body", (
        f"`codeflow {subcommand} --config <X>` IGNORED skill_forge from X. "
        f"Got injection_mode={sf.injection_mode!r} (from global config), "
        f"expected 'full_body' (from --config).\n\n"
        f"Root cause: load_codeflow_config() was called BEFORE "
        f"_load_runtime_config() in codeflow/cli/commands.py:{subcommand}(). "
        f"_load_runtime_config calls set_config_path(--config); "
        f"load_codeflow_config reads that global. Reverse the order to fix."
    )
