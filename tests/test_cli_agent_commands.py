"""CLI tests for ``codeflow run``.

The ``run`` command is an interactive REPL with optional ``-m`` single-turn
mode. Smoke-level coverage: ``--help`` works, options are surfaced, the
``no-API-key`` path exits cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeflow.cli.commands import app
from codeflow.config.loader import set_config_path

runner = CliRunner()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    set_config_path(cfg)
    yield cfg
    set_config_path(None)  # type: ignore[arg-type]


def test_run_help_works() -> None:
    """``codeflow run --help`` lists the key options."""
    r = runner.invoke(app, ["run", "--help"])
    assert r.exit_code == 0
    assert "Interact with the agent" in r.stdout

    assert "--message" in r.stdout
    assert "--session" in r.stdout
    assert "--workspace" in r.stdout
    assert "--config" in r.stdout
    assert "--markdown" in r.stdout


def test_run_without_api_key_exits_cleanly(tmp_config: Path) -> None:
    """With no provider configured, the command must exit non-zero — and
    crucially must not raise a *crash* exception (NameError / AttributeError /
    ImportError). ``typer.testing.CliRunner`` captures the exception, so the
    only reliable way to detect a regression like a missing import is to
    inspect ``r.exception`` directly.
    """
    from codeflow.config.loader import save_config
    from codeflow.config.schema import Config

    save_config(Config())

    r = runner.invoke(app, ["run", "-m", "hello"])

    if r.exception is not None:
        assert not isinstance(r.exception, (NameError, AttributeError, ImportError)), (
            f"Crash-class exception leaked through: {r.exception!r}"
        )
    assert r.exit_code != 0


# ============================================================================

# ============================================================================


def test_run_help_shows_continue_flag() -> None:
    """--continue flag appears in run --help."""
    r = runner.invoke(app, ["run", "--help"])
    assert r.exit_code == 0
    assert "--continue" in r.stdout


def test_run_help_shows_resume_flag() -> None:
    """--resume flag appears in run --help."""
    r = runner.invoke(app, ["run", "--help"])
    assert r.exit_code == 0
    assert "--resume" in r.stdout


def _invoke_agent_capturing_session(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path | None,
    extra_args: list[str],
    *,
    fail_turn: bool = False,
) -> tuple[object, dict[str, str]]:
    """Run ``run -m`` with the provider and AgentLoop stubbed out, capturing
    the session_id that reaches the spine turn (req.conversation is the session
    key, mirroring the old session_key arg)."""
    import os as _os

    from codeflow.config.loader import save_config
    from codeflow.config.schema import Config
    from codeflow.spine import Text, TurnOutcome, Usage

    cfg = Config()
    cfg.providers.openrouter.api_key = "stub-test-key"
    save_config(cfg)

    captured: dict[str, str] = {}

    class _StubSubagents:
        def set_submit(self, _submit) -> None:
            pass

    class _StubAgentLoop:
        def __init__(self, **kwargs):
            self.channels_config = kwargs.get("channels_config")
            self.subagents = _StubSubagents()

        def configure_personalization(self, *_args) -> None:
            pass

        async def run_turn(self, req, emit, drain, *, stream, **_kw) -> TurnOutcome:
            captured["session_id"] = req.conversation
            if fail_turn:
                raise RuntimeError("provider offline")
            await emit(Text(content="stub-response", source=req.source))
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

        async def close(self) -> None:
            pass

        def begin_close(self) -> None:
            pass

    monkeypatch.setattr(_os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr("codeflow.cli.agent_commands.make_provider", lambda _: object())
    monkeypatch.setattr("codeflow.agent.loop.AgentLoop", _StubAgentLoop)

    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.maybe_build_memory_backend",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        lambda *a, **k: [],
    )
    args = ["run", "-m", "hi"]
    if workspace is not None:
        args.extend(["-w", str(workspace)])
    r = runner.invoke(app, [*args, *extra_args])
    return r, captured


def test_one_shot_turn_failure_is_visible_and_nonzero(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    result, _captured = _invoke_agent_capturing_session(
        monkeypatch,
        workspace,
        [],
        fail_turn=True,
    )

    assert result.exit_code == 1
    assert "Turn failed: provider offline" in result.stdout


def test_agent_adapter_uses_shared_runtime_assembly(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli import _runtime_assembly

    workspace = tmp_path / "ws"
    workspace.mkdir()
    original = _runtime_assembly.assemble_runtime
    calls: list[tuple[tuple, dict]] = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(_runtime_assembly, "assemble_runtime", _spy)

    result, _captured = _invoke_agent_capturing_session(
        monkeypatch,
        workspace,
        [],
    )

    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1
    assert calls[0][1]["interactive"] is False
    assert calls[0][1]["session_manager"] is not None
    assert "cron_service" in calls[0][1]
    assert "provider" in calls[0][1]
    assert calls[0][1]["paths"].workspace == workspace
    assert calls[0][1]["paths"].state == workspace


def test_agent_defaults_to_current_project_and_keeps_state_hidden(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli import _runtime_assembly

    project = tmp_path / "project"
    project.mkdir()
    product_home = tmp_path / "codeflow-home"
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    monkeypatch.chdir(project)
    original = _runtime_assembly.assemble_runtime
    calls: list[tuple[tuple, dict]] = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(_runtime_assembly, "assemble_runtime", _spy)

    result, _captured = _invoke_agent_capturing_session(
        monkeypatch,
        None,
        [],
    )

    assert result.exit_code == 0, result.stdout
    assert calls[0][1]["paths"].workspace == project.resolve()
    assert calls[0][1]["paths"].state.parent == product_home / "projects"
    assert calls[0][1]["paths"].state.name.startswith("project-")
    assert list(project.iterdir()) == []


def test_invalid_resume_stops_before_runtime_assembly(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli import _runtime_assembly

    workspace = tmp_path / "ws"
    workspace.mkdir()
    calls = 0
    original = _runtime_assembly.assemble_runtime

    def _spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_runtime_assembly, "assemble_runtime", _spy)

    result, _captured = _invoke_agent_capturing_session(
        monkeypatch,
        workspace,
        ["--resume", "missing"],
    )

    assert result.exit_code != 0
    assert calls == 0


def test_interactive_setup_failure_closes_runtime(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from codeflow.cli import _runtime_assembly

    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent_loop = SimpleNamespace(
        channels_config=None,
        run=AsyncMock(),
        stop=MagicMock(),
    )
    runtime = SimpleNamespace(
        agent_loop=agent_loop,
        session_manager=None,
        start_memory_backend=AsyncMock(return_value=True),
        close=AsyncMock(),
    )

    def _assemble(*args, **kwargs):
        runtime.session_manager = kwargs["session_manager"]
        return runtime

    monkeypatch.setattr(_runtime_assembly, "assemble_runtime", _assemble)
    monkeypatch.setattr("codeflow.cli.agent_commands.make_provider", lambda cfg: object())
    monkeypatch.setattr("codeflow.cli.agent_commands._init_prompt_session", lambda: None)
    monkeypatch.setattr("codeflow.cli.agent_commands.signal.signal", lambda *args: None)

    class _SetupFailure(RuntimeError):
        pass

    def _fail_build(*args, **kwargs):
        raise _SetupFailure

    monkeypatch.setattr("codeflow.cli._repl_spine.build_repl", _fail_build)

    result = runner.invoke(
        app,
        ["run", "--workspace", str(workspace)],
    )

    assert isinstance(result.exception, _SetupFailure)
    runtime.close.assert_awaited_once()
    agent_loop.stop.assert_called_once()
    agent_loop.run.assert_not_awaited()


def test_agent_default_mints_fresh_session(tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare ``run -m`` mints a fresh ``cli:{chat_id}`` per invocation."""
    import re

    ws = tmp_path / "ws"
    ws.mkdir()

    r1, cap1 = _invoke_agent_capturing_session(monkeypatch, ws, [])
    assert r1.exit_code == 0, r1.stdout
    assert re.fullmatch(r"cli:\d{8}_\d{6}_[0-9a-f]{6}", cap1["session_id"]), (
        f"expected freshly minted cli session key, got {cap1['session_id']!r}"
    )

    r2, cap2 = _invoke_agent_capturing_session(monkeypatch, ws, [])
    assert r2.exit_code == 0
    assert cap1["session_id"] != cap2["session_id"], "each bare invocation must mint a NEW session"


def test_agent_continue_binds_most_recent_cli_session(
    tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-c`` binds the agent to the most-recent persisted cli session."""
    from codeflow.session.manager import SessionManager

    ws = tmp_path / "ws"
    ws.mkdir()
    mgr = SessionManager(ws)
    seeded = "20990101_000000_aaaaaa"
    s = mgr.get_or_create(f"cli:{seeded}")
    s.add_message("user", "earlier turn")
    mgr.save(s)

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["-c"])
    assert r.exit_code == 0, r.stdout
    assert captured["session_id"] == f"cli:{seeded}"


def test_agent_resume_binds_resolved_session(tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--resume <prefix>`` resolves and binds that cli session."""
    from codeflow.session.manager import SessionManager

    ws = tmp_path / "ws"
    ws.mkdir()
    mgr = SessionManager(ws)
    seeded = "20990101_000000_bbbbbb"
    s = mgr.get_or_create(f"cli:{seeded}")
    s.add_message("user", "earlier turn")
    mgr.save(s)

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["--resume", seeded[:20]])
    assert r.exit_code == 0, r.stdout
    assert captured["session_id"] == f"cli:{seeded}"


def test_agent_session_key_passthrough(tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--session <key>`` passes a full key through unchanged (any channel)."""
    ws = tmp_path / "ws"
    ws.mkdir()

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["--session", "feishu:ou_xyz"])
    assert r.exit_code == 0, r.stdout
    assert captured["session_id"] == "feishu:ou_xyz"


def test_agent_bare_session_resolves_cross_channel(
    tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--session <bare id>`` resolves to an existing session on a non-cli
    channel — it must NOT be mis-routed to a colon-less/malformed key."""
    from codeflow.session.manager import SessionManager

    ws = tmp_path / "ws"
    ws.mkdir()
    mgr = SessionManager(ws)
    cid = "20990101_000000_cccccc"
    s = mgr.get_or_create(f"tui:{cid}")
    s.add_message("user", "earlier turn")
    mgr.save(s)

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["--session", cid])
    assert r.exit_code == 0, r.stdout
    assert captured["session_id"] == f"tui:{cid}"


def test_agent_unknown_bare_session_falls_back_to_cli(
    tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--session <bare id>`` with no matching session falls back to a proper
    ``cli:<id>`` key — never a colon-less/malformed path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    cid = "20990101_000000_dddddd"

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["--session", cid])
    assert r.exit_code == 0, r.stdout
    assert captured["session_id"] == f"cli:{cid}"


@pytest.mark.parametrize(
    "args",
    [
        ["-c", "--resume", "x"],
        ["--session", "cli:abc", "-c"],
        ["--session", "cli:abc", "--resume", "x"],
        ["--session", "cli:abc", "-c", "--resume", "x"],
    ],
)
def test_agent_session_binding_flags_mutually_exclusive(tmp_config: Path, args: list[str]) -> None:
    """More than one of --session/--continue/--resume exits with usage error."""
    r = runner.invoke(app, ["run", "-m", "hi", *args])
    assert r.exit_code == 2, f"expected usage error, got {r.exit_code}: {r.stdout}"
    assert "mutually exclusive" in r.stderr


def test_agent_continue_without_prior_session_starts_fresh(
    tmp_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-c`` with no stored cli session prints a notice and mints fresh."""
    import re

    ws = tmp_path / "ws"
    ws.mkdir()

    r, captured = _invoke_agent_capturing_session(monkeypatch, ws, ["-c"])
    assert r.exit_code == 0, r.stdout
    assert re.fullmatch(r"cli:\d{8}_\d{6}_[0-9a-f]{6}", captured["session_id"])
    assert "no previous cli session" in r.stdout


# ============================================================================

# ============================================================================


@pytest.mark.parametrize(
    "command,expected",
    [
        ("exit", True),
        ("quit", True),
        ("/exit", True),
        ("/quit", True),
        (":q", True),
        ("EXIT", True),
        ("Quit", True),
        (" exit", False),
        ("hello", False),
        ("", False),
        ("exit later", False),
    ],
)
def test_is_exit_command(command: str, expected: bool) -> None:
    """``_is_exit_command`` detects the canonical exit triggers (case-insensitive)."""
    from codeflow.cli.agent_commands import _is_exit_command

    assert _is_exit_command(command) is expected


def test_exit_commands_set_contents() -> None:
    """The canonical exit triggers stay in sync with documented behavior."""
    from codeflow.cli.agent_commands import EXIT_COMMANDS

    assert EXIT_COMMANDS == {"exit", "quit", "/exit", "/quit", ":q"}


def test_print_agent_response_with_markdown(capsys: pytest.CaptureFixture) -> None:
    """``_print_agent_response`` renders the body — markdown mode."""
    from codeflow.cli.agent_commands import _print_agent_response

    _print_agent_response("# hi", render_markdown=True)
    out = capsys.readouterr().out

    assert "hi" in out


def test_print_agent_response_plain(capsys: pytest.CaptureFixture) -> None:
    """``_print_agent_response`` renders plain text — markdown disabled."""
    from codeflow.cli.agent_commands import _print_agent_response

    _print_agent_response("hello world", render_markdown=False)
    out = capsys.readouterr().out
    assert "hello world" in out


# ============================================================================

# ============================================================================


def test_agent_message_mode_mocked_provider(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run -m 'hi'`` with a mocked provider must reach a clean exit
    (no traceback). We mock ``make_provider`` so the agent loop builds
    without contacting any LLM."""
    from codeflow.config.loader import save_config
    from codeflow.config.schema import Config

    cfg = Config()
    cfg.providers.openrouter.api_key = "stub-test-key"
    save_config(cfg)

    monkeypatch.setattr(
        "codeflow.cli.agent_commands.make_provider",
        lambda _: (_ for _ in ()).throw(RuntimeError("mock-no-provider")),
    )

    r = runner.invoke(app, ["run", "-m", "hello"])
    if r.exception is not None:
        assert not isinstance(r.exception, (NameError, AttributeError, ImportError)), (
            f"Crash-class exception leaked through: {r.exception!r}"
        )


# ---------------------------------------------------------------------------

#


# ---------------------------------------------------------------------------


class _RecordingConsole:
    """Captures only what ``_repl_slash`` itself prints. Delegated CLI
    commands print to their own module-level console (real stdout), which
    is irrelevant to the routing assertions here."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **_kwargs: object) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def isolated_runtime(tmp_path: Path, monkeypatch) -> Path:
    """Point the Cron runtime directory at a throwaway config.

    ``set_config_path`` covers the read path. Config writers resolve the path via the update module's own
    ``get_config_path`` binding, so pin that too — otherwise a leaked
    monkeypatch from another test file could redirect our writes elsewhere.
    """
    cfg = tmp_path / "config.json"
    set_config_path(cfg)
    monkeypatch.setattr("codeflow.config.update.get_config_path", lambda: cfg)
    yield tmp_path
    set_config_path(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["hello there", "/stop", "/restart", "/unknowntop"])
def test_slash_non_commands_fall_through(text: str) -> None:
    """Plain chat and bus-level commands (/stop, /restart) must fall through
    to the LLM/bus path — handler returns False."""
    from codeflow.cli._repl_slash import handle_repl_slash

    assert handle_repl_slash(text, console=_RecordingConsole()) is False


def test_slash_help_lists_namespaces() -> None:
    from codeflow.cli._repl_slash import handle_repl_slash

    con = _RecordingConsole()
    assert handle_repl_slash("/help", console=con) is True
    assert "/cron" in con.text
    assert "/sentinel" not in con.text


def test_cron_help_is_handled_and_sentinel_falls_through() -> None:
    from codeflow.cli._repl_slash import handle_repl_slash

    con = _RecordingConsole()
    assert handle_repl_slash("/cron", console=con) is True
    assert handle_repl_slash("/sentinel help", console=con) is False
    assert "/cron list" in con.text


def test_cron_run_is_shell_only(isolated_runtime: Path) -> None:
    from codeflow.cli._repl_slash import handle_repl_slash

    con = _RecordingConsole()
    assert handle_repl_slash("/cron run abc123", console=con) is True
    assert "shell-only" in con.text


def test_cron_config_write_is_shell_only(isolated_runtime: Path) -> None:
    from codeflow.cli._repl_slash import handle_repl_slash

    con = _RecordingConsole()
    assert handle_repl_slash("/cron config set --forward-channels '*'", console=con) is True
    assert "shell-only" in con.text


def test_cron_list_runs_against_empty_store(isolated_runtime: Path) -> None:
    from codeflow.cli._repl_slash import handle_repl_slash

    assert handle_repl_slash("/cron list", console=_RecordingConsole()) is True


def _make_cron_job():
    from codeflow.config.paths import get_cron_dir
    from codeflow.proactive_engine.schedulers.cron.service import CronService
    from codeflow.proactive_engine.schedulers.cron.types import CronSchedule

    svc = CronService(get_cron_dir() / "jobs.json", allowed_channels=None)
    job = svc.add_job(
        name="testjob",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hi",
        channel="cli",
        to="direct",
    )
    return svc, job


def test_cron_delete_requires_inline_yes(isolated_runtime: Path) -> None:
    """Without -y, destructive ops only preview and keep the job."""
    from codeflow.cli._repl_slash import handle_repl_slash

    svc, job = _make_cron_job()
    con = _RecordingConsole()
    assert handle_repl_slash(f"/cron delete {job.id}", console=con) is True
    assert "-y" in con.text
    assert len(svc.list_jobs(include_disabled=True)) == 1


def test_cron_delete_with_yes_removes_job(isolated_runtime: Path) -> None:
    from codeflow.cli._repl_slash import handle_repl_slash
    from codeflow.config.paths import get_cron_dir
    from codeflow.proactive_engine.schedulers.cron.service import CronService

    svc, job = _make_cron_job()
    assert handle_repl_slash(f"/cron delete {job.id} -y", console=_RecordingConsole()) is True

    fresh = CronService(get_cron_dir() / "jobs.json", allowed_channels=None)
    assert fresh.list_jobs(include_disabled=True) == []
