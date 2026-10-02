"""Bug2 deep tests — pathological inputs, edge state, and concurrency.

The base file ``test_runtime_checkpoint_bug2.py`` validates the happy path and
core regressions; this file is the fail-safe hardening tier:

- D1: filesystem pathology (unicode names, symlinks, deep nesting, file<->dir
      transitions, chmod, binary).
- D2: edge state (missing/read-only workspace, corrupted/blocked shadow dir,
      pre-existing files where dirs are expected).
- D3: concurrency (interleaved sessions, repeated stash, async-parallel
      commits sharing one shadow git).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.agent.loop.checkpoint import CheckpointService
from codeflow.agent.loop.recovery import RecoveryLimits
from codeflow.config.codeflow import CheckpointConfig, RuntimeConfig
from codeflow.providers.base import LLMProvider, LLMResponse


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# =============================================================================

# =============================================================================


async def test_d1_unicode_and_space_in_filename(workspace):
    """add -A is name-agnostic — unicode + spaces still snapshot cleanly."""
    svc = CheckpointService(workspace)
    (workspace / "测试 文件.py").write_text("# hi\n", encoding="utf-8")
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None

    assert any("测试" in p for p in changed), f"got: {changed!r}"


async def test_d1_symlink_snapshotted_as_symlink(workspace):
    """Symlinks are first-class in git (mode 120000). add -A stages the link
    itself, not the target — so the snapshot doesn't accidentally inline a
    large target or pull in out-of-workspace files."""
    (workspace / "target.txt").write_text("real\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to("target.txt")
    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None
    assert "link.txt" in changed and "target.txt" in changed


async def test_d1_deeply_nested_paths(workspace):
    p = workspace / "a" / "b" / "c" / "d" / "e.py"
    p.parent.mkdir(parents=True)
    p.write_text("ok\n", encoding="utf-8")
    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None
    assert "a/b/c/d/e.py" in changed


async def test_d1_file_to_directory_transition(workspace):
    """Turn 1: file ``thing``. Turn 2: thing is now a directory holding x.py.
    The transition must snapshot — this is a realistic agent edit pattern when
    refactoring a single-file module into a package."""
    svc = CheckpointService(workspace)
    (workspace / "thing").write_text("a = 1\n", encoding="utf-8")
    cid1, _ = await svc.commit_turn("t1")
    assert cid1 is not None

    (workspace / "thing").unlink()
    (workspace / "thing").mkdir()
    (workspace / "thing" / "x.py").write_text("b = 2\n", encoding="utf-8")
    cid2, changed = await svc.commit_turn("t2")
    assert cid2 is not None and cid2 != cid1

    assert "thing" in changed or "thing/x.py" in changed


async def test_d1_chmod_only_change_captured(workspace):
    """Mode-bit change (e.g. chmod +x on a script) without content change is
    still a recoverable diff — git tracks executable bit and we want recovery
    to know about it."""
    p = workspace / "script.sh"
    p.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    svc = CheckpointService(workspace)
    await svc.commit_turn("t1")
    os.chmod(p, 0o755)
    cid2, changed = await svc.commit_turn("t2")

    assert cid2 is not None
    assert "script.sh" in changed


async def test_d1_binary_file(workspace):
    """Binary blobs (small, ~64 KiB) shouldn't choke the snapshot pipeline."""
    (workspace / "blob.bin").write_bytes(os.urandom(64 * 1024))
    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None
    assert "blob.bin" in changed


# =============================================================================

# =============================================================================


async def test_d2_workspace_does_not_exist():
    """A bogus workspace path must degrade — no crash, no half-built shadow."""
    bogus = Path(tempfile.gettempdir()) / "codeflow_nonexistent_xyz"
    if bogus.exists():
        shutil.rmtree(bogus)
    svc = CheckpointService(bogus)
    cid, changed = await svc.commit_turn("t1")
    assert cid is None
    assert changed == []


async def test_d2_shadow_path_blocked_by_a_regular_file(workspace):
    """If something pre-existing sits at the shadow git-dir path as a file
    (not a directory), init must fail gracefully — checkpoint is best-effort."""
    blocker = workspace / ".codeflow"
    blocker.write_text("not a dir", encoding="utf-8")
    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is None
    assert changed == []


async def test_d2_corrupted_shadow_repo(workspace):
    """A pre-existing but corrupted shadow .git → commit_turn doesn't raise."""
    gd = workspace / ".codeflow" / "shadow.git"
    gd.mkdir(parents=True)
    (gd / "HEAD").write_text("garbage\n", encoding="utf-8")

    (gd / "objects").mkdir(exist_ok=True)
    svc = CheckpointService(workspace)
    (workspace / "x.py").write_text("ok\n", encoding="utf-8")
    cid, _changed = await svc.commit_turn("t1")

    assert cid is None or isinstance(cid, str)


async def test_d2_very_long_label(workspace):
    svc = CheckpointService(workspace)
    (workspace / "x.py").write_text("ok\n", encoding="utf-8")
    cid, changed = await svc.commit_turn("L" * 4000)
    assert cid is not None
    assert "x.py" in changed


# =============================================================================

# =============================================================================


class _NoopProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        return LLMResponse(content="", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


def _agent_with_checkpoint(workspace: Path) -> AgentLoop:
    return AgentLoop(
        provider=_NoopProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        empty_recovery=RecoveryLimits(enabled=False),
        runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy="always")),
    )


def test_d3_interleaved_sessions_keep_recovery_isolated(workspace):
    """Two sessions A and B both interrupted; assembling for B must NOT
    inject A's recovery (and vice-versa). Key isolation in _pending_recovery."""
    agent = _agent_with_checkpoint(workspace)
    agent._pending_recovery["A"] = {"checkpoint_id": "aaa", "files": ["a.py"]}
    agent._pending_recovery["B"] = {"checkpoint_id": "bbb", "files": ["b.py"]}

    msgs_b = [{"role": "user", "content": "bbb-turn"}]
    agent._inject_recovery_block("B", msgs_b)
    text_b = msgs_b[-1]["content"]
    assert "b.py" in text_b and "bbb" in text_b
    assert "a.py" not in text_b and "aaa" not in text_b

    msgs_a = [{"role": "user", "content": "aaa-turn"}]
    agent._inject_recovery_block("A", msgs_a)
    text_a = msgs_a[-1]["content"]
    assert "a.py" in text_a and "aaa" in text_a
    assert "b.py" not in text_a


def test_d3_repeated_stash_latest_wins(workspace):
    """If a session is interrupted twice before consumption, the latter stash
    must replace the former — recovery prompt should reflect *current* state,
    not stale files."""
    agent = _agent_with_checkpoint(workspace)
    from codeflow.agent.loop import TurnOutcome

    out1 = TurnOutcome(status="interrupted", checkpoint_id="old", edited_files=["stale.py"])
    out2 = TurnOutcome(status="interrupted", checkpoint_id="new", edited_files=["fresh.py"])
    agent._stash_recovery("s", out1)
    agent._stash_recovery("s", out2)
    pending = agent._pending_recovery["s"]
    assert pending["checkpoint_id"] == "new"
    assert pending["files"] == ["fresh.py"]


async def test_d3_concurrent_commits_serialize_or_degrade(workspace):
    """Two concurrent commit_turn calls share one shadow .git. Git's index lock
    means at most one can hold the lock at a time. Either both succeed (one
    waits) or one degrades to (None, []) — but neither must raise."""
    svc = CheckpointService(workspace)

    (workspace / "seed.py").write_text("0\n", encoding="utf-8")
    await svc.commit_turn("seed")

    (workspace / "a.py").write_text("1\n", encoding="utf-8")
    (workspace / "b.py").write_text("2\n", encoding="utf-8")

    r1, r2 = await asyncio.gather(
        svc.commit_turn("race-1"),
        svc.commit_turn("race-2"),
        return_exceptions=False,
    )

    ids = [r1[0], r2[0]]
    assert any(cid is not None for cid in ids), f"at least one concurrent commit should land; got {r1!r}, {r2!r}"


# =============================================================================

# =============================================================================


# =============================================================================

# =============================================================================


async def test_d5_perf_1k_files_commit(workspace, capsys):
    """Measure commit_turn latency on a 1000-file workspace:
    - cold commit (full snapshot)
    - warm commit after modifying 10 files (typical-turn diff)
    Generous bounds: this is a sanity check, not a strict perf gate.
    """
    import time

    for i in range(1000):
        (workspace / f"f{i:04d}.py").write_text(f"# file {i}\n", encoding="utf-8")

    svc = CheckpointService(workspace)
    t0 = time.perf_counter()
    cid1, changed1 = await svc.commit_turn("cold")
    t_cold = time.perf_counter() - t0
    assert cid1 is not None
    assert len(changed1) == 1000

    for i in range(10):
        (workspace / f"f{i:04d}.py").write_text(f"# mod {i}\n", encoding="utf-8")
    t0 = time.perf_counter()
    cid2, changed2 = await svc.commit_turn("warm")
    t_warm = time.perf_counter() - t0
    assert cid2 is not None and cid2 != cid1
    assert len(changed2) == 10

    with capsys.disabled():
        print(f"\n[D5 perf] 1000-file cold commit: {t_cold:.2f}s | 10-file diff after 1000 baseline: {t_warm:.2f}s")

    assert t_cold < 30.0, f"cold commit took {t_cold:.2f}s, too slow"
    assert t_warm < 10.0, f"warm commit took {t_warm:.2f}s, too slow"


async def test_d2_loop_unaffected_when_shadow_blocked(workspace):
    """Tying D2 back to the loop: if shadow is unusable (blocked by a file),
    the loop still returns cleanly — the safety net never breaks the turn."""
    (workspace / ".codeflow").write_text("blocker", encoding="utf-8")
    agent = _agent_with_checkpoint(workspace)
    final, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )

    assert outcome.status == "completed"
    assert outcome.checkpoint_id is None


# =============================================================================

# =============================================================================


def _agent(workspace: Path, *, policy: str, interactive: bool) -> AgentLoop:
    return AgentLoop(
        provider=_NoopProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        empty_recovery=RecoveryLimits(enabled=False),
        runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy=policy)),
        interactive=interactive,
    )


def test_d6_policy_never_disables_checkpoint(workspace):
    """``policy="never"`` is the kill switch — no shadow git regardless of
    interactive. Loop is byte-identical to the pre-Bug2 baseline."""
    a_inter = _agent(workspace, policy="never", interactive=True)
    a_one_shot = _agent(workspace, policy="never", interactive=False)
    assert a_inter._checkpoint is None
    assert a_one_shot._checkpoint is None


def test_d6_policy_always_overrides_interactive(workspace):
    """``policy="always"`` keeps the checkpoint on even for ``-m`` one-shots.
    Useful for users who want the safety net everywhere."""
    a_inter = _agent(workspace, policy="always", interactive=True)
    a_one_shot = _agent(workspace, policy="always", interactive=False)
    assert a_inter._checkpoint is not None
    assert a_one_shot._checkpoint is not None


def test_d6_policy_interactive_gated_by_call_site(workspace):
    """Default ``policy="interactive"`` follows the caller's signal:
    REPL/TUI/gateway get the snapshot, ``-m`` one-shot doesn't (no next turn
    to inject recovery into anyway)."""
    a_inter = _agent(workspace, policy="interactive", interactive=True)
    a_one_shot = _agent(workspace, policy="interactive", interactive=False)
    assert a_inter._checkpoint is not None
    assert a_one_shot._checkpoint is None


def test_d6_default_policy_is_interactive(workspace):
    """Sanity check on the rollout decision: default is ``"interactive"`` so
    out-of-the-box behavior matches the table the team approved."""
    cfg = CheckpointConfig()
    assert cfg.policy == "interactive"


# =============================================================================

# =============================================================================


async def test_d7_ephemeral_dirs_never_snapshotted(workspace):
    """S4-B: built-in excludes keep build artifacts out of the shadow git
    even when the user has no ``.gitignore``. node_modules, venv, build/,
    *.log all share the same pattern; we sample three categories."""
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "react").mkdir()
    (workspace / "node_modules" / "react" / "index.js").write_text("// huge", encoding="utf-8")
    (workspace / "venv").mkdir()
    (workspace / "venv" / "bin").mkdir()
    (workspace / "venv" / "bin" / "python").write_text("#!/bin/sh", encoding="utf-8")
    (workspace / "app.log").write_text("startup\n", encoding="utf-8")
    (workspace / "real.py").write_text("x=1\n", encoding="utf-8")

    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None

    assert "real.py" in changed

    assert not any(p.startswith("node_modules/") for p in changed), changed
    assert not any(p.startswith("venv/") for p in changed), changed
    assert "app.log" not in changed, changed


async def test_d7_dotenv_and_secrets_never_snapshotted(workspace):
    """S4-B: credential-shaped files (``.env``, ``*.key``, ``*.pem``) ship
    in the default exclude list. Defense in depth — the user's own
    ``.gitignore`` usually covers these, but a fresh workspace might not."""
    (workspace / ".env").write_text("API_KEY=hunter2\n", encoding="utf-8")
    (workspace / ".env.production").write_text("API_KEY=prod\n", encoding="utf-8")
    (workspace / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    (workspace / "main.py").write_text("ok\n", encoding="utf-8")

    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None
    assert "main.py" in changed
    assert ".env" not in changed, changed
    assert ".env.production" not in changed, changed
    assert "deploy.pem" not in changed, changed


async def test_d7_respects_user_gitignore_in_workspace(workspace):
    """S4-A: when the workspace itself has a ``.gitignore``, the shadow git
    honors it automatically (standard ``add -A`` semantics on the work-tree).
    This is the load-bearing safety property for credential-leak protection
    — verified end-to-end here so a future change can't quietly remove it."""
    (workspace / ".gitignore").write_text("private_data.yml\nlocal_cache/\n", encoding="utf-8")
    (workspace / "private_data.yml").write_text("token: hunter2\n", encoding="utf-8")
    (workspace / "local_cache").mkdir()
    (workspace / "local_cache" / "blob.bin").write_text("xxx", encoding="utf-8")
    (workspace / "code.py").write_text("ok\n", encoding="utf-8")

    svc = CheckpointService(workspace)
    cid, changed = await svc.commit_turn("t1")
    assert cid is not None
    assert "code.py" in changed

    assert ".gitignore" in changed

    assert "private_data.yml" not in changed, changed
    assert not any(p.startswith("local_cache/") for p in changed), changed


async def test_d7_notice_txt_written_next_to_shadow(workspace):
    """S4 discoverability: a user noticing ``.codeflow/`` should be able to
    identify what it is without grepping our codebase."""
    svc = CheckpointService(workspace)
    (workspace / "x.py").write_text("ok\n", encoding="utf-8")
    await svc.commit_turn("t1")
    notice = workspace / ".codeflow" / "NOTICE.txt"
    assert notice.exists()
    text = notice.read_text()
    assert "CodeFlow" in text and "checkpoint" in text
    assert "policy" in text


async def test_d7_gc_auto_config_set_on_shadow_repo(workspace):
    """S4-C: ``gc.auto`` is set in the shadow repo's git config at init so
    long-lived sessions can prune loose objects. We don't measure GC firing
    (slow + flaky); the config bit is the load-bearing piece."""
    svc = CheckpointService(workspace)
    (workspace / "x.py").write_text("ok\n", encoding="utf-8")
    await svc.commit_turn("t1")
    rc, out, _ = await svc._git("config", "--get", "gc.auto")
    assert rc == 0
    assert out.strip() == "256"


async def test_d7_gc_invoked_every_n_commits(workspace, monkeypatch):
    """S4-C heartbeat: every N successful commits the service fires
    ``git gc --auto``. We spy on ``_git`` to count invocations rather than
    pay for a real GC (which is cheap but still noisier than necessary)."""
    import codeflow.agent.loop.checkpoint as cp_module

    monkeypatch.setattr(cp_module, "_GC_EVERY_N_COMMITS", 3)
    svc = CheckpointService(workspace)

    real_git = svc._git
    gc_calls: list[tuple] = []

    async def _spy(*args):
        if args[:1] == ("gc",):
            gc_calls.append(args)
        return await real_git(*args)

    monkeypatch.setattr(svc, "_git", _spy)

    for i in range(7):
        (workspace / f"f{i}.py").write_text(f"{i}\n", encoding="utf-8")
        cid, _ = await svc.commit_turn(f"t{i}")
        assert cid is not None

    assert len(gc_calls) == 2, f"expected 2 gc fires, got {gc_calls}"
    assert all(call[1] == "--auto" for call in gc_calls)


# =============================================================================

# =============================================================================


async def test_d8_one_shot_mode_creates_no_shadow_dir(workspace):
    """S3 end-to-end: ``interactive=False`` + default policy means
    ``codeflow run -m "..."`` leaves no ``.codeflow/`` artifact on disk —
    one-shot commands don't pay the shadow-git cost."""
    agent = _agent(workspace, policy="interactive", interactive=False)
    _f, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "completed"
    assert outcome.checkpoint_id is None
    assert not (workspace / ".codeflow").exists()


# =============================================================================

# =============================================================================
#


def test_d9_separate_state_keeps_shadow_git_outside_workspace(workspace, tmp_path: Path):
    state = tmp_path / "state"

    svc = CheckpointService(workspace, state=state)

    assert svc._git_dir == state.resolve() / "shadow.git"
    assert not (workspace / ".codeflow").exists()


def test_d9_parent_escape_rejected(workspace):
    """``shadow_dir="../foo"`` resolves outside the workspace → rejected."""
    with pytest.raises(ValueError, match="strictly under the workspace"):
        CheckpointService(workspace, shadow_dir="../escape")


def test_d9_absolute_path_rejected(workspace):
    """``shadow_dir`` as an absolute path replaces the workspace prefix
    entirely (standard ``Path /`` semantics) → also rejected."""
    with pytest.raises(ValueError, match="strictly under the workspace"):
        CheckpointService(workspace, shadow_dir="/tmp/some_global_path")


def test_d9_empty_string_rejected(workspace):
    """Empty ``shadow_dir`` collapses to the workspace itself — a degenerate
    case where the shadow's git-dir would equal the work-tree and a normal
    ``git init`` could clobber the user's files."""
    with pytest.raises(ValueError, match="strictly under the workspace"):
        CheckpointService(workspace, shadow_dir="")


def test_d9_dot_path_rejected(workspace):
    """``"."`` is the same degenerate case as the empty string — git-dir
    would equal the workspace."""
    with pytest.raises(ValueError, match="strictly under the workspace"):
        CheckpointService(workspace, shadow_dir=".")


def test_d9_nested_relative_path_accepted(workspace):
    """Sanity: a normal nested relative path is fine."""
    svc = CheckpointService(workspace, shadow_dir="deep/nested/shadow.git")

    assert svc._git_dir.is_relative_to(workspace.resolve())
    assert svc._git_dir.name == "shadow.git"


async def test_d9_agent_loop_degrades_when_shadow_dir_invalid(workspace):
    """AgentLoop must not crash when the user config contains a bad
    ``shadow_dir`` — checkpoint silently disables so the turn still runs."""
    bad_cfg = RuntimeConfig(
        checkpoint=CheckpointConfig(
            policy="always",
            shadow_dir="../escape_via_config",
        )
    )
    agent = AgentLoop(
        provider=_NoopProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        empty_recovery=RecoveryLimits(enabled=False),
        runtime_config=bad_cfg,
        interactive=True,
    )
    assert agent._checkpoint is None, "bad config should degrade to no checkpoint"
    final, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "completed"
    assert outcome.checkpoint_id is None

    escaped = workspace.parent / "escape_via_config"
    assert not escaped.exists(), f"shadow leaked to {escaped}"


# =============================================================================

# =============================================================================


def test_d10_recovery_kept_when_content_unknown_type(workspace):
    """Regression: previously ``_inject_recovery_block`` ``pop()``'d the
    pending entry BEFORE checking whether ``content`` was a shape it could
    actually write to (str / list). A user message with ``content=None``
    (or any other shape) would therefore consume the recovery silently —
    the user got a "Please continue" prompt with no clue what was lost.

    Now the pop only happens after a successful mutation; an unknown
    content shape keeps the recovery pending for the next assembly."""
    agent = _agent_with_checkpoint(workspace)
    agent._pending_recovery["s"] = {"checkpoint_id": "abc123", "files": ["x.py"]}

    msgs_bad = [{"role": "user", "content": None}]
    agent._inject_recovery_block("s", msgs_bad)
    assert msgs_bad[-1]["content"] is None, "content must not be mutated"
    assert "s" in agent._pending_recovery, "recovery must NOT be dropped"

    msgs_ok = [{"role": "user", "content": "go"}]
    agent._inject_recovery_block("s", msgs_ok)
    assert "abc123" in msgs_ok[-1]["content"]
    assert "x.py" in msgs_ok[-1]["content"]
    assert "s" not in agent._pending_recovery


async def test_d10_git_subprocess_times_out_and_kills_proc(workspace, monkeypatch):
    """If the git child hangs (NFS lock / index-lock / disk full), ``_git``
    must time out instead of blocking the whole turn. Verified by stubbing
    ``asyncio.create_subprocess_exec`` to return a fake proc whose
    ``communicate`` never returns, plus shrinking the timeout so the test
    runs in milliseconds. The hanging proc must also be killed (no zombie)."""
    import codeflow.agent.loop.checkpoint as cp_module

    monkeypatch.setattr(cp_module, "_GIT_TIMEOUT_SECONDS", 0.05)

    class _HangingProc:
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        async def communicate(self):
            await asyncio.sleep(60)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return -9

    fake = _HangingProc()

    async def _fake_create(*_args, **_kwargs):
        return fake

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    svc = CheckpointService(workspace)
    rc, out, err = await svc._git("status")
    assert rc == -1, f"timeout must surface as non-zero rc, got {rc}"
    assert err == "timeout"
    assert out == ""
    assert fake.killed, "hanging proc must be killed to avoid zombies"


async def test_d10_commit_turn_degrades_on_git_hang(workspace, monkeypatch):
    """End-to-end: if any git call hangs, ``commit_turn`` returns
    ``(None, [])`` and does not raise — preserving the "never break a turn"
    contract that the rest of the safety net relies on."""
    import codeflow.agent.loop.checkpoint as cp_module

    monkeypatch.setattr(cp_module, "_GIT_TIMEOUT_SECONDS", 0.05)

    class _HangingProc:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(60)
            return b"", b""

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return -9

    async def _fake_create(*_args, **_kwargs):
        return _HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    svc = CheckpointService(workspace)
    (workspace / "a.py").write_text("ok\n", encoding="utf-8")
    cid, changed = await svc.commit_turn("t1")
    assert cid is None and changed == [], f"hang must degrade to no-op, got {(cid, changed)!r}"


async def test_d8_interactive_mode_with_user_gitignore_end_to_end(workspace):
    """S3 + S4-A together: an interactive session in a workspace with a
    ``.gitignore`` snapshots the source files but never the user-marked
    private ones. This is the load-bearing user-facing guarantee."""
    (workspace / ".gitignore").write_text(".env\n", encoding="utf-8")
    (workspace / ".env").write_text("SECRET=abc\n", encoding="utf-8")
    (workspace / "src.py").write_text("ok\n", encoding="utf-8")

    agent = _agent(workspace, policy="interactive", interactive=True)
    _f, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )

    assert outcome.status == "completed"
    assert outcome.checkpoint_id is not None

    svc = agent._checkpoint
    rc, out, _ = await svc._git("ls-tree", "-r", "--name-only", "HEAD")
    assert rc == 0
    tracked = out.splitlines()
    assert "src.py" in tracked
    assert ".gitignore" in tracked
    assert ".env" not in tracked, f"secret leaked into shadow: {tracked}"
