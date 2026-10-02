"""Bug2 — per-turn shadow-git checkpoint + max-iter interrupted handling.

Covers the runtime-discipline safety net gated by
``config.runtime.checkpoint.policy`` × ``AgentLoop(interactive=...)``:
- CheckpointService snapshots the worktree without touching the user's .git.
- A max-iteration turn reports ``status="interrupted"`` (not a fake
  completion) and the workspace is snapshotted for recovery.
- With policy="never" (or "interactive" + interactive=False), the loop is
  behaviorally unchanged from baseline.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.agent.loop.checkpoint import CheckpointService
from codeflow.config.codeflow import CheckpointConfig, RuntimeConfig
from codeflow.providers.base import LLMProvider, LLMResponse, ToolCallRequest


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


async def test_checkpoint_commits_then_noops_when_unchanged(workspace):
    svc = CheckpointService(workspace)
    (workspace / "a.py").write_text("print(1)\n", encoding="utf-8")

    cid, changed = await svc.commit_turn("turn 1")
    assert cid is not None
    assert "a.py" in changed

    cid2, changed2 = await svc.commit_turn("turn 2")
    assert cid2 is None
    assert changed2 == []


async def test_checkpoint_does_not_touch_user_git(workspace):
    """When the workspace is itself a git repo, the shadow checkpoint must
    not add commits to the user's real history (plan §7 fixture)."""
    env = {"GIT_AUTHOR_NAME": "u", "GIT_AUTHOR_EMAIL": "u@u", "GIT_COMMITTER_NAME": "u", "GIT_COMMITTER_EMAIL": "u@u"}
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "base"], cwd=workspace, check=True, env={**_os_environ(), **env}
    )
    head_before = _git_head(workspace)
    count_before = _git_count(workspace)

    svc = CheckpointService(workspace)
    (workspace / "b.py").write_text("x = 1\n", encoding="utf-8")
    cid, changed = await svc.commit_turn("snap")
    assert cid is not None
    assert "b.py" in changed

    assert _git_head(workspace) == head_before, "user HEAD must be unchanged"
    assert _git_count(workspace) == count_before, "no commits added to user repo"


def _os_environ():
    import os

    return dict(os.environ)


def _git_head(cwd: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _git_count(cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class _ToolLoopProvider(LLMProvider):
    """Always asks to write the same file — never finishes, forcing max-iter."""

    def __init__(self, path: str = "a.py") -> None:
        super().__init__(api_key="test")
        self._path = path

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
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name="write_file",
                    arguments={"path": self._path, "content": "x = 1\n"},
                )
            ],
            finish_reason="tool_calls",
        )

    def get_default_model(self) -> str:
        return "stub"


def _loop_agent(workspace: Path, *, checkpoint_enabled: bool) -> AgentLoop:
    return AgentLoop(
        provider=_ToolLoopProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        runtime_config=RuntimeConfig(
            checkpoint=CheckpointConfig(
                policy="always" if checkpoint_enabled else "never",
            ),
        ),
    )


async def test_max_iter_interrupted_with_checkpoint(workspace):
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    final, _used, _msgs, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "interrupted"

    assert "maximum number of tool call iterations" in (final or "")

    assert outcome.checkpoint_id is not None
    assert "a.py" in outcome.edited_files


async def test_max_iter_baseline_preserved_when_disabled(workspace):
    agent = _loop_agent(workspace, checkpoint_enabled=False)
    final, _used, _msgs, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )

    assert outcome.status == "interrupted"
    assert "maximum number of tool call iterations" in (final or "")
    assert outcome.checkpoint_id is None
    assert outcome.edited_files == []


class _WriteThenStopProvider(LLMProvider):
    """First turn writes a file, then stops — a normal completion that also
    left an edit to snapshot."""

    def __init__(self) -> None:
        super().__init__(api_key="test")
        self._n = 0

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
        self._n += 1
        if self._n == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="c1",
                        name="write_file",
                        arguments={"path": "done.py", "content": "ok\n"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="all done", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


class _ErrorProvider(LLMProvider):
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
        return LLMResponse(content="fatal model error: bad request", finish_reason="error")

    def get_default_model(self) -> str:
        return "stub"


async def test_completed_status_and_snapshot(workspace):
    agent = AgentLoop(
        provider=_WriteThenStopProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=5,
        restrict_to_workspace=True,
        runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy="always")),
    )
    final, _used, _msgs, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "completed"
    assert "all done" in (final or "")

    assert outcome.checkpoint_id is not None

    assert outcome.edited_files == []


async def test_error_status(workspace):
    agent = AgentLoop(
        provider=_ErrorProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=5,
        restrict_to_workspace=True,
        runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy="always")),
    )
    _final, _used, _msgs, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "error"
    assert outcome.error_category == "unknown"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_recovery_block_injected_into_next_user_message(workspace):
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    agent._pending_recovery["sess"] = {"checkpoint_id": "abc123", "files": ["a.py", "b.py"]}

    messages = [{"role": "user", "content": "continue please"}]
    agent._inject_recovery_block("sess", messages)

    injected = messages[-1]["content"]
    assert "previous turn was interrupted" in injected.lower()
    assert "a.py" in injected and "b.py" in injected
    assert "abc123" in injected
    assert "continue please" in injected

    assert "sess" not in agent._pending_recovery


def test_recovery_block_noop_without_pending(workspace):
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    messages = [{"role": "user", "content": "hello"}]
    agent._inject_recovery_block("sess", messages)
    assert messages[-1]["content"] == "hello"


async def test_checkpoint_captures_deletion(workspace):
    """add -A is edit-source-agnostic: a deletion (or any non-tool change) is
    still snapshotted — the advantage over edit-tool-triggered approaches."""
    svc = CheckpointService(workspace)
    (workspace / "x.py").write_text("a\n", encoding="utf-8")
    await svc.commit_turn("t1")
    (workspace / "x.py").unlink()
    cid, changed = await svc.commit_turn("t2")
    assert cid is not None
    assert "x.py" in changed


def test_recovery_block_injects_into_list_content(workspace):
    """Multimodal user message (content is a list) → recovery prepended as a
    leading text block."""
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    agent._pending_recovery["s"] = {"checkpoint_id": "cid9", "files": ["a.py"]}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    ]
    agent._inject_recovery_block("s", messages)
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "interrupted" in content[0]["text"].lower()

    assert content[1]["text"] == "hi"


def test_recovery_block_kept_when_last_not_user(workspace):
    """If the last message isn't the user turn, don't inject and don't lose the
    pending recovery — keep it for the next (user-terminated) assembly."""
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    agent._pending_recovery["s"] = {"checkpoint_id": "c", "files": ["a.py"]}
    messages = [{"role": "assistant", "content": "x"}]
    agent._inject_recovery_block("s", messages)
    assert messages[-1]["content"] == "x"
    assert "s" in agent._pending_recovery


async def test_recovery_flows_interrupt_to_next_assembly(workspace):
    """End-to-end: an interrupted turn stashes recovery (as the caller does),
    and the NEXT context assembly injects it into the user message — exercising
    the real stash -> _assemble_context_messages -> inject wiring."""
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    key = "tui:default"

    _f, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "interrupted"
    agent._stash_recovery(key, outcome)
    assert key in agent._pending_recovery

    session = agent.sessions.get_or_create(key)
    messages = await agent._assemble_context_messages(
        session=session,
        session_key=key,
        current_message="continue",
    )
    last_user = messages[-1]
    assert last_user["role"] == "user"
    text = (
        last_user["content"]
        if isinstance(last_user["content"], str)
        else next(b["text"] for b in last_user["content"] if b.get("type") == "text")
    )
    assert "interrupted" in text.lower()
    assert "continue" in text


async def test_recovery_consumed_once(workspace):
    """A third assembly (after consumption) must not re-inject."""
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    key = "tui:default"
    agent._pending_recovery[key] = {"checkpoint_id": "c", "files": ["a.py"]}
    session = agent.sessions.get_or_create(key)

    m1 = await agent._assemble_context_messages(
        session=session,
        session_key=key,
        current_message="first",
    )
    t1 = m1[-1]["content"]
    assert "interrupted" in (t1 if isinstance(t1, str) else str(t1)).lower()

    m2 = await agent._assemble_context_messages(
        session=session,
        session_key=key,
        current_message="second",
    )
    t2 = m2[-1]["content"]
    assert "interrupted" not in (t2 if isinstance(t2, str) else str(t2)).lower()


async def test_checkpoint_degrades_when_git_missing(workspace, monkeypatch):
    """If git is unavailable, commit_turn returns (None, []) and never raises —
    the safety net must not break the turn it protects."""
    import codeflow.agent.loop.checkpoint as ckpt_mod

    async def _boom(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(ckpt_mod.asyncio, "create_subprocess_exec", _boom)
    svc = ckpt_mod.CheckpointService(workspace)
    (workspace / "a.py").write_text("x\n", encoding="utf-8")
    cid, changed = await svc.commit_turn("t1")
    assert cid is None
    assert changed == []


async def test_loop_survives_checkpoint_failure(workspace, monkeypatch):
    """Same failure, but through the loop: the turn still returns a result."""
    import codeflow.agent.loop.checkpoint as ckpt_mod

    async def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(ckpt_mod.asyncio, "create_subprocess_exec", _boom)
    agent = _loop_agent(workspace, checkpoint_enabled=True)
    final, _u, _m, outcome = await agent._run_agent_loop(
        [{"role": "user", "content": "go"}],
    )
    assert outcome.status == "interrupted"
    assert outcome.checkpoint_id is None
